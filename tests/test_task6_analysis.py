from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

_FIXTURE_SHA = "1" * 64
_GATE_SHA = "2" * 64


def _selection(policy: dict[str, object], *, random: bool) -> dict[str, object]:
    value: dict[str, object] = {
        "policy": policy,
        "visual_population": 10,
        "geometry_count": 10,
        "geometry_sha256": "3" * 64,
        "prefill_cache_lengths": [20, 20],
        "retained_mrope_position_shape": [3, 1, 20],
        "retained_mrope_position_sha256": "4" * 64,
        "requested_budget": 6,
        "achieved_budget": 6,
        "boundary": "B_1",
    }
    if random:
        value["seed_sha256"] = "5" * 64
    return value


def _write_shard(
    root: Path, shard: int, qid: str, kind: str, *, gate_sha256: str = _GATE_SHA
) -> None:
    from docprune.task6_runtime import task6_policy_matrix

    cells = task6_policy_matrix(kind)
    shard_dir = root / f"shard-{shard:04d}"
    shard_dir.mkdir(parents=True)
    rows = []
    for index, cell in enumerate(cells):
        random = cell.policy.family in {"random-top-m", "coverage-top-m"}
        predicted = "gold"
        if random and (cell.repetition or 0) % 2:
            predicted = "wrong"
        rows.append(
            {
                "matrix_cell": index,
                "matrix_kind": kind,
                "question_id": qid,
                "question": f"question {qid}",
                "answers": ["gold"],
                "predicted_answer": predicted,
                "retrieved_pages": [["doc", 0]],
                "fixed_page_fixture_sha256": _FIXTURE_SHA,
                "fixed_page_provenance": True,
                "global_index_loaded": False,
                "policy_context": {
                    "experiment_version": f"task6-{kind}-v1" if random else None,
                    "repetition": cell.repetition,
                },
                "policy_selection": _selection(cell.policy.to_dict(), random=random),
            }
        )
    manifest = {
        "matrix_kind": kind,
        "qid": qid,
        "cell_count": len(cells),
        "fixture_sha256": _FIXTURE_SHA,
        "gate_sha256": gate_sha256,
        "cells": [
            {
                "cell": index,
                "policy": cell.policy.to_dict(),
                "experiment_version": (
                    f"task6-{kind}-v1"
                    if cell.policy.family in {"random-top-m", "coverage-top-m"}
                    else None
                ),
                "repetition": cell.repetition,
            }
            for index, cell in enumerate(cells)
        ],
    }
    manifest["run_manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "utf-8"
        )
    ).hexdigest()
    (shard_dir / "run_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    (shard_dir / "results.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )


def _gate(qids: list[str]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "sealed-development-gate",
        "fixture_sha256": _FIXTURE_SHA,
        "fixed_page_provenance": True,
        "global_index_loaded": False,
        "qid_shards": [{"shard": index, "qid": qid} for index, qid in enumerate(qids)],
    }


def test_combined_analysis_uses_all_twenty_masks_and_authenticates_members(
    tmp_path: Path,
) -> None:
    from docprune.task6_analysis import analyze_task6_native_extension

    native, extension = tmp_path / "native", tmp_path / "extension"
    qids = ["q0", "q1"]
    for shard, qid in enumerate(qids):
        _write_shard(native, shard, qid, "native")
        _write_shard(extension, shard, qid, "native-extension")

    report = analyze_task6_native_extension(
        native_root=native,
        extension_root=extension,
        gate=_gate(qids),
        gate_sha256=_GATE_SHA,
        fixture_sha256=_FIXTURE_SHA,
        native_job_id="base-job",
        extension_job_id="extension-job",
        draws=200,
        seed=7,
    )

    assert report["status"] == "admitted-development-r20"
    assert report["admission"] == {
        "native_rows": 90,
        "extension_rows": 80,
        "shards": 2,
        "member_file_count": 8,
        "fixed_page_provenance": True,
        "global_index_loaded": False,
        "member_digest_sha256": report["admission"]["member_digest_sha256"],
    }
    assert len(report["admission"]["member_digest_sha256"]) == 64
    assert report["policy_summary"]["global-uniform-random"]["rows"] == 40
    assert report["policy_summary"]["global-uniform-random"]["f1"] == 50.0
    assert report["random_calibrations"]["global-uniform-random"]["initial_repetitions"] == 10
    assert report["random_calibrations"]["global-uniform-random"]["observed_repetitions"] == 20
    assert report["random_calibrations"]["global-uniform-random"]["frozen_repetitions"] == 20
    assert report["primary_r20"]["draws"] == 200
    assert "bootstrap_draws" not in report["primary_r20"]
    assert report["job_ids"] == {"native": "base-job", "extension": "extension-job"}
    assert len(report["analysis_sha256"]) == 64


def test_combined_analysis_rejects_extension_repetition_drift(tmp_path: Path) -> None:
    from docprune.task6_analysis import analyze_task6_native_extension

    native, extension = tmp_path / "native", tmp_path / "extension"
    _write_shard(native, 0, "q0", "native")
    _write_shard(extension, 0, "q0", "native-extension")
    path = extension / "shard-0000" / "results.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["policy_context"]["repetition"] = 9
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    with pytest.raises(ValueError, match="policy context"):
        analyze_task6_native_extension(
            native_root=native,
            extension_root=extension,
            gate=_gate(["q0"]),
            gate_sha256=_GATE_SHA,
            fixture_sha256=_FIXTURE_SHA,
            native_job_id="base-job",
            extension_job_id="extension-job",
            draws=20,
            seed=7,
        )


def test_combined_analysis_rejects_symlinked_member(tmp_path: Path) -> None:
    from docprune.task6_analysis import analyze_task6_native_extension

    native, extension = tmp_path / "native", tmp_path / "extension"
    _write_shard(native, 0, "q0", "native")
    _write_shard(extension, 0, "q0", "native-extension")
    member = extension / "shard-0000" / "results.jsonl"
    target = tmp_path / "target.jsonl"
    member.rename(target)
    member.symlink_to(target)

    with pytest.raises(ValueError, match="regular member"):
        analyze_task6_native_extension(
            native_root=native,
            extension_root=extension,
            gate=_gate(["q0"]),
            gate_sha256=_GATE_SHA,
            fixture_sha256=_FIXTURE_SHA,
            native_job_id="base-job",
            extension_job_id="extension-job",
            draws=20,
            seed=7,
        )


def test_combined_analysis_rejects_tampered_run_manifest_cells(tmp_path: Path) -> None:
    from docprune.task6_analysis import analyze_task6_native_extension

    native, extension = tmp_path / "native", tmp_path / "extension"
    _write_shard(native, 0, "q0", "native")
    _write_shard(extension, 0, "q0", "native-extension")
    path = extension / "shard-0000" / "run_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["cells"][0]["repetition"] = 9
    path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="manifest (digest|cells)"):
        analyze_task6_native_extension(
            native_root=native,
            extension_root=extension,
            gate=_gate(["q0"]),
            gate_sha256=_GATE_SHA,
            fixture_sha256=_FIXTURE_SHA,
            native_job_id="base-job",
            extension_job_id="extension-job",
            draws=20,
            seed=7,
        )


def test_combined_analysis_publication_is_no_replace(tmp_path: Path) -> None:
    from docprune.task6_analysis import publish_task6_analysis

    output = tmp_path / "analysis.json"
    report = {"schema_version": 1, "status": "admitted-development-r20"}

    digest = publish_task6_analysis(report, output)

    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert digest == hashlib.sha256(output.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        publish_task6_analysis(report, output)


def test_analysis_cli_authenticates_gate_and_publishes_report(tmp_path: Path) -> None:
    native, extension = tmp_path / "native", tmp_path / "extension"
    gate_path = tmp_path / "gate.json"
    qids = ["q0", "q1"]
    gate_path.write_text(json.dumps(_gate(qids), sort_keys=True) + "\n", encoding="utf-8")
    gate_sha256 = hashlib.sha256(gate_path.read_bytes()).hexdigest()
    for shard, qid in enumerate(qids):
        _write_shard(native, shard, qid, "native", gate_sha256=gate_sha256)
        _write_shard(extension, shard, qid, "native-extension", gate_sha256=gate_sha256)
    output = tmp_path / "analysis.json"

    completed = subprocess.run(
        [
            sys.executable,
            "examples/analyze_task6_extension.py",
            "--native-root",
            str(native),
            "--extension-root",
            str(extension),
            "--gate-manifest",
            str(gate_path),
            "--gate-manifest-sha256",
            gate_sha256,
            "--fixture-sha256",
            _FIXTURE_SHA,
            "--native-job-id",
            "base-job",
            "--extension-job-id",
            "extension-job",
            "--draws",
            "20",
            "--output",
            str(output),
        ],
        cwd=Path(__file__).parents[1],
        env={"PYTHONPATH": str(Path(__file__).parents[1] / "src")},
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == ("admitted-development-r20")
