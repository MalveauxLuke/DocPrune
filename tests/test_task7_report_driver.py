"""Artifact-only orchestration tests for the canonical Task 7 report driver."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from docprune import task7_report_driver


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _write_authority(tmp_path: Path) -> tuple[Path, str, Path, str, Path, tuple[str, ...]]:
    eligible_path = tmp_path / "eligible.jsonl"
    eligible_path.write_text('{"qid":"unused"}\n', encoding="utf-8")
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(
        json.dumps(
            {
                "eligible_questions_path": str(eligible_path),
                "eligible_questions_sha256": hashlib.sha256(eligible_path.read_bytes()).hexdigest(),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    fixture_sha256 = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
    qids = tuple(f"q-{index:02d}" for index in range(64))
    gate_path = tmp_path / "gate.json"
    gate = {
        "schema_version": 1,
        "status": "sealed-development-gate",
        "fixture_path": str(fixture_path),
        "fixture_sha256": fixture_sha256,
        "fixed_page_provenance": True,
        "global_index_loaded": False,
        "qid_shards": [{"shard": index, "qid": qid} for index, qid in enumerate(qids)],
    }
    gate_path.write_text(json.dumps(gate, sort_keys=True) + "\n", encoding="utf-8")
    gate_sha256 = hashlib.sha256(gate_path.read_bytes()).hexdigest()
    shard_root = tmp_path / "shards"
    shard_root.mkdir()
    for index, qid in enumerate(qids):
        shard = shard_root / f"shard-{index:04d}"
        shard.mkdir()
        manifest = {
            "qid": qid,
            "shard": index,
            "gate_manifest_path": str(gate_path),
            "gate_manifest_sha256": gate_sha256,
        }
        (shard / "run_manifest.json").write_text(
            json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
        )
        (shard / "results.jsonl").write_text(f'{{"qid":"{qid}"}}\n', encoding="utf-8")
        (shard / "task7-likelihood.jsonl").write_text(f'{{"qid":"{qid}"}}\n', encoding="utf-8")
    return fixture_path, fixture_sha256, gate_path, gate_sha256, shard_root, qids


def _write_scheduler_logs(root: Path, job_id: str) -> None:
    for index in range(64):
        for suffix in ("out", "err"):
            (root / f"slurm-docprune-task6-l40s-{job_id}_{index}.{suffix}").write_text(
                f"scheduler record {index} {suffix}\n", encoding="utf-8"
            )


def _bundle(qid: str, marker: int) -> dict[str, object]:
    digest = f"{marker % 16:x}" * 64
    provenance = {
        "fixture_sha256": "a" * 64,
        "run_manifest_file_sha256": digest,
        "run_manifest_sha256": "b" * 64,
        "results_file_sha256": "c" * 64,
        "likelihood_file_sha256": "d" * 64,
        "reference_result_sha256": "e" * 64,
        "input_all_drop_result_sha256": "f" * 64,
    }
    curve = {
        "qid": qid,
        "reference_f1": 50.0,
        "all_drop_f1": {
            boundary: 50.0 for boundary in ("B_input", "B_0", "B_6", "B_13", "B_20", "B_23", "B_26")
        },
    }
    opportunity = {
        "qid": qid,
        "supporting_document_ids": [f"doc-{qid}"],
        "retrieved_document_ids": [f"doc-{qid}", "d2", "d3", "d4"],
        "reference": {
            "name": "btp-qtp-no-ctp",
            "result_sha256": "e" * 64,
            "em_correct": True,
            "f1": 50.0,
        },
        "input_all_drop": {
            "name": "all-visual-drop-B_input",
            "boundary": "B_input",
            "mode": "physical_delete",
            "retained_visual_ids": [],
            "result_sha256": "f" * 64,
            "em_correct": True,
            "best_reference_loglikelihood_drop_per_token": 0.0,
            "likelihood_target": "best-reference-full-gold-sequence",
            "likelihood_target_sha256": "1" * 64,
        },
    }
    value: dict[str, object] = {
        "schema_version": 1,
        "qid": qid,
        "curve": {"provenance": dict(provenance), "row": curve},
        "opportunity": {"provenance": dict(provenance), "row": opportunity},
    }
    value["analysis_bundle_sha256"] = _canonical_sha256(value)
    return value


def _write_member_authority(
    tmp_path: Path,
    *,
    fixture_sha256: str,
    gate_sha256: str,
    shard_root: Path,
    qids: tuple[str, ...],
) -> tuple[Path, str]:
    members = []
    for index, qid in enumerate(qids):
        shard = shard_root / f"shard-{index:04d}"
        members.append(
            {
                "shard": index,
                "qid": qid,
                "files": {
                    name: hashlib.sha256((shard / name).read_bytes()).hexdigest()
                    for name in task7_report_driver._SHARD_MEMBERS
                },
            }
        )
    authority: dict[str, object] = {
        "schema_version": 1,
        "status": "sealed-task7-member-hashes",
        "fixture_sha256": fixture_sha256,
        "gate_sha256": gate_sha256,
        "shard_root": str(shard_root),
        "qids": list(qids),
        "members": members,
    }
    authority["member_manifest_sha256"] = _canonical_sha256(authority)
    authority_path = tmp_path / "member-hashes.json"
    authority_path.write_text(json.dumps(authority, sort_keys=True) + "\n", encoding="utf-8")
    return authority_path, hashlib.sha256(authority_path.read_bytes()).hexdigest()


def test_driver_admits_exact_sealed_order_and_publishes_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, fixture_sha, gate, gate_sha, root, qids = _write_authority(tmp_path)
    calls: list[dict[str, object]] = []

    def admit(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        index = int(Path(kwargs["run_manifest_path"]).parent.name.removeprefix("shard-"))
        return _bundle(qids[index], index)

    monkeypatch.setattr(task7_report_driver, "assemble_task7_analysis_bundle_from_artifacts", admit)
    authority, authority_sha = _write_member_authority(
        tmp_path,
        fixture_sha256=fixture_sha,
        gate_sha256=gate_sha,
        shard_root=root,
        qids=qids,
    )
    output = tmp_path / "analysis-bundle"
    validated = task7_report_driver.compile_task7_report_from_shards(
        shard_root=root,
        fixture_path=fixture,
        fixture_sha256=fixture_sha,
        gate_path=gate,
        gate_sha256=gate_sha,
        expected_members_path=authority,
        expected_members_sha256=authority_sha,
        output_path=output,
        draws=4,
        seed=3,
        validate_only=True,
    )
    assert not output.exists()
    assert len(calls) == 64
    calls.clear()
    report = task7_report_driver.compile_task7_report_from_shards(
        shard_root=root,
        fixture_path=fixture,
        fixture_sha256=fixture_sha,
        gate_path=gate,
        gate_sha256=gate_sha,
        expected_members_path=authority,
        expected_members_sha256=authority_sha,
        output_path=output,
        draws=4,
        seed=3,
    )

    assert len(calls) == 64
    assert report == validated
    assert calls[0]["fixture_sha256"] == fixture_sha
    assert (
        calls[0]["run_manifest_file_sha256"]
        == hashlib.sha256((root / "shard-0000" / "run_manifest.json").read_bytes()).hexdigest()
    )
    assert (
        calls[0]["results_sha256"]
        == hashlib.sha256((root / "shard-0000" / "results.jsonl").read_bytes()).hexdigest()
    )
    assert report["qids"] == list(qids)
    assert report["member_count"] == 64
    assert report["analysis"]["curve"]["draw_count"] == 4
    assert report["fixed_page_provenance"] is True
    assert report["global_index_loaded"] is False
    assert output.is_dir()
    stored = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert stored == report
    for identity_name in ("fixture", "gate", "eligible_questions"):
        identity = report[identity_name]
        assert not Path(identity["path"]).is_absolute()
        assert (
            hashlib.sha256((output / identity["path"]).read_bytes()).hexdigest()
            == identity["sha256"]
        )
    member_authority = report["member_hash_authority"]
    assert not Path(member_authority["path"]).is_absolute()
    assert (
        hashlib.sha256((output / member_authority["path"]).read_bytes()).hexdigest()
        == member_authority["file_sha256"]
    )
    for member in report["members"]:
        assert not Path(member["path"]).is_absolute()
        for identity in member["files"].values():
            assert not Path(identity["path"]).is_absolute()
            assert (
                hashlib.sha256((output / identity["path"]).read_bytes()).hexdigest()
                == identity["sha256"]
            )
    unsigned = dict(stored)
    assert unsigned.pop("canonical_report_sha256") == _canonical_sha256(unsigned)
    with pytest.raises(FileExistsError):
        task7_report_driver.compile_task7_report_from_shards(
            shard_root=root,
            fixture_path=fixture,
            fixture_sha256=fixture_sha,
            gate_path=gate,
            gate_sha256=gate_sha,
            expected_members_path=authority,
            expected_members_sha256=authority_sha,
            output_path=output,
            draws=4,
            seed=3,
        )


def test_member_sealer_accepts_only_exact_job_bound_scheduler_logs(tmp_path: Path) -> None:
    """Catch rejecting the launcher-prescribed Slurm namespace or allowing unrelated extras."""

    fixture, fixture_sha, gate, gate_sha, root, _ = _write_authority(tmp_path)
    _write_scheduler_logs(root, "62314816")
    output = tmp_path / "member-hashes.json"

    authority, _ = task7_report_driver.seal_task7_member_hash_authority(
        shard_root=root,
        fixture_path=fixture,
        fixture_sha256=fixture_sha,
        gate_path=gate,
        gate_sha256=gate_sha,
        output_path=output,
        scheduler_job_id="62314816",
    )

    assert len(authority["members"]) == 64
    (root / "unrelated.log").write_text("must fail\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing or extra entries"):
        task7_report_driver._require_exact_shard_tree(root, scheduler_job_id="62314816")


@pytest.mark.parametrize("mutation", ["missing", "extra", "qid", "symlink"])
def test_driver_rejects_incomplete_extra_substituted_or_misordered_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    fixture, fixture_sha, gate, gate_sha, root, qids = _write_authority(tmp_path)
    if mutation == "missing":
        (root / "shard-0063").rename(tmp_path / "missing-shard")
    elif mutation == "extra":
        (root / "unexpected").mkdir()
    elif mutation == "qid":
        manifest_path = root / "shard-0001" / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["qid"] = qids[0]
        manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    else:
        results = root / "shard-0001" / "results.jsonl"
        target = tmp_path / "substituted-results.jsonl"
        results.rename(target)
        results.symlink_to(target)

    monkeypatch.setattr(
        task7_report_driver,
        "assemble_task7_analysis_bundle_from_artifacts",
        lambda **kwargs: _bundle(
            qids[int(Path(kwargs["run_manifest_path"]).parent.name.removeprefix("shard-"))],
            int(Path(kwargs["run_manifest_path"]).parent.name.removeprefix("shard-")),
        ),
    )
    with pytest.raises(ValueError):
        task7_report_driver.compile_task7_report_from_shards(
            shard_root=root,
            fixture_path=fixture,
            fixture_sha256=fixture_sha,
            gate_path=gate,
            gate_sha256=gate_sha,
            output_path=tmp_path / "analysis.json",
            draws=4,
            seed=3,
            validate_only=True,
        )
    assert not (tmp_path / "analysis.json").exists()


def test_driver_rejects_a_member_that_differs_from_explicit_hash_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, fixture_sha, gate, gate_sha, root, qids = _write_authority(tmp_path)
    members = []
    for index, qid in enumerate(qids):
        shard = root / f"shard-{index:04d}"
        members.append(
            {
                "shard": index,
                "qid": qid,
                "files": {
                    name: hashlib.sha256((shard / name).read_bytes()).hexdigest()
                    for name in (
                        "run_manifest.json",
                        "results.jsonl",
                        "task7-likelihood.jsonl",
                    )
                },
            }
        )
    members[7]["files"]["results.jsonl"] = "0" * 64
    authority: dict[str, object] = {
        "schema_version": 1,
        "status": "sealed-task7-member-hashes",
        "fixture_sha256": fixture_sha,
        "gate_sha256": gate_sha,
        "shard_root": str(root),
        "qids": list(qids),
        "members": members,
    }
    authority["member_manifest_sha256"] = _canonical_sha256(authority)
    authority_path = tmp_path / "member-hashes.json"
    authority_path.write_text(json.dumps(authority, sort_keys=True) + "\n", encoding="utf-8")
    authority_file_sha = hashlib.sha256(authority_path.read_bytes()).hexdigest()
    monkeypatch.setattr(
        task7_report_driver,
        "assemble_task7_analysis_bundle_from_artifacts",
        lambda **kwargs: _bundle(
            qids[int(Path(kwargs["run_manifest_path"]).parent.name.removeprefix("shard-"))],
            int(Path(kwargs["run_manifest_path"]).parent.name.removeprefix("shard-")),
        ),
    )

    with pytest.raises(ValueError, match="explicit member hash"):
        task7_report_driver.compile_task7_report_from_shards(
            shard_root=root,
            fixture_path=fixture,
            fixture_sha256=fixture_sha,
            gate_path=gate,
            gate_sha256=gate_sha,
            expected_members_path=authority_path,
            expected_members_sha256=authority_file_sha,
            output_path=tmp_path / "analysis.json",
            draws=4,
            seed=3,
            validate_only=True,
        )


def test_report_and_sealer_reject_a_fifo_member_without_a_blocking_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, fixture_sha, gate, gate_sha, root, qids = _write_authority(tmp_path)
    authority, authority_sha = _write_member_authority(
        tmp_path,
        fixture_sha256=fixture_sha,
        gate_sha256=gate_sha,
        shard_root=root,
        qids=qids,
    )
    fifo = root / "shard-0000" / "results.jsonl"
    fifo.unlink()
    os.mkfifo(fifo)
    original_open = task7_report_driver.os.open

    def require_nonblocking_leaf(path: object, flags: int, *args: object, **kwargs: object) -> int:
        if path == "results.jsonl" and not flags & os.O_NONBLOCK:
            raise AssertionError("FIFO leaf was opened without O_NONBLOCK")
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(task7_report_driver.os, "open", require_nonblocking_leaf)

    with pytest.raises(ValueError, match="regular file"):
        task7_report_driver.compile_task7_report_from_shards(
            shard_root=root,
            fixture_path=fixture,
            fixture_sha256=fixture_sha,
            gate_path=gate,
            gate_sha256=gate_sha,
            expected_members_path=authority,
            expected_members_sha256=authority_sha,
            output_path=tmp_path / "analysis-bundle",
            validate_only=True,
        )
    with pytest.raises(ValueError, match="regular file"):
        task7_report_driver.seal_task7_member_hash_authority(
            shard_root=root,
            fixture_path=fixture,
            fixture_sha256=fixture_sha,
            gate_path=gate,
            gate_sha256=gate_sha,
            output_path=tmp_path / "authority.json",
        )


def test_driver_never_publishes_inside_the_sealed_shard_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, fixture_sha, gate, gate_sha, root, qids = _write_authority(tmp_path)
    monkeypatch.setattr(
        task7_report_driver,
        "assemble_task7_analysis_bundle_from_artifacts",
        lambda **kwargs: _bundle(
            qids[int(Path(kwargs["run_manifest_path"]).parent.name.removeprefix("shard-"))],
            int(Path(kwargs["run_manifest_path"]).parent.name.removeprefix("shard-")),
        ),
    )
    with pytest.raises(ValueError, match="outside the sealed shard root"):
        task7_report_driver.compile_task7_report_from_shards(
            shard_root=root,
            fixture_path=fixture,
            fixture_sha256=fixture_sha,
            gate_path=gate,
            gate_sha256=gate_sha,
            output_path=root / "analysis.json",
            draws=4,
            seed=3,
            validate_only=True,
        )


def test_snapshot_closes_early_members_before_late_shard_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, fixture_sha, gate, gate_sha, root, qids = _write_authority(tmp_path)
    authority, authority_sha = _write_member_authority(
        tmp_path,
        fixture_sha256=fixture_sha,
        gate_sha256=gate_sha,
        shard_root=root,
        qids=qids,
    )

    def admit(**kwargs: object) -> dict[str, object]:
        index = int(Path(kwargs["run_manifest_path"]).parent.name.removeprefix("shard-"))
        if index == 63:
            (root / "shard-0000" / "results.jsonl").write_text(
                '{"qid":"q-00","replaced":true}\n', encoding="utf-8"
            )
        return _bundle(qids[index], index)

    monkeypatch.setattr(task7_report_driver, "assemble_task7_analysis_bundle_from_artifacts", admit)
    output = tmp_path / "analysis-bundle"
    report = task7_report_driver.compile_task7_report_from_shards(
        shard_root=root,
        fixture_path=fixture,
        fixture_sha256=fixture_sha,
        gate_path=gate,
        gate_sha256=gate_sha,
        expected_members_path=authority,
        expected_members_sha256=authority_sha,
        output_path=output,
        draws=4,
        seed=3,
    )
    member = report["members"][0]["files"]["results.jsonl"]
    assert hashlib.sha256((output / member["path"]).read_bytes()).hexdigest() == member["sha256"]
    assert (
        hashlib.sha256((root / "shard-0000" / "results.jsonl").read_bytes()).hexdigest()
        != (member["sha256"])
    )


def test_snapshot_remains_closed_after_analysis_compilation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, fixture_sha, gate, gate_sha, root, qids = _write_authority(tmp_path)
    authority, authority_sha = _write_member_authority(
        tmp_path,
        fixture_sha256=fixture_sha,
        gate_sha256=gate_sha,
        shard_root=root,
        qids=qids,
    )
    monkeypatch.setattr(
        task7_report_driver,
        "assemble_task7_analysis_bundle_from_artifacts",
        lambda **kwargs: _bundle(
            qids[int(Path(kwargs["run_manifest_path"]).parent.name.removeprefix("shard-"))],
            int(Path(kwargs["run_manifest_path"]).parent.name.removeprefix("shard-")),
        ),
    )
    compile_report = task7_report_driver.compile_task7_visual_state_report

    def compile_then_replace(*args: object, **kwargs: object) -> dict[str, object]:
        report = compile_report(*args, **kwargs)
        (root / "shard-0000" / "results.jsonl").write_text(
            '{"qid":"q-00","replaced":"after-analysis"}\n', encoding="utf-8"
        )
        return report

    monkeypatch.setattr(
        task7_report_driver, "compile_task7_visual_state_report", compile_then_replace
    )
    output = tmp_path / "analysis-bundle"
    report = task7_report_driver.compile_task7_report_from_shards(
        shard_root=root,
        fixture_path=fixture,
        fixture_sha256=fixture_sha,
        gate_path=gate,
        gate_sha256=gate_sha,
        expected_members_path=authority,
        expected_members_sha256=authority_sha,
        output_path=output,
        draws=4,
        seed=3,
    )
    member = report["members"][0]["files"]["results.jsonl"]
    assert hashlib.sha256((output / member["path"]).read_bytes()).hexdigest() == member["sha256"]
    assert (
        hashlib.sha256((root / "shard-0000" / "results.jsonl").read_bytes()).hexdigest()
        != (member["sha256"])
    )


def test_validate_only_builds_and_discards_snapshot_without_reserving_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, fixture_sha, gate, gate_sha, root, qids = _write_authority(tmp_path)
    monkeypatch.setattr(
        task7_report_driver,
        "assemble_task7_analysis_bundle_from_artifacts",
        lambda **kwargs: _bundle(
            qids[int(Path(kwargs["run_manifest_path"]).parent.name.removeprefix("shard-"))],
            int(Path(kwargs["run_manifest_path"]).parent.name.removeprefix("shard-")),
        ),
    )
    authority, authority_sha = _write_member_authority(
        tmp_path,
        fixture_sha256=fixture_sha,
        gate_sha256=gate_sha,
        shard_root=root,
        qids=qids,
    )
    output = tmp_path / "analysis-bundle"
    output.write_text("existing\n", encoding="utf-8")
    report = task7_report_driver.compile_task7_report_from_shards(
        shard_root=root,
        fixture_path=fixture,
        fixture_sha256=fixture_sha,
        gate_path=gate,
        gate_sha256=gate_sha,
        expected_members_path=authority,
        expected_members_sha256=authority_sha,
        output_path=output,
        draws=4,
        seed=3,
        validate_only=True,
    )
    assert report["status"] == "validated-task7-explicit-visual-state-report"
    assert output.read_text(encoding="utf-8") == "existing\n"
    assert not list(tmp_path.glob(".analysis-bundle.*"))


def test_driver_requires_explicit_member_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, fixture_sha, gate, gate_sha, root, qids = _write_authority(tmp_path)
    monkeypatch.setattr(
        task7_report_driver,
        "assemble_task7_analysis_bundle_from_artifacts",
        lambda **kwargs: _bundle(
            qids[int(Path(kwargs["run_manifest_path"]).parent.name.removeprefix("shard-"))],
            int(Path(kwargs["run_manifest_path"]).parent.name.removeprefix("shard-")),
        ),
    )

    with pytest.raises(ValueError, match="explicit member authority is required"):
        task7_report_driver.compile_task7_report_from_shards(
            shard_root=root,
            fixture_path=fixture,
            fixture_sha256=fixture_sha,
            gate_path=gate,
            gate_sha256=gate_sha,
            output_path=tmp_path / "analysis-bundle",
            draws=4,
            seed=3,
            validate_only=True,
        )


def test_outcome_blind_sealer_creates_the_explicit_authority_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, fixture_sha, gate, gate_sha, root, qids = _write_authority(tmp_path)
    authority_path = tmp_path / "sealed-member-hashes.json"

    authority, authority_file_sha256 = task7_report_driver.seal_task7_member_hash_authority(
        shard_root=root,
        fixture_path=fixture,
        fixture_sha256=fixture_sha,
        gate_path=gate,
        gate_sha256=gate_sha,
        output_path=authority_path,
    )

    assert authority_path.is_file()
    assert json.loads(authority_path.read_text(encoding="utf-8")) == authority
    assert hashlib.sha256(authority_path.read_bytes()).hexdigest() == authority_file_sha256
    assert authority["status"] == "sealed-task7-member-hashes"
    assert authority["qids"] == list(qids)
    assert len(authority["members"]) == 64
    assert authority["member_manifest_sha256"] == _canonical_sha256(
        {key: value for key, value in authority.items() if key != "member_manifest_sha256"}
    )
    for index, member in enumerate(authority["members"]):
        shard = root / f"shard-{index:04d}"
        assert member["qid"] == qids[index]
        assert member["files"] == {
            name: hashlib.sha256((shard / name).read_bytes()).hexdigest()
            for name in task7_report_driver._SHARD_MEMBERS
        }
    with pytest.raises(FileExistsError):
        task7_report_driver.seal_task7_member_hash_authority(
            shard_root=root,
            fixture_path=fixture,
            fixture_sha256=fixture_sha,
            gate_path=gate,
            gate_sha256=gate_sha,
            output_path=authority_path,
        )

    monkeypatch.setattr(
        task7_report_driver,
        "assemble_task7_analysis_bundle_from_artifacts",
        lambda **kwargs: _bundle(
            qids[int(Path(kwargs["run_manifest_path"]).parent.name.removeprefix("shard-"))],
            int(Path(kwargs["run_manifest_path"]).parent.name.removeprefix("shard-")),
        ),
    )
    report = task7_report_driver.compile_task7_report_from_shards(
        shard_root=root,
        fixture_path=fixture,
        fixture_sha256=fixture_sha,
        gate_path=gate,
        gate_sha256=gate_sha,
        expected_members_path=authority_path,
        expected_members_sha256=authority_file_sha256,
        output_path=tmp_path / "analysis-bundle",
        draws=4,
        seed=3,
        validate_only=True,
    )
    assert report["member_hash_authority"]["kind"] == "explicit"


def test_member_authority_sealer_reauthenticates_destination_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, fixture_sha, gate, gate_sha, root, _ = _write_authority(tmp_path)
    output = tmp_path / "sealed-member-hashes.json"
    original_rename = task7_report_driver._renameat2_noreplace

    def rename_then_mutate(
        source_parent_fd: int,
        source_name: str,
        destination_parent_fd: int,
        destination_name: str,
    ) -> None:
        original_rename(
            source_parent_fd,
            source_name,
            destination_parent_fd,
            destination_name,
        )
        descriptor = os.open(
            destination_name,
            os.O_WRONLY | os.O_TRUNC | os.O_NOFOLLOW,
            dir_fd=destination_parent_fd,
        )
        try:
            os.write(descriptor, b"substituted\n")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    monkeypatch.setattr(task7_report_driver, "_renameat2_noreplace", rename_then_mutate)

    with pytest.raises(RuntimeError, match="post-publication gate failed; file preserved"):
        task7_report_driver.seal_task7_member_hash_authority(
            shard_root=root,
            fixture_path=fixture,
            fixture_sha256=fixture_sha,
            gate_path=gate,
            gate_sha256=gate_sha,
            output_path=output,
        )

    assert output.read_bytes() == b"substituted\n"


def test_failed_member_authority_stage_is_preserved_without_name_unlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed private stage must not delete a replacement through its name."""

    output = tmp_path / "sealed-member-hashes.json"
    content = b'{"status":"staged"}\n'

    def fail_file_fsync(descriptor: int) -> None:
        raise OSError("injected pre-rename fsync failure")

    def forbid_name_unlink(*args: object, **kwargs: object) -> None:
        raise AssertionError("failed publication must not unlink a staged name")

    monkeypatch.setattr(task7_report_driver.os, "fsync", fail_file_fsync)
    monkeypatch.setattr(task7_report_driver.os, "unlink", forbid_name_unlink)

    with pytest.raises(OSError, match="injected pre-rename fsync failure"):
        task7_report_driver._publish_regular_file_noreplace(
            output,
            content,
            "Task 7 member authority",
        )

    staged = list(tmp_path.glob(".sealed-member-hashes.json.*"))
    assert len(staged) == 1
    assert staged[0].read_bytes() == content
    assert not output.exists()


def test_snapshot_uses_relative_authority_and_survives_source_mutation_at_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, fixture_sha, gate, gate_sha, root, qids = _write_authority(tmp_path)
    authority, authority_sha = _write_member_authority(
        tmp_path,
        fixture_sha256=fixture_sha,
        gate_sha256=gate_sha,
        shard_root=root,
        qids=qids,
    )
    monkeypatch.setattr(
        task7_report_driver,
        "assemble_task7_analysis_bundle_from_artifacts",
        lambda **kwargs: _bundle(
            qids[int(Path(kwargs["run_manifest_path"]).parent.name.removeprefix("shard-"))],
            int(Path(kwargs["run_manifest_path"]).parent.name.removeprefix("shard-")),
        ),
    )
    source = root / "shard-0000" / "results.jsonl"
    source_bytes = source.read_bytes()
    original_rename = task7_report_driver._renameat2_noreplace

    def mutate_source_then_publish(*args: object) -> None:
        source.write_text('{"qid":"q-00","changed":"at-publish"}\n', encoding="utf-8")
        original_rename(*args)

    monkeypatch.setattr(task7_report_driver, "_renameat2_noreplace", mutate_source_then_publish)
    output = tmp_path / "analysis-bundle"
    report = task7_report_driver.compile_task7_report_from_shards(
        shard_root=root,
        fixture_path=fixture,
        fixture_sha256=fixture_sha,
        gate_path=gate,
        gate_sha256=gate_sha,
        expected_members_path=authority,
        expected_members_sha256=authority_sha,
        output_path=output,
        draws=4,
        seed=3,
    )

    member = report["members"][0]["files"]["results.jsonl"]
    assert member["path"] == "shards/shard-0000/results.jsonl"
    assert member["historical_locator"] == str(source)
    assert (output / member["path"]).read_bytes() == source_bytes
    assert hashlib.sha256((output / member["path"]).read_bytes()).hexdigest() == member["sha256"]
    assert report["fixture"]["path"] == "inputs/fixture.json"
    assert report["fixture"]["historical_locator"] == str(fixture)
    assert report["member_hash_authority"]["path"] == "inputs/member-authority.json"
    assert report["locator_semantics"] == {
        "authoritative": "bundle-relative-paths-with-pinned-sha256",
        "embedded_absolute_paths": "historical-locators-only",
    }


def test_atomic_publication_never_replaces_a_competing_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, fixture_sha, gate, gate_sha, root, qids = _write_authority(tmp_path)
    authority, authority_sha = _write_member_authority(
        tmp_path,
        fixture_sha256=fixture_sha,
        gate_sha256=gate_sha,
        shard_root=root,
        qids=qids,
    )
    monkeypatch.setattr(
        task7_report_driver,
        "assemble_task7_analysis_bundle_from_artifacts",
        lambda **kwargs: _bundle(
            qids[int(Path(kwargs["run_manifest_path"]).parent.name.removeprefix("shard-"))],
            int(Path(kwargs["run_manifest_path"]).parent.name.removeprefix("shard-")),
        ),
    )
    original_rename = task7_report_driver._renameat2_noreplace

    def publish_competitor_then_attempt_rename(
        source_parent_fd: int,
        source_name: str,
        destination_parent_fd: int,
        destination_name: str,
    ) -> None:
        os.mkdir(destination_name, dir_fd=destination_parent_fd)
        competitor_fd = os.open(
            destination_name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=destination_parent_fd,
        )
        try:
            descriptor = os.open(
                "owner.txt",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=competitor_fd,
            )
            try:
                os.write(descriptor, b"competitor\n")
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.fsync(competitor_fd)
        finally:
            os.close(competitor_fd)
        original_rename(
            source_parent_fd,
            source_name,
            destination_parent_fd,
            destination_name,
        )

    monkeypatch.setattr(
        task7_report_driver,
        "_renameat2_noreplace",
        publish_competitor_then_attempt_rename,
    )
    output = tmp_path / "analysis-bundle"

    with pytest.raises(FileExistsError):
        task7_report_driver.compile_task7_report_from_shards(
            shard_root=root,
            fixture_path=fixture,
            fixture_sha256=fixture_sha,
            gate_path=gate,
            gate_sha256=gate_sha,
            expected_members_path=authority,
            expected_members_sha256=authority_sha,
            output_path=output,
            draws=4,
            seed=3,
        )

    assert (output / "owner.txt").read_text(encoding="utf-8") == "competitor\n"
    assert not (output / "report.json").exists()
    assert not list(tmp_path.glob(".analysis-bundle.*"))


@pytest.mark.parametrize(
    "relative_path",
    (
        "inputs/gate.json",
        "inputs/member-authority.json",
        "shards/shard-0000/results.jsonl",
    ),
)
def test_final_authentication_rejects_any_staged_member_changed_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_path: str,
) -> None:
    fixture, fixture_sha, gate, gate_sha, root, qids = _write_authority(tmp_path)
    authority, authority_sha = _write_member_authority(
        tmp_path,
        fixture_sha256=fixture_sha,
        gate_sha256=gate_sha,
        shard_root=root,
        qids=qids,
    )
    monkeypatch.setattr(
        task7_report_driver,
        "assemble_task7_analysis_bundle_from_artifacts",
        lambda **kwargs: _bundle(
            qids[int(Path(kwargs["run_manifest_path"]).parent.name.removeprefix("shard-"))],
            int(Path(kwargs["run_manifest_path"]).parent.name.removeprefix("shard-")),
        ),
    )
    compile_report = task7_report_driver.compile_task7_visual_state_report

    def compile_then_mutate(*args: object, **kwargs: object) -> dict[str, object]:
        report = compile_report(*args, **kwargs)
        staging = next(tmp_path.glob(".analysis-bundle.*"))
        (staging / relative_path).write_bytes(b"substituted\n")
        return report

    monkeypatch.setattr(
        task7_report_driver,
        "compile_task7_visual_state_report",
        compile_then_mutate,
    )

    with pytest.raises(ValueError, match="snapshot"):
        task7_report_driver.compile_task7_report_from_shards(
            shard_root=root,
            fixture_path=fixture,
            fixture_sha256=fixture_sha,
            gate_path=gate,
            gate_sha256=gate_sha,
            expected_members_path=authority,
            expected_members_sha256=authority_sha,
            output_path=tmp_path / "analysis-bundle",
            draws=4,
            seed=3,
        )

    assert not (tmp_path / "analysis-bundle").exists()
    assert not list(tmp_path.glob(".analysis-bundle.*"))


def test_source_name_swap_is_detected_and_no_unverified_tree_is_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, fixture_sha, gate, gate_sha, root, qids = _write_authority(tmp_path)
    authority, authority_sha = _write_member_authority(
        tmp_path,
        fixture_sha256=fixture_sha,
        gate_sha256=gate_sha,
        shard_root=root,
        qids=qids,
    )
    monkeypatch.setattr(
        task7_report_driver,
        "assemble_task7_analysis_bundle_from_artifacts",
        lambda **kwargs: _bundle(
            qids[int(Path(kwargs["run_manifest_path"]).parent.name.removeprefix("shard-"))],
            int(Path(kwargs["run_manifest_path"]).parent.name.removeprefix("shard-")),
        ),
    )
    original_rename = task7_report_driver._renameat2_noreplace
    preserved_name = ".preserved-authenticated-staging"

    def substitute_source_name(
        source_parent_fd: int,
        source_name: str,
        destination_parent_fd: int,
        destination_name: str,
    ) -> None:
        os.rename(
            source_name,
            preserved_name,
            src_dir_fd=source_parent_fd,
            dst_dir_fd=source_parent_fd,
        )
        os.mkdir(source_name, mode=0o700, dir_fd=source_parent_fd)
        foreign_fd = os.open(
            source_name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=source_parent_fd,
        )
        try:
            descriptor = os.open(
                "foreign.txt",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=foreign_fd,
            )
            os.close(descriptor)
        finally:
            os.close(foreign_fd)
        original_rename(
            source_parent_fd,
            source_name,
            destination_parent_fd,
            destination_name,
        )

    monkeypatch.setattr(
        task7_report_driver,
        "_renameat2_noreplace",
        substitute_source_name,
    )

    with pytest.raises(RuntimeError, match="identity|preserved"):
        task7_report_driver.compile_task7_report_from_shards(
            shard_root=root,
            fixture_path=fixture,
            fixture_sha256=fixture_sha,
            gate_path=gate,
            gate_sha256=gate_sha,
            expected_members_path=authority,
            expected_members_sha256=authority_sha,
            output_path=tmp_path / "analysis-bundle",
            draws=4,
            seed=3,
        )

    assert (tmp_path / preserved_name / "report.json").is_file()
    assert (tmp_path / "analysis-bundle" / "foreign.txt").is_file()


def test_destination_swap_during_parent_fsync_cannot_return_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, fixture_sha, gate, gate_sha, root, qids = _write_authority(tmp_path)
    authority, authority_sha = _write_member_authority(
        tmp_path,
        fixture_sha256=fixture_sha,
        gate_sha256=gate_sha,
        shard_root=root,
        qids=qids,
    )
    monkeypatch.setattr(
        task7_report_driver,
        "assemble_task7_analysis_bundle_from_artifacts",
        lambda **kwargs: _bundle(
            qids[int(Path(kwargs["run_manifest_path"]).parent.name.removeprefix("shard-"))],
            int(Path(kwargs["run_manifest_path"]).parent.name.removeprefix("shard-")),
        ),
    )
    output = tmp_path / "analysis-bundle"
    preserved = tmp_path / ".preserved-after-fsync-swap"
    original_fsync = task7_report_driver.os.fsync
    swapped = False

    def fsync_then_swap(descriptor: int) -> None:
        nonlocal swapped
        original_fsync(descriptor)
        if not swapped and (output / "report.json").is_file():
            swapped = True
            output.rename(preserved)
            output.mkdir(mode=0o700)
            (output / "foreign.txt").write_text("foreign\n", encoding="utf-8")

    monkeypatch.setattr(task7_report_driver.os, "fsync", fsync_then_swap)

    with pytest.raises(RuntimeError, match="preserved"):
        task7_report_driver.compile_task7_report_from_shards(
            shard_root=root,
            fixture_path=fixture,
            fixture_sha256=fixture_sha,
            gate_path=gate,
            gate_sha256=gate_sha,
            expected_members_path=authority,
            expected_members_sha256=authority_sha,
            output_path=output,
            draws=4,
            seed=3,
        )

    assert (preserved / "report.json").is_file()
    assert (output / "foreign.txt").read_text(encoding="utf-8") == "foreign\n"


def test_failure_cleanup_preserves_an_unverified_replacement_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, fixture_sha, gate, gate_sha, root, qids = _write_authority(tmp_path)
    authority, authority_sha = _write_member_authority(
        tmp_path,
        fixture_sha256=fixture_sha,
        gate_sha256=gate_sha,
        shard_root=root,
        qids=qids,
    )
    monkeypatch.setattr(
        task7_report_driver,
        "assemble_task7_analysis_bundle_from_artifacts",
        lambda **kwargs: _bundle(
            qids[int(Path(kwargs["run_manifest_path"]).parent.name.removeprefix("shard-"))],
            int(Path(kwargs["run_manifest_path"]).parent.name.removeprefix("shard-")),
        ),
    )
    preserved = tmp_path / ".preserved-on-failure"

    def swap_then_fail(*args: object, **kwargs: object) -> dict[str, object]:
        staging = next(tmp_path.glob(".analysis-bundle.*"))
        staging.rename(preserved)
        staging.mkdir(mode=0o700)
        (staging / "foreign.txt").write_text("do not delete\n", encoding="utf-8")
        raise ValueError("forced analysis failure")

    monkeypatch.setattr(
        task7_report_driver,
        "compile_task7_visual_state_report",
        swap_then_fail,
    )

    with pytest.raises(RuntimeError, match="identity changed|preserved"):
        task7_report_driver.compile_task7_report_from_shards(
            shard_root=root,
            fixture_path=fixture,
            fixture_sha256=fixture_sha,
            gate_path=gate,
            gate_sha256=gate_sha,
            expected_members_path=authority,
            expected_members_sha256=authority_sha,
            output_path=tmp_path / "analysis-bundle",
            draws=4,
            seed=3,
        )

    replacement = next(tmp_path.glob(".analysis-bundle.*"))
    assert (replacement / "foreign.txt").read_text(encoding="utf-8") == "do not delete\n"
    assert (preserved / "inputs" / "fixture.json").is_file()


def test_report_driver_import_is_artifact_only() -> None:
    script = """
import json
import sys
import docprune.task7_report_driver
prefixes = (
    'torch',
    'transformers',
    'docprune.qwen2vl',
    'docprune.m3docrag',
    'docprune.m3docvqa_factory',
    'docprune.indexing',
)
print(json.dumps(sorted(name for name in sys.modules if name.startswith(prefixes))))
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert json.loads(completed.stdout) == []
