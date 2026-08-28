"""Artifact-only orchestration tests for the canonical Task 7 report driver."""

from __future__ import annotations

import hashlib
import json
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
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text("{}\n", encoding="utf-8")
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
    output = tmp_path / "analysis.json"
    validated = task7_report_driver.compile_task7_report_from_shards(
        shard_root=root,
        fixture_path=fixture,
        fixture_sha256=fixture_sha,
        gate_path=gate,
        gate_sha256=gate_sha,
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
    assert output.is_file()
    stored = json.loads(output.read_text(encoding="utf-8"))
    assert stored == report
    unsigned = dict(stored)
    assert unsigned.pop("canonical_report_sha256") == _canonical_sha256(unsigned)
    with pytest.raises(FileExistsError):
        task7_report_driver.compile_task7_report_from_shards(
            shard_root=root,
            fixture_path=fixture,
            fixture_sha256=fixture_sha,
            gate_path=gate,
            gate_sha256=gate_sha,
            output_path=output,
            draws=4,
            seed=3,
        )


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


def test_driver_never_publishes_inside_the_sealed_shard_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, fixture_sha, gate, gate_sha, root, qids = _write_authority(tmp_path)
    monkeypatch.setattr(
        task7_report_driver,
        "assemble_task7_analysis_bundle_from_artifacts",
        lambda **kwargs: _bundle(qids[0], 0),
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
