from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from docprune.m3docrag import SampleInput
from docprune.m3docvqa_factory import (
    _write_run_manifest,
    build_workload,
    filter_samples,
    load_completed_qids,
    validate_processor_contract_file,
)


def test_filter_samples_is_deterministic_and_rejects_duplicate_requested_ids() -> None:
    samples = tuple(SampleInput(f"q-{i}", f"question-{i}") for i in range(4))

    assert [sample.question_id for sample in filter_samples(samples, limit=2)] == ["q-0", "q-1"]
    assert [
        sample.question_id for sample in filter_samples(samples, sample_ids=("q-3", "q-1"))
    ] == [
        "q-1",
        "q-3",
    ]
    with pytest.raises(ValueError, match="duplicate"):
        filter_samples(samples, sample_ids=("q-1", "q-1"))


def test_load_completed_qids_rejects_duplicate_and_unknown_records(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    path.write_text(
        json.dumps({"question_id": "q-1"}) + "\n" + json.dumps({"question_id": "q-2"}) + "\n"
    )
    assert load_completed_qids(path, expected_qids=("q-1", "q-2", "q-3")) == {"q-1", "q-2"}
    path.write_text(
        json.dumps({"question_id": "q-1"}) + "\n" + json.dumps({"question_id": "q-1"}) + "\n"
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_completed_qids(path, expected_qids=("q-1", "q-2"))
    path.write_text(json.dumps({"question_id": "q-x"}) + "\n")
    with pytest.raises(ValueError, match="unexpected"):
        load_completed_qids(path, expected_qids=("q-1", "q-2"))

    path.write_text(json.dumps({"question_id": "q-1", "question": "changed", "answers": []}) + "\n")
    with pytest.raises(ValueError, match="question drift"):
        load_completed_qids(
            path,
            expected_samples=(SampleInput("q-1", "original", ("answer",)),),
        )


def test_validate_processor_contract_file_is_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "contract.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "mapping_checks": {"raster_order_verified": True},
            }
        )
    )
    with pytest.raises(ValueError, match="structural"):
        validate_processor_contract_file(path)


def test_resume_manifest_requires_complete_identity_and_valid_digest(tmp_path: Path) -> None:
    payload = {
        "schema_version": 2,
        "status": "configured",
        "operation": "evaluate",
        "output": str(tmp_path.resolve()),
        "mode": "all-kept",
        "page_count": 1,
        "runtime_commit": "a" * 40,
        "m3docrag_commit": "b" * 40,
        "resources": {"qwen": {"revision": "c" * 40}},
        "processor_contract_path": str(tmp_path / "contract.json"),
        "processor_contract_sha256": "d" * 64,
        "corpus": {"integrity_sha256": "e" * 64},
        "generation": {"max_new_tokens": 128},
        "index_manifest": {"manifest_sha256": "f" * 64},
    }
    _write_run_manifest(tmp_path, payload, resume=False)
    manifest_path = tmp_path / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["resources"]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="missing an immutable"):
        _write_run_manifest(tmp_path, payload, resume=True)

    _write_run_manifest(tmp_path, payload, resume=False)
    manifest = json.loads(manifest_path.read_text())
    manifest["run_manifest_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="digest"):
        _write_run_manifest(tmp_path, payload, resume=True)


def test_build_workload_rejects_unknown_operation_without_loading_models() -> None:
    with pytest.raises(ValueError, match="operation"):
        build_workload(
            operation="unknown",
            config=SimpleNamespace(),
            page_count=1,
            output=Path("/tmp/unused"),
        )
