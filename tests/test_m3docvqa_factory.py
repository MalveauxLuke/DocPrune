from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from docprune.benchmark_config import (
    COLPALI_BACKBONE_MODEL,
    COLPALI_BACKBONE_REVISION,
    COLPALI_MODEL,
    COLPALI_REVISION,
    QWEN_MODEL,
)
from docprune.m3docrag import SampleInput
from docprune.m3docvqa_factory import (
    _load_index_manifest,
    _write_run_manifest,
    build_workload,
    filter_samples,
    load_completed_qids,
    validate_processor_contract_file,
)


def result_record(qid: str, *, question: str = "question", answers: list[str] | None = None):
    return {
        "question_id": qid,
        "question": question,
        "answers": [] if answers is None else answers,
        "predicted_answer": "answer",
        "retrieved_pages": [{"doc_id": "doc", "page_index": 0, "score": 1.0}],
        "trace": {
            "original_visual_tokens": 4,
            "post_btp_visual_tokens": 3,
            "post_qtp_visual_tokens": 2,
            "post_ctp_visual_tokens": 1,
            "ctp_layer": None,
        },
        "timing": {"retrieval_seconds": 0.1, "qa_seconds": 0.2},
    }


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
        json.dumps(result_record("q-1")) + "\n" + json.dumps(result_record("q-2")) + "\n"
    )
    assert load_completed_qids(path, expected_qids=("q-1", "q-2", "q-3")) == {"q-1", "q-2"}
    path.write_text(
        json.dumps(result_record("q-1")) + "\n" + json.dumps(result_record("q-1")) + "\n"
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_completed_qids(path, expected_qids=("q-1", "q-2"))
    path.write_text(json.dumps(result_record("q-x")) + "\n")
    with pytest.raises(ValueError, match="unexpected"):
        load_completed_qids(path, expected_qids=("q-1", "q-2"))

    path.write_text(json.dumps(result_record("q-1", question="changed")) + "\n")
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

    with pytest.raises(FileExistsError, match="complete run"):
        _write_run_manifest(tmp_path, payload, resume=False)
    manifest_path.unlink()
    _write_run_manifest(tmp_path, payload, resume=False)
    manifest = json.loads(manifest_path.read_text())
    manifest["run_manifest_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="digest"):
        _write_run_manifest(tmp_path, payload, resume=True)


def test_resume_requires_existing_output_and_incomplete_direct_manifests_collide(
    tmp_path: Path,
) -> None:
    payload = {"mode": "all-kept", "page_count": 1}
    with pytest.raises(FileNotFoundError, match="existing"):
        _write_run_manifest(tmp_path / "missing", payload, resume=True)
    output = tmp_path / "existing"
    output.mkdir()
    (output / "run_manifest.json").write_text(
        json.dumps({"status": "configured", "command": "evaluate", "factory": "x"})
    )
    with pytest.raises(FileExistsError, match="unrelated"):
        _write_run_manifest(output, payload, resume=False)


def test_json_run_config_is_authoritative_and_complete(tmp_path: Path, monkeypatch) -> None:
    contract = tmp_path / "contract.json"
    corpus_root = tmp_path / "corpus"
    path_values = {
        "root": str(corpus_root),
        "questions_path": str(corpus_root / "questions.jsonl"),
        "document_ids_path": str(corpus_root / "doc_ids.json"),
        "pdf_dir": str(corpus_root / "pdfs"),
        "integrity_report_path": str(corpus_root / "integrity.json"),
        "archive_checksum_manifest_path": str(corpus_root / "archives.sha256"),
        "integrity_sha256": "a" * 64,
        "archive_checksum_manifest_sha256": "b" * 64,
        "questions_sha256": "c" * 64,
        "document_ids_sha256": "d" * 64,
        "expected_question_count": 1,
        "expected_pdf_count": 1,
        "expected_page_count": 1,
        "archive_hashes": {},
        "is_fixture": True,
    }
    payload = {
        "mode": "all-kept",
        "page_count": 1,
        "runtime_commit": "e" * 40,
        "m3docrag_commit": "f" * 40,
        "resources": {
            "qwen": {"model": "qwen", "revision": "1" * 40},
            "colpali": {"model": "colpali", "revision": "2" * 40},
            "colpali_backbone": {"model": "backbone", "revision": "3" * 40},
        },
        "processor_contract_path": str(contract),
        "corpus": path_values,
        "generation": {
            "max_new_tokens": 128,
            "do_sample": False,
            "num_beams": 1,
            "prompt": "question: $question\noutput only answer.",
        },
    }
    contract.write_text("{}")
    path = tmp_path / "run-config.json"
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(
        "docprune.m3docvqa_factory.BenchmarkRunConfig.from_env",
        lambda *args, **kwargs: pytest.fail("authoritative JSON must not consult the environment"),
    )
    from docprune.m3docvqa_factory import _resolve_run_config

    resolved = _resolve_run_config(path, mode="all-kept", page_count=1)
    assert resolved.runtime_commit == "e" * 40
    assert resolved.corpus.root == corpus_root


def test_index_manifest_schema_and_digest_are_checked_before_construction(tmp_path: Path) -> None:
    wrong_schema = tmp_path / "wrong-schema.json"
    wrong_schema.write_text(json.dumps({"schema_version": 3, "manifest_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="schema_version"):
        _load_index_manifest(wrong_schema)
    wrong_digest = tmp_path / "wrong-digest.json"
    wrong_digest.write_text(json.dumps({"schema_version": 4, "manifest_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="canonical"):
        _load_index_manifest(wrong_digest)


def test_completed_results_require_full_schema_and_regular_file(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    path.write_text(json.dumps({"question_id": "q-1"}) + "\n")
    with pytest.raises(ValueError, match="missing"):
        load_completed_qids(path, expected_qids=("q-1",))
    target = tmp_path / "target.jsonl"
    target.write_text(json.dumps(result_record("q-1")) + "\n")
    path.unlink()
    path.symlink_to(target)
    with pytest.raises(ValueError, match="regular"):
        load_completed_qids(path, expected_qids=("q-1",))


def test_completed_results_require_exact_requested_page_count(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    path.write_text(json.dumps(result_record("q-1")) + "\n")
    with pytest.raises(ValueError, match="page count"):
        load_completed_qids(path, expected_qids=("q-1",), expected_page_count=2)


def test_completed_results_canonicalize_legacy_qid_and_reject_conflicts(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    legacy = result_record("q-1")
    legacy["qid"] = legacy.pop("question_id")
    path.write_text(json.dumps(legacy) + "\n")
    assert load_completed_qids(path, expected_qids=("q-1",)) == {"q-1"}

    conflict = result_record("q-1")
    conflict["qid"] = "q-other"
    path.write_text(json.dumps(conflict) + "\n")
    with pytest.raises(ValueError, match="conflicting"):
        load_completed_qids(path, expected_qids=("q-1",))


def test_processor_contract_resources_are_exactly_pinned(tmp_path: Path) -> None:
    path = tmp_path / "contract.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "resources": {
                    "qwen": {"model": QWEN_MODEL, "revision": "wrong"},
                    "colpali": {"model": COLPALI_MODEL, "revision": COLPALI_REVISION},
                    "colpali_backbone": {
                        "model": COLPALI_BACKBONE_MODEL,
                        "revision": COLPALI_BACKBONE_REVISION,
                    },
                },
                "mapping_checks": {
                    "colpali_visual_grid_inferred": True,
                    "qwen_merge_groups_valid": True,
                    "raster_order_verified": True,
                },
            }
        )
    )
    with pytest.raises(ValueError, match="resources"):
        validate_processor_contract_file(path)


def test_build_workload_rejects_unknown_operation_without_loading_models() -> None:
    with pytest.raises(ValueError, match="operation"):
        build_workload(
            operation="unknown",
            config=SimpleNamespace(),
            page_count=1,
            output=Path("/tmp/unused"),
        )
