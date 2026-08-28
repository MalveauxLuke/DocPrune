"""Outcome-blind projection of cached full-run pages into holdout eligibility."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from docprune.experiment_design import _canonical_json_sha256
from docprune.holdout_eligibility import (
    project_cached_holdout_eligibility,
    publish_cached_holdout_eligibility,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture(tmp_path: Path) -> dict[str, object]:
    pdf_dir = tmp_path / "pdfs"
    feature_dir = tmp_path / "features"
    pdf_dir.mkdir(parents=True)
    feature_dir.mkdir(parents=True)
    ledgers = []
    for index in range(4):
        doc_id = f"doc-{index}"
        (pdf_dir / f"{doc_id}.pdf").write_bytes(f"pdf-{index}".encode())
        feature = feature_dir / f"{index:06d}.safetensors"
        feature.write_bytes(f"features-{index}".encode())
        ledgers.append(
            {
                "doc_id": doc_id,
                "document_path": str(feature),
                "sha256": _sha(feature),
                "pages": [{"doc_id": doc_id, "page_index": 0, "source_hw": [2, 2]}],
            }
        )
    completion = tmp_path / "completion.json"
    completion.write_text(json.dumps(ledgers) + "\n")
    questions = tmp_path / "questions.jsonl"
    questions.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in (
                {
                    "qid": "development-qid",
                    "question": "must not be projected",
                    "answers": [{"answer": "secret-development-answer"}],
                    "metadata": {"type": "TextQ"},
                    "supporting_context": [{"doc_id": "doc-0", "doc_part": "text"}],
                },
                {
                    "qid": "candidate-qid",
                    "question": "must not be projected either",
                    "answers": [{"answer": "secret-candidate-answer"}],
                    "metadata": {"type": "ImageQ"},
                    "supporting_context": [{"doc_id": "doc-0", "doc_part": "image"}],
                },
            )
        )
    )
    results = tmp_path / "results.jsonl"
    pages = [
        {"doc_id": f"doc-{index}", "page_index": 0, "score": 999.0 - index} for index in range(4)
    ]
    results.write_text(
        "".join(
            json.dumps(
                {
                    "question_id": qid,
                    "question": "outcome-bearing source text",
                    "answers": ["gold must stay out"],
                    "predicted_answer": "prediction must stay out",
                    "retrieved_pages": pages,
                    "timing": {"total_seconds": 1.0},
                    "trace": {"post_ctp_visual_tokens": 1},
                }
            )
            + "\n"
            for qid in ("development-qid", "candidate-qid")
        )
    )
    quality = tmp_path / "quality.json"
    quality.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scope": "quality-only",
                "mode": "docprune",
                "page_count": 4,
                "question_count": 2,
                "question_ids": ["development-qid", "candidate-qid"],
                "results_sha256": _sha(results),
            }
        )
        + "\n"
    )
    index = tmp_path / "index-manifest.json"
    index.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "mode": "docprune",
                "page_count": 4,
                "completion_ledger_path": str(completion),
                "completion_ledger_sha256": _sha(completion),
            }
        )
        + "\n"
    )
    registry = tmp_path / "development-registry.json"
    registry_payload = {
        "schema_version": 1,
        "status": "complete",
        "projection_policy": "explicit QID fields only; outcome fields forbidden",
        "union_qids": ["development-qid"],
        "union_count": 1,
    }
    registry_payload["registry_sha256"] = _canonical_json_sha256(registry_payload)
    registry.write_text(json.dumps(registry_payload) + "\n")
    return {
        "questions_path": questions,
        "questions_sha256": _sha(questions),
        "cached_results_path": results,
        "cached_results_sha256": _sha(results),
        "quality_manifest_path": quality,
        "quality_manifest_sha256": _sha(quality),
        "index_manifest_path": index,
        "index_manifest_sha256": _sha(index),
        "completion_ledger_path": completion,
        "completion_ledger_sha256": _sha(completion),
        "development_registry_path": registry,
        "development_registry_file_sha256": _sha(registry),
        "pdf_dir": pdf_dir,
    }


def test_projection_uses_cached_pages_without_outcomes_or_retrieval(tmp_path: Path) -> None:
    """Catch selecting on QA outcomes, retrieval scores, or a fresh page search."""

    inputs = _write_fixture(tmp_path)
    projection = project_cached_holdout_eligibility(**inputs)

    assert projection["status"] == "outcome-blind-cached-page-eligibility"
    assert projection["global_index_loaded"] is False
    assert projection["retrieval_run"] is False
    assert projection["development_qids_registered"] == 1
    assert projection["development_qids_excluded"] == 1
    assert projection["eligible_count"] == 1
    record = projection["eligible_records"][0]
    assert record["qid"] == "candidate-qid"
    assert record["supporting_document_ids"] == ["doc-0"]
    assert [(row["doc_id"], row["page_index"]) for row in record["cached_pages"]] == [
        (f"doc-{index}", 0) for index in range(4)
    ]
    assert [row["doc_id"] for row in record["persisted_features"]] == [
        f"doc-{index}" for index in range(4)
    ]
    serialized = json.dumps(projection, sort_keys=True)
    for forbidden in (
        "secret-candidate-answer",
        "prediction must stay out",
        "outcome-bearing source text",
        "999.0",
        "post_ctp_visual_tokens",
    ):
        assert forbidden not in serialized
    assert len(projection["projection_sha256"]) == 64


def test_projection_rejects_modified_development_registry(tmp_path: Path) -> None:
    """Catch changing the diagnostic exclusion union after its file identity was fixed."""

    inputs = _write_fixture(tmp_path)
    registry = Path(inputs["development_registry_path"])
    value = json.loads(registry.read_text())
    value["union_qids"] = ["different-qid"]
    registry.write_text(json.dumps(value) + "\n")

    with pytest.raises(ValueError, match="development registry"):
        project_cached_holdout_eligibility(**inputs)


def test_projection_rejects_non_docprune_or_unsealed_cached_results(tmp_path: Path) -> None:
    """Catch projecting all-kept pages or results not bound by the quality manifest."""

    inputs = _write_fixture(tmp_path)
    quality = Path(inputs["quality_manifest_path"])
    value = json.loads(quality.read_text())
    value["mode"] = "all-kept"
    quality.write_text(json.dumps(value) + "\n")
    inputs["quality_manifest_sha256"] = _sha(quality)
    with pytest.raises(ValueError, match="DocPrune top-4"):
        project_cached_holdout_eligibility(**inputs)


def test_projection_publication_is_atomic_no_replace(tmp_path: Path) -> None:
    """Catch replacing the outcome-blind eligible-pool projection after publication."""

    inputs = _write_fixture(tmp_path / "inputs")
    projection = project_cached_holdout_eligibility(**inputs)
    output = tmp_path / "eligible.json"

    digest = publish_cached_holdout_eligibility(projection, output)

    assert digest == _sha(output)
    assert json.loads(output.read_text())["projection_sha256"] == projection["projection_sha256"]
    with pytest.raises(FileExistsError):
        publish_cached_holdout_eligibility(projection, output)

    inputs = _write_fixture(tmp_path / "second")
    inputs["cached_results_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="cached result"):
        project_cached_holdout_eligibility(**inputs)
