"""Strict contract tests for the Task 9 shared-probe cohort."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from docprune.experiment_design import _canonical_json_sha256
from docprune.task9_shared_probe import (
    DEFAULT_SHARED_PROBE_SPLITS,
    build_task9_shared_probe_cohort,
    validate_task9_shared_probe_cohort,
)

ROOT = Path(__file__).resolve().parents[2]
SEALER = ROOT / "archive/experiments/task6_9_2026_09_10" / "examples" / "seal_task9_shared_probe_cohort.py"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sealed_prior(selected_qids: object) -> dict[str, object]:
    value = {"schema_version": 1, "status": "sealed", "selected_qids": selected_qids}
    value["cohort_sha256"] = _canonical_json_sha256(value)
    return value


def _prior_cohorts() -> tuple[dict[str, object], dict[str, object]]:
    preliminary = _sealed_prior(
        {
            "baseline_correct": [f"pre-c-{i:02d}" for i in range(24)],
            "baseline_wrong": [f"pre-w-{i:02d}" for i in range(24)],
        }
    )
    confirmation = _sealed_prior([f"confirm-{i:03d}" for i in range(100)])
    return preliminary, confirmation


def _fixture_files(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    source = tmp_path / "questions.jsonl"
    pdf = tmp_path / "page.pdf"
    feature = tmp_path / "features.bin"
    index = tmp_path / "index.json"
    source.write_bytes(b"authenticated question source\n")
    pdf.write_bytes(b"authenticated page source\n")
    feature.write_bytes(b"authenticated persisted feature\n")
    index.write_bytes(b"authenticated feature index manifest\n")
    return source, pdf, feature, index


def _rows(tmp_path: Path, count: int = 600) -> list[dict[str, object]]:
    source, pdf, feature, index = _fixture_files(tmp_path)
    pages = [
        {
            "doc_id": f"page-doc-{i}",
            "page_index": i,
            "source_path": str(pdf),
            "source_sha256": _sha(pdf),
        }
        for i in range(4)
    ]
    persisted = [
        {
            "doc_id": page["doc_id"],
            "page_index": page["page_index"],
            "feature_path": str(feature),
            "feature_sha256": _sha(feature),
            "index_manifest_path": str(index),
            "index_manifest_sha256": _sha(index),
        }
        for page in pages
    ]
    rows: list[dict[str, object]] = []
    for i in range(count):
        qid = f"q-{i:04d}"
        correct = i % 2 == 0
        retrieved = [
            {"doc_id": page["doc_id"], "page_index": page["page_index"], "score": 4 - rank}
            for rank, page in enumerate(pages)
        ]
        rows.append(
            {
                "question_id": qid,
                "question": f"Question {qid}",
                "answers": ["gold"],
                "predicted_answer": "gold" if correct else "wrong",
                "retrieved_pages": retrieved,
                "supporting_document_ids": [f"support-doc-{i:04d}"],
                "cached_pages": deepcopy(pages),
                "persisted_features": deepcopy(persisted),
                "source": {
                    "path": str(source),
                    "sha256": _sha(source),
                    "type": "single_hop",
                },
                "template_id": f"template-{i % 7}",
                "vendor": f"vendor-{i % 3}",
            }
        )
    return rows


def _tiny_split_counts() -> dict[str, dict[str, int]]:
    return {
        "train": {"baseline_correct": 6, "baseline_wrong": 6},
        "validation": {"baseline_correct": 2, "baseline_wrong": 2},
        "primary_test": {"baseline_correct": 1, "baseline_wrong": 1},
        "secondary_test": {"total": 2},
    }


def test_shared_probe_cohort_has_exact_document_disjoint_stratified_splits(tmp_path: Path) -> None:
    preliminary, confirmation = _prior_cohorts()
    cohort = build_task9_shared_probe_cohort(
        _rows(tmp_path),
        preliminary_cohort=preliminary,
        confirmation_cohort=confirmation,
        selection_seed="shared-probe-test-v1",
    )

    assert cohort["question_count"] == 600
    assert cohort["split_counts"] == DEFAULT_SHARED_PROBE_SPLITS
    assert {name: len(qids) for name, qids in cohort["selected_qids"].items()} == {
        "train": 360,
        "validation": 120,
        "primary_test": 60,
        "secondary_test": 60,
    }
    assert all(
        split["baseline_counts"] == expected
        for split, expected in (
            (cohort["splits"]["train"], {"baseline_correct": 180, "baseline_wrong": 180}),
            (cohort["splits"]["validation"], {"baseline_correct": 60, "baseline_wrong": 60}),
            (cohort["splits"]["primary_test"], {"baseline_correct": 30, "baseline_wrong": 30}),
        )
    )
    assert cohort["splits"]["secondary_test"]["baseline_counts"] == {
        "baseline_correct": 30,
        "baseline_wrong": 30,
    }
    selected = cohort["selected_records"]
    assert len({record["qid"] for records in selected.values() for record in records}) == 600
    assert len(cohort["support_components"]["components"]) == 600
    supports_by_split = {
        split: {
            doc
            for record in records
            for doc in record["supporting_document_ids"]
        }
        for split, records in selected.items()
    }
    assert all(
        left.isdisjoint(right)
        for index, left in enumerate(supports_by_split.values())
        for right in list(supports_by_split.values())[index + 1 :]
    )
    assert not set(cohort["selected_qids"]["train"]) & {
        *preliminary["selected_qids"]["baseline_correct"],
        *preliminary["selected_qids"]["baseline_wrong"],
        *confirmation["selected_qids"],
    }
    assert validate_task9_shared_probe_cohort(cohort) == cohort


def test_shared_probe_selection_is_deterministic_and_preserves_metadata(tmp_path: Path) -> None:
    preliminary, confirmation = _prior_cohorts()
    rows = _rows(tmp_path)
    first = build_task9_shared_probe_cohort(
        rows,
        preliminary_cohort=preliminary,
        confirmation_cohort=confirmation,
        selection_seed="shared-probe-test-v1",
    )
    second = build_task9_shared_probe_cohort(
        list(reversed(rows)),
        preliminary_cohort=preliminary,
        confirmation_cohort=confirmation,
        selection_seed="shared-probe-test-v1",
    )
    assert first == second
    assert first["selected_records"]["train"][0]["template_id"].startswith("template-")
    assert first["selected_records"]["train"][0]["vendor"].startswith("vendor-")
    assert first["cohort_sha256"] == _canonical_json_sha256(
        {key: value for key, value in first.items() if key != "cohort_sha256"}
    )


def test_shared_probe_selects_one_question_from_a_component(tmp_path: Path) -> None:
    preliminary, confirmation = _prior_cohorts()
    rows = _rows(tmp_path, 8)
    rows[1]["supporting_document_ids"] = list(rows[0]["supporting_document_ids"])
    split_counts = {
        "train": {"baseline_correct": 2, "baseline_wrong": 0},
        "validation": {"baseline_correct": 2, "baseline_wrong": 0},
        "primary_test": {"baseline_correct": 0, "baseline_wrong": 1},
        "secondary_test": {"total": 0},
    }
    cohort = build_task9_shared_probe_cohort(
        rows,
        preliminary_cohort=preliminary,
        confirmation_cohort=confirmation,
        split_counts=split_counts,
    )
    selected = [qid for qids in cohort["selected_qids"].values() for qid in qids]
    assert len(selected) == 5
    assert bool({"q-0000", "q-0001"} & set(selected))
    assert not {"q-0000", "q-0001"}.issubset(selected)
    assert len(cohort["support_components"]["components"]) == 5


def test_shared_probe_rejects_insufficient_independent_components_and_duplicates(tmp_path: Path) -> None:
    preliminary, confirmation = _prior_cohorts()
    with pytest.raises(ValueError, match="at least 600 independent"):
        build_task9_shared_probe_cohort(
            _rows(tmp_path, 599),
            preliminary_cohort=preliminary,
            confirmation_cohort=confirmation,
        )
    rows = _rows(tmp_path)
    rows[-1] = deepcopy(rows[0])
    with pytest.raises(ValueError, match="duplicate question"):
        build_task9_shared_probe_cohort(
            rows,
            preliminary_cohort=preliminary,
            confirmation_cohort=confirmation,
        )


def test_shared_probe_rejects_missing_identity_malformed_hash_and_leaked_outcome(tmp_path: Path) -> None:
    preliminary, confirmation = _prior_cohorts()
    rows = _rows(tmp_path)
    del rows[0]["cached_pages"][0]["source_sha256"]
    with pytest.raises(ValueError, match="cached page"):
        build_task9_shared_probe_cohort(
            rows,
            preliminary_cohort=preliminary,
            confirmation_cohort=confirmation,
        )

    rows = _rows(tmp_path)
    rows[0]["source"]["sha256"] = "z" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        build_task9_shared_probe_cohort(
            rows,
            preliminary_cohort=preliminary,
            confirmation_cohort=confirmation,
        )

    rows = _rows(tmp_path)
    rows[0]["f1"] = 0.8
    with pytest.raises(ValueError, match="outcome field"):
        build_task9_shared_probe_cohort(
            rows,
            preliminary_cohort=preliminary,
            confirmation_cohort=confirmation,
        )


def test_shared_probe_rejects_retrieval_or_global_index_declarations(tmp_path: Path) -> None:
    preliminary, confirmation = _prior_cohorts()
    rows = _rows(tmp_path)
    rows[0]["retrieval_run"] = True
    with pytest.raises(ValueError, match="retrieval"):
        build_task9_shared_probe_cohort(
            rows,
            preliminary_cohort=preliminary,
            confirmation_cohort=confirmation,
        )
    rows = _rows(tmp_path)
    rows[0]["global_index_loaded"] = True
    with pytest.raises(ValueError, match="global index"):
        build_task9_shared_probe_cohort(
            rows,
            preliminary_cohort=preliminary,
            confirmation_cohort=confirmation,
        )


def test_shared_probe_cli_publishes_atomically_and_never_replaces(tmp_path: Path) -> None:
    preliminary, confirmation = _prior_cohorts()
    preliminary_path = tmp_path / "preliminary.json"
    confirmation_path = tmp_path / "confirmation.json"
    rows_path = tmp_path / "results.jsonl"
    output_path = tmp_path / "cohort.json"
    preliminary_path.write_text(json.dumps(preliminary), encoding="utf-8")
    confirmation_path.write_text(json.dumps(confirmation), encoding="utf-8")
    rows = _rows(tmp_path, 20)
    rows_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    split_path = tmp_path / "split-counts.json"
    split_path.write_text(json.dumps(_tiny_split_counts()), encoding="utf-8")
    command = [
        sys.executable,
        str(SEALER),
        "--cached-results",
        str(rows_path),
        "--preliminary-cohort",
        str(preliminary_path),
        "--confirmation-cohort",
        str(confirmation_path),
        "--split-counts",
        str(split_path),
        "--selection-seed",
        "cli-test-v1",
        "--output",
        str(output_path),
    ]
    environment = {"PYTHONPATH": str(ROOT / "src")}
    completed = subprocess.run(command, cwd=ROOT, env=environment, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    published = json.loads(output_path.read_text(encoding="utf-8"))
    assert published["question_count"] == 20
    assert published["input_file_hashes"][str(rows_path)] == _sha(rows_path)
    repeated = subprocess.run(command, cwd=ROOT, env=environment, text=True, capture_output=True)
    assert repeated.returncode != 0
    assert "already exists" in repeated.stderr
