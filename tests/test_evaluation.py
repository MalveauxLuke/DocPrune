import hashlib
import json
from pathlib import Path

import pytest

from docprune.evaluation import (
    M3DocVQAMetrics,
    ValidationReport,
    evaluate_m3docvqa,
    summarize_benchmark_run,
    validate_benchmark_run,
)


def _source_rows():
    return [
        {
            "qid": "q1",
            "question": "Where?",
            "answers": [{"answer": "New York", "modality": "text"}],
            "metadata": {"type": "TextQ"},
            "supporting_context": [{"doc_id": "doc-a", "doc_part": "text"}],
        },
        {
            "qid": "q2",
            "question": "How many?",
            "answers": [{"answer": "2", "modality": "table"}],
            "metadata": {"type": "Compose(TextQ,TableQ)"},
            "supporting_context": [
                {"doc_id": "doc-b", "doc_part": "table"},
                {"doc_id": "doc-c", "doc_part": "text"},
            ],
        },
        {
            "qid": "q3",
            "question": "Which colors?",
            "answers": [
                {"answer": "red", "modality": "image"},
                {"answer": "blue", "modality": "image"},
            ],
            "metadata": {"type": "ImageListQ"},
            "supporting_context": [{"doc_id": "doc-d", "doc_part": "image"}],
        },
        {
            "qid": "q4",
            "question": "What?",
            "answers": [{"answer": "answer", "modality": "chart"}],
            "metadata": {"type": "Compose(ImageQ,TextQ)"},
            "supporting_context": [{"doc_id": "doc-e", "doc_part": "figure"}],
        },
    ]


def _result(qid, answer, pages):
    return {
        "question_id": qid,
        "question": qid,
        "answers": [],
        "predicted_answer": answer,
        "retrieved_pages": [
            {"doc_id": doc_id, "page_index": page, "score": score} for doc_id, page, score in pages
        ],
        "trace": {
            "original_visual_tokens": 100,
            "post_btp_visual_tokens": 80,
            "post_qtp_visual_tokens": 60,
            "post_ctp_visual_tokens": 40,
            "ctp_layer": 7,
        },
        "timing": {
            "retrieval_seconds": 1.0,
            "qa_seconds": 2.0,
            "peak_allocated_gpu_bytes": 123,
            "warmup_excluded": True,
        },
    }


def test_evaluate_m3docvqa_matches_official_lists_and_slices():
    results = [
        _result("q1", "new york", [("doc-a", 0, 1.0), ("other", 0, 0.5)]),
        _result("q2", "2", [("doc-b", 0, 1.0), ("doc-c", 0, 0.5)]),
        _result("q3", "red", [("doc-d", 0, 1.0)]),
        _result("q4", "wrong", [("other", 0, 1.0)]),
    ]

    metrics = evaluate_m3docvqa(results, _source_rows())

    assert isinstance(metrics, M3DocVQAMetrics)
    assert metrics.count == 4
    assert metrics.overall == {"list_em": 50.0, "list_f1": 62.5}
    assert metrics.modalities["text"]["list_em"] == 100.0
    assert metrics.modalities["table"]["list_f1"] == 100.0
    assert metrics.modalities["image"] == {"list_em": 0.0, "list_f1": 50.0}
    assert metrics.hop_types["Single-hop"]["list_em"] == 50.0
    assert metrics.hop_types["Multi-hop"]["list_em"] == 50.0
    assert metrics.question_types["TextQ"]["list_f1"] == 100.0
    assert metrics.question_types["Compose(TextQ,TableQ)"]["list_f1"] == 100.0
    assert metrics.question_types["Compose(ImageQ,TextQ)"]["list_f1"] == 0.0
    assert metrics.document_recall == {
        1: 0.625,
        2: 0.75,
        4: 0.75,
        5: 0.75,
        10: 0.75,
    }


def _run_manifest(qids):
    manifest = {
        "schema_version": 2,
        "status": "configured",
        "operation": "evaluate",
        "selection": {"resolved_question_ids": qids},
        "index_manifest_source_path": None,
        "index_manifest_source_sha256": None,
        "index_manifest": None,
        "measurement": {
            "warmup_excluded": True,
            "profiler_enabled": False,
        },
    }
    unsigned = dict(manifest)
    manifest["run_manifest_sha256"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return manifest


def _write_valid_run(path: Path):
    path.mkdir()
    records = [
        _result("q1", "answer", [("doc-a", 0, 1.0)]),
        _result("q2", "answer", [("doc-a", 0, 1.0)]),
    ]
    source_path = path / "MMQA_dev.jsonl"
    source_path.write_text(
        "".join(
            json.dumps({"qid": qid, "question": qid, "answers": [{"answer": "answer"}]}) + "\n"
            for qid in ("q1", "q2")
        )
    )
    for record in records:
        record["answers"] = ["answer"]
    (path / "results.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    )
    manifest = _run_manifest(["q1", "q2"])
    manifest["corpus"] = {
        "questions_path": str(source_path),
        "questions_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
    }
    unsigned = dict(manifest)
    unsigned.pop("run_manifest_sha256")
    manifest["run_manifest_sha256"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (path / "run_manifest.json").write_text(json.dumps(manifest))
    (path / "summary.json").write_text(
        json.dumps(
            summarize_benchmark_run(
                path / "results.jsonl",
                [
                    {"qid": "q1", "question": "q1", "answers": [{"answer": "answer"}]},
                    {"qid": "q2", "question": "q2", "answers": [{"answer": "answer"}]},
                ],
            ),
            sort_keys=True,
        )
    )


def test_validate_benchmark_run_accepts_sealed_two_record_fixture(tmp_path):
    run = tmp_path / "run"
    _write_valid_run(run)

    report = validate_benchmark_run(run, expected_questions=2)

    assert isinstance(report, ValidationReport)
    assert report.valid is True
    assert report.errors == ()
    assert report.question_count == 2
    assert report.summary["quality"]["overall"] == {"list_em": 100.0, "list_f1": 100.0}
    assert report.summary["quality"]["observed_retrieval_depth"] == 1


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("duplicate", "duplicate"),
        ("order", "source order"),
        ("trace", "monotonic"),
        ("warmup", "warmup"),
        ("summary", "summary"),
    ],
)
def test_validate_benchmark_run_reports_integrity_failures(tmp_path, mutation, message):
    run = tmp_path / "run"
    _write_valid_run(run)
    result_path = run / "results.jsonl"
    records = [json.loads(line) for line in result_path.read_text().splitlines()]
    if mutation == "duplicate":
        records[1]["question_id"] = "q1"
    elif mutation == "order":
        records.reverse()
    elif mutation == "trace":
        records[0]["trace"]["post_ctp_visual_tokens"] = 90
    elif mutation == "warmup":
        records[0]["timing"]["warmup_excluded"] = False
    elif mutation == "summary":
        (run / "summary.json").write_text("{}")
    if mutation != "summary":
        result_path.write_text("".join(json.dumps(record) + "\n" for record in records))

    report = validate_benchmark_run(run, expected_questions=2)

    assert report.valid is False
    assert any(message in error.lower() for error in report.errors)


def test_validate_benchmark_run_checks_run_and_index_digests(tmp_path):
    run = tmp_path / "run"
    _write_valid_run(run)
    index_path = tmp_path / "index-manifest.json"
    index_payload = {"schema_version": 4, "value": "index"}
    index_payload["manifest_sha256"] = hashlib.sha256(
        json.dumps(
            {"schema_version": 4, "value": "index"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    index_path.write_text(json.dumps(index_payload, sort_keys=True))
    manifest = json.loads((run / "run_manifest.json").read_text())
    manifest["index_manifest_source_path"] = str(index_path)
    manifest["index_manifest_source_sha256"] = hashlib.sha256(index_path.read_bytes()).hexdigest()
    manifest["index_manifest"] = index_payload
    unsigned = dict(manifest)
    unsigned.pop("run_manifest_sha256")
    manifest["run_manifest_sha256"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (run / "run_manifest.json").write_text(json.dumps(manifest))
    index_path.write_text(index_path.read_text() + "tampered")

    report = validate_benchmark_run(run, expected_questions=2)

    assert report.valid is False
    assert any("index manifest digest mismatch" in error for error in report.errors)


def test_validate_benchmark_run_rejects_profiler_data_when_disabled(tmp_path):
    run = tmp_path / "run"
    _write_valid_run(run)
    result_path = run / "results.jsonl"
    record = json.loads(result_path.read_text().splitlines()[0])
    record["timing"]["profiler_definition"] = "disabled profiler"
    result_path.write_text(
        json.dumps(record) + "\n" + result_path.read_text().splitlines()[1] + "\n"
    )

    report = validate_benchmark_run(run, expected_questions=2)

    assert report.valid is False
    assert any("profiler" in error.lower() for error in report.errors)


def test_validate_benchmark_run_checks_nested_index_manifest_digest(tmp_path):
    run = tmp_path / "run"
    _write_valid_run(run)
    index_path = tmp_path / "index-manifest.json"
    index_payload = {"schema_version": 4, "value": "index"}
    index_payload["manifest_sha256"] = hashlib.sha256(
        json.dumps(index_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    index_path.write_text(json.dumps(index_payload, sort_keys=True))
    manifest = json.loads((run / "run_manifest.json").read_text())
    manifest["index_manifest_source_path"] = str(index_path)
    manifest["index_manifest_source_sha256"] = hashlib.sha256(index_path.read_bytes()).hexdigest()
    manifest["index_manifest"] = dict(index_payload, manifest_sha256="0" * 64)
    unsigned = dict(manifest)
    unsigned.pop("run_manifest_sha256")
    manifest["run_manifest_sha256"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (run / "run_manifest.json").write_text(json.dumps(manifest))

    report = validate_benchmark_run(run, expected_questions=2)

    assert report.valid is False
    assert any("canonical digest" in error for error in report.errors)


def test_validate_benchmark_run_checks_source_question_content(tmp_path):
    run = tmp_path / "run"
    _write_valid_run(run)
    source_path = tmp_path / "MMQA_dev.jsonl"
    source_rows = [
        {"qid": "q1", "question": "q1", "answers": [{"answer": "a"}]},
        {"qid": "q2", "question": "q2", "answers": [{"answer": "b"}]},
    ]
    source_path.write_text("".join(json.dumps(row) + "\n" for row in source_rows))
    manifest = json.loads((run / "run_manifest.json").read_text())
    manifest["corpus"] = {
        "questions_path": str(source_path),
        "questions_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
    }
    unsigned = dict(manifest)
    unsigned.pop("run_manifest_sha256")
    manifest["run_manifest_sha256"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (run / "run_manifest.json").write_text(json.dumps(manifest))

    report = validate_benchmark_run(run, expected_questions=2)

    assert report.valid is False
    assert any("source" in error for error in report.errors)


def test_validate_benchmark_run_requires_production_source_identity(tmp_path):
    run = tmp_path / "run"
    _write_valid_run(run)
    manifest = json.loads((run / "run_manifest.json").read_text())
    manifest.pop("corpus")
    unsigned = dict(manifest)
    unsigned.pop("run_manifest_sha256")
    manifest["run_manifest_sha256"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (run / "run_manifest.json").write_text(json.dumps(manifest))

    report = validate_benchmark_run(run, expected_questions=2)

    assert report.valid is False
    assert any("corpus source identity" in error for error in report.errors)
