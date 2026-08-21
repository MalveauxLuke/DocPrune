"""Contract tests for the independently validated six-cell comparison."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from docprune.metrics import MEASUREMENT_DEFINITION


def _pdf_bytes(page_count: int = 1) -> bytes:
    """A tiny parseable-enough PDF; comparison counts /Type /Page objects itself."""

    objects = [
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n",
        f"2 0 obj << /Type /Pages /Kids [{3} 0 R] /Count {page_count} >> endobj\n".encode(),
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 10 10] >> endobj\n",
    ]
    body = b"%PDF-1.4\n" + b"".join(objects) + b"%%EOF\n"
    return body


def _corpus(root: Path) -> dict[str, object]:
    root.mkdir()
    pdf_dir = root / "pdfs_dev"
    pdf_dir.mkdir()
    docs = [f"doc-{index}" for index in range(1, 5)]
    for doc_id in docs:
        (pdf_dir / f"{doc_id}.pdf").write_bytes(_pdf_bytes())
    questions = root / "MMQA_dev.jsonl"
    questions.write_text(
        "".join(
            json.dumps(
                {
                    "qid": qid,
                    "question": qid,
                    "answers": [{"answer": "answer", "modality": "text"}],
                    "metadata": {"type": "TextQ"},
                    "supporting_context": [{"doc_id": docs[0]}],
                }
            )
            + "\n"
            for qid in ("q1", "q2")
        )
    )
    document_ids = root / "dev_doc_ids.json"
    document_ids.write_text(json.dumps(docs))
    integrity = root / "attempt-3-integrity.json"
    integrity.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dev_questions": 2,
                "expected_pdf_count": 4,
                "actual_pdf_count": 4,
                "missing_pdf_ids": [],
                "extra_pdf_ids": [],
                "corrupt_pdfs": [],
                "observed_page_count": 4,
                "within_ten_percent_of_published_page_count": True,
            }
        )
    )
    archive = root / "mmqa-archives.sha256"
    archive.write_text("")
    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "root": str(root),
        "questions_path": str(questions),
        "document_ids_path": str(document_ids),
        "pdf_dir": str(pdf_dir),
        "integrity_report_path": str(integrity),
        "integrity_sha256": digest(integrity),
        "archive_checksum_manifest_path": str(archive),
        "archive_checksum_manifest_sha256": digest(archive),
        "questions_sha256": digest(questions),
        "document_ids_sha256": digest(document_ids),
        "expected_question_count": 2,
        "expected_pdf_count": 4,
        "expected_page_count": 4,
        "is_fixture": True,
        "archive_hashes": {},
    }


def _write_run(
    path: Path,
    corpus: dict[str, object],
    mode: str,
    page_count: int,
    *,
    profiler_enabled: bool = False,
    profiler_definition: str = "profile-v1",
    flops: float = 2.0e12,
) -> None:
    path.mkdir()
    docs = [f"doc-{index}" for index in range(1, 5)]
    pages = [
        {"doc_id": doc_id, "page_index": 0, "score": 1.0 - index / 10}
        for index, doc_id in enumerate(docs[:page_count])
    ]
    records = []
    for qid in ("q1", "q2"):
        counts = (100, 100, 100, 100) if mode == "all-kept" else (100, 80, 60, 40)
        timing = {
            "retrieval_seconds": 1.0,
            "page_load_seconds": 1.0,
            "qa_seconds": 2.0,
            "encoder_seconds": 1.0,
            "decoder_seconds": 1.0,
            "total_sample_seconds": 4.0,
            "peak_allocated_gpu_bytes": 100,
            "warmup_excluded": True,
            "profiler_enabled": profiler_enabled,
        }
        if profiler_enabled:
            timing["profiler_definition"] = profiler_definition
            timing["flops"] = flops
        records.append(
            {
                "question_id": qid,
                "question": qid,
                "answers": ["answer"],
                "predicted_answer": "answer",
                "retrieved_pages": pages,
                "trace": {
                    "original_visual_tokens": counts[0],
                    "post_btp_visual_tokens": counts[1],
                    "post_qtp_visual_tokens": counts[2],
                    "post_ctp_visual_tokens": counts[3],
                    "ctp_layer": None if mode == "all-kept" else 7,
                },
                "timing": timing,
            }
        )
    results = path / "results.jsonl"
    results.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records))
    manifest = {
        "schema_version": 2,
        "status": "configured",
        "operation": "evaluate",
        "output": str(path),
        "mode": mode,
        "page_count": page_count,
        "runtime_commit": "runtime",
        "m3docrag_commit": "upstream",
        "resources": {"model": "model"},
        "processor_contract_path": "processor.json",
        "processor_contract_sha256": "processor-sha",
        "processor_contract": {"processor": "processor"},
        "run_config_source_path": None,
        "run_config_source_sha256": None,
        "index_manifest_source_path": None,
        "index_manifest_source_sha256": None,
        "index_manifest": None,
        "corpus": corpus,
        "generation": {"max_new_tokens": 128},
        "pruning_config": {"mode": mode},
        "selection": {"resolved_question_ids": ["q1", "q2"], "count": 2},
        "measurement": {
            "definition": MEASUREMENT_DEFINITION,
            "warmup_required": True,
            "profiler_enabled": profiler_enabled,
        },
        "fixture_mode": True,
    }
    unsigned = dict(manifest)
    manifest["run_manifest_sha256"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (path / "run_manifest.json").write_text(json.dumps(manifest, sort_keys=True))
    from docprune.evaluation import summarize_benchmark_run

    source = [json.loads(line) for line in corpus["questions_path"] and Path(corpus["questions_path"]).read_text().splitlines()]
    summary = summarize_benchmark_run(results, source)
    (path / "summary.json").write_text(json.dumps(summary, sort_keys=True))


def _matrix(
    tmp_path: Path,
    *,
    profiler_enabled: bool = False,
    profiler_definition: str = "profile-v1",
    flops: float = 2.0e12,
) -> tuple[Path, dict[str, object], dict[tuple[str, int], Path]]:
    corpus = _corpus(tmp_path / "corpus")
    runs: dict[tuple[str, int], Path] = {}
    for mode in ("all-kept", "docprune"):
        for page_count in (1, 2, 4):
            run = tmp_path / f"{mode}-{page_count}"
            _write_run(
                run,
                corpus,
                mode,
                page_count,
                profiler_enabled=profiler_enabled,
                profiler_definition=profiler_definition,
                flops=flops,
            )
            runs[(mode, page_count)] = run
    return Path(str(corpus["root"])), corpus, runs


def _refresh_summary(run: Path, corpus: dict[str, object]) -> None:
    from docprune.evaluation import summarize_benchmark_run

    source = [
        json.loads(line)
        for line in Path(str(corpus["questions_path"])).read_text().splitlines()
    ]
    summary = summarize_benchmark_run(run / "results.jsonl", source)
    (run / "summary.json").write_text(json.dumps(summary, sort_keys=True))


def test_complete_matrix_reports_six_cells_and_three_deltas(tmp_path: Path) -> None:
    corpus_root, _, runs = _matrix(tmp_path)

    from docprune.comparison import validate_comparison_matrix

    report = validate_comparison_matrix(runs, corpus_root=corpus_root, expected_questions=2, allow_fixture=True)

    assert report.valid
    assert len(report.cells) == 6
    assert [delta["page_count"] for delta in report.deltas] == [1, 2, 4]
    assert report.payload is not None
    assert "profiling" not in report.payload
    payload = report.payload
    unsigned = dict(payload)
    unsigned.pop("comparison_sha256")
    expected_digest = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()
    assert payload["comparison_sha256"] == expected_digest
    from docprune.comparison import comparison_markdown

    markdown = comparison_markdown(payload)
    stable_markdown = markdown.replace(
        f"`{payload['comparison_sha256']}`", "`<digest>`"
    ).encode()
    assert hashlib.sha256(stable_markdown).hexdigest() == (
        "6c65df3c10f96f5ccdc08bd9c657ea695ace78a362dd75376c767178e59e0ce6"
    )
    assert comparison_markdown(payload) == markdown
    all_kept = report.cells[0]
    docprune = report.cells[1]
    assert all_kept["quality"]["overall"] == {"list_em": 100.0, "list_f1": 100.0}
    assert all_kept["quality"]["retrieval"] == {"1": 1.0, "2": 1.0, "4": 1.0, "5": 1.0, "10": 1.0}
    assert all_kept["efficiency"]["timing_seconds"] == {
        "retrieval": 2.0,
        "page_load": 2.0,
        "qa": 4.0,
        "encoder": 2.0,
        "decoder": 2.0,
        "total": 8.0,
    }
    assert docprune["efficiency"]["retention_rates"] == {"btp": 0.8, "qtp": 0.6, "ctp": 0.4}
    assert report.deltas[0]["metrics"]["efficiency"]["drop_rates"] == {
        "btp": 0.19999999999999996,
        "qtp": 0.4,
        "ctp": 0.6,
    }


def test_complete_profile_reports_exact_shared_identity_and_tflops(tmp_path: Path) -> None:
    corpus_root, _, runs = _matrix(tmp_path, profiler_enabled=True, flops=2.0e12)

    from docprune.comparison import validate_comparison_matrix

    report = validate_comparison_matrix(
        runs, corpus_root=corpus_root, expected_questions=2, allow_fixture=True
    )

    assert report.valid
    assert report.payload is not None
    assert report.payload["profiling"] == {
        "profiler_definition": "profile-v1",
        "tflops": [0.5] * 6,
    }


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("definition", "profiler definition"),
        ("partial", "mismatched shared identity"),
        ("nonpositive", "finite and positive"),
    ],
)
def test_matrix_rejects_invalid_or_inconsistent_profiling(
    tmp_path: Path, mutation: str, expected: str
) -> None:
    corpus_root, corpus, runs = _matrix(tmp_path, profiler_enabled=True)
    run = runs[("docprune", 1)]
    records = [json.loads(line) for line in (run / "results.jsonl").read_text().splitlines()]
    manifest_path = run / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if mutation == "definition":
        for record in records:
            record["timing"]["profiler_definition"] = "profile-v2"
    elif mutation == "partial":
        for record in records:
            record["timing"].pop("profiler_definition")
            record["timing"].pop("flops")
            record["timing"]["profiler_enabled"] = False
        manifest["measurement"]["profiler_enabled"] = False
    else:
        for record in records:
            record["timing"]["flops"] = 0.0
    (run / "results.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    )
    unsigned = dict(manifest)
    unsigned.pop("run_manifest_sha256", None)
    manifest["run_manifest_sha256"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    _refresh_summary(run, corpus)

    from docprune.comparison import validate_comparison_matrix

    report = validate_comparison_matrix(
        runs, corpus_root=corpus_root, expected_questions=2, allow_fixture=True
    )
    assert not report.valid
    assert report.payload is None
    assert any(expected in error for error in report.errors)


@pytest.mark.parametrize("mutation", ["incomplete", "duplicate", "extra"])
def test_matrix_rejects_wrong_cell_set(tmp_path: Path, mutation: str) -> None:
    corpus_root, _, runs = _matrix(tmp_path)
    inputs: object = runs
    if mutation == "incomplete":
        runs.pop(("docprune", 4))
    elif mutation == "duplicate":
        inputs = list(runs.values()) + [runs[("docprune", 1)]]
    else:
        runs[("extra", 8)] = runs[("docprune", 4)]

    from docprune.comparison import validate_comparison_matrix

    report = validate_comparison_matrix(inputs, corpus_root=corpus_root, expected_questions=2, allow_fixture=True)
    assert not report.valid
    assert any("six" in error or "cell" in error or "mode" in error for error in report.errors)


def test_matrix_rejects_reordered_qids(tmp_path: Path) -> None:
    corpus_root, _, runs = _matrix(tmp_path)
    docprune = runs[("docprune", 1)]
    manifest_path = docprune / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["selection"]["resolved_question_ids"] = ["q2", "q1"]
    manifest.pop("run_manifest_sha256")
    manifest["run_manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))

    from docprune.comparison import validate_comparison_matrix

    report = validate_comparison_matrix(runs, corpus_root=corpus_root, expected_questions=2, allow_fixture=True)
    assert not report.valid
    assert any("order" in error for error in report.errors)


@pytest.mark.parametrize(
    ("label", "field", "value"),
    [
        ("runtime", "runtime_commit", "runtime-other"),
        ("upstream", "m3docrag_commit", "upstream-other"),
        ("model", "resources", {"model": "model-other"}),
        ("processor", "processor_contract", {"processor": "processor-other"}),
        ("generation", "generation", {"max_new_tokens": 64}),
        ("measurement", "measurement", {"measurement_identity": "measurement-other"}),
    ],
)
def test_matrix_rejects_each_shared_identity_mismatch(
    tmp_path: Path, label: str, field: str, value: object
) -> None:
    corpus_root, _, runs = _matrix(tmp_path)
    manifest_path = runs[("docprune", 1)] / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if field == "measurement":
        manifest[field] = dict(manifest[field], **value)
    else:
        manifest[field] = value
    unsigned = dict(manifest)
    unsigned.pop("run_manifest_sha256", None)
    manifest["run_manifest_sha256"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))

    from docprune.comparison import validate_comparison_matrix

    report = validate_comparison_matrix(
        runs, corpus_root=corpus_root, expected_questions=2, allow_fixture=True
    )
    assert not report.valid, label
    assert any("mismatched shared identity" in error for error in report.errors)


def test_matrix_rejects_fabricated_document_and_out_of_range_page(tmp_path: Path) -> None:
    corpus_root, _, runs = _matrix(tmp_path)
    result_path = runs[("all-kept", 1)] / "results.jsonl"
    record = json.loads(result_path.read_text().splitlines()[0])
    record["retrieved_pages"][0]["doc_id"] = "fabricated"
    result_path.write_text(
        json.dumps(record, sort_keys=True) + "\n" + result_path.read_text().splitlines()[1] + "\n"
    )

    from docprune.comparison import validate_comparison_matrix

    report = validate_comparison_matrix(runs, corpus_root=corpus_root, expected_questions=2, allow_fixture=True)
    assert not report.valid
    assert any("corpus" in error or "document" in error for error in report.errors)


def test_matrix_rejects_page_index_outside_actual_pdf(tmp_path: Path) -> None:
    corpus_root, _, runs = _matrix(tmp_path)
    result_path = runs[("docprune", 2)] / "results.jsonl"
    records = [json.loads(line) for line in result_path.read_text().splitlines()]
    records[0]["retrieved_pages"][0]["page_index"] = 1
    result_path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records))

    from docprune.comparison import validate_comparison_matrix

    report = validate_comparison_matrix(runs, corpus_root=corpus_root, expected_questions=2, allow_fixture=True)
    assert not report.valid
    assert any("actual page count" in error for error in report.errors)


def test_matrix_rejects_altered_summary(tmp_path: Path) -> None:
    corpus_root, _, runs = _matrix(tmp_path)
    summary_path = runs[("docprune", 4)] / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["quality"]["overall"]["list_f1"] = 0.0
    summary_path.write_text(json.dumps(summary))

    from docprune.comparison import validate_comparison_matrix

    report = validate_comparison_matrix(runs, corpus_root=corpus_root, expected_questions=2, allow_fixture=True)
    assert not report.valid
    assert any("summary" in error for error in report.errors)


def test_publication_rolls_back_when_second_publish_fails(tmp_path: Path, monkeypatch) -> None:
    corpus_root, _, runs = _matrix(tmp_path)
    json_output = tmp_path / "comparison.json"
    markdown_output = tmp_path / "comparison.md"
    import docprune.comparison as comparison

    original = comparison._publish_noreplace
    calls = 0

    def fail_second(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected second publish failure")
        original(source, destination)

    monkeypatch.setattr(comparison, "_publish_noreplace", fail_second)
    with pytest.raises(OSError, match="injected second publish failure"):
        comparison.write_comparison_report(
            runs,
            corpus_root=corpus_root,
            json_path=json_output,
            markdown_path=markdown_output,
            expected_questions=2,
            allow_fixture=True,
        )
    assert not json_output.exists()
    assert not markdown_output.exists()


def test_publication_refuses_preexisting_destination_without_changing_bytes(tmp_path: Path) -> None:
    corpus_root, _, runs = _matrix(tmp_path)
    json_output = tmp_path / "comparison.json"
    markdown_output = tmp_path / "comparison.md"
    sentinel = b"preexisting bytes"
    json_output.write_bytes(sentinel)

    from docprune.comparison import write_comparison_report

    with pytest.raises(FileExistsError):
        write_comparison_report(
            runs,
            corpus_root=corpus_root,
            json_path=json_output,
            markdown_path=markdown_output,
            expected_questions=2,
            allow_fixture=True,
        )
    assert json_output.read_bytes() == sentinel
    assert not markdown_output.exists()


def test_publication_rolls_back_around_concurrent_second_destination(
    tmp_path: Path, monkeypatch
) -> None:
    corpus_root, _, runs = _matrix(tmp_path)
    json_output = tmp_path / "comparison.json"
    markdown_output = tmp_path / "comparison.md"
    import docprune.comparison as comparison

    original = comparison._publish_noreplace
    calls = 0
    concurrent_bytes = b"concurrent destination"

    def race_on_second(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            markdown_output.write_bytes(concurrent_bytes)
        original(source, destination)

    monkeypatch.setattr(comparison, "_publish_noreplace", race_on_second)
    with pytest.raises(FileExistsError):
        comparison.write_comparison_report(
            runs,
            corpus_root=corpus_root,
            json_path=json_output,
            markdown_path=markdown_output,
            expected_questions=2,
            allow_fixture=True,
        )
    assert not json_output.exists()
    assert markdown_output.read_bytes() == concurrent_bytes
