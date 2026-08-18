from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from docprune.benchmark_config import CorpusIdentity
from docprune.m3docvqa_dataset import M3DocVQADevDataset


def make_dataset_root(tmp_path: Path) -> CorpusIdentity:
    root = tmp_path / "corpus"
    questions = root / "multimodalqa" / "MMQA_dev.jsonl"
    questions.parent.mkdir(parents=True)
    rows = [
        {"qid": "q-2", "question": "second", "answers": [{"answer": "two"}]},
        {"qid": "q-1", "question": "first", "answers": [{"answer": "one"}, {"answer": "uno"}]},
    ]
    questions.write_text("".join(json.dumps(row) + "\n" for row in rows))
    document_ids = root / "dev_doc_ids.json"
    document_ids.write_text(json.dumps(["doc-2", "doc-1"]))
    pdfs = root / "pdfs_dev"
    pdfs.mkdir()
    for doc_id in ("doc-1", "doc-2"):
        (pdfs / f"{doc_id}.pdf").write_bytes(b"%PDF-1.4\n")
    integrity = root / "attempt-integrity.json"
    integrity.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dev_questions": 2,
                "expected_pdf_count": 2,
                "actual_pdf_count": 2,
                "missing_pdf_ids": [],
                "extra_pdf_ids": [],
                "corrupt_pdfs": [],
                "observed_page_count": 1,
                "within_ten_percent_of_published_page_count": True,
            }
        )
    )
    return CorpusIdentity.fixture(
        root=root,
        questions_path=questions,
        document_ids_path=document_ids,
        pdf_dir=pdfs,
        integrity_report_path=integrity,
        integrity_sha256=hashlib.sha256(integrity.read_bytes()).hexdigest(),
        archive_checksum_manifest_path=integrity,
        archive_checksum_manifest_sha256=hashlib.sha256(integrity.read_bytes()).hexdigest(),
        questions_sha256=hashlib.sha256(questions.read_bytes()).hexdigest(),
        document_ids_sha256=hashlib.sha256(document_ids.read_bytes()).hexdigest(),
        expected_question_count=2,
        expected_pdf_count=2,
        expected_page_count=1,
    )


def test_iter_samples_preserves_jsonl_order_and_extracts_answer_objects(tmp_path: Path) -> None:
    dataset = M3DocVQADevDataset(make_dataset_root(tmp_path), expected_question_count=2)

    samples = list(dataset)

    assert [sample.question_id for sample in samples] == ["q-2", "q-1"]
    assert [sample.question for sample in samples] == ["second", "first"]
    assert samples[0].answers == ("two",)
    assert samples[1].answers == ("one", "uno")


def test_dataset_rejects_duplicate_question_ids_and_unknown_pdf_documents(tmp_path: Path) -> None:
    corpus = make_dataset_root(tmp_path)
    corpus.questions_path.write_text(
        '{"qid": "q-1", "question": "a", "answers": []}\n'
        '{"qid": "q-1", "question": "b", "answers": []}\n'
    )
    object.__setattr__(
        corpus, "questions_sha256", hashlib.sha256(corpus.questions_path.read_bytes()).hexdigest()
    )
    with pytest.raises(ValueError, match="duplicate qid"):
        M3DocVQADevDataset(corpus, expected_question_count=2)

    corpus = make_dataset_root(tmp_path / "second")
    corpus.document_ids_path.write_text(json.dumps(["doc-1"]))
    object.__setattr__(
        corpus,
        "document_ids_sha256",
        hashlib.sha256(corpus.document_ids_path.read_bytes()).hexdigest(),
    )
    with pytest.raises(ValueError, match="not listed"):
        M3DocVQADevDataset(corpus, expected_question_count=2)


def test_load_document_pages_uses_144_dpi_rgb_and_modal_size_rule(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset = M3DocVQADevDataset(make_dataset_root(tmp_path), expected_question_count=2)
    calls: list[tuple[str, int]] = []

    def fake_convert(path: str, *, dpi: int) -> list[Image.Image]:
        calls.append((path, dpi))
        return [
            Image.new("L", (10, 20)),
            Image.new("RGB", (20, 10)),
            Image.new("RGBA", (10, 20)),
        ]

    monkeypatch.setattr("docprune.m3docvqa_dataset.convert_from_path", fake_convert)

    pages = dataset.load_pages("doc-1")

    assert calls == [(str(dataset.pdf_path("doc-1")), 144)]
    assert [page.mode for page in pages] == ["RGB", "RGB", "RGB"]
    assert [page.size for page in pages] == [(10, 20), (10, 20), (10, 20)]
    assert dataset.load_page("doc-1", 1).size == (10, 20)


def test_source_order_digest_binds_document_order_and_pdf_contents(tmp_path: Path) -> None:
    dataset = M3DocVQADevDataset(make_dataset_root(tmp_path), expected_question_count=2)

    original = dataset.source_order_sha256
    dataset.pdf_path("doc-1").write_bytes(b"%PDF-1.4\nchanged")

    assert dataset.source_order_sha256 != original


def test_dataset_requires_full_dev_question_count_without_fixture_override(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="2441"):
        M3DocVQADevDataset(make_dataset_root(tmp_path))
