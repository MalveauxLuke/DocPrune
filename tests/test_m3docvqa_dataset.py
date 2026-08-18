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
    integrity.write_text("{}")
    return CorpusIdentity(
        root=root,
        questions_path=questions,
        document_ids_path=document_ids,
        pdf_dir=pdfs,
        integrity_report_path=integrity,
        integrity_sha256=hashlib.sha256(integrity.read_bytes()).hexdigest(),
    )


def test_iter_samples_preserves_jsonl_order_and_extracts_answer_objects(tmp_path: Path) -> None:
    dataset = M3DocVQADevDataset(make_dataset_root(tmp_path))

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
    with pytest.raises(ValueError, match="duplicate qid"):
        M3DocVQADevDataset(corpus)

    corpus = make_dataset_root(tmp_path / "second")
    corpus.document_ids_path.write_text(json.dumps(["doc-1"]))
    with pytest.raises(ValueError, match="not listed"):
        M3DocVQADevDataset(corpus)


def test_load_document_pages_uses_144_dpi_rgb_and_modal_size_rule(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset = M3DocVQADevDataset(make_dataset_root(tmp_path))
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
    dataset = M3DocVQADevDataset(make_dataset_root(tmp_path))

    original = dataset.source_order_sha256
    dataset.pdf_path("doc-1").write_bytes(b"%PDF-1.4\nchanged")

    assert dataset.source_order_sha256 != original
