from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from docprune.benchmark_config import (
    COLPALI_BACKBONE_MODEL,
    COLPALI_BACKBONE_REVISION,
    COLPALI_MODEL,
    COLPALI_REVISION,
    M3DOCRAG_COMMIT,
    QWEN_MODEL,
    QWEN_REVISION,
    BenchmarkRunConfig,
    CorpusIdentity,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_corpus(root: Path) -> CorpusIdentity:
    questions = root / "multimodalqa" / "MMQA_dev.jsonl"
    questions.parent.mkdir(parents=True)
    questions.write_text('{"qid": "q1", "question": "one", "answers": []}\n')
    doc_ids = root / "dev_doc_ids.json"
    doc_ids.write_text(json.dumps(["doc-1"]))
    pdfs = root / "pdfs_dev"
    pdfs.mkdir()
    (pdfs / "doc-1.pdf").write_bytes(b"%PDF-1.4\n")
    integrity = root / "attempt-3-integrity.json"
    integrity.write_text(
        json.dumps(
            {
                "dev_questions": 1,
                "expected_pdf_count": 1,
                "actual_pdf_count": 1,
                "missing_pdf_ids": [],
                "extra_pdf_ids": [],
                "corrupt_pdfs": [],
            }
        )
    )
    return CorpusIdentity(
        root=root,
        questions_path=questions,
        document_ids_path=doc_ids,
        pdf_dir=pdfs,
        integrity_report_path=integrity,
        integrity_sha256=sha256(integrity),
    )


def test_corpus_identity_requires_verified_files_and_integrity_hash(tmp_path: Path) -> None:
    corpus = make_corpus(tmp_path / "corpus")

    corpus.validate()

    corpus.integrity_report_path.write_text("changed")
    with pytest.raises(ValueError, match="integrity SHA-256 mismatch"):
        corpus.validate()


@pytest.mark.parametrize("attribute", ["questions_path", "document_ids_path", "pdf_dir"])
def test_corpus_identity_rejects_missing_required_source(attribute: str, tmp_path: Path) -> None:
    corpus = make_corpus(tmp_path / "corpus")
    target = getattr(corpus, attribute)
    if target.is_dir():
        for child in target.iterdir():
            child.unlink()
        target.rmdir()
    else:
        target.unlink()

    with pytest.raises(FileNotFoundError, match="required corpus"):
        corpus.validate()


def test_run_config_from_env_binds_exact_resources_and_generation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    corpus = make_corpus(tmp_path / "corpus")
    values = {
        "DOCPRUNE_CORPUS_ROOT": str(corpus.root),
        "DOCPRUNE_CORPUS_INTEGRITY_SHA256": corpus.integrity_sha256,
        "DOCPRUNE_RUNTIME_COMMIT": "a" * 40,
        "M3DOCRAG_COMMIT": M3DOCRAG_COMMIT,
        "QWEN_MODEL": QWEN_MODEL,
        "QWEN_REVISION": QWEN_REVISION,
        "COLPALI_MODEL": COLPALI_MODEL,
        "COLPALI_REVISION": COLPALI_REVISION,
        "COLPALI_BACKBONE_MODEL": COLPALI_BACKBONE_MODEL,
        "COLPALI_BACKBONE_REVISION": COLPALI_BACKBONE_REVISION,
        "DOCPRUNE_PROCESSOR_CONTRACT": str(tmp_path / "processor-contract.json"),
    }
    (tmp_path / "processor-contract.json").write_text("{}")
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    config = BenchmarkRunConfig.from_env(mode="docprune", page_count=4)

    assert config.mode == "docprune"
    assert config.page_count == 4
    assert config.corpus == corpus
    assert config.max_new_tokens == 128
    assert config.do_sample is False
    assert config.num_beams == 1
    assert config.prompt == "Answer the question using the image. Answer concisely."
    assert config.m3docrag_commit == M3DOCRAG_COMMIT
    assert config.qwen_model == QWEN_MODEL
    assert config.qwen_revision == QWEN_REVISION
    assert config.colpali_model == COLPALI_MODEL
    assert config.colpali_revision == COLPALI_REVISION
    assert config.colpali_backbone_model == COLPALI_BACKBONE_MODEL
    assert config.colpali_backbone_revision == COLPALI_BACKBONE_REVISION


@pytest.mark.parametrize(
    ("mode", "page_count", "message"),
    [("wrong", 1, "mode"), ("all-kept", 3, "page_count")],
)
def test_run_config_rejects_unknown_mode_or_page_count(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str, page_count: int, message: str
) -> None:
    corpus = make_corpus(tmp_path / "corpus")
    contract = tmp_path / "contract.json"
    contract.write_text("{}")
    for name, value in {
        "DOCPRUNE_CORPUS_ROOT": str(corpus.root),
        "DOCPRUNE_CORPUS_INTEGRITY_SHA256": corpus.integrity_sha256,
        "DOCPRUNE_RUNTIME_COMMIT": "a" * 40,
        "M3DOCRAG_COMMIT": M3DOCRAG_COMMIT,
        "QWEN_MODEL": QWEN_MODEL,
        "QWEN_REVISION": QWEN_REVISION,
        "COLPALI_MODEL": COLPALI_MODEL,
        "COLPALI_REVISION": COLPALI_REVISION,
        "COLPALI_BACKBONE_MODEL": COLPALI_BACKBONE_MODEL,
        "COLPALI_BACKBONE_REVISION": COLPALI_BACKBONE_REVISION,
        "DOCPRUNE_PROCESSOR_CONTRACT": str(contract),
    }.items():
        monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=message):
        BenchmarkRunConfig.from_env(mode=mode, page_count=page_count)


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("M3DOCRAG_COMMIT", "b" * 40, "pinned"),
        ("QWEN_REVISION", "b" * 40, "pinned"),
        ("COLPALI_REVISION", "b" * 40, "pinned"),
        ("COLPALI_BACKBONE_REVISION", "b" * 40, "pinned"),
        ("DOCPRUNE_RUNTIME_COMMIT", "short", "40-character"),
    ],
)
def test_run_config_rejects_unpinned_or_nonimmutable_resources(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, name: str, value: str, message: str
) -> None:
    corpus = make_corpus(tmp_path / "corpus")
    contract = tmp_path / "contract.json"
    contract.write_text("{}")
    values = {
        "DOCPRUNE_CORPUS_ROOT": str(corpus.root),
        "DOCPRUNE_CORPUS_INTEGRITY_SHA256": corpus.integrity_sha256,
        "DOCPRUNE_RUNTIME_COMMIT": "a" * 40,
        "M3DOCRAG_COMMIT": M3DOCRAG_COMMIT,
        "QWEN_MODEL": QWEN_MODEL,
        "QWEN_REVISION": QWEN_REVISION,
        "COLPALI_MODEL": COLPALI_MODEL,
        "COLPALI_REVISION": COLPALI_REVISION,
        "COLPALI_BACKBONE_MODEL": COLPALI_BACKBONE_MODEL,
        "COLPALI_BACKBONE_REVISION": COLPALI_BACKBONE_REVISION,
        "DOCPRUNE_PROCESSOR_CONTRACT": str(contract),
    }
    values[name] = value
    for variable, setting in values.items():
        monkeypatch.setenv(variable, setting)

    with pytest.raises(ValueError, match=message):
        BenchmarkRunConfig.from_env(mode="all-kept", page_count=1)
