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
    DEV_DOC_IDS_SHA256,
    FINAL_INTEGRITY_SHA256,
    M3DOCRAG_COMMIT,
    MMQA_ARCHIVES_SHA256,
    MMQA_DEV_SHA256,
    QWEN_MODEL,
    QWEN_REVISION,
    SHORT_ANSWER_TEMPLATE,
    BenchmarkRunConfig,
    CorpusIdentity,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_archive_fixture(directory: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in (
        "MMQA_dev.jsonl.gz",
        "MMQA_images.jsonl.gz",
        "MMQA_tables.jsonl.gz",
        "MMQA_texts.jsonl.gz",
        "MMQA_train.jsonl.gz",
    ):
        path = directory / name
        path.write_bytes(name.encode("utf-8"))
        hashes[name] = sha256(path)
    return hashes


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
                "observed_page_count": 1,
                "within_ten_percent_of_published_page_count": True,
                "schema_version": 1,
                "attempt": "fixture",
            }
        )
    )
    archives = root / "setup" / "mmqa-archives.sha256"
    archives.parent.mkdir()
    archive_hashes = write_archive_fixture(questions.parent)
    archives.write_text(
        "".join(
            f"{digest}  {questions.parent / name}\n" for name, digest in archive_hashes.items()
        )
    )
    return CorpusIdentity.fixture(
        root=root,
        questions_path=questions,
        document_ids_path=doc_ids,
        pdf_dir=pdfs,
        integrity_report_path=integrity,
        archive_checksum_manifest_path=archives,
        integrity_sha256=sha256(integrity),
        archive_checksum_manifest_sha256=sha256(archives),
        archive_hashes=archive_hashes,
        questions_sha256=sha256(questions),
        document_ids_sha256=sha256(doc_ids),
        expected_question_count=1,
        expected_pdf_count=1,
        expected_page_count=1,
    )


def test_corpus_identity_requires_verified_files_and_integrity_hash(tmp_path: Path) -> None:
    corpus = make_corpus(tmp_path / "corpus")

    corpus.validate()

    corpus.integrity_report_path.write_text("changed")
    with pytest.raises(ValueError, match="integrity SHA-256 mismatch"):
        corpus.validate()


def test_production_corpus_identity_uses_fixed_acquisition_identities(tmp_path: Path) -> None:
    corpus = CorpusIdentity.from_root(tmp_path / "m3docvqa")

    assert corpus.integrity_sha256 == FINAL_INTEGRITY_SHA256
    assert corpus.archive_checksum_manifest_sha256 == MMQA_ARCHIVES_SHA256
    assert corpus.questions_sha256 == MMQA_DEV_SHA256
    assert corpus.document_ids_sha256 == DEV_DOC_IDS_SHA256
    assert corpus.expected_question_count == 2441
    assert corpus.expected_pdf_count == 3366
    assert corpus.expected_page_count == 44638


def test_corpus_identity_rejects_incomplete_or_wrong_integrity_report(tmp_path: Path) -> None:
    corpus = make_corpus(tmp_path / "corpus")
    corpus.integrity_report_path.write_text(json.dumps({"schema_version": 1}))
    object.__setattr__(corpus, "integrity_sha256", sha256(corpus.integrity_report_path))

    with pytest.raises(ValueError, match="integrity report"):
        corpus.validate()


@pytest.mark.parametrize(
    ("path_name", "message"),
    [
        ("archive_checksum_manifest_path", "archive manifest SHA-256 mismatch"),
        ("questions_path", "MMQA_dev.jsonl SHA-256 mismatch"),
        ("document_ids_path", "dev_doc_ids.json SHA-256 mismatch"),
    ],
)
def test_corpus_identity_binds_every_acquisition_identity(
    tmp_path: Path, path_name: str, message: str
) -> None:
    corpus = make_corpus(tmp_path / "corpus")
    getattr(corpus, path_name).write_text("changed")

    with pytest.raises(ValueError, match=message):
        corpus.validate()


@pytest.mark.parametrize("mutation", ["missing", "extra", "renamed", "content"])
def test_corpus_identity_requires_exact_archive_manifest_and_preserved_archives(
    tmp_path: Path, mutation: str
) -> None:
    corpus = make_corpus(tmp_path / "corpus")
    if mutation == "missing":
        lines = corpus.archive_checksum_manifest_path.read_text().splitlines()
        corpus.archive_checksum_manifest_path.write_text("\n".join(lines[:-1]) + "\n")
        object.__setattr__(
            corpus,
            "archive_checksum_manifest_sha256",
            sha256(corpus.archive_checksum_manifest_path),
        )
    elif mutation == "extra":
        with corpus.archive_checksum_manifest_path.open("a") as stream:
            stream.write("a" * 64 + "  " + str(corpus.questions_path.parent / "extra.jsonl.gz") + "\n")
        object.__setattr__(
            corpus,
            "archive_checksum_manifest_sha256",
            sha256(corpus.archive_checksum_manifest_path),
        )
    elif mutation == "renamed":
        corpus.archive_checksum_manifest_path.write_text(
            corpus.archive_checksum_manifest_path.read_text().replace(
                "MMQA_dev.jsonl.gz", "renamed.jsonl.gz", 1
            )
        )
        object.__setattr__(
            corpus,
            "archive_checksum_manifest_sha256",
            sha256(corpus.archive_checksum_manifest_path),
        )
    else:
        (corpus.questions_path.parent / "MMQA_dev.jsonl.gz").write_bytes(b"changed")

    with pytest.raises(ValueError, match="archive"):
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


def test_run_config_binds_exact_resources_and_renders_upstream_prompt(tmp_path: Path) -> None:
    corpus = make_corpus(tmp_path / "corpus")
    contract = tmp_path / "processor-contract.json"
    contract.write_text("{}")
    config = BenchmarkRunConfig(
        mode="docprune", page_count=4, corpus=corpus, runtime_commit="a" * 40,
        m3docrag_commit=M3DOCRAG_COMMIT, qwen_model=QWEN_MODEL, qwen_revision=QWEN_REVISION,
        colpali_model=COLPALI_MODEL, colpali_revision=COLPALI_REVISION,
        colpali_backbone_model=COLPALI_BACKBONE_MODEL,
        colpali_backbone_revision=COLPALI_BACKBONE_REVISION, processor_contract_path=contract,
    )

    assert config.mode == "docprune"
    assert config.page_count == 4
    assert config.corpus == corpus
    assert config.max_new_tokens == 128
    assert config.do_sample is False
    assert config.num_beams == 1
    assert config.prompt == SHORT_ANSWER_TEMPLATE
    assert config.render_prompt("What is shown?") == "question: What is shown?\noutput only answer."
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
        "DOCPRUNE_CORPUS_ROOT": str(corpus.root), "DOCPRUNE_RUNTIME_COMMIT": "a" * 40,
        "M3DOCRAG_COMMIT": M3DOCRAG_COMMIT, "QWEN_MODEL": QWEN_MODEL,
        "QWEN_REVISION": QWEN_REVISION, "COLPALI_MODEL": COLPALI_MODEL,
        "COLPALI_REVISION": COLPALI_REVISION, "COLPALI_BACKBONE_MODEL": COLPALI_BACKBONE_MODEL,
        "COLPALI_BACKBONE_REVISION": COLPALI_BACKBONE_REVISION, "DOCPRUNE_PROCESSOR_CONTRACT": str(contract),
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
