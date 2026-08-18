"""Fail-closed immutable inputs for an M3DocVQA benchmark run."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from docprune.processor_probe import (
    COLPALI_BACKBONE_MODEL,
    COLPALI_BACKBONE_REVISION,
    COLPALI_MODEL,
    COLPALI_REVISION,
    QWEN_MODEL,
    QWEN_REVISION,
    require_immutable_revision,
    require_pinned_processor_resources,
)

M3DOCRAG_COMMIT = "29e6ac2294d6b87075a1d45b8a8df175b214248a"
MODES = ("all-kept", "docprune")
PAGE_COUNTS = (1, 2, 4)
MAX_NEW_TOKENS = 128
SHORT_ANSWER_PROMPT = "Answer the question using the image. Answer concisely."

__all__ = [
    "COLPALI_BACKBONE_MODEL",
    "COLPALI_BACKBONE_REVISION",
    "COLPALI_MODEL",
    "COLPALI_REVISION",
    "M3DOCRAG_COMMIT",
    "MAX_NEW_TOKENS",
    "BenchmarkRunConfig",
    "CorpusIdentity",
    "QWEN_MODEL",
    "QWEN_REVISION",
    "SHORT_ANSWER_PROMPT",
    "sha256_file",
]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of a regular file without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: str, *, name: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdefABCDEF" for character in value):
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA-256")
    return value.lower()


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"required environment variable {name} is not set")
    return value


@dataclass(frozen=True)
class CorpusIdentity:
    """The acquired M3DocVQA dev corpus, addressed only by immutable files."""

    root: Path
    questions_path: Path
    document_ids_path: Path
    pdf_dir: Path
    integrity_report_path: Path
    integrity_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        object.__setattr__(self, "questions_path", Path(self.questions_path))
        object.__setattr__(self, "document_ids_path", Path(self.document_ids_path))
        object.__setattr__(self, "pdf_dir", Path(self.pdf_dir))
        object.__setattr__(self, "integrity_report_path", Path(self.integrity_report_path))
        object.__setattr__(
            self,
            "integrity_sha256",
            _require_sha256(self.integrity_sha256, name="integrity_sha256"),
        )

    @classmethod
    def from_root(cls, root: Path, *, integrity_sha256: str) -> CorpusIdentity:
        root = Path(root)
        return cls(
            root=root,
            questions_path=root / "multimodalqa" / "MMQA_dev.jsonl",
            document_ids_path=root / "dev_doc_ids.json",
            pdf_dir=root / "pdfs_dev",
            integrity_report_path=root / "attempt-3-integrity.json",
            integrity_sha256=integrity_sha256,
        )

    def validate(self) -> None:
        required = (
            (self.root, "root", True),
            (self.questions_path, "questions", False),
            (self.document_ids_path, "document IDs", False),
            (self.pdf_dir, "PDF directory", True),
            (self.integrity_report_path, "integrity report", False),
        )
        for path, label, directory in required:
            if not path.exists() or (path.is_dir() != directory):
                raise FileNotFoundError(f"required corpus {label} is missing: {path}")
        actual = sha256_file(self.integrity_report_path)
        if actual != self.integrity_sha256:
            raise ValueError(
                "corpus integrity SHA-256 mismatch: "
                f"expected {self.integrity_sha256}, got {actual}"
            )
        try:
            report = json.loads(self.integrity_report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"corpus integrity report is not JSON: {self.integrity_report_path}") from error
        if not isinstance(report, dict):
            raise ValueError("corpus integrity report must be a JSON object")
        for name in ("missing_pdf_ids", "extra_pdf_ids", "corrupt_pdfs"):
            if name in report and report[name]:
                raise ValueError(f"corpus integrity report records {name}")


@dataclass(frozen=True)
class BenchmarkRunConfig:
    """Fully pinned model, corpus, and decoding inputs for one comparison cell."""

    mode: str
    page_count: int
    corpus: CorpusIdentity
    runtime_commit: str
    m3docrag_commit: str
    qwen_model: str
    qwen_revision: str
    colpali_model: str
    colpali_revision: str
    colpali_backbone_model: str
    colpali_backbone_revision: str
    processor_contract_path: Path
    max_new_tokens: int = MAX_NEW_TOKENS
    do_sample: bool = False
    num_beams: int = 1
    prompt: str = SHORT_ANSWER_PROMPT

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}")
        if self.page_count not in PAGE_COUNTS:
            raise ValueError("page_count must be 1, 2, or 4")
        require_immutable_revision(self.runtime_commit, name="runtime_commit")
        if self.m3docrag_commit != M3DOCRAG_COMMIT:
            raise ValueError(f"m3docrag_commit must equal the pinned {M3DOCRAG_COMMIT}")
        require_pinned_processor_resources(
            qwen_model=self.qwen_model,
            qwen_revision=self.qwen_revision,
            colpali_model=self.colpali_model,
            colpali_revision=self.colpali_revision,
            colpali_backbone_model=self.colpali_backbone_model,
            colpali_backbone_revision=self.colpali_backbone_revision,
        )
        object.__setattr__(self, "processor_contract_path", Path(self.processor_contract_path))
        if not self.processor_contract_path.is_file():
            raise FileNotFoundError(
                f"required processor contract is missing: {self.processor_contract_path}"
            )
        if self.max_new_tokens != MAX_NEW_TOKENS:
            raise ValueError(f"max_new_tokens must equal {MAX_NEW_TOKENS}")
        if self.do_sample or self.num_beams != 1:
            raise ValueError("generation must use greedy decoding (do_sample=False, num_beams=1)")
        if self.prompt != SHORT_ANSWER_PROMPT:
            raise ValueError("prompt must equal the official short-answer prompt")
        self.corpus.validate()

    @classmethod
    def from_env(cls, mode: str, page_count: int) -> BenchmarkRunConfig:
        corpus = CorpusIdentity.from_root(
            Path(_required_env("DOCPRUNE_CORPUS_ROOT")),
            integrity_sha256=_required_env("DOCPRUNE_CORPUS_INTEGRITY_SHA256"),
        )
        return cls(
            mode=mode,
            page_count=page_count,
            corpus=corpus,
            runtime_commit=_required_env("DOCPRUNE_RUNTIME_COMMIT"),
            m3docrag_commit=_required_env("M3DOCRAG_COMMIT"),
            qwen_model=_required_env("QWEN_MODEL"),
            qwen_revision=_required_env("QWEN_REVISION"),
            colpali_model=_required_env("COLPALI_MODEL"),
            colpali_revision=_required_env("COLPALI_REVISION"),
            colpali_backbone_model=_required_env("COLPALI_BACKBONE_MODEL"),
            colpali_backbone_revision=_required_env("COLPALI_BACKBONE_REVISION"),
            processor_contract_path=Path(_required_env("DOCPRUNE_PROCESSOR_CONTRACT")),
        )
