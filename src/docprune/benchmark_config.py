"""Fail-closed immutable inputs for an M3DocVQA benchmark run."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from string import Template
from types import MappingProxyType

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
SHORT_ANSWER_TEMPLATE = "question: $question\noutput only answer."
FINAL_INTEGRITY_SHA256 = "e2581c9766157e800ba195d37905c0c25611cf4057d03e2fa32fc23e79280e39"
MMQA_ARCHIVES_SHA256 = "8ff8f1dca284a16d9a0a726ea5f0dac58f7e05a2aafaa7c2d88a56dfecd46f6d"
MMQA_DEV_SHA256 = "31192a64bfc4ffc23123c1e6657a5b57dbb9515e9a37ecbbc8578cf894dd0e3b"
DEV_DOC_IDS_SHA256 = "2d9e09689b2d1e867c566e0e893e9b53955487202921bdd8664aaa6e4037e429"
MMQA_ARCHIVE_HASHES = {
    "MMQA_dev.jsonl.gz": "2e348ca574b2dc368e84671709689070f943a59f2e08e6b6e374705deb712d31",
    "MMQA_images.jsonl.gz": "691227e1758140c51a5617a904b5e363cd44b2582a0c0e9f9fe9153d4e62d5e7",
    "MMQA_tables.jsonl.gz": "8d082d254fd0bfa19bca7e2da20369c15ccb25867db3edfe46b33d59e8dbf7b1",
    "MMQA_texts.jsonl.gz": "cae3808ccc6c258e91131a3ca3dce43e629936e1496c97a288c4abb04a6ef905",
    "MMQA_train.jsonl.gz": "2d7c8f6f1659df69f8dcdb79e4701c48e53e8674ab8573244b3ffe2e4b1da565",
}

__all__ = [
    "COLPALI_BACKBONE_MODEL",
    "COLPALI_BACKBONE_REVISION",
    "COLPALI_MODEL",
    "COLPALI_REVISION",
    "M3DOCRAG_COMMIT",
    "MAX_NEW_TOKENS",
    "BenchmarkRunConfig",
    "CorpusIdentity",
    "DEV_DOC_IDS_SHA256",
    "FINAL_INTEGRITY_SHA256",
    "MMQA_ARCHIVES_SHA256",
    "MMQA_ARCHIVE_HASHES",
    "MMQA_DEV_SHA256",
    "QWEN_MODEL",
    "QWEN_REVISION",
    "SHORT_ANSWER_TEMPLATE",
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
    archive_checksum_manifest_path: Path
    archive_checksum_manifest_sha256: str
    questions_sha256: str
    document_ids_sha256: str
    expected_question_count: int = 2441
    expected_pdf_count: int = 3366
    expected_page_count: int = 44638
    is_fixture: bool = False
    archive_hashes: Mapping[str, str] = MappingProxyType({})

    def __post_init__(self) -> None:
        if type(self.is_fixture) is not bool:
            raise TypeError("is_fixture must be a boolean")
        object.__setattr__(self, "root", Path(self.root))
        object.__setattr__(self, "questions_path", Path(self.questions_path))
        object.__setattr__(self, "document_ids_path", Path(self.document_ids_path))
        object.__setattr__(self, "pdf_dir", Path(self.pdf_dir))
        object.__setattr__(self, "integrity_report_path", Path(self.integrity_report_path))
        object.__setattr__(
            self, "archive_checksum_manifest_path", Path(self.archive_checksum_manifest_path)
        )
        for name in (
            "integrity_sha256",
            "archive_checksum_manifest_sha256",
            "questions_sha256",
            "document_ids_sha256",
        ):
            object.__setattr__(self, name, _require_sha256(getattr(self, name), name=name))
        archive_hashes = {
            str(name): _require_sha256(digest, name=f"archive hash for {name}")
            for name, digest in self.archive_hashes.items()
        }
        object.__setattr__(self, "archive_hashes", MappingProxyType(archive_hashes))
        if (
            self.expected_question_count < 1
            or self.expected_pdf_count < 1
            or self.expected_page_count < 1
        ):
            raise ValueError("expected corpus counts must be positive")
        if not self.is_fixture and (
            self.integrity_sha256 != FINAL_INTEGRITY_SHA256
            or self.archive_checksum_manifest_sha256 != MMQA_ARCHIVES_SHA256
            or self.questions_sha256 != MMQA_DEV_SHA256
            or self.document_ids_sha256 != DEV_DOC_IDS_SHA256
            or self.expected_question_count != 2441
            or self.expected_pdf_count != 3366
            or self.expected_page_count != 44638
            or dict(self.archive_hashes) != MMQA_ARCHIVE_HASHES
        ):
            raise ValueError("production corpus identity must use the pinned M3DocVQA dev values")

    @classmethod
    def from_root(cls, root: Path) -> CorpusIdentity:
        root = Path(root)
        return cls(
            root=root,
            questions_path=root / "multimodalqa" / "MMQA_dev.jsonl",
            document_ids_path=root / "dev_doc_ids.json",
            pdf_dir=root / "pdfs_dev",
            integrity_report_path=root / "attempt-3-integrity.json",
            integrity_sha256=FINAL_INTEGRITY_SHA256,
            archive_checksum_manifest_path=root / "setup" / "mmqa-archives.sha256",
            archive_checksum_manifest_sha256=MMQA_ARCHIVES_SHA256,
            questions_sha256=MMQA_DEV_SHA256,
            document_ids_sha256=DEV_DOC_IDS_SHA256,
            archive_hashes=MMQA_ARCHIVE_HASHES,
        )

    @classmethod
    def fixture(cls, **values: object) -> CorpusIdentity:
        """Create an explicitly non-production identity for small test fixtures."""

        return cls(**{**values, "is_fixture": True})  # type: ignore[arg-type]

    def validate(self) -> None:
        required = (
            (self.root, "root", True),
            (self.questions_path, "questions", False),
            (self.document_ids_path, "document IDs", False),
            (self.pdf_dir, "PDF directory", True),
            (self.integrity_report_path, "integrity report", False),
            (self.archive_checksum_manifest_path, "archive checksum manifest", False),
        )
        for path, label, directory in required:
            if not path.exists() or (path.is_dir() != directory):
                raise FileNotFoundError(f"required corpus {label} is missing: {path}")
        for path, expected, label in (
            (self.integrity_report_path, self.integrity_sha256, "integrity"),
            (
                self.archive_checksum_manifest_path,
                self.archive_checksum_manifest_sha256,
                "archive manifest",
            ),
            (self.questions_path, self.questions_sha256, "MMQA_dev.jsonl"),
            (self.document_ids_path, self.document_ids_sha256, "dev_doc_ids.json"),
        ):
            actual = sha256_file(path)
            if actual != expected:
                raise ValueError(
                    f"corpus {label} SHA-256 mismatch: expected {expected}, got {actual}"
                )
        self._validate_archives()
        try:
            report = json.loads(self.integrity_report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(
                f"corpus integrity report is not JSON: {self.integrity_report_path}"
            ) from error
        if not isinstance(report, dict):
            raise ValueError("corpus integrity report must be a JSON object")
        expected = {
            "schema_version": 1,
            "dev_questions": self.expected_question_count,
            "expected_pdf_count": self.expected_pdf_count,
            "actual_pdf_count": self.expected_pdf_count,
            "missing_pdf_ids": [],
            "extra_pdf_ids": [],
            "corrupt_pdfs": [],
            "observed_page_count": self.expected_page_count,
            "within_ten_percent_of_published_page_count": True,
        }
        if not self.is_fixture:
            expected["attempt"] = "attempt-3"
        for name, value in expected.items():
            if report.get(name) != value:
                raise ValueError(f"corpus integrity report has invalid {name!r}")

    def _validate_archives(self) -> None:
        if not self.archive_hashes:
            return
        archive_root = self.questions_path.parent
        observed: dict[str, tuple[str, Path]] = {}
        for raw_line in self.archive_checksum_manifest_path.read_text(
            encoding="utf-8"
        ).splitlines():
            if not raw_line.strip():
                continue
            parts = raw_line.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError("archive checksum manifest has an invalid entry")
            digest, raw_path = parts
            path = Path(raw_path)
            if path.name in observed:
                raise ValueError("archive checksum manifest has duplicate entries")
            observed[path.name] = (digest.lower(), path)
        if set(observed) != set(self.archive_hashes):
            raise ValueError(
                "archive checksum manifest entries do not match the pinned archive set"
            )
        actual_paths = {path.name for path in archive_root.glob("*.jsonl.gz")}
        if actual_paths != set(self.archive_hashes):
            raise ValueError("preserved MMQA archive files do not match the pinned archive set")
        for name, expected_digest in self.archive_hashes.items():
            manifest_digest, manifest_path = observed[name]
            expected_path = archive_root / name
            if manifest_path != expected_path or manifest_digest != expected_digest:
                raise ValueError(f"archive checksum manifest entry is invalid for {name}")
            actual_digest = sha256_file(expected_path)
            if actual_digest != expected_digest:
                raise ValueError(f"archive SHA-256 mismatch for {name}")


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
    prompt: str = SHORT_ANSWER_TEMPLATE

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
        if self.prompt != SHORT_ANSWER_TEMPLATE:
            raise ValueError("prompt must equal the official short-answer prompt")
        self.corpus.validate()

    @classmethod
    def from_env(cls, mode: str, page_count: int) -> BenchmarkRunConfig:
        corpus = CorpusIdentity.from_root(Path(_required_env("DOCPRUNE_CORPUS_ROOT")))
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

    def render_prompt(self, question: str) -> str:
        return Template(self.prompt).substitute(question=question)

    def make_dataset(self):
        """Construct the production corpus adapter with its fixed 2,441-row contract."""

        from docprune.m3docvqa_dataset import M3DocVQADevDataset

        return M3DocVQADevDataset(self.corpus, expected_question_count=2441)
