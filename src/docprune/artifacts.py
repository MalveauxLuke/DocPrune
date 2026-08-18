"""Canonical, immutable manifests for derived benchmark artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from docprune.benchmark_config import M3DOCRAG_COMMIT, MODES, PAGE_COUNTS, sha256_file
from docprune.processor_probe import require_immutable_revision, require_pinned_processor_resources


def canonical_json_sha256(value: object) -> str:
    """Hash JSON with stable keys and no insignificant whitespace."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _require_sha256(value: str, *, name: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdefABCDEF" for character in value):
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA-256")
    return value.lower()


@dataclass(frozen=True)
class IndexManifest:
    """Identity of one mode/page-count index and its checked derived files."""

    mode: str
    page_count: int
    corpus_integrity_sha256: str
    source_order_sha256: str
    runtime_commit: str
    m3docrag_commit: str
    qwen_model: str
    qwen_revision: str
    colpali_model: str
    colpali_revision: str
    colpali_backbone_model: str
    colpali_backbone_revision: str
    processor_contract_path: Path
    processor_contract_sha256: str
    pruning_config: Mapping[str, object]
    artifact_root: Path
    embeddings_path: Path
    embedding_metadata_path: Path
    embedding_shape: tuple[int, int]
    embedding_dtype: str
    index_path: Path
    index_sha256: str

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}")
        if self.page_count not in PAGE_COUNTS:
            raise ValueError("page_count must be 1, 2, or 4")
        for name in (
            "corpus_integrity_sha256",
            "source_order_sha256",
            "processor_contract_sha256",
            "index_sha256",
        ):
            object.__setattr__(self, name, _require_sha256(getattr(self, name), name=name))
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
        object.__setattr__(self, "artifact_root", Path(self.artifact_root).resolve())
        object.__setattr__(self, "embeddings_path", Path(self.embeddings_path).resolve())
        object.__setattr__(
            self, "embedding_metadata_path", Path(self.embedding_metadata_path).resolve()
        )
        object.__setattr__(self, "processor_contract_path", Path(self.processor_contract_path))
        object.__setattr__(self, "index_path", Path(self.index_path).resolve())
        object.__setattr__(self, "pruning_config", MappingProxyType(dict(self.pruning_config)))
        _require_mode_artifact_paths(
            self.mode,
            self.artifact_root,
            (self.embeddings_path, self.embedding_metadata_path, self.index_path),
        )
        if len(self.embedding_shape) != 2 or self.embedding_shape[0] < 1:
            raise ValueError("embedding_shape must contain positive token and width dimensions")
        if self.embedding_shape[1] != 128:
            raise ValueError("embedding_shape must have width 128")
        if self.embedding_dtype not in {"float16", "float32", "bfloat16"}:
            raise ValueError("embedding_dtype must be float16, float32, or bfloat16")

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": 2,
            "mode": self.mode,
            "page_count": self.page_count,
            "corpus_integrity_sha256": self.corpus_integrity_sha256,
            "source_order_sha256": self.source_order_sha256,
            "runtime_commit": self.runtime_commit,
            "m3docrag_commit": self.m3docrag_commit,
            "resources": {
                "qwen": {"model": self.qwen_model, "revision": self.qwen_revision},
                "colpali": {"model": self.colpali_model, "revision": self.colpali_revision},
                "colpali_backbone": {
                    "model": self.colpali_backbone_model,
                    "revision": self.colpali_backbone_revision,
                },
            },
            "processor_contract_path": str(self.processor_contract_path),
            "processor_contract_sha256": self.processor_contract_sha256,
            "pruning_config": dict(self.pruning_config),
            "artifact_root": str(self.artifact_root),
            "embeddings_path": str(self.embeddings_path),
            "embedding_metadata_path": str(self.embedding_metadata_path),
            "embeddings": {
                "shape": list(self.embedding_shape),
                "dtype": self.embedding_dtype,
            },
            "index_path": str(self.index_path),
            "index_sha256": self.index_sha256,
        }
        payload["manifest_sha256"] = canonical_json_sha256(payload)
        return payload

    def validate_files(self) -> None:
        _require_mode_artifact_paths(
            self.mode,
            self.artifact_root.resolve(),
            (
                self.embeddings_path.resolve(),
                self.embedding_metadata_path.resolve(),
                self.index_path.resolve(),
            ),
        )
        if not self.embeddings_path.is_file():
            raise FileNotFoundError(f"embedding artifact is missing: {self.embeddings_path}")
        if not self.embedding_metadata_path.is_file():
            raise FileNotFoundError(
                f"embedding metadata is missing: {self.embedding_metadata_path}"
            )
        if not self.processor_contract_path.is_file():
            raise FileNotFoundError(
                f"processor contract is missing: {self.processor_contract_path}"
            )
        if not self.index_path.is_file():
            raise FileNotFoundError(f"index artifact is missing: {self.index_path}")
        actual_contract = sha256_file(self.processor_contract_path)
        if actual_contract != self.processor_contract_sha256:
            raise ValueError(
                "processor contract SHA-256 mismatch: "
                f"expected {self.processor_contract_sha256}, got {actual_contract}"
            )
        try:
            metadata = json.loads(self.embedding_metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(
                f"embedding metadata is not JSON: {self.embedding_metadata_path}"
            ) from error
        if not isinstance(metadata, dict):
            raise ValueError("embedding metadata must be a JSON object")
        if metadata.get("shape") != list(self.embedding_shape):
            raise ValueError("embedding metadata shape mismatch")
        if metadata.get("dtype") != self.embedding_dtype:
            raise ValueError("embedding metadata dtype mismatch")
        actual = sha256_file(self.index_path)
        if actual != self.index_sha256:
            raise ValueError(f"index SHA-256 mismatch: expected {self.index_sha256}, got {actual}")


def validate_index_manifest_pair(first: IndexManifest, second: IndexManifest) -> None:
    """Reject artifact reuse between the baseline and pruning comparison modes."""

    if first.mode == second.mode:
        raise ValueError("manifest pair must contain one all-kept and one docprune manifest")
    if first.embeddings_path.parent.resolve() == second.embeddings_path.parent.resolve():
        raise ValueError("manifest pair must use distinct embedding roots")
    if first.index_path.resolve() == second.index_path.resolve():
        raise ValueError("manifest pair must use distinct index paths")
    for manifest in (first, second):
        _require_mode_artifact_paths(
            manifest.mode,
            manifest.artifact_root,
            (manifest.embeddings_path, manifest.embedding_metadata_path, manifest.index_path),
        )


def _require_mode_artifact_paths(mode: str, artifact_root: Path, paths: tuple[Path, ...]) -> None:
    """Require mode-scoped artifacts to remain under one resolved root."""

    other = "docprune" if mode == "all-kept" else "all-kept"
    if mode not in artifact_root.parts or other in artifact_root.parts:
        raise ValueError(f"manifest mode/path mismatch for {mode}")
    for path in paths:
        try:
            path.relative_to(artifact_root)
        except ValueError as error:
            raise ValueError(
                f"artifact path must be contained under artifact root: {artifact_root}"
            ) from error
    if any(mode not in path.parts or other in path.parts for path in paths):
        raise ValueError(f"manifest mode/path mismatch for {mode}")
