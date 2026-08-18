"""Canonical, immutable manifests for derived benchmark artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
import torch

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


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


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
    embeddings_sha256: str
    token2pageuid_path: Path
    token2pageuid_sha256: str
    completion_ledger_path: Path
    completion_ledger_sha256: str
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
            "embeddings_sha256",
            "token2pageuid_sha256",
            "completion_ledger_sha256",
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
        object.__setattr__(self, "token2pageuid_path", Path(self.token2pageuid_path).resolve())
        object.__setattr__(
            self, "completion_ledger_path", Path(self.completion_ledger_path).resolve()
        )
        object.__setattr__(self, "index_path", Path(self.index_path).resolve())
        object.__setattr__(self, "pruning_config", _freeze_json(self.pruning_config))
        _require_mode_artifact_paths(
            self.mode,
            self.artifact_root,
            (
                self.embeddings_path,
                self.embedding_metadata_path,
                self.token2pageuid_path,
                self.completion_ledger_path,
                self.index_path,
            ),
        )
        if len(self.embedding_shape) != 2 or self.embedding_shape[0] < 1:
            raise ValueError("embedding_shape must contain positive token and width dimensions")
        if self.embedding_shape[1] != 128:
            raise ValueError("embedding_shape must have width 128")
        if self.embedding_dtype not in {"float16", "float32", "bfloat16"}:
            raise ValueError("embedding_dtype must be float16, float32, or bfloat16")

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": 4,
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
            "pruning_config": _thaw_json(self.pruning_config),
            "artifact_root": str(self.artifact_root),
            "embeddings_path": str(self.embeddings_path),
            "embedding_metadata_path": str(self.embedding_metadata_path),
            "embeddings": {
                "shape": list(self.embedding_shape),
                "dtype": self.embedding_dtype,
            },
            "embeddings_sha256": self.embeddings_sha256,
            "token2pageuid_path": str(self.token2pageuid_path),
            "token2pageuid_sha256": self.token2pageuid_sha256,
            "completion_ledger_path": str(self.completion_ledger_path),
            "completion_ledger_sha256": self.completion_ledger_sha256,
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
                self.token2pageuid_path.resolve(),
                self.completion_ledger_path.resolve(),
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
        if not self.token2pageuid_path.is_file():
            raise FileNotFoundError(f"token2pageuid artifact is missing: {self.token2pageuid_path}")
        if not self.completion_ledger_path.is_file():
            raise FileNotFoundError(f"completion ledger is missing: {self.completion_ledger_path}")
        if not self.index_path.is_file():
            raise FileNotFoundError(f"index artifact is missing: {self.index_path}")
        actual_contract = sha256_file(self.processor_contract_path)
        if actual_contract != self.processor_contract_sha256:
            raise ValueError(
                "processor contract SHA-256 mismatch: "
                f"expected {self.processor_contract_sha256}, got {actual_contract}"
            )
        actual_embeddings = sha256_file(self.embeddings_path)
        if actual_embeddings != self.embeddings_sha256:
            raise ValueError(
                "embeddings SHA-256 mismatch: "
                f"expected {self.embeddings_sha256}, got {actual_embeddings}"
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
        embeddings, raster_indices = self._validate_safetensors()
        self._validate_token2pageuid(raster_indices)
        for path, expected, label in (
            (self.token2pageuid_path, self.token2pageuid_sha256, "token2pageuid"),
            (self.completion_ledger_path, self.completion_ledger_sha256, "completion ledger"),
        ):
            actual = sha256_file(path)
            if actual != expected:
                raise ValueError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")
        actual = sha256_file(self.index_path)
        if actual != self.index_sha256:
            raise ValueError(f"index SHA-256 mismatch: expected {self.index_sha256}, got {actual}")
        self._validate_faiss(embeddings)

    def _validate_safetensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            from safetensors.torch import load_file

            tensors = load_file(self.embeddings_path, device="cpu")
        except Exception as error:
            raise ValueError(
                f"embedding artifact is not valid safetensors: {self.embeddings_path}"
            ) from error
        if set(tensors) != {"embeddings", "raster_indices"}:
            raise ValueError("safetensors must contain embeddings and raster_indices")
        embeddings = tensors["embeddings"]
        raster_indices = tensors["raster_indices"]
        physical_dtype = str(embeddings.dtype).removeprefix("torch.")
        if tuple(embeddings.shape) != self.embedding_shape:
            raise ValueError("safetensors embedding shape mismatch")
        if physical_dtype != self.embedding_dtype:
            raise ValueError("safetensors embedding dtype mismatch")
        if tuple(raster_indices.shape) != (self.embedding_shape[0],):
            raise ValueError("safetensors raster row count mismatch")
        if str(raster_indices.dtype) != "torch.int64":
            raise ValueError("safetensors raster indices must use int64")
        if not bool(embeddings.isfinite().all()):
            raise ValueError("safetensors embeddings must be finite")
        if not bool(((raster_indices >= 0) & (raster_indices < 1024)).all()):
            raise ValueError("safetensors raster indices must be in [0, 1024)")
        return embeddings, raster_indices

    def _validate_token2pageuid(self, raster_indices: torch.Tensor) -> None:
        try:
            rows = json.loads(self.token2pageuid_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError("token2pageuid artifact must be JSON") from error
        if not isinstance(rows, list) or len(rows) != self.embedding_shape[0]:
            raise ValueError("token2pageuid row count must match embeddings")
        for row in rows:
            if (
                not isinstance(row, dict)
                or set(row) != {"doc_id", "page_index"}
                or not isinstance(row["doc_id"], str)
                or not row["doc_id"]
                or not isinstance(row["page_index"], int)
                or isinstance(row["page_index"], bool)
                or row["page_index"] < 0
            ):
                raise ValueError(
                    "token2pageuid rows must contain structured document/page identities"
                )

        start = 0
        seen_pages: set[tuple[str, int]] = set()
        while start < len(rows):
            identity = (rows[start]["doc_id"], rows[start]["page_index"])
            if identity in seen_pages:
                raise ValueError("token2pageuid page segments must remain contiguous")
            seen_pages.add(identity)
            stop = start + 1
            while stop < len(rows) and (rows[stop]["doc_id"], rows[stop]["page_index"]) == identity:
                stop += 1
            segment = raster_indices[start:stop]
            if segment.numel() == 0 or (
                segment.numel() > 1 and not bool((segment[1:] > segment[:-1]).all())
            ):
                raise ValueError(
                    "token2pageuid page segment raster indices must be strictly increasing"
                )
            start = stop

    def _validate_faiss(self, embeddings: torch.Tensor) -> None:
        try:
            import faiss

            index = faiss.read_index(str(self.index_path))
        except Exception as error:
            raise ValueError(f"index artifact is not valid FAISS: {self.index_path}") from error
        if index.d != 128 or index.ntotal != self.embedding_shape[0]:
            raise ValueError("FAISS dimensions or row count do not match embeddings")
        reconstructed = np.asarray(index.reconstruct_n(0, index.ntotal), dtype=np.float32)
        expected = embeddings.detach().cpu().numpy().astype(np.float32, copy=False)
        if not np.array_equal(reconstructed, expected):
            raise ValueError("FAISS rows do not match embeddings in stored order")


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
            (
                manifest.embeddings_path,
                manifest.embedding_metadata_path,
                manifest.token2pageuid_path,
                manifest.completion_ledger_path,
                manifest.index_path,
            ),
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
