"""Fail-closed developmental runtime contracts for Task 6.

The module contains only deterministic CPU-side identities and fixed-page
loading. It never imports or opens FAISS, a global token map, or a global
embedding tensor.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from docprune.ctp_controls import VisualTokenGeometry
from docprune.ctp_policy import (
    CTPPolicy,
    aggregate_native_threshold_policy,
    aggregate_score_top_m_policy,
    btp_qtp_no_ctp_policy,
    fixed_retention_policy,
    fixed_retention_random_policy,
    literal_native_threshold_policy,
    literal_score_top_m_policy,
    random_top_m_policy,
)
from docprune.m3docrag import RetrievalOutput, RetrievedPage, RetrievedPageFeatures, SampleInput

TASK6_RENDERER_CONTRACT = (
    "pdf2image-1.17.0|poppler-pdftoppm-26.05.0|dpi-144|rgb-contiguous|"
    "pdftoppm-sha256-1102bc3f4a12f3d3d207ac8e39f463d0fb3c403511fb4c817e776e5251b53f33"
)
TASK6_POPPLER_BIN = Path("/home/lmalveau/mamba-envs/m3docvqa-acquisition/bin")
TASK6_PDFTOPPM_SHA256 = "1102bc3f4a12f3d3d207ac8e39f463d0fb3c403511fb4c817e776e5251b53f33"
_SHA256_LENGTH = 64
_FIXTURE_KEYS = {
    "schema_version",
    "fixture_version",
    "fixed_page_provenance",
    "global_index_loaded",
    "reference_path",
    "reference_sha256",
    "eligible_questions_path",
    "eligible_questions_sha256",
    "feature_manifest_path",
    "feature_manifest_sha256",
    "completion_ledger_path",
    "completion_ledger_sha256",
    "questions",
}
_QUESTION_KEYS = {"qid", "question_sha256", "pages"}
_PAGE_KEYS = {
    "rank",
    "doc_id",
    "page_index",
    "score",
    "source_pdf_path",
    "source_pdf_sha256",
    "rendered_rgb_width",
    "rendered_rgb_height",
    "rendered_rgb_sha256",
    "renderer_contract",
    "feature_shard_path",
    "feature_shard_sha256",
    "feature_page_index",
}


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: object, label: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return str(value)


def _hash_file(path: Path) -> str:
    file_path = Path(path)
    if not file_path.is_absolute() or file_path.is_symlink() or not file_path.is_file():
        raise ValueError(f"authenticated input is missing or not a regular file: {file_path}")
    digest = hashlib.sha256()
    with file_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class GeometryIdentity:
    geometry: tuple[VisualTokenGeometry, ...]
    count: int
    sha256: str


def derive_post_qtp_geometry(
    image_grid_thw: object,
    combined_keep_mask: object,
    *,
    merge_size: int,
) -> GeometryIdentity:
    """Derive compact visual geometry from the exact page-major keep mask."""

    grid = torch.as_tensor(image_grid_thw, dtype=torch.long)
    keep = torch.as_tensor(combined_keep_mask, dtype=torch.bool)
    if grid.ndim != 2 or grid.shape[1] != 3:
        raise ValueError("image grid must have shape [pages, 3]")
    if type(merge_size) is not int or merge_size <= 0:
        raise ValueError("merge_size must be a positive integer")
    if bool((grid[:, 0] != 1).any()):
        raise ValueError("Task 6 document geometry requires temporal-one grids")
    if bool((grid[:, 1:] <= 0).any()) or bool((grid[:, 1:] % merge_size != 0).any()):
        raise ValueError("Task 6 image grids must be positive and merge divisible")
    group_counts = [
        int(height // merge_size) * int(width // merge_size) for _, height, width in grid.tolist()
    ]
    if keep.ndim != 1 or keep.numel() != sum(group_counts):
        raise ValueError("combined keep mask length does not match merged visual population")
    full: list[VisualTokenGeometry] = []
    for page_index, (_, height, width) in enumerate(grid.tolist()):
        merged_height, merged_width = int(height // merge_size), int(width // merge_size)
        full.extend(
            VisualTokenGeometry(page_index, row, column, merged_height, merged_width)
            for row in range(merged_height)
            for column in range(merged_width)
        )
    retained = tuple(token for token, selected in zip(full, keep.tolist(), strict=True) if selected)
    rows = [
        [token.page_index, token.row, token.column, token.height, token.width] for token in retained
    ]
    return GeometryIdentity(retained, len(retained), _canonical_sha256(rows))


@dataclass(frozen=True, slots=True)
class Task6PolicyCell:
    policy: CTPPolicy
    repetition: int | None


_RANDOM_NAMES = (
    "global-uniform-random",
    "page-stratified-random",
    "grid-stratified-random",
    "coverage-matched-identity-shuffle",
)
_RETENTION_NAMES = {"55": "11/20", "65": "13/20", "80": "4/5"}


def task6_policy(name: str) -> CTPPolicy:
    """Resolve only the closed Task 6 policy namespace."""

    static: dict[str, Callable[[], CTPPolicy]] = {
        "btp-qtp-no-ctp": btp_qtp_no_ctp_policy,
        "literal-native-threshold": literal_native_threshold_policy,
        "aggregate-native-threshold": aggregate_native_threshold_policy,
        "literal-score-top-m": literal_score_top_m_policy,
        "aggregate-score-top-m": aggregate_score_top_m_policy,
        **{
            selector: (lambda value=selector: random_top_m_policy(value))
            for selector in _RANDOM_NAMES
        },
    }
    factory = static.get(name)
    if factory is not None:
        return factory()
    for suffix, retention in _RETENTION_NAMES.items():
        ending = f"-retain-{suffix}"
        if not name.endswith(ending):
            continue
        base = name[: -len(ending)]
        if base in {"literal-score-top-m", "aggregate-score-top-m"}:
            return fixed_retention_policy(base.removesuffix("-score-top-m"), retention)
        if base in _RANDOM_NAMES:
            return fixed_retention_random_policy(base, retention)
    raise ValueError(f"name is not an approved Task 6 policy: {name}")


def task6_policy_matrix(kind: str) -> tuple[Task6PolicyCell, ...]:
    """Return a frozen Task 6 policy matrix."""

    if kind == "native":
        deterministic = (
            "btp-qtp-no-ctp",
            "literal-native-threshold",
            "aggregate-native-threshold",
            "literal-score-top-m",
            "aggregate-score-top-m",
        )
        return tuple(Task6PolicyCell(task6_policy(name), None) for name in deterministic) + tuple(
            Task6PolicyCell(task6_policy(name), repetition)
            for name in _RANDOM_NAMES
            for repetition in range(10)
        )
    if kind == "native-extension":
        return tuple(
            Task6PolicyCell(task6_policy(name), repetition)
            for name in _RANDOM_NAMES
            for repetition in range(10, 20)
        )
    if kind == "fixed":
        cells: list[Task6PolicyCell] = []
        for suffix in _RETENTION_NAMES:
            cells.extend(
                Task6PolicyCell(task6_policy(f"{score}-score-top-m-retain-{suffix}"), None)
                for score in ("literal", "aggregate")
            )
            cells.extend(
                Task6PolicyCell(task6_policy(f"{name}-retain-{suffix}"), repetition)
                for name in _RANDOM_NAMES
                for repetition in range(3)
            )
        return tuple(cells)
    raise ValueError("Task 6 matrix kind must be native, native-extension, or fixed")


def build_task6_gate_manifest(
    fixture: FixedPageFixture,
    *,
    fixture_path: Path,
    fixture_sha256: str,
    smoke_qid: str,
) -> dict[str, object]:
    """Build the closed 64-shard developmental policy and hardware authority."""

    _require_sha256(fixture_sha256, "fixture checksum")
    fixture_path = Path(fixture_path)
    if not fixture_path.is_absolute():
        raise ValueError("gate fixture path must be absolute")
    qids = [question.qid for question in fixture.questions]
    if smoke_qid not in qids:
        raise ValueError("smoke QID must belong to the fixed-page fixture")

    def cells(kind: str) -> list[dict[str, object]]:
        version = f"task6-{kind}-v1"
        return [
            {
                "cell": index,
                "policy": cell.policy.to_dict(),
                "experiment_version": (
                    version if cell.policy.family in {"random-top-m", "coverage-top-m"} else None
                ),
                "repetition": cell.repetition,
            }
            for index, cell in enumerate(task6_policy_matrix(kind))
        ]

    native = cells("native")
    native_extension = cells("native-extension")
    fixed = cells("fixed")
    smoke = [
        dict(native[index], cell=smoke_index)
        for smoke_index, index in enumerate((0, 1, 2, 3, 4, 5, 15, 25, 35))
    ]
    count = len(qids)
    return {
        "schema_version": 1,
        "status": "sealed-development-gate",
        "fixture_path": str(fixture_path),
        "fixture_sha256": fixture_sha256,
        "fixture_version": fixture.fixture_version,
        "fixed_page_provenance": True,
        "global_index_loaded": False,
        "renderer_contract": TASK6_RENDERER_CONTRACT,
        "smoke_qid": smoke_qid,
        "qid_shards": [{"shard": index, "qid": qid} for index, qid in enumerate(qids)],
        "native_cells": native,
        "native_extension_cells": native_extension,
        "fixed_cells": fixed,
        "smoke_cells": smoke,
        "generation_counts": {
            "smoke_per_qid": len(smoke),
            "native_per_qid": len(native),
            "fixed_per_qid": len(fixed),
            "native_total": len(native) * count,
            "native_extension_total": len(native_extension) * count,
            "fixed_total": len(fixed) * count,
        },
        "canonical_gpu_family": "L40S",
        "portability_gpu_families": ["A30", "A100-40GB", "H100", "L40S"],
        "native_random_mcse_f1_limit": 0.25,
        "conditional_additional_native_repetitions": list(range(10, 20)),
        "holdout_status": "unsealed-until-development-analysis",
        "required_result_evidence": [
            "fixed_page_fixture_sha256",
            "fixed_page_provenance",
            "global_index_loaded",
            "policy_context",
            "policy_selection.geometry_count",
            "policy_selection.geometry_sha256",
            "policy_selection.prefill_cache_lengths",
            "policy_selection.retained_mrope_position_shape",
            "policy_selection.retained_mrope_position_sha256",
        ],
    }


@dataclass(frozen=True, slots=True)
class FixedPageRecord:
    rank: int
    doc_id: str
    page_index: int
    score: float
    source_pdf_path: Path
    source_pdf_sha256: str
    rendered_rgb_width: int
    rendered_rgb_height: int
    rendered_rgb_sha256: str
    renderer_contract: str
    feature_shard_path: Path
    feature_shard_sha256: str
    feature_page_index: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_pdf_path", Path(self.source_pdf_path))
        object.__setattr__(self, "feature_shard_path", Path(self.feature_shard_path))
        if type(self.rank) is not int or self.rank < 0:
            raise ValueError("fixed page rank must be non-negative")
        if not isinstance(self.doc_id, str) or not self.doc_id:
            raise ValueError("fixed page doc_id must be nonempty")
        if type(self.page_index) is not int or self.page_index < 0:
            raise ValueError("fixed page index must be non-negative")
        if type(self.feature_page_index) is not int or self.feature_page_index != self.page_index:
            raise ValueError("feature page index must match the fixed page")
        if not isinstance(self.score, int | float) or not math.isfinite(float(self.score)):
            raise ValueError("fixed page score must be finite")
        if not self.source_pdf_path.is_absolute() or not self.feature_shard_path.is_absolute():
            raise ValueError("fixed page paths must be absolute")
        _require_sha256(self.source_pdf_sha256, "source PDF checksum")
        _require_sha256(self.rendered_rgb_sha256, "rendered RGB checksum")
        _require_sha256(self.feature_shard_sha256, "feature shard checksum")
        if type(self.rendered_rgb_width) is not int or self.rendered_rgb_width <= 0:
            raise ValueError("rendered RGB width must be positive")
        if type(self.rendered_rgb_height) is not int or self.rendered_rgb_height <= 0:
            raise ValueError("rendered RGB height must be positive")
        if self.renderer_contract != TASK6_RENDERER_CONTRACT:
            raise ValueError("fixed page renderer contract is not the pinned Task 6 contract")

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "doc_id": self.doc_id,
            "page_index": self.page_index,
            "score": float(self.score),
            "source_pdf_path": str(self.source_pdf_path),
            "source_pdf_sha256": self.source_pdf_sha256,
            "rendered_rgb_width": self.rendered_rgb_width,
            "rendered_rgb_height": self.rendered_rgb_height,
            "rendered_rgb_sha256": self.rendered_rgb_sha256,
            "renderer_contract": self.renderer_contract,
            "feature_shard_path": str(self.feature_shard_path),
            "feature_shard_sha256": self.feature_shard_sha256,
            "feature_page_index": self.feature_page_index,
        }

    @classmethod
    def from_dict(cls, value: object) -> FixedPageRecord:
        if not isinstance(value, Mapping) or set(value) != _PAGE_KEYS:
            raise ValueError("fixed page record schema is invalid")
        return cls(
            rank=value["rank"],
            doc_id=value["doc_id"],
            page_index=value["page_index"],
            score=value["score"],
            source_pdf_path=Path(value["source_pdf_path"]),
            source_pdf_sha256=value["source_pdf_sha256"],
            rendered_rgb_width=value["rendered_rgb_width"],
            rendered_rgb_height=value["rendered_rgb_height"],
            rendered_rgb_sha256=value["rendered_rgb_sha256"],
            renderer_contract=value["renderer_contract"],
            feature_shard_path=Path(value["feature_shard_path"]),
            feature_shard_sha256=value["feature_shard_sha256"],
            feature_page_index=value["feature_page_index"],
        )


@dataclass(frozen=True, slots=True)
class FixedPageQuestion:
    qid: str
    question_sha256: str
    pages: tuple[FixedPageRecord, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.qid, str) or not self.qid:
            raise ValueError("fixed question QID must be nonempty")
        _require_sha256(self.question_sha256, "question checksum")
        pages = tuple(self.pages)
        object.__setattr__(self, "pages", pages)
        if not pages or tuple(page.rank for page in pages) != tuple(range(len(pages))):
            raise ValueError("fixed pages must be in exact contiguous rank order")
        identities = [(page.doc_id, page.page_index) for page in pages]
        if len(set(identities)) != len(identities):
            raise ValueError("fixed question contains duplicate pages")

    def to_dict(self) -> dict[str, object]:
        return {
            "qid": self.qid,
            "question_sha256": self.question_sha256,
            "pages": [page.to_dict() for page in self.pages],
        }

    @classmethod
    def from_dict(cls, value: object) -> FixedPageQuestion:
        if not isinstance(value, Mapping) or set(value) != _QUESTION_KEYS:
            raise ValueError("fixed question schema is invalid")
        pages = value["pages"]
        if not isinstance(pages, Sequence) or isinstance(pages, str | bytes):
            raise ValueError("fixed question pages are invalid")
        return cls(
            value["qid"],
            value["question_sha256"],
            tuple(FixedPageRecord.from_dict(page) for page in pages),
        )


@dataclass(frozen=True, slots=True)
class FixedPageFixture:
    fixture_version: str
    reference_path: Path
    reference_sha256: str
    eligible_questions_path: Path
    eligible_questions_sha256: str
    feature_manifest_path: Path
    feature_manifest_sha256: str
    completion_ledger_path: Path
    completion_ledger_sha256: str
    questions: tuple[FixedPageQuestion, ...]

    def __post_init__(self) -> None:
        for name in (
            "reference_path",
            "eligible_questions_path",
            "feature_manifest_path",
            "completion_ledger_path",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))
        object.__setattr__(self, "questions", tuple(self.questions))
        if not isinstance(self.fixture_version, str) or not self.fixture_version:
            raise ValueError("fixture version must be nonempty")
        for name in (
            "reference_path",
            "eligible_questions_path",
            "feature_manifest_path",
            "completion_ledger_path",
        ):
            if not getattr(self, name).is_absolute():
                raise ValueError("fixture provenance paths must be absolute")
        for name in (
            "reference_sha256",
            "eligible_questions_sha256",
            "feature_manifest_sha256",
            "completion_ledger_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if not self.questions or len({question.qid for question in self.questions}) != len(
            self.questions
        ):
            raise ValueError("fixture questions must be nonempty with unique QIDs")
        if len({question.question_sha256 for question in self.questions}) != len(self.questions):
            raise ValueError("fixture question text digests must be unique")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 2,
            "fixture_version": self.fixture_version,
            "fixed_page_provenance": True,
            "global_index_loaded": False,
            "reference_path": str(self.reference_path),
            "reference_sha256": self.reference_sha256,
            "eligible_questions_path": str(self.eligible_questions_path),
            "eligible_questions_sha256": self.eligible_questions_sha256,
            "feature_manifest_path": str(self.feature_manifest_path),
            "feature_manifest_sha256": self.feature_manifest_sha256,
            "completion_ledger_path": str(self.completion_ledger_path),
            "completion_ledger_sha256": self.completion_ledger_sha256,
            "questions": [question.to_dict() for question in self.questions],
        }

    @classmethod
    def from_dict(cls, value: object) -> FixedPageFixture:
        if not isinstance(value, Mapping) or set(value) != _FIXTURE_KEYS:
            raise ValueError("fixed page fixture schema is invalid")
        if value["schema_version"] != 2 or value["fixed_page_provenance"] is not True:
            raise ValueError("fixed page fixture provenance is invalid")
        if value["global_index_loaded"] is not False:
            raise ValueError("fixed page fixture permits a global index")
        questions = value["questions"]
        if not isinstance(questions, Sequence) or isinstance(questions, str | bytes):
            raise ValueError("fixed page fixture questions are invalid")
        return cls(
            fixture_version=value["fixture_version"],
            reference_path=Path(value["reference_path"]),
            reference_sha256=value["reference_sha256"],
            eligible_questions_path=Path(value["eligible_questions_path"]),
            eligible_questions_sha256=value["eligible_questions_sha256"],
            feature_manifest_path=Path(value["feature_manifest_path"]),
            feature_manifest_sha256=value["feature_manifest_sha256"],
            completion_ledger_path=Path(value["completion_ledger_path"]),
            completion_ledger_sha256=value["completion_ledger_sha256"],
            questions=tuple(FixedPageQuestion.from_dict(question) for question in questions),
        )

    def validate_external_bytes(self, *, selected_qids: Sequence[str] | None = None) -> None:
        expected = (
            (self.reference_path, self.reference_sha256, "reference"),
            (
                self.eligible_questions_path,
                self.eligible_questions_sha256,
                "eligible questions",
            ),
            (self.feature_manifest_path, self.feature_manifest_sha256, "feature manifest"),
            (self.completion_ledger_path, self.completion_ledger_sha256, "completion ledger"),
        )
        for path, digest, label in expected:
            if _hash_file(path) != digest:
                raise ValueError(f"{label} checksum mismatch")
        selected = self.questions
        if selected_qids is not None:
            requested = tuple(selected_qids)
            if len(requested) != len(set(requested)):
                raise ValueError("selected fixture QIDs contain a duplicate")
            requested_set = set(requested)
            fixture_qids = {question.qid for question in self.questions}
            if requested_set - fixture_qids:
                raise ValueError("selected QID is absent from the fixed-page fixture")
            selected = tuple(question for question in self.questions if question.qid in requested_set)
        seen: set[tuple[Path, str, str]] = set()
        for question in selected:
            for page in question.pages:
                for path, digest, label in (
                    (page.source_pdf_path, page.source_pdf_sha256, "source PDF"),
                    (page.feature_shard_path, page.feature_shard_sha256, "feature shard"),
                ):
                    key = (path, digest, label)
                    if key not in seen:
                        if _hash_file(path) != digest:
                            raise ValueError(f"{label} checksum mismatch")
                        seen.add(key)

    def selected_samples(self, qids: Sequence[str]) -> tuple[SampleInput, ...]:
        """Read only fixture-bound questions, never the global benchmark dataset."""

        requested = tuple(qids)
        if not requested:
            raise ValueError("selected fixture QIDs must be nonempty")
        if len(requested) != len(set(requested)):
            raise ValueError("selected fixture QIDs contain a duplicate")
        requested_set = set(requested)
        fixture_by_qid = {question.qid: question for question in self.questions}
        if requested_set - set(fixture_by_qid):
            raise ValueError("selected QID is absent from the fixed-page fixture")
        if _hash_file(self.eligible_questions_path) != self.eligible_questions_sha256:
            raise ValueError("eligible questions checksum mismatch")
        rows: dict[str, Mapping[str, object]] = {}
        try:
            with self.eligible_questions_path.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if not isinstance(row, Mapping):
                        raise ValueError(f"eligible question row {line_number} is invalid")
                    qid = row.get("qid", row.get("question_id"))
                    if not isinstance(qid, str) or not qid:
                        raise ValueError(f"eligible question row {line_number} is invalid")
                    if qid in rows:
                        raise ValueError("eligible questions contain duplicate QIDs")
                    if qid in requested_set:
                        rows[qid] = row
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError("eligible questions are not valid JSONL") from error
        if set(rows) != requested_set:
            raise ValueError("selected QID is absent from eligible questions")
        samples: list[SampleInput] = []
        for question in self.questions:
            if question.qid not in requested_set:
                continue
            sample = SampleInput.from_mapping(rows[question.qid])
            if hashlib.sha256(sample.question.encode("utf-8")).hexdigest() != (
                fixture_by_qid[question.qid].question_sha256
            ):
                raise ValueError("Task 6 question text does not match the fixed-page fixture")
            samples.append(sample)
        return tuple(samples)

    def question(self, qid: str) -> FixedPageQuestion:
        for question in self.questions:
            if question.qid == qid:
                return question
        raise KeyError(f"QID is absent from fixed page fixture: {qid}")


class FixedPageFeatureStore:
    """Load only fixture-bound page slices from authenticated document shards."""

    def __init__(self, fixture: FixedPageFixture, *, validate_external_bytes: bool = True) -> None:
        if validate_external_bytes:
            fixture.validate_external_bytes()
        self.fixture = fixture
        self._cache: dict[Path, Mapping[str, torch.Tensor]] = {}

    def load(self, qid: str, rank: int) -> RetrievedPageFeatures:
        from safetensors.torch import load_file

        question = self.fixture.question(qid)
        if type(rank) is not int or not 0 <= rank < len(question.pages):
            raise IndexError("fixed page rank is outside the question")
        page = question.pages[rank]
        tensors = self._cache.get(page.feature_shard_path)
        if tensors is None:
            tensors = load_file(page.feature_shard_path, device="cpu")
            if set(tensors) != {"embeddings", "page_offsets", "raster_indices"}:
                raise ValueError("fixed feature shard has invalid keys")
            self._cache[page.feature_shard_path] = tensors
        embeddings = torch.as_tensor(tensors["embeddings"])
        offsets = torch.as_tensor(tensors["page_offsets"], dtype=torch.int64)
        rasters = torch.as_tensor(tensors["raster_indices"], dtype=torch.int64)
        index = page.feature_page_index
        if (
            embeddings.ndim != 2
            or embeddings.shape[1] != 128
            or offsets.ndim != 1
            or index + 1 >= offsets.numel()
            or int(offsets[0]) != 0
            or int(offsets[-1]) != len(embeddings)
            or rasters.ndim != 1
            or len(rasters) != len(embeddings)
        ):
            raise ValueError("fixed feature shard offsets or shapes are invalid")
        start, stop = int(offsets[index]), int(offsets[index + 1])
        page_rasters = rasters[start:stop]
        valid = page_rasters >= 0
        if not bool(valid.any()):
            raise ValueError("fixed page has no persisted visual feature rows")
        selected_rasters = page_rasters[valid]
        if bool((selected_rasters >= 1024).any()) or (
            selected_rasters.numel() > 1
            and not bool((selected_rasters[1:] > selected_rasters[:-1]).all())
        ):
            raise ValueError("fixed page raster indices are invalid")
        selected_embeddings = embeddings[start:stop][valid].to(torch.float32)
        return RetrievedPageFeatures(
            page.doc_id,
            page.page_index,
            selected_embeddings,
            selected_rasters,
            (32, 32),
        )


class AuthenticatedFixedPageLoader:
    """Render only fixture-bound pages and authenticate exact RGB bytes."""

    def __init__(
        self,
        fixture: FixedPageFixture,
        render_page: Callable[[Path, int], Image.Image],
        *,
        validate_external_bytes: bool = True,
    ) -> None:
        if validate_external_bytes:
            fixture.validate_external_bytes()
        self.fixture = fixture
        self.render_page = render_page
        self._by_identity: dict[tuple[str, int], FixedPageRecord] = {}
        self._image_cache: dict[tuple[str, int], Image.Image] = {}
        for question in fixture.questions:
            for page in question.pages:
                identity = (page.doc_id, page.page_index)
                previous = self._by_identity.get(identity)
                if previous is not None and (
                    previous.source_pdf_path,
                    previous.source_pdf_sha256,
                    previous.rendered_rgb_width,
                    previous.rendered_rgb_height,
                    previous.rendered_rgb_sha256,
                    previous.renderer_contract,
                ) != (
                    page.source_pdf_path,
                    page.source_pdf_sha256,
                    page.rendered_rgb_width,
                    page.rendered_rgb_height,
                    page.rendered_rgb_sha256,
                    page.renderer_contract,
                ):
                    raise ValueError(
                        "duplicate fixed page identity has conflicting raster provenance"
                    )
                self._by_identity.setdefault(identity, page)

    def _load_record(self, page: FixedPageRecord) -> Image.Image:
        identity = (page.doc_id, page.page_index)
        cached = self._image_cache.get(identity)
        if cached is not None:
            return cached.copy()
        image = self.render_page(page.source_pdf_path, page.page_index).convert("RGB")
        if image.size != (page.rendered_rgb_width, page.rendered_rgb_height):
            raise ValueError("rendered RGB dimensions do not match the fixture")
        if hashlib.sha256(image.tobytes()).hexdigest() != page.rendered_rgb_sha256:
            raise ValueError("rendered RGB checksum mismatch")
        self._image_cache[identity] = image.copy()
        return image

    def load_page_for_question(self, qid: str, rank: int) -> Image.Image:
        question = self.fixture.question(qid)
        if type(rank) is not int or not 0 <= rank < len(question.pages):
            raise IndexError("fixed page rank is outside the question")
        return self._load_record(question.pages[rank])

    def load_page(self, doc_id: str, page_index: int) -> Image.Image:
        try:
            page = self._by_identity[(doc_id, page_index)]
        except KeyError as error:
            raise KeyError("page identity is absent from the fixed-page fixture") from error
        return self._load_record(page)


class AuthenticatedFixedPageRetriever:
    """Return sealed page order and persisted features after query encoding only."""

    def __init__(
        self,
        fixture: FixedPageFixture,
        query_encoder: object,
        *,
        validate_external_bytes: bool = True,
    ) -> None:
        if validate_external_bytes:
            fixture.validate_external_bytes()
        encode_queries = getattr(query_encoder, "encode_queries", None)
        if not callable(encode_queries):
            raise ValueError("fixed-page query encoder must expose encode_queries")
        self.fixture = fixture
        self.query_encoder = query_encoder
        self.feature_store = FixedPageFeatureStore(fixture, validate_external_bytes=False)
        self._by_question_sha256 = {
            question.question_sha256: question for question in fixture.questions
        }
        self._cache: dict[tuple[str, int], RetrievalOutput] = {}

    def release_query_encoder(self) -> None:
        """Release ColPali after every selected question has been primed."""

        self.query_encoder = None

    def retrieve(self, question: str, top_k: int) -> RetrievalOutput:
        if not isinstance(question, str):
            raise TypeError("fixed-page question must be text")
        digest = hashlib.sha256(question.encode("utf-8")).hexdigest()
        fixed = self._by_question_sha256.get(digest)
        if fixed is None:
            raise KeyError("question is absent from the fixed-page fixture")
        if type(top_k) is not int or top_k != len(fixed.pages):
            raise ValueError("requested top_k does not match the sealed page count")
        cached = self._cache.get((digest, top_k))
        if cached is not None:
            return cached
        if self.query_encoder is None:
            raise RuntimeError(
                "fixed-page query encoder was released before this question was primed"
            )
        encoded = self.query_encoder.encode_queries([question])
        query = torch.as_tensor(encoded)
        if query.ndim == 3 and query.shape[0] == 1:
            query = query[0]
        if query.ndim != 2 or query.shape[1] != 128:
            raise ValueError("fixed-page query embeddings must have shape [tokens, 128]")
        pages = tuple(
            RetrievedPage(page.doc_id, page.page_index, float(page.score)) for page in fixed.pages
        )
        features = tuple(
            self.feature_store.load(fixed.qid, rank) for rank in range(len(fixed.pages))
        )
        output = RetrievalOutput(pages, query.to(torch.float32), features)
        self._cache[(digest, top_k)] = output
        return output


def render_task6_pdf_page(path: Path, page_index: int) -> Image.Image:
    """Render one page with the exact historical 144-DPI Poppler contract."""

    if type(page_index) is not int or page_index < 0:
        raise ValueError("PDF page index must be non-negative")
    from pdf2image import convert_from_path

    pdftoppm = TASK6_POPPLER_BIN / "pdftoppm"
    if _hash_file(pdftoppm) != TASK6_PDFTOPPM_SHA256:
        raise ValueError("Task 6 pdftoppm binary checksum mismatch")

    pages = convert_from_path(
        str(path),
        dpi=144,
        first_page=page_index + 1,
        last_page=page_index + 1,
        thread_count=1,
        poppler_path=str(TASK6_POPPLER_BIN),
    )
    if len(pages) != 1:
        raise ValueError("fixed PDF page did not render exactly once")
    return pages[0].convert("RGB")


def _load_json_object(path: Path, label: str) -> Mapping[str, object]:
    _hash_file(path)
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def build_fixed_page_fixture(
    *,
    reference_path: Path,
    eligible_questions_path: Path,
    feature_manifest_path: Path,
    pdf_dir: Path,
    fixture_version: str,
    page_count: int,
    render_page: Callable[[Path, int], Image.Image] = render_task6_pdf_page,
) -> FixedPageFixture:
    """Seal existing fixed pages and document shards without search or rebuilding."""

    reference_path = Path(reference_path).resolve()
    eligible_questions_path = Path(eligible_questions_path).resolve()
    feature_manifest_path = Path(feature_manifest_path).resolve()
    pdf_dir = Path(pdf_dir).resolve()
    if type(page_count) is not int or page_count not in {1, 2, 4}:
        raise ValueError("fixed fixture page_count must be 1, 2, or 4")
    if not pdf_dir.is_dir() or pdf_dir.is_symlink():
        raise ValueError("fixed fixture PDF directory must be a real directory")

    reference = _load_json_object(reference_path, "fixed-page reference")
    if reference.get("selection_is_outcome_blind") is not True:
        raise ValueError("fixed-page reference must be outcome blind")
    qids = reference.get("question_ids")
    rows = reference.get("rows")
    if (
        not isinstance(qids, list)
        or not qids
        or any(not isinstance(qid, str) or not qid for qid in qids)
        or len(qids) != len(set(qids))
        or not isinstance(rows, Mapping)
        or set(rows) != set(qids)
    ):
        raise ValueError("fixed-page reference QID schema is invalid")

    questions_by_qid: dict[str, str] = {}
    _hash_file(eligible_questions_path)
    try:
        with eligible_questions_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                item = json.loads(line)
                if not isinstance(item, Mapping):
                    raise ValueError("eligible question row must be a JSON object")
                qid = item.get("qid", item.get("question_id"))
                question = item.get("question")
                if not isinstance(qid, str) or not qid or not isinstance(question, str):
                    raise ValueError(f"eligible question row {line_number} is invalid")
                if qid in questions_by_qid:
                    raise ValueError("eligible questions contain duplicate QIDs")
                questions_by_qid[qid] = question
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("eligible questions are not valid JSONL") from error
    if any(qid not in questions_by_qid for qid in qids):
        raise ValueError("fixed-page reference QID is absent from eligible questions")

    feature_manifest = _load_json_object(feature_manifest_path, "feature manifest")
    if (
        feature_manifest.get("schema_version") != 5
        or feature_manifest.get("mode") != "docprune"
        or feature_manifest.get("page_count") != page_count
    ):
        raise ValueError("feature manifest is not the requested schema-5 DocPrune identity")
    raw_ledger_path = feature_manifest.get("completion_ledger_path")
    ledger_sha256 = feature_manifest.get("completion_ledger_sha256")
    if not isinstance(raw_ledger_path, str) or not _is_sha256(ledger_sha256):
        raise ValueError("feature manifest completion-ledger identity is invalid")
    ledger_path = Path(raw_ledger_path).resolve()
    if _hash_file(ledger_path) != ledger_sha256:
        raise ValueError("completion ledger checksum mismatch")
    try:
        ledger_rows = json.loads(ledger_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("completion ledger is not valid JSON") from error
    if not isinstance(ledger_rows, list):
        raise ValueError("completion ledger must be a JSON list")
    ledger_by_doc: dict[str, Mapping[str, object]] = {}
    for ledger_row in ledger_rows:
        if not isinstance(ledger_row, Mapping):
            raise ValueError("completion ledger row is invalid")
        doc_id = ledger_row.get("doc_id")
        if not isinstance(doc_id, str) or not doc_id or doc_id in ledger_by_doc:
            raise ValueError("completion ledger document identity is invalid")
        ledger_by_doc[doc_id] = ledger_row

    pdf_hashes: dict[Path, str] = {}
    shard_hashes: dict[Path, str] = {}
    rendered: dict[tuple[str, int], tuple[int, int, str]] = {}
    sealed_questions: list[FixedPageQuestion] = []
    for qid in qids:
        row = rows[qid]
        pages = row.get("retrieved_pages") if isinstance(row, Mapping) else None
        if not isinstance(pages, list) or len(pages) != page_count:
            raise ValueError("fixed-page reference has the wrong ordered page count")
        sealed_pages: list[FixedPageRecord] = []
        for rank, page in enumerate(pages):
            if not isinstance(page, Mapping) or set(page) != {"doc_id", "page_index", "score"}:
                raise ValueError("fixed-page reference page schema is invalid")
            doc_id = page["doc_id"]
            page_index = page["page_index"]
            score = page["score"]
            if (
                not isinstance(doc_id, str)
                or not doc_id
                or type(page_index) is not int
                or page_index < 0
                or not isinstance(score, int | float)
                or isinstance(score, bool)
                or not math.isfinite(float(score))
            ):
                raise ValueError("fixed-page reference page identity is invalid")
            ledger = ledger_by_doc.get(doc_id)
            if ledger is None:
                raise ValueError("fixed page document is absent from the completion ledger")
            ledger_pages = ledger.get("pages")
            if (
                ledger.get("schema_version") != 5
                or not isinstance(ledger_pages, list)
                or page_index >= len(ledger_pages)
                or ledger_pages[page_index]
                != {"doc_id": doc_id, "page_index": page_index, "source_hw": [32, 32]}
            ):
                raise ValueError("fixed page does not match its feature-shard page identity")
            shard_path = Path(str(ledger.get("document_path"))).resolve()
            shard_sha256 = ledger.get("sha256")
            if not _is_sha256(shard_sha256):
                raise ValueError("feature shard checksum in completion ledger is invalid")
            actual_shard = shard_hashes.get(shard_path)
            if actual_shard is None:
                actual_shard = _hash_file(shard_path)
                shard_hashes[shard_path] = actual_shard
            if actual_shard != shard_sha256:
                raise ValueError("feature shard checksum mismatch")
            pdf_path = (pdf_dir / f"{doc_id}.pdf").resolve()
            pdf_sha256 = pdf_hashes.get(pdf_path)
            if pdf_sha256 is None:
                pdf_sha256 = _hash_file(pdf_path)
                pdf_hashes[pdf_path] = pdf_sha256
            render_key = (doc_id, page_index)
            if render_key not in rendered:
                image = render_page(pdf_path, page_index).convert("RGB")
                rendered[render_key] = (
                    image.width,
                    image.height,
                    hashlib.sha256(image.tobytes()).hexdigest(),
                )
            width, height, rendered_sha256 = rendered[render_key]
            sealed_pages.append(
                FixedPageRecord(
                    rank=rank,
                    doc_id=doc_id,
                    page_index=page_index,
                    score=float(score),
                    source_pdf_path=pdf_path,
                    source_pdf_sha256=pdf_sha256,
                    rendered_rgb_width=width,
                    rendered_rgb_height=height,
                    rendered_rgb_sha256=rendered_sha256,
                    renderer_contract=TASK6_RENDERER_CONTRACT,
                    feature_shard_path=shard_path,
                    feature_shard_sha256=str(shard_sha256),
                    feature_page_index=page_index,
                )
            )
        sealed_questions.append(
            FixedPageQuestion(
                qid,
                hashlib.sha256(questions_by_qid[qid].encode("utf-8")).hexdigest(),
                tuple(sealed_pages),
            )
        )
    return FixedPageFixture(
        fixture_version=fixture_version,
        reference_path=reference_path,
        reference_sha256=_hash_file(reference_path),
        eligible_questions_path=eligible_questions_path,
        eligible_questions_sha256=_hash_file(eligible_questions_path),
        feature_manifest_path=feature_manifest_path,
        feature_manifest_sha256=_hash_file(feature_manifest_path),
        completion_ledger_path=ledger_path,
        completion_ledger_sha256=str(ledger_sha256),
        questions=tuple(sealed_questions),
    )


def _publish_json_no_replace(payload: object, destination: Path, *, label: str) -> str:
    target = Path(destination).absolute()
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"{label} already exists: {target}")
    if not target.parent.is_dir() or target.parent.is_symlink():
        raise ValueError(f"{label} parent must be a real existing directory")
    content = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")
    digest = hashlib.sha256(content).hexdigest()
    descriptor, raw = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        descriptor = -1
        os.link(temporary, target)
        directory = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor != -1:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
    return digest


def publish_fixed_page_fixture(fixture: FixedPageFixture, destination: Path) -> str:
    """Publish canonical fixture JSON atomically without replacing any path."""

    fixture.validate_external_bytes()
    return _publish_json_no_replace(fixture.to_dict(), destination, label="fixed page fixture")


def publish_task6_gate_manifest(manifest: Mapping[str, object], destination: Path) -> str:
    """Publish a closed Task 6 developmental gate atomically without replacement."""

    if manifest.get("schema_version") != 1 or manifest.get("status") != ("sealed-development-gate"):
        raise ValueError("Task 6 gate manifest schema/status is invalid")
    if (
        manifest.get("fixed_page_provenance") is not True
        or manifest.get("global_index_loaded") is not False
    ):
        raise ValueError("Task 6 gate manifest does not enforce fixed-page no-search input")
    return _publish_json_no_replace(manifest, destination, label="Task 6 gate manifest")


def load_fixed_page_fixture(
    path: Path, *, expected_sha256: str, validate_external_bytes: bool = True
) -> FixedPageFixture:
    _require_sha256(expected_sha256, "fixture checksum")
    fixture_path = Path(path)
    actual = _hash_file(fixture_path)
    if actual != expected_sha256:
        raise ValueError("fixture checksum mismatch")
    try:
        payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("fixed page fixture is not valid JSON") from error
    fixture = FixedPageFixture.from_dict(payload)
    if validate_external_bytes:
        fixture.validate_external_bytes()
    return fixture


@dataclass(frozen=True, slots=True)
class Task6ResultIdentity:
    fixture_sha256: str
    policy_name: str
    experiment_version: object
    repetition: object

    def __post_init__(self) -> None:
        _require_sha256(self.fixture_sha256, "fixture checksum")
        task6_policy(self.policy_name)


def validate_task6_result_record(
    record: Mapping[str, object], expected: Task6ResultIdentity
) -> None:
    """Reject fixed-input, policy, seed-context, or geometry drift."""

    if record.get("fixed_page_fixture_sha256") != expected.fixture_sha256:
        raise ValueError("Task 6 result fixture checksum mismatch")
    if record.get("fixed_page_provenance") is not True:
        raise ValueError("Task 6 result lacks fixed page provenance")
    if record.get("global_index_loaded") is not False:
        raise ValueError("Task 6 result permits or loaded a global index")
    selection = record.get("policy_selection")
    if not isinstance(selection, Mapping):
        raise ValueError("Task 6 result lacks policy selection evidence")
    policy = selection.get("policy")
    if not isinstance(policy, Mapping) or policy.get("name") != expected.policy_name:
        raise ValueError("Task 6 result policy identity mismatch")
    population = selection.get("visual_population")
    geometry_count = selection.get("geometry_count")
    if (
        type(population) is not int
        or type(geometry_count) is not int
        or population != geometry_count
        or not _is_sha256(selection.get("geometry_sha256"))
    ):
        raise ValueError("Task 6 result geometry identity is invalid")
    cache_lengths = selection.get("prefill_cache_lengths")
    position_shape = selection.get("retained_mrope_position_shape")
    if (
        not isinstance(cache_lengths, list)
        or not cache_lengths
        or any(type(length) is not int or length < 0 for length in cache_lengths)
        or not isinstance(position_shape, list)
        or len(position_shape) != 3
        or any(type(size) is not int or size < 0 for size in position_shape)
        or not _is_sha256(selection.get("retained_mrope_position_sha256"))
    ):
        raise ValueError("Task 6 result cache or M-RoPE identity is invalid")
    context = record.get("policy_context")
    if (
        not isinstance(context, Mapping)
        or context.get("experiment_version") != expected.experiment_version
        or context.get("repetition") != expected.repetition
    ):
        raise ValueError("Task 6 result policy context mismatch")
    resolved = task6_policy(expected.policy_name)
    if resolved.family in {"random-top-m", "coverage-top-m"}:
        seed = selection.get("seed_sha256")
        if selection.get("boundary") is None:
            if seed is not None:
                raise ValueError("Task 6 random no-crossing result has a seed checksum")
        elif not _is_sha256(seed):
            raise ValueError("Task 6 random result lacks a seed checksum")


__all__ = [
    "TASK6_RENDERER_CONTRACT",
    "TASK6_POPPLER_BIN",
    "AuthenticatedFixedPageLoader",
    "AuthenticatedFixedPageRetriever",
    "FixedPageFeatureStore",
    "FixedPageFixture",
    "FixedPageQuestion",
    "FixedPageRecord",
    "GeometryIdentity",
    "Task6PolicyCell",
    "Task6ResultIdentity",
    "build_task6_gate_manifest",
    "build_fixed_page_fixture",
    "derive_post_qtp_geometry",
    "load_fixed_page_fixture",
    "publish_fixed_page_fixture",
    "publish_task6_gate_manifest",
    "render_task6_pdf_page",
    "task6_policy",
    "task6_policy_matrix",
    "validate_task6_result_record",
]
