"""Deterministic MinerU page regions and whole-region visual-token masks."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path

from docprune.ctp_controls import VisualTokenGeometry
from docprune.task6_runtime import FixedPageFixture

REGION_MAPPING_SCHEMA_VERSION = "docprune-mineru-region-mapping-v2"
ASSIGNMENT_CONTRACT = (
    "maximum-positive-normalized-intersection;"
    "ties=reading-order-then-region-id;boundary-only=residual;"
    "residual-grid=normalized-token-center"
)


def _finite_number(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _require_revision(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase 40-character revision")
    return value


def _page_size(value: object) -> tuple[float, float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, str | bytes)
        or len(value) != 2
        or any(not _finite_number(item) or float(item) <= 0 for item in value)
    ):
        raise ValueError("MinerU page_size must contain positive finite width and height")
    return (float(value[0]), float(value[1]))


def _box(value: object, *, page_size: tuple[float, float], label: str) -> tuple[float, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, str | bytes)
        or len(value) != 4
        or any(not _finite_number(item) for item in value)
    ):
        raise ValueError(f"{label} bbox must contain four finite coordinates")
    x0, y0, x1, y1 = (float(item) for item in value)
    width, height = page_size
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f"{label} bbox must be a positive box within the page")
    return (x0, y0, x1, y1)


def _canonical_sha256(value: object) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(data).hexdigest()


def _validate_page_coordinates(
    input_page_index: object,
    mineru_page_index: object,
    document_id: object,
    source_page_index: object,
) -> None:
    if type(input_page_index) is not int or input_page_index < 0:
        raise ValueError("input page index must be non-negative")
    if type(mineru_page_index) is not int or mineru_page_index < 0:
        raise ValueError("MinerU-local page index must be non-negative")
    if not isinstance(document_id, str) or not document_id or ":" in document_id:
        raise ValueError("document ID must be nonempty and contain no colon")
    if type(source_page_index) is not int or source_page_index < 0:
        raise ValueError("source page index must be non-negative")


@dataclass(frozen=True, slots=True)
class InputPageIdentity:
    """Keep Qwen slot, MinerU-local page, and original source page distinct."""

    input_page_index: int
    mineru_page_index: int
    document_id: str
    source_page_index: int
    source_pdf_path: Path
    source_pdf_sha256: str
    fixture_question_id: str
    fixture_rank: int
    fixed_page_fixture_path: Path
    fixed_page_fixture_sha256: str
    fixed_page_record_sha256: str
    rendered_rgb_width: int
    rendered_rgb_height: int
    rendered_rgb_sha256: str
    renderer_contract: str

    def __post_init__(self) -> None:
        source_path = Path(self.source_pdf_path)
        fixture_path = Path(self.fixed_page_fixture_path)
        object.__setattr__(self, "source_pdf_path", source_path)
        object.__setattr__(self, "fixed_page_fixture_path", fixture_path)
        _validate_page_coordinates(
            self.input_page_index,
            self.mineru_page_index,
            self.document_id,
            self.source_page_index,
        )
        if not source_path.is_absolute() or not fixture_path.is_absolute():
            raise ValueError("fixed input page provenance paths must be absolute")
        _require_sha256(self.source_pdf_sha256, "source PDF SHA-256")
        _require_sha256(self.fixed_page_fixture_sha256, "fixed page fixture SHA-256")
        _require_sha256(self.fixed_page_record_sha256, "fixed page record SHA-256")
        _require_sha256(self.rendered_rgb_sha256, "rendered RGB SHA-256")
        if not isinstance(self.fixture_question_id, str) or not self.fixture_question_id:
            raise ValueError("fixed fixture question ID must be nonempty")
        if type(self.fixture_rank) is not int or self.fixture_rank < 0:
            raise ValueError("fixed fixture page rank must be non-negative")
        if type(self.rendered_rgb_width) is not int or self.rendered_rgb_width <= 0:
            raise ValueError("rendered RGB width must be positive")
        if type(self.rendered_rgb_height) is not int or self.rendered_rgb_height <= 0:
            raise ValueError("rendered RGB height must be positive")
        if not isinstance(self.renderer_contract, str) or not self.renderer_contract:
            raise ValueError("renderer contract must be nonempty")

    def to_dict(self) -> dict[str, object]:
        return {
            "input_page_index": self.input_page_index,
            "mineru_page_index": self.mineru_page_index,
            "document_id": self.document_id,
            "source_page_index": self.source_page_index,
            "source_pdf_path": str(self.source_pdf_path),
            "source_pdf_sha256": self.source_pdf_sha256,
            "fixture_question_id": self.fixture_question_id,
            "fixture_rank": self.fixture_rank,
            "fixed_page_fixture_path": str(self.fixed_page_fixture_path),
            "fixed_page_fixture_sha256": self.fixed_page_fixture_sha256,
            "fixed_page_record_sha256": self.fixed_page_record_sha256,
            "rendered_rgb_width": self.rendered_rgb_width,
            "rendered_rgb_height": self.rendered_rgb_height,
            "rendered_rgb_sha256": self.rendered_rgb_sha256,
            "renderer_contract": self.renderer_contract,
        }

    @classmethod
    def from_dict(cls, value: object) -> InputPageIdentity:
        if not isinstance(value, Mapping) or set(value) != {
            "input_page_index",
            "mineru_page_index",
            "document_id",
            "source_page_index",
            "source_pdf_path",
            "source_pdf_sha256",
            "fixture_question_id",
            "fixture_rank",
            "fixed_page_fixture_path",
            "fixed_page_fixture_sha256",
            "fixed_page_record_sha256",
            "rendered_rgb_width",
            "rendered_rgb_height",
            "rendered_rgb_sha256",
            "renderer_contract",
        }:
            raise ValueError("region mapping has an invalid input page identity")
        return cls(
            value["input_page_index"],
            value["mineru_page_index"],
            value["document_id"],
            value["source_page_index"],
            Path(value["source_pdf_path"]),
            value["source_pdf_sha256"],
            value["fixture_question_id"],
            value["fixture_rank"],
            Path(value["fixed_page_fixture_path"]),
            value["fixed_page_fixture_sha256"],
            value["fixed_page_record_sha256"],
            value["rendered_rgb_width"],
            value["rendered_rgb_height"],
            value["rendered_rgb_sha256"],
            value["renderer_contract"],
        )


@dataclass(frozen=True, slots=True)
class MinerUArtifactIdentity:
    """Authenticated raw MinerU output plus all code/model/configuration pins."""

    raw_middle_json_path: Path
    raw_middle_json_sha256: str
    backend: str
    version: str
    repository_id: str
    repository_revision: str
    model_repository_id: str
    model_revision: str
    configuration_path: Path
    configuration_sha256: str
    tool_manifest_path: Path
    tool_manifest_sha256: str
    pages: tuple[InputPageIdentity, ...]

    def __post_init__(self) -> None:
        path = Path(self.raw_middle_json_path)
        config_path = Path(self.configuration_path)
        manifest_path = Path(self.tool_manifest_path)
        pages = tuple(self.pages)
        object.__setattr__(self, "raw_middle_json_path", path)
        object.__setattr__(self, "configuration_path", config_path)
        object.__setattr__(self, "tool_manifest_path", manifest_path)
        object.__setattr__(self, "pages", pages)
        if (
            not path.is_absolute()
            or not config_path.is_absolute()
            or not manifest_path.is_absolute()
        ):
            raise ValueError("MinerU provenance paths must be absolute")
        if config_path == manifest_path:
            raise ValueError("MinerU configuration and tool manifest paths must be distinct")
        _require_sha256(self.raw_middle_json_sha256, "MinerU raw output SHA-256")
        _require_sha256(self.configuration_sha256, "MinerU configuration SHA-256")
        _require_sha256(self.tool_manifest_sha256, "MinerU tool manifest SHA-256")
        _require_revision(self.repository_revision, "MinerU repository revision")
        _require_revision(self.model_revision, "MinerU model revision")
        if self.backend not in {"vlm", "pipeline"}:
            raise ValueError("MinerU backend must be vlm or pipeline")
        if not isinstance(self.version, str) or not self.version:
            raise ValueError("MinerU version must be nonempty")
        if not isinstance(self.repository_id, str) or not self.repository_id:
            raise ValueError("MinerU repository ID must be nonempty")
        if not isinstance(self.model_repository_id, str) or not self.model_repository_id:
            raise ValueError("MinerU model repository ID must be nonempty")
        if not pages or any(not isinstance(page, InputPageIdentity) for page in pages):
            raise ValueError("MinerU artifact must bind at least one input page")
        local = [page.mineru_page_index for page in pages]
        slots = [page.input_page_index for page in pages]
        if len(set(local)) != len(local) or local != sorted(local):
            raise ValueError("MinerU artifact local page indices must be unique and ordered")
        if len(set(slots)) != len(slots):
            raise ValueError("MinerU artifact input page indices must be unique")

    def to_dict(self) -> dict[str, object]:
        return {
            "raw_middle_json_path": str(self.raw_middle_json_path),
            "raw_middle_json_sha256": self.raw_middle_json_sha256,
            "backend": self.backend,
            "version": self.version,
            "repository_id": self.repository_id,
            "repository_revision": self.repository_revision,
            "model_repository_id": self.model_repository_id,
            "model_revision": self.model_revision,
            "configuration_path": str(self.configuration_path),
            "configuration_sha256": self.configuration_sha256,
            "tool_manifest_path": str(self.tool_manifest_path),
            "tool_manifest_sha256": self.tool_manifest_sha256,
            "pages": [page.to_dict() for page in self.pages],
        }

    @classmethod
    def from_dict(cls, value: object) -> MinerUArtifactIdentity:
        required = {
            "raw_middle_json_path",
            "raw_middle_json_sha256",
            "backend",
            "version",
            "repository_id",
            "repository_revision",
            "model_repository_id",
            "model_revision",
            "configuration_path",
            "configuration_sha256",
            "tool_manifest_path",
            "tool_manifest_sha256",
            "pages",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError("region mapping has an invalid MinerU artifact identity")
        pages = value["pages"]
        if not isinstance(pages, list):
            raise ValueError("region mapping has invalid MinerU artifact pages")
        return cls(
            Path(value["raw_middle_json_path"]),
            value["raw_middle_json_sha256"],
            value["backend"],
            value["version"],
            value["repository_id"],
            value["repository_revision"],
            value["model_repository_id"],
            value["model_revision"],
            Path(value["configuration_path"]),
            value["configuration_sha256"],
            Path(value["tool_manifest_path"]),
            value["tool_manifest_sha256"],
            tuple(InputPageIdentity.from_dict(page) for page in pages),
        )


@dataclass(frozen=True, slots=True)
class MinerURegion:
    """One ordered page-space region emitted by the pinned MinerU parser."""

    input_page_index: int
    mineru_page_index: int
    document_id: str
    source_page_index: int
    region_id: str
    region_type: str
    bbox: tuple[float, float, float, float]
    page_size: tuple[float, float]
    reading_order: int

    def __post_init__(self) -> None:
        _validate_page_coordinates(
            self.input_page_index,
            self.mineru_page_index,
            self.document_id,
            self.source_page_index,
        )
        if not isinstance(self.region_id, str) or not self.region_id or ":" in self.region_id:
            raise ValueError("MinerU region ID must be nonempty and contain no colon")
        if not isinstance(self.region_type, str) or not self.region_type:
            raise ValueError("MinerU region type must be nonempty")
        if type(self.reading_order) is not int or self.reading_order < 0:
            raise ValueError("MinerU reading order must be non-negative")
        size = _page_size(self.page_size)
        object.__setattr__(self, "page_size", size)
        object.__setattr__(self, "bbox", _box(self.bbox, page_size=size, label="MinerU region"))

    def source_id(self) -> str:
        return (
            f"mineru:q{self.input_page_index}:{self.document_id}:"
            f"{self.source_page_index}:{self.region_id}"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "input_page_index": self.input_page_index,
            "mineru_page_index": self.mineru_page_index,
            "document_id": self.document_id,
            "source_page_index": self.source_page_index,
            "region_id": self.region_id,
            "region_type": self.region_type,
            "bbox": list(self.bbox),
            "page_size": list(self.page_size),
            "reading_order": self.reading_order,
        }

    @classmethod
    def from_dict(cls, value: object) -> MinerURegion:
        required = {
            "input_page_index",
            "mineru_page_index",
            "document_id",
            "source_page_index",
            "region_id",
            "region_type",
            "bbox",
            "page_size",
            "reading_order",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError("region mapping has invalid audited region evidence")
        return cls(
            value["input_page_index"],
            value["mineru_page_index"],
            value["document_id"],
            value["source_page_index"],
            value["region_id"],
            value["region_type"],
            tuple(value["bbox"]),
            tuple(value["page_size"]),
            value["reading_order"],
        )


@dataclass(frozen=True, slots=True)
class RegionTokenSource:
    """One nonempty intervention source over actual post-QTP token ordinals."""

    source_id: str
    source_kind: str
    input_page_index: int
    mineru_page_index: int
    document_id: str
    source_page_index: int
    region_type: str
    bbox: tuple[float, float, float, float]
    page_size: tuple[float, float]
    reading_order: int | None
    token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        _validate_page_coordinates(
            self.input_page_index,
            self.mineru_page_index,
            self.document_id,
            self.source_page_index,
        )
        if self.source_kind not in {"mineru-region", "residual-grid"}:
            raise ValueError("region mapping source kind is invalid")
        if not isinstance(self.source_id, str) or not self.source_id:
            raise ValueError("region mapping source ID is invalid")
        if not isinstance(self.region_type, str) or not self.region_type:
            raise ValueError("region mapping source region type is invalid")
        size = _page_size(self.page_size)
        object.__setattr__(self, "page_size", size)
        object.__setattr__(self, "bbox", _box(self.bbox, page_size=size, label="source"))
        if self.reading_order is not None and (
            type(self.reading_order) is not int or self.reading_order < 0
        ):
            raise ValueError("region mapping source reading order is invalid")
        token_ids = tuple(self.token_ids)
        object.__setattr__(self, "token_ids", token_ids)
        if (
            not token_ids
            or any(type(token_id) is not int or token_id < 0 for token_id in token_ids)
            or token_ids != tuple(sorted(set(token_ids)))
        ):
            raise ValueError("region mapping source token membership is invalid")

    @property
    def token_cost(self) -> int:
        return len(self.token_ids)

    def to_dict(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "source_kind": self.source_kind,
            "input_page_index": self.input_page_index,
            "mineru_page_index": self.mineru_page_index,
            "document_id": self.document_id,
            "source_page_index": self.source_page_index,
            "region_type": self.region_type,
            "bbox": list(self.bbox),
            "page_size": list(self.page_size),
            "reading_order": self.reading_order,
            "token_ids": list(self.token_ids),
            "token_cost": self.token_cost,
        }

    @classmethod
    def from_dict(cls, value: object) -> RegionTokenSource:
        required = {
            "source_id",
            "source_kind",
            "input_page_index",
            "mineru_page_index",
            "document_id",
            "source_page_index",
            "region_type",
            "bbox",
            "page_size",
            "reading_order",
            "token_ids",
            "token_cost",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError("region mapping has an invalid intervention source")
        token_ids = value["token_ids"]
        if not isinstance(token_ids, list) or value["token_cost"] != len(token_ids):
            raise ValueError("region mapping source token cost is invalid")
        source = cls(
            value["source_id"],
            value["source_kind"],
            value["input_page_index"],
            value["mineru_page_index"],
            value["document_id"],
            value["source_page_index"],
            value["region_type"],
            tuple(value["bbox"]),
            tuple(value["page_size"]),
            value["reading_order"],
            tuple(token_ids),
        )
        return source


def _geometry_rows(geometry: Sequence[VisualTokenGeometry]) -> list[list[int]]:
    return [
        [token.page_index, token.row, token.column, token.height, token.width] for token in geometry
    ]


def _mapping_payload(
    *,
    artifacts: Sequence[MinerUArtifactIdentity],
    geometry: Sequence[VisualTokenGeometry],
    residual_grid_size: int,
    audited_regions: Sequence[MinerURegion],
    empty_region_source_ids: Sequence[str],
    sources: Sequence[RegionTokenSource],
    token_to_source: Sequence[str],
) -> dict[str, object]:
    rows = _geometry_rows(geometry)
    return {
        "schema_version": REGION_MAPPING_SCHEMA_VERSION,
        "assignment_contract": ASSIGNMENT_CONTRACT,
        "artifacts": [artifact.to_dict() for artifact in artifacts],
        "geometry": rows,
        "geometry_count": len(rows),
        "geometry_sha256": _canonical_sha256(rows),
        "residual_grid_size": residual_grid_size,
        "audited_regions": [region.to_dict() for region in audited_regions],
        "empty_region_source_ids": list(empty_region_source_ids),
        "sources": [source.to_dict() for source in sources],
        "token_to_source": list(token_to_source),
    }


@dataclass(frozen=True, slots=True)
class RegionTokenMapping:
    """Authenticated replayable partition of the post-QTP visual population."""

    artifacts: tuple[MinerUArtifactIdentity, ...]
    geometry: tuple[VisualTokenGeometry, ...]
    geometry_count: int
    geometry_sha256: str
    residual_grid_size: int
    assignment_contract: str
    audited_regions: tuple[MinerURegion, ...]
    empty_region_source_ids: tuple[str, ...]
    sources: tuple[RegionTokenSource, ...]
    token_to_source: tuple[str, ...]
    sha256: str

    def to_dict(self) -> dict[str, object]:
        payload = _mapping_payload(
            artifacts=self.artifacts,
            geometry=self.geometry,
            residual_grid_size=self.residual_grid_size,
            audited_regions=self.audited_regions,
            empty_region_source_ids=self.empty_region_source_ids,
            sources=self.sources,
            token_to_source=self.token_to_source,
        )
        payload["sha256"] = self.sha256
        return payload

    @classmethod
    def from_dict(cls, value: object) -> RegionTokenMapping:
        required = {
            "schema_version",
            "assignment_contract",
            "artifacts",
            "geometry",
            "geometry_count",
            "geometry_sha256",
            "residual_grid_size",
            "audited_regions",
            "empty_region_source_ids",
            "sources",
            "token_to_source",
            "sha256",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError("region mapping schema is invalid")
        try:
            artifacts = tuple(MinerUArtifactIdentity.from_dict(item) for item in value["artifacts"])
            geometry = tuple(VisualTokenGeometry(*row) for row in value["geometry"])
            audited = tuple(MinerURegion.from_dict(item) for item in value["audited_regions"])
            tuple(RegionTokenSource.from_dict(item) for item in value["sources"])
        except (TypeError, KeyError, ValueError) as error:
            raise ValueError("region mapping content is invalid") from error
        try:
            expected = map_post_qtp_tokens_to_regions(
                geometry,
                artifacts,
                audited,
                residual_grid_size=value["residual_grid_size"],
            )
        except (TypeError, KeyError, ValueError) as error:
            raise ValueError("region mapping content is invalid") from error
        if expected.to_dict() != dict(value):
            raise ValueError("region mapping identity is invalid")
        return expected


def mineru_regions_from_middle_json(
    raw_middle_json: bytes, artifact: MinerUArtifactIdentity
) -> tuple[MinerURegion, ...]:
    """Parse raw MinerU output while applying an explicit local-to-source page map."""

    if not isinstance(raw_middle_json, bytes):
        raise TypeError("raw MinerU middle JSON must be bytes")
    if hashlib.sha256(raw_middle_json).hexdigest() != artifact.raw_middle_json_sha256:
        raise ValueError("MinerU raw output SHA-256 mismatch")
    try:
        payload = json.loads(raw_middle_json)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("MinerU middle JSON is invalid") from error
    if not isinstance(payload, Mapping):
        raise ValueError("MinerU middle JSON must be an object")
    if (
        payload.get("_backend") != artifact.backend
        or payload.get("_version_name") != artifact.version
    ):
        raise ValueError("MinerU backend or version does not match artifact identity")
    raw_pages = payload.get("pdf_info")
    if not isinstance(raw_pages, list):
        raise ValueError("MinerU middle JSON pdf_info must be a list")
    page_by_local = {page.mineru_page_index: page for page in artifact.pages}
    observed: set[int] = set()
    regions: list[MinerURegion] = []
    for raw_page in raw_pages:
        if not isinstance(raw_page, Mapping):
            raise ValueError("MinerU page must be an object")
        local_page = raw_page.get("page_idx")
        if type(local_page) is not int or local_page in observed or local_page not in page_by_local:
            raise ValueError("MinerU page coverage does not match the artifact page map")
        observed.add(local_page)
        identity = page_by_local[local_page]
        size = _page_size(raw_page.get("page_size"))
        blocks = raw_page.get("para_blocks")
        if not isinstance(blocks, list):
            raise ValueError("MinerU para_blocks must be a list")
        for order, block in enumerate(blocks):
            if not isinstance(block, Mapping):
                raise ValueError("MinerU paragraph block must be an object")
            region_type = block.get("type")
            if not isinstance(region_type, str) or not region_type:
                raise ValueError("MinerU paragraph block type must be nonempty")
            regions.append(
                MinerURegion(
                    identity.input_page_index,
                    local_page,
                    identity.document_id,
                    identity.source_page_index,
                    f"m{local_page}-r{order}",
                    region_type,
                    _box(block.get("bbox"), page_size=size, label="MinerU paragraph block"),
                    size,
                    order,
                )
            )
    if observed != set(page_by_local):
        raise ValueError("MinerU page coverage does not match the artifact page map")
    return tuple(regions)


def _intersection_area(
    left: tuple[float, float, float, float], right: tuple[float, float, float, float]
) -> float:
    return max(0.0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0.0, min(left[3], right[3]) - max(left[1], right[1])
    )


def _validate_geometry(geometry: tuple[VisualTokenGeometry, ...], page_count: int) -> None:
    identities: list[tuple[int, int, int]] = []
    grids: dict[int, tuple[int, int]] = {}
    for token in geometry:
        if (
            not isinstance(token, VisualTokenGeometry)
            or not 0 <= token.page_index < page_count
            or token.height <= 0
            or token.width <= 0
            or not 0 <= token.row < token.height
            or not 0 <= token.column < token.width
        ):
            raise ValueError("post-QTP token geometry is invalid")
        grid = (token.height, token.width)
        if token.page_index in grids and grids[token.page_index] != grid:
            raise ValueError("post-QTP token geometry has inconsistent page grids")
        grids[token.page_index] = grid
        identities.append((token.page_index, token.row, token.column))
    if identities != sorted(set(identities)):
        raise ValueError("post-QTP token geometry must be unique and page-major")


def _validate_mapping_membership(
    sources: Sequence[RegionTokenSource], token_to_source: Sequence[str], population: int
) -> None:
    source_ids = [source.source_id for source in sources]
    if len(source_ids) != len(set(source_ids)) or any(not source.token_ids for source in sources):
        raise ValueError("region mapping intervention sources are invalid")
    assigned: list[int] = []
    for source in sources:
        if tuple(sorted(set(source.token_ids))) != source.token_ids:
            raise ValueError("region mapping source token membership is invalid")
        assigned.extend(source.token_ids)
        if any(token_to_source[token_id] != source.source_id for token_id in source.token_ids):
            raise ValueError("region mapping token-to-source membership disagrees")
    if sorted(assigned) != list(range(population)) or len(assigned) != population:
        raise ValueError("region mapping does not partition every token exactly once")
    if len(token_to_source) != population or any(
        value not in set(source_ids) for value in token_to_source
    ):
        raise ValueError("region mapping token-to-source vector is invalid")


def map_post_qtp_tokens_to_regions(
    geometry: Sequence[VisualTokenGeometry],
    artifacts: Sequence[MinerUArtifactIdentity],
    regions: Sequence[MinerURegion],
    *,
    residual_grid_size: int = 4,
) -> RegionTokenMapping:
    """Assign each post-QTP token to one nonempty region source or residual cell."""

    if type(residual_grid_size) is not int or residual_grid_size <= 0:
        raise ValueError("residual grid size must be a positive integer")
    artifact_tuple = tuple(artifacts)
    if not artifact_tuple or any(
        not isinstance(artifact, MinerUArtifactIdentity) for artifact in artifact_tuple
    ):
        raise ValueError("region mapping requires authenticated MinerU artifacts")
    tool_identities = {
        (
            artifact.backend,
            artifact.version,
            artifact.repository_id,
            artifact.repository_revision,
            artifact.model_repository_id,
            artifact.model_revision,
            artifact.configuration_path,
            artifact.configuration_sha256,
            artifact.tool_manifest_path,
            artifact.tool_manifest_sha256,
        )
        for artifact in artifact_tuple
    }
    if len(tool_identities) != 1:
        raise ValueError("MinerU artifacts must use one homogeneous pinned tool identity")
    pages = tuple(page for artifact in artifact_tuple for page in artifact.pages)
    if tuple(page.input_page_index for page in pages) != tuple(range(len(pages))):
        raise ValueError("MinerU artifact pages must follow exact Qwen page-major order")
    if tuple(page.fixture_rank for page in pages) != tuple(range(len(pages))):
        raise ValueError("MinerU artifact fixture ranks must follow exact page-major order")
    if len({(page.document_id, page.source_page_index) for page in pages}) != len(pages):
        raise ValueError("MinerU artifact source pages must be unique")
    if (
        len(
            {
                (
                    page.fixed_page_fixture_path,
                    page.fixed_page_fixture_sha256,
                    page.fixture_question_id,
                )
                for page in pages
            }
        )
        != 1
    ):
        raise ValueError("MinerU artifact pages must use one homogeneous fixed-page fixture")
    if len({artifact.raw_middle_json_path for artifact in artifact_tuple}) != len(artifact_tuple):
        raise ValueError("MinerU artifact paths must be unique")
    geometry_tuple = tuple(geometry)
    if not geometry_tuple:
        raise ValueError("post-QTP token geometry must be nonempty")
    _validate_geometry(geometry_tuple, len(pages))

    page_by_slot = {page.input_page_index: page for page in pages}
    region_tuple = tuple(
        sorted(
            regions, key=lambda item: (item.input_page_index, item.reading_order, item.region_id)
        )
    )
    region_ids: set[str] = set()
    page_sizes: dict[int, tuple[float, float]] = {}
    reading_orders: set[tuple[int, int]] = set()
    for region in region_tuple:
        page = page_by_slot.get(region.input_page_index)
        if page is None or (
            region.mineru_page_index,
            region.document_id,
            region.source_page_index,
        ) != (page.mineru_page_index, page.document_id, page.source_page_index):
            raise ValueError("MinerU region does not match an authenticated input page")
        if region.source_id() in region_ids:
            raise ValueError("MinerU region source IDs must be unique")
        region_ids.add(region.source_id())
        if (
            region.input_page_index in page_sizes
            and page_sizes[region.input_page_index] != region.page_size
        ):
            raise ValueError("MinerU regions have inconsistent page sizes")
        page_sizes[region.input_page_index] = region.page_size
        order = (region.input_page_index, region.reading_order)
        if order in reading_orders:
            raise ValueError("MinerU reading order must be unique within each input page")
        reading_orders.add(order)

    regions_by_slot: dict[int, list[MinerURegion]] = defaultdict(list)
    for region in region_tuple:
        regions_by_slot[region.input_page_index].append(region)
    assigned: dict[str, list[int]] = defaultdict(list)
    residual_cells: dict[str, tuple[InputPageIdentity, int, int]] = {}
    token_to_source: list[str] = []
    for token_id, token in enumerate(geometry_tuple):
        token_box = (
            token.column / token.width,
            token.row / token.height,
            (token.column + 1) / token.width,
            (token.row + 1) / token.height,
        )
        candidates: list[tuple[float, int, str, MinerURegion]] = []
        for region in regions_by_slot[token.page_index]:
            width, height = region.page_size
            region_box = (
                region.bbox[0] / width,
                region.bbox[1] / height,
                region.bbox[2] / width,
                region.bbox[3] / height,
            )
            overlap = _intersection_area(token_box, region_box)
            if overlap > 0:
                candidates.append((-overlap, region.reading_order, region.region_id, region))
        if candidates:
            source_id = min(candidates)[3].source_id()
        else:
            page = page_by_slot[token.page_index]
            row = min(
                residual_grid_size - 1,
                int(residual_grid_size * (token.row + 0.5) / token.height),
            )
            column = min(
                residual_grid_size - 1,
                int(residual_grid_size * (token.column + 0.5) / token.width),
            )
            source_id = (
                f"residual:q{page.input_page_index}:{page.document_id}:"
                f"{page.source_page_index}:r{row}:c{column}"
            )
            residual_cells[source_id] = (page, row, column)
        token_to_source.append(source_id)
        assigned[source_id].append(token_id)

    sources: list[RegionTokenSource] = []
    empty: list[str] = []
    for region in region_tuple:
        token_ids = tuple(assigned[region.source_id()])
        if not token_ids:
            empty.append(region.source_id())
            continue
        sources.append(
            RegionTokenSource(
                region.source_id(),
                "mineru-region",
                region.input_page_index,
                region.mineru_page_index,
                region.document_id,
                region.source_page_index,
                region.region_type,
                region.bbox,
                region.page_size,
                region.reading_order,
                token_ids,
            )
        )
    for source_id, (page, row, column) in sorted(
        residual_cells.items(), key=lambda item: (item[1][0].input_page_index, item[1][1:])
    ):
        sources.append(
            RegionTokenSource(
                source_id,
                "residual-grid",
                page.input_page_index,
                page.mineru_page_index,
                page.document_id,
                page.source_page_index,
                "residual",
                (
                    column / residual_grid_size,
                    row / residual_grid_size,
                    (column + 1) / residual_grid_size,
                    (row + 1) / residual_grid_size,
                ),
                (1.0, 1.0),
                None,
                tuple(assigned[source_id]),
            )
        )
    _validate_mapping_membership(sources, token_to_source, len(geometry_tuple))
    payload = _mapping_payload(
        artifacts=artifact_tuple,
        geometry=geometry_tuple,
        residual_grid_size=residual_grid_size,
        audited_regions=region_tuple,
        empty_region_source_ids=empty,
        sources=sources,
        token_to_source=token_to_source,
    )
    return RegionTokenMapping(
        artifact_tuple,
        geometry_tuple,
        len(geometry_tuple),
        payload["geometry_sha256"],
        residual_grid_size,
        ASSIGNMENT_CONTRACT,
        region_tuple,
        tuple(empty),
        tuple(sources),
        tuple(token_to_source),
        _canonical_sha256(payload),
    )


def build_region_mapping_from_artifacts(
    geometry: Sequence[VisualTokenGeometry],
    artifact_bytes: Sequence[tuple[MinerUArtifactIdentity, bytes]],
    *,
    residual_grid_size: int = 4,
) -> RegionTokenMapping:
    """Build only from authenticated raw MinerU bytes, never hand-selected regions."""

    pairs = tuple(artifact_bytes)
    if not pairs:
        raise ValueError("region mapping requires raw MinerU artifacts")
    artifacts: list[MinerUArtifactIdentity] = []
    regions: list[MinerURegion] = []
    for pair in pairs:
        if (
            not isinstance(pair, tuple)
            or len(pair) != 2
            or not isinstance(pair[0], MinerUArtifactIdentity)
            or not isinstance(pair[1], bytes)
        ):
            raise ValueError("raw MinerU artifact input is invalid")
        artifact, raw = pair
        artifacts.append(artifact)
        regions.extend(mineru_regions_from_middle_json(raw, artifact))
    return map_post_qtp_tokens_to_regions(
        geometry,
        artifacts,
        regions,
        residual_grid_size=residual_grid_size,
    )


def region_mapping_json_bytes(mapping: RegionTokenMapping) -> bytes:
    """Return the one canonical byte representation used for cached replay."""

    if not isinstance(mapping, RegionTokenMapping):
        raise TypeError("mapping must be a RegionTokenMapping")
    return (
        json.dumps(mapping.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode()


def _read_regular_bytes(path: Path, *, label: str) -> bytes:
    file_path = Path(path)
    if not file_path.is_absolute():
        raise ValueError(f"{label} path must be absolute")
    descriptor: int | None = None
    try:
        descriptor = os.open(file_path, os.O_RDONLY | os.O_NOFOLLOW)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"{label} must be a regular file")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        return b"".join(chunks)
    except OSError as error:
        raise ValueError(f"{label} is missing or not a regular file") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _authenticated_bytes(path: Path, expected_sha256: str, *, label: str) -> bytes:
    raw = _read_regular_bytes(path, label=label)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch")
    return raw


def _authenticate_input_page(page: InputPageIdentity) -> None:
    fixture_raw = _authenticated_bytes(
        page.fixed_page_fixture_path,
        page.fixed_page_fixture_sha256,
        label="fixed page fixture",
    )
    try:
        fixture_value = json.loads(fixture_raw)
        fixture = FixedPageFixture.from_dict(fixture_value)
        fixture.validate_external_bytes(selected_qids=(page.fixture_question_id,))
        record = fixture.question(page.fixture_question_id).pages[page.fixture_rank]
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        KeyError,
        IndexError,
        TypeError,
        ValueError,
    ) as error:
        raise ValueError("fixed page fixture record is unavailable") from error
    record_value = record.to_dict()
    expected_fields = {
        "rank": page.fixture_rank,
        "doc_id": page.document_id,
        "page_index": page.source_page_index,
        "source_pdf_path": str(page.source_pdf_path),
        "source_pdf_sha256": page.source_pdf_sha256,
        "rendered_rgb_width": page.rendered_rgb_width,
        "rendered_rgb_height": page.rendered_rgb_height,
        "rendered_rgb_sha256": page.rendered_rgb_sha256,
        "renderer_contract": page.renderer_contract,
    }
    if _canonical_sha256(record_value) != page.fixed_page_record_sha256 or any(
        record_value.get(name) != expected for name, expected in expected_fields.items()
    ):
        raise ValueError("fixed page fixture record does not match the input page identity")


def _expected_tool_manifest(artifact: MinerUArtifactIdentity) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "pinned-mineru-tool",
        "backend": artifact.backend,
        "version": artifact.version,
        "repository_id": artifact.repository_id,
        "repository_revision": artifact.repository_revision,
        "model_repository_id": artifact.model_repository_id,
        "model_revision": artifact.model_revision,
        "configuration_path": str(artifact.configuration_path),
        "configuration_sha256": artifact.configuration_sha256,
    }


def _authenticate_tool_manifest(artifact: MinerUArtifactIdentity) -> None:
    raw = _authenticated_bytes(
        artifact.tool_manifest_path,
        artifact.tool_manifest_sha256,
        label="MinerU tool manifest",
    )
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("MinerU tool manifest is invalid") from error
    if value != _expected_tool_manifest(artifact):
        raise ValueError("MinerU tool manifest contradicts the pinned tool identity")


def _reauthenticate_mapping_inputs(mapping: RegionTokenMapping) -> None:
    audited: list[MinerURegion] = []
    seen_configs: set[tuple[Path, str]] = set()
    seen_manifests: set[tuple[Path, str]] = set()
    seen_pages: set[tuple[Path, str, int]] = set()
    for artifact in mapping.artifacts:
        config_key = (artifact.configuration_path, artifact.configuration_sha256)
        if config_key not in seen_configs:
            _authenticated_bytes(
                artifact.configuration_path,
                artifact.configuration_sha256,
                label="MinerU configuration",
            )
            seen_configs.add(config_key)
        manifest_key = (artifact.tool_manifest_path, artifact.tool_manifest_sha256)
        if manifest_key not in seen_manifests:
            _authenticate_tool_manifest(artifact)
            seen_manifests.add(manifest_key)
        for page in artifact.pages:
            key = (
                page.fixed_page_fixture_path,
                page.fixed_page_fixture_sha256,
                page.fixture_rank,
            )
            if key not in seen_pages:
                _authenticate_input_page(page)
                seen_pages.add(key)
        raw = _authenticated_bytes(
            artifact.raw_middle_json_path,
            artifact.raw_middle_json_sha256,
            label="raw MinerU artifact",
        )
        audited.extend(mineru_regions_from_middle_json(raw, artifact))
    expected = map_post_qtp_tokens_to_regions(
        mapping.geometry,
        mapping.artifacts,
        audited,
        residual_grid_size=mapping.residual_grid_size,
    )
    if expected != mapping:
        raise ValueError("raw MinerU artifact does not reproduce the cached mapping")


def load_region_mapping(path: Path, *, validate_raw_artifacts: bool = True) -> RegionTokenMapping:
    """Load canonical cached bytes and optionally reauthenticate every raw MinerU file."""

    raw_mapping = _read_regular_bytes(path, label="region mapping")
    try:
        value = json.loads(raw_mapping)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("region mapping JSON is invalid") from error
    mapping = RegionTokenMapping.from_dict(value)
    if raw_mapping != region_mapping_json_bytes(mapping):
        raise ValueError("region mapping bytes are not canonical")
    if not isinstance(validate_raw_artifacts, bool):
        raise TypeError("validate_raw_artifacts must be a boolean")
    if validate_raw_artifacts:
        try:
            _reauthenticate_mapping_inputs(mapping)
        except ValueError as error:
            raise ValueError("raw MinerU artifact authentication failed") from error
    return mapping


def _open_directory_nofollow(path: Path) -> int:
    absolute = Path(os.path.abspath(path))
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in absolute.parts[1:]:
            next_descriptor = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
    except OSError:
        os.close(descriptor)
        raise
    return descriptor


def publish_region_mapping(path: Path, mapping: RegionTokenMapping) -> None:
    """Publish canonical mapping bytes atomically without replacing prior evidence."""

    destination = Path(path)
    if not destination.is_absolute():
        raise ValueError("region mapping output path must be absolute")
    verified = RegionTokenMapping.from_dict(mapping.to_dict())
    try:
        _reauthenticate_mapping_inputs(verified)
    except ValueError as error:
        raise ValueError("raw MinerU artifact authentication failed") from error
    parent_descriptor: int | None = None
    temporary_name: str | None = None
    try:
        parent_descriptor = _open_directory_nofollow(destination.parent)
        temporary_descriptor, temporary_path = tempfile.mkstemp(
            prefix=f".{destination.name}.", dir=destination.parent
        )
        temporary_name = Path(temporary_path).name
        try:
            with os.fdopen(temporary_descriptor, "wb") as stream:
                stream.write(region_mapping_json_bytes(mapping))
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            temporary_descriptor = -1
            raise
        os.link(
            temporary_name,
            destination.name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        os.fsync(parent_descriptor)
    finally:
        if parent_descriptor is not None:
            if temporary_name is not None:
                try:
                    os.unlink(temporary_name, dir_fd=parent_descriptor)
                except FileNotFoundError:
                    pass
            os.close(parent_descriptor)


__all__ = [
    "ASSIGNMENT_CONTRACT",
    "InputPageIdentity",
    "MinerUArtifactIdentity",
    "MinerURegion",
    "REGION_MAPPING_SCHEMA_VERSION",
    "RegionTokenMapping",
    "RegionTokenSource",
    "build_region_mapping_from_artifacts",
    "load_region_mapping",
    "map_post_qtp_tokens_to_regions",
    "mineru_regions_from_middle_json",
    "publish_region_mapping",
    "region_mapping_json_bytes",
]
