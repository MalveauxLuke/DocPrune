"""Compose admitted Task 8 MinerU and post-QTP geometry evidence."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Mapping
from pathlib import Path

from docprune.ctp_controls import VisualTokenGeometry
from docprune.segmentation import (
    MinerUArtifactIdentity,
    RegionTokenMapping,
    build_region_mapping_from_artifacts,
    publish_region_mapping,
)
from docprune.task8_geometry import load_task8_geometry_capture
from docprune.task8_runtime import load_task8_mineru_completion


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _read_regular(path: Path, expected_sha256: str) -> bytes:
    file_path = Path(path)
    descriptor: int | None = None
    try:
        descriptor = os.open(file_path, os.O_RDONLY | os.O_NOFOLLOW)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("raw MinerU artifact must be a regular file")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        raw = b"".join(chunks)
    except OSError as error:
        raise ValueError("raw MinerU artifact is missing or unsafe") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("raw MinerU artifact checksum mismatch")
    return raw


def publish_task8_region_mapping(
    *,
    mineru_completion_path: Path,
    mineru_completion_sha256: str,
    geometry_capture_path: Path,
    geometry_capture_sha256: str,
    output_path: Path,
    expected_qid: str,
    expected_geometry_count: int,
    expected_geometry_sha256: str,
    required_geometry_gpu_substring: str | None = "L40S",
) -> RegionTokenMapping:
    """Publish one mapping only when both live evidence streams agree exactly."""

    if not isinstance(expected_qid, str) or not expected_qid:
        raise ValueError("expected Task 8 QID must be nonempty")
    if type(expected_geometry_count) is not int or expected_geometry_count <= 0:
        raise ValueError("expected Task 8 geometry count must be positive")
    expected_geometry_sha = _require_sha256(expected_geometry_sha256, "expected geometry checksum")
    completion = load_task8_mineru_completion(
        Path(mineru_completion_path),
        expected_sha256=_require_sha256(mineru_completion_sha256, "MinerU completion checksum"),
    )
    geometry_capture = load_task8_geometry_capture(
        Path(geometry_capture_path),
        expected_sha256=_require_sha256(geometry_capture_sha256, "geometry capture checksum"),
    )
    gpu = geometry_capture.get("gpu")
    device_name = gpu.get("device_name") if isinstance(gpu, Mapping) else None
    if not isinstance(device_name, str) or not device_name:
        raise ValueError("Task 8 mapping requires authenticated CUDA geometry")
    if required_geometry_gpu_substring is not None and (
        not isinstance(required_geometry_gpu_substring, str)
        or not required_geometry_gpu_substring
        or required_geometry_gpu_substring not in device_name
    ):
        raise ValueError(
            f"Task 8 mapping requires {required_geometry_gpu_substring!r} geometry"
        )
    if (
        completion.get("global_index_loaded") is not False
        or completion.get("retrieval_run") is not False
        or geometry_capture.get("global_index_loaded") is not False
        or geometry_capture.get("retrieval_search_run") is not False
        or geometry_capture.get("qwen_generation_model_loaded") is not False
        or geometry_capture.get("qid") != expected_qid
        or geometry_capture.get("geometry_count") != expected_geometry_count
        or geometry_capture.get("geometry_sha256") != expected_geometry_sha
    ):
        raise ValueError("Task 8 mapping input execution identity mismatch")

    rows = geometry_capture.get("geometry")
    if not isinstance(rows, list):
        raise ValueError("Task 8 mapping geometry rows are invalid")
    try:
        geometry = tuple(VisualTokenGeometry(*row) for row in rows)
    except (TypeError, ValueError) as error:
        raise ValueError("Task 8 mapping geometry rows are invalid") from error
    if len(geometry) != expected_geometry_count:
        raise ValueError("Task 8 mapping geometry count mismatch")

    completion_artifacts = completion.get("artifacts")
    if not isinstance(completion_artifacts, list) or not completion_artifacts:
        raise ValueError("Task 8 mapping has no admitted MinerU artifacts")
    artifacts: list[MinerUArtifactIdentity] = []
    raw_pairs: list[tuple[MinerUArtifactIdentity, bytes]] = []
    for row in completion_artifacts:
        if not isinstance(row, Mapping) or not isinstance(row.get("identity"), Mapping):
            raise ValueError("Task 8 mapping MinerU artifact schema is invalid")
        artifact = MinerUArtifactIdentity.from_dict(row["identity"])
        if len(artifact.pages) != 1:
            raise ValueError("Task 8 mapping requires one MinerU artifact per fixed page")
        artifacts.append(artifact)
        raw_pairs.append(
            (
                artifact,
                _read_regular(
                    artifact.raw_middle_json_path,
                    artifact.raw_middle_json_sha256,
                ),
            )
        )
    pages = [artifact.pages[0] for artifact in artifacts]
    if [page.input_page_index for page in pages] != list(range(len(pages))):
        raise ValueError("Task 8 mapping MinerU pages are not in fixed input order")
    fixture_path = geometry_capture.get("fixed_page_fixture_path")
    fixture_sha = geometry_capture.get("fixed_page_fixture_sha256")
    smoke_path = geometry_capture.get("smoke_input_manifest_path")
    smoke_sha = geometry_capture.get("smoke_input_manifest_sha256")
    if any(
        page.fixture_question_id != expected_qid
        or str(page.fixed_page_fixture_path) != fixture_path
        or page.fixed_page_fixture_sha256 != fixture_sha
        or str(artifact.smoke_input_manifest_path) != smoke_path
        or artifact.smoke_input_manifest_sha256 != smoke_sha
        for artifact, page in zip(artifacts, pages, strict=True)
    ):
        raise ValueError("Task 8 MinerU and geometry provenance do not agree")
    if geometry and max(token.page_index for token in geometry) >= len(pages):
        raise ValueError("Task 8 geometry refers to a page without MinerU output")

    mapping = build_region_mapping_from_artifacts(geometry, tuple(raw_pairs))
    publish_region_mapping(Path(output_path), mapping)
    return mapping


__all__ = ["publish_task8_region_mapping"]
