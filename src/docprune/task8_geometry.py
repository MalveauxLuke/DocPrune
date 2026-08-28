"""Authenticated post-BTP+QTP geometry capture for the Task 8 smoke QID."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

import torch
from PIL import Image

from docprune.answerers import derive_btp_qtp_geometry_without_qwen_model
from docprune.benchmark_config import (
    COLPALI_BACKBONE_MODEL,
    COLPALI_BACKBONE_REVISION,
    COLPALI_MODEL,
    COLPALI_REVISION,
    QWEN_MODEL,
    QWEN_REVISION,
)
from docprune.task6_runtime import AuthenticatedFixedPageRetriever, load_fixed_page_fixture
from docprune.task8_runtime import load_task8_smoke_inputs

TASK8_GEOMETRY_SCHEMA_VERSION = "docprune-task8-btp-qtp-geometry-v1"


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: object, label: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return str(value)


def _require_commit(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("runtime commit must be a lowercase 40-character Git revision")
    return value


def _regular_bytes(path: Path, label: str) -> bytes:
    value = Path(path)
    if not value.is_absolute() or value.is_symlink() or not value.is_file():
        raise ValueError(f"{label} must be an absolute regular file")
    return value.read_bytes()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _load_reference_row(
    path: Path,
    expected_sha256: str,
    *,
    qid: str,
) -> Mapping[str, object]:
    raw = _regular_bytes(path, "Task 6 reference results")
    if _sha256_bytes(raw) != _require_sha256(expected_sha256, "reference results checksum"):
        raise ValueError("Task 6 reference results checksum mismatch")
    matches: list[Mapping[str, object]] = []
    try:
        for line in raw.decode("utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError("Task 6 reference result row is invalid")
            selection = row.get("policy_selection")
            policy = selection.get("policy") if isinstance(selection, Mapping) else None
            if (
                row.get("question_id") == qid
                and isinstance(policy, Mapping)
                and policy.get("name") == "btp-qtp-no-ctp"
            ):
                matches.append(row)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Task 6 reference results are not valid JSONL") from error
    if len(matches) != 1:
        raise ValueError("Task 6 reference must contain one BTP+QTP no-CTP row for the QID")
    return matches[0]


def _reference_identity(
    row: Mapping[str, object],
    *,
    fixture_sha256: str,
    expected_geometry_count: int,
    expected_geometry_sha256: str,
) -> tuple[str, list[Mapping[str, object]], Mapping[str, object]]:
    if row.get("fixed_page_fixture_sha256") != fixture_sha256:
        raise ValueError("Task 6 reference fixture identity mismatch")
    if row.get("fixed_page_provenance") is not True or row.get("global_index_loaded") is not False:
        raise ValueError("Task 6 reference does not prove fixed-page no-search provenance")
    question = row.get("question")
    pages = row.get("retrieved_pages")
    trace = row.get("trace")
    selection = row.get("policy_selection")
    if not isinstance(question, str) or not question:
        raise ValueError("Task 6 reference question is invalid")
    if (
        not isinstance(pages, list)
        or not pages
        or any(not isinstance(page, Mapping) for page in pages)
        or not isinstance(trace, Mapping)
        or not isinstance(selection, Mapping)
    ):
        raise ValueError("Task 6 reference structure is invalid")
    expected_sha = _require_sha256(expected_geometry_sha256, "expected geometry checksum")
    if (
        type(expected_geometry_count) is not int
        or expected_geometry_count <= 0
        or selection.get("geometry_count") != expected_geometry_count
        or selection.get("geometry_sha256") != expected_sha
    ):
        raise ValueError("Task 6 reference geometry identity mismatch")
    if (
        trace.get("post_qtp_visual_tokens") != expected_geometry_count
        or trace.get("post_ctp_visual_tokens") != expected_geometry_count
        or trace.get("ctp_layer") is not None
    ):
        raise ValueError("Task 6 reference is not a no-CTP geometry control")
    return question, pages, trace


def _load_images_and_validate_pages(
    smoke: Mapping[str, object],
    *,
    fixture_path: Path,
    fixture_sha256: str,
    qid: str,
) -> tuple[Image.Image, ...]:
    if (
        smoke.get("fixed_page_fixture_path") != str(fixture_path)
        or smoke.get("fixed_page_fixture_sha256") != fixture_sha256
        or smoke.get("qid") != qid
        or smoke.get("global_index_loaded") is not False
    ):
        raise ValueError("Task 8 smoke input identity mismatch")
    pages = smoke.get("pages")
    if not isinstance(pages, list) or not pages:
        raise ValueError("Task 8 smoke input pages are invalid")
    images: list[Image.Image] = []
    for expected_rank, page in enumerate(pages):
        if not isinstance(page, Mapping) or page.get("fixture_rank") != expected_rank:
            raise ValueError("Task 8 smoke pages are not in fixed rank order")
        path = Path(str(page.get("mineru_input_path")))
        raw = _regular_bytes(path, "Task 8 sealed page image")
        if _sha256_bytes(raw) != page.get("mineru_input_sha256"):
            raise ValueError("Task 8 sealed page image checksum mismatch")
        try:
            with Image.open(path) as source:
                image = source.convert("RGB")
                image.load()
        except OSError as error:
            raise ValueError("Task 8 sealed page image is invalid") from error
        if image.size != (
            page.get("rendered_rgb_width"),
            page.get("rendered_rgb_height"),
        ) or _sha256_bytes(image.tobytes()) != page.get("rendered_rgb_sha256"):
            raise ValueError("Task 8 sealed page RGB identity mismatch")
        images.append(image)
    return tuple(images)


def _publish_no_replace(payload: dict[str, object], destination: Path) -> None:
    output = Path(destination)
    if not output.is_absolute():
        raise ValueError("Task 8 geometry output path must be absolute")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Task 8 geometry output already exists: {output}")
    fd, name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    stage = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(_canonical_bytes(payload))
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(stage, output)
        except FileExistsError:
            raise FileExistsError(f"Task 8 geometry output already exists: {output}") from None
    finally:
        stage.unlink(missing_ok=True)


def capture_task8_btp_qtp_geometry(
    *,
    fixture_path: Path,
    fixture_sha256: str,
    smoke_input_manifest_path: Path,
    smoke_input_manifest_sha256: str,
    reference_results_path: Path,
    reference_results_sha256: str,
    qid: str,
    expected_geometry_count: int,
    expected_geometry_sha256: str,
    output_path: Path,
    runtime_commit: str,
    query_encoder: object,
    qwen_processor: object,
    require_cuda: bool = True,
) -> dict[str, object]:
    """Capture exact post-QTP geometry without retrieval search or Qwen weights."""

    _require_commit(runtime_commit)
    fixture_sha = _require_sha256(fixture_sha256, "fixture checksum")
    smoke_sha = _require_sha256(smoke_input_manifest_sha256, "smoke input checksum")
    expected_geometry_sha = _require_sha256(expected_geometry_sha256, "expected geometry checksum")
    if not isinstance(qid, str) or not qid:
        raise ValueError("Task 8 geometry QID must be nonempty")
    output = Path(output_path)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Task 8 geometry output already exists: {output}")
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("Task 8 geometry capture requires CUDA")

    fixture_file = Path(fixture_path)
    fixture = load_fixed_page_fixture(
        fixture_file,
        expected_sha256=fixture_sha,
        validate_external_bytes=False,
    )
    fixture.validate_external_bytes(selected_qids=(qid,))
    samples = fixture.selected_samples((qid,))
    if len(samples) != 1:
        raise ValueError("Task 8 geometry fixture did not resolve exactly one sample")
    sample = samples[0]

    smoke_raw = _regular_bytes(Path(smoke_input_manifest_path), "Task 8 smoke manifest")
    if _sha256_bytes(smoke_raw) != smoke_sha:
        raise ValueError("Task 8 smoke manifest checksum mismatch")
    smoke = load_task8_smoke_inputs(Path(smoke_input_manifest_path))
    images = _load_images_and_validate_pages(
        smoke,
        fixture_path=fixture_file,
        fixture_sha256=fixture_sha,
        qid=qid,
    )

    reference = _load_reference_row(
        Path(reference_results_path),
        reference_results_sha256,
        qid=qid,
    )
    question, reference_pages, reference_trace = _reference_identity(
        reference,
        fixture_sha256=fixture_sha,
        expected_geometry_count=expected_geometry_count,
        expected_geometry_sha256=expected_geometry_sha,
    )
    if sample.question != question:
        raise ValueError("Task 8 fixture question differs from the Task 6 reference")

    retriever = AuthenticatedFixedPageRetriever(
        fixture,
        query_encoder,
        validate_external_bytes=False,
    )
    retrieval_output = retriever.retrieve(question, len(images))
    live_pages = [
        {"doc_id": page.doc_id, "page_index": page.page_index, "score": page.score}
        for page in retrieval_output.pages
    ]
    if live_pages != reference_pages:
        raise ValueError("Task 8 fixed pages differ from the Task 6 reference")
    capture = derive_btp_qtp_geometry_without_qwen_model(
        qwen_processor,
        images,
        question,
        retrieval_output,
    )
    if (
        capture.geometry_count != expected_geometry_count
        or capture.geometry_sha256 != expected_geometry_sha
    ):
        raise ValueError("live post-QTP geometry identity differs from the Task 6 reference")
    live_trace = {
        "original_visual_tokens": capture.original_visual_tokens,
        "post_btp_visual_tokens": capture.post_btp_visual_tokens,
        "post_qtp_visual_tokens": capture.post_qtp_visual_tokens,
        "post_ctp_visual_tokens": capture.post_qtp_visual_tokens,
        "ctp_layer": None,
    }
    if live_trace != dict(reference_trace):
        raise ValueError("live BTP/QTP token counts differ from the Task 6 reference")

    gpu = None
    if torch.cuda.is_available():
        gpu = {
            "device_index": int(torch.cuda.current_device()),
            "device_name": torch.cuda.get_device_name(torch.cuda.current_device()),
        }
    geometry_rows = [
        [token.page_index, token.row, token.column, token.height, token.width]
        for token in capture.geometry
    ]
    if (
        len(geometry_rows) != capture.geometry_count
        or _sha256_bytes(_canonical_bytes(geometry_rows)) != capture.geometry_sha256
    ):
        raise ValueError("live post-QTP geometry rows do not match their identity")
    for label, digest in (
        ("background keep mask", capture.background_keep_sha256),
        ("question keep mask", capture.question_keep_sha256),
        ("combined keep mask", capture.combined_keep_sha256),
    ):
        _require_sha256(digest, label)
    unsigned: dict[str, object] = {
        "schema_version": TASK8_GEOMETRY_SCHEMA_VERSION,
        "status": "complete",
        "runtime_commit": runtime_commit,
        "qid": qid,
        "question_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
        "fixed_page_fixture_path": str(fixture_file),
        "fixed_page_fixture_sha256": fixture_sha,
        "smoke_input_manifest_path": str(smoke_input_manifest_path),
        "smoke_input_manifest_sha256": smoke_sha,
        "reference_results_path": str(reference_results_path),
        "reference_results_sha256": reference_results_sha256,
        "global_index_loaded": False,
        "retrieval_search_run": False,
        "fixed_page_query_encoding_run": True,
        "qwen_generation_model_loaded": False,
        "gpu": gpu,
        "resources": {
            "colpali_model": COLPALI_MODEL,
            "colpali_revision": COLPALI_REVISION,
            "colpali_backbone_model": COLPALI_BACKBONE_MODEL,
            "colpali_backbone_revision": COLPALI_BACKBONE_REVISION,
            "qwen_processor_model": QWEN_MODEL,
            "qwen_processor_revision": QWEN_REVISION,
        },
        "retrieved_pages": live_pages,
        "image_grid_thw": [list(row) for row in capture.image_grid_thw],
        "trace": live_trace,
        "mask_sha256": {
            "background_keep": capture.background_keep_sha256,
            "question_keep": capture.question_keep_sha256,
            "combined_keep": capture.combined_keep_sha256,
        },
        "geometry_count": capture.geometry_count,
        "geometry_sha256": capture.geometry_sha256,
        "geometry": geometry_rows,
    }
    payload = dict(unsigned)
    payload["manifest_sha256"] = _sha256_bytes(_canonical_bytes(unsigned))
    _publish_no_replace(payload, output)
    return payload


__all__ = [
    "TASK8_GEOMETRY_SCHEMA_VERSION",
    "capture_task8_btp_qtp_geometry",
]
