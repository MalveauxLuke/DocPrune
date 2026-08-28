"""Sealed fixed-page inputs for the bounded Task 8 MinerU smoke."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path

from PIL import Image

from docprune.task6_runtime import load_fixed_page_fixture, render_task6_pdf_page

TASK8_SMOKE_SCHEMA_VERSION = "docprune-task8-mineru-smoke-input-v1"


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _read_regular(path: Path, label: str) -> bytes:
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
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


def _reject_symlink_components(path: Path, label: str) -> None:
    current = path
    while True:
        if current.is_symlink():
            raise ValueError(f"{label} path contains a symlink")
        if current == current.parent:
            break
        current = current.parent


def _publish_staged_directory(stage: Path, destination: Path) -> None:
    """Claim a new sibling directory through a pinned no-follow parent handle."""

    parent_descriptor: int | None = None
    output_descriptor: int | None = None
    try:
        parent_descriptor = os.open(
            destination.parent,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        os.mkdir(destination.name, dir_fd=parent_descriptor)
        output_descriptor = os.open(
            destination.name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_descriptor,
        )
        for staged_path in sorted(stage.iterdir(), key=lambda path: path.name):
            if staged_path.name != "smoke-input-manifest.json":
                os.replace(staged_path, staged_path.name, dst_dir_fd=output_descriptor)
        os.replace(
            stage / "smoke-input-manifest.json",
            "smoke-input-manifest.json",
            dst_dir_fd=output_descriptor,
        )
        stage.rmdir()
    finally:
        if output_descriptor is not None:
            os.close(output_descriptor)
        if parent_descriptor is not None:
            os.close(parent_descriptor)


def _require_commit(value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("Task 8 runtime commit must be a lowercase 40-character revision")
    return value


def _record_sha256(record: object) -> str:
    return _sha256_bytes(_canonical_bytes(record))


def _validate_manifest(value: object, manifest_path: Path) -> dict[str, object]:
    required = {
        "schema_version",
        "status",
        "runtime_commit",
        "fixed_page_fixture_path",
        "fixed_page_fixture_sha256",
        "qid",
        "global_index_loaded",
        "pages",
        "manifest_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("Task 8 smoke input manifest has an invalid schema")
    payload = dict(value)
    supplied = payload.pop("manifest_sha256")
    if supplied != _sha256_bytes(_canonical_bytes(payload)):
        raise ValueError("Task 8 smoke input manifest digest is invalid")
    if (
        value["schema_version"] != TASK8_SMOKE_SCHEMA_VERSION
        or value["status"] != "sealed-fixed-pages"
        or value["global_index_loaded"] is not False
    ):
        raise ValueError("Task 8 smoke input manifest permits an invalid execution mode")
    _require_commit(value["runtime_commit"])
    fixture_path = Path(value["fixed_page_fixture_path"])
    if not fixture_path.is_absolute():
        raise ValueError("Task 8 fixed-page fixture path must be absolute")
    fixture = load_fixed_page_fixture(
        fixture_path,
        expected_sha256=value["fixed_page_fixture_sha256"],
        validate_external_bytes=False,
    )
    qid = value["qid"]
    if not isinstance(qid, str) or not qid:
        raise ValueError("Task 8 smoke QID must be nonempty")
    fixture.validate_external_bytes(selected_qids=(qid,))
    fixed = fixture.question(qid)
    pages = value["pages"]
    if not isinstance(pages, list) or len(pages) != len(fixed.pages) or not pages:
        raise ValueError("Task 8 smoke pages do not match the fixed-page fixture")
    expected_page_fields = {
        "input_page_index",
        "mineru_page_index",
        "document_id",
        "source_page_index",
        "fixture_rank",
        "fixed_page_record",
        "fixed_page_record_sha256",
        "mineru_input_path",
        "mineru_input_sha256",
        "rendered_rgb_width",
        "rendered_rgb_height",
        "rendered_rgb_sha256",
    }
    root = manifest_path.parent.resolve()
    for rank, (row, record) in enumerate(zip(pages, fixed.pages, strict=True)):
        if not isinstance(row, Mapping) or set(row) != expected_page_fields:
            raise ValueError("Task 8 smoke page row has an invalid schema")
        record_value = record.to_dict()
        input_path = Path(row["mineru_input_path"])
        expected_path = root / f"page-{rank:02d}.png"
        if (
            row["input_page_index"] != rank
            or row["mineru_page_index"] != 0
            or row["document_id"] != record.doc_id
            or row["source_page_index"] != record.page_index
            or row["fixture_rank"] != rank
            or row["fixed_page_record"] != record_value
            or row["fixed_page_record_sha256"] != _record_sha256(record_value)
            or input_path != expected_path
            or row["rendered_rgb_width"] != record.rendered_rgb_width
            or row["rendered_rgb_height"] != record.rendered_rgb_height
            or row["rendered_rgb_sha256"] != record.rendered_rgb_sha256
        ):
            raise ValueError("Task 8 smoke page row contradicts the fixed-page fixture")
        raw = _read_regular(input_path, "MinerU input")
        if _sha256_bytes(raw) != row["mineru_input_sha256"]:
            raise ValueError("MinerU input SHA-256 mismatch")
        try:
            with Image.open(input_path) as opened:
                image = opened.convert("RGB")
                image.load()
        except OSError as error:
            raise ValueError("MinerU input is not a valid image") from error
        if image.size != (record.rendered_rgb_width, record.rendered_rgb_height) or (
            _sha256_bytes(image.tobytes()) != record.rendered_rgb_sha256
        ):
            raise ValueError("MinerU input decoded RGB does not match the fixed page")
    return dict(value)


def load_task8_smoke_inputs(path: Path) -> dict[str, object]:
    """Load and reauthenticate a completed smoke-input seal."""

    manifest_path = Path(path)
    if not manifest_path.is_absolute():
        raise ValueError("Task 8 smoke input manifest path must be absolute")
    _reject_symlink_components(manifest_path.parent, "Task 8 smoke input manifest")
    raw = _read_regular(manifest_path, "Task 8 smoke input manifest")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Task 8 smoke input manifest is invalid JSON") from error
    if raw != _canonical_bytes(value):
        raise ValueError("Task 8 smoke input manifest bytes are not canonical")
    return _validate_manifest(value, manifest_path)


def seal_task8_smoke_inputs(
    *,
    fixture_path: Path,
    fixture_sha256: str,
    qid: str,
    output_root: Path,
    runtime_commit: str,
    render_page: Callable[[Path, int], Image.Image] = render_task6_pdf_page,
) -> dict[str, object]:
    """Render exact fixture pages to immutable per-page MinerU image inputs."""

    requested_root = Path(output_root)
    if not requested_root.is_absolute():
        raise ValueError("Task 8 smoke output root must be absolute")
    _reject_symlink_components(requested_root.parent, "Task 8 smoke output")
    if requested_root.exists() or requested_root.is_symlink():
        raise FileExistsError(f"Task 8 smoke output already exists: {requested_root}")
    root = requested_root.resolve()
    fixture_path = Path(fixture_path)
    if not fixture_path.is_absolute():
        raise ValueError("Task 8 fixed-page fixture path must be absolute")
    _reject_symlink_components(fixture_path.parent, "Task 8 fixed-page fixture")
    _require_commit(runtime_commit)
    fixture = load_fixed_page_fixture(
        fixture_path,
        expected_sha256=fixture_sha256,
        validate_external_bytes=False,
    )
    fixture.validate_external_bytes(selected_qids=(qid,))
    fixed = fixture.question(qid)
    root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    try:
        pages: list[dict[str, object]] = []
        for rank, record in enumerate(fixed.pages):
            image = render_page(record.source_pdf_path, record.page_index).convert("RGB")
            if image.size != (record.rendered_rgb_width, record.rendered_rgb_height) or (
                _sha256_bytes(image.tobytes()) != record.rendered_rgb_sha256
            ):
                raise ValueError("rendered RGB does not match the fixed-page fixture")
            staged_path = stage / f"page-{rank:02d}.png"
            image.save(staged_path, format="PNG", optimize=False, compress_level=9)
            final_path = root / staged_path.name
            record_value = record.to_dict()
            pages.append(
                {
                    "input_page_index": rank,
                    "mineru_page_index": 0,
                    "document_id": record.doc_id,
                    "source_page_index": record.page_index,
                    "fixture_rank": rank,
                    "fixed_page_record": record_value,
                    "fixed_page_record_sha256": _record_sha256(record_value),
                    "mineru_input_path": str(final_path),
                    "mineru_input_sha256": _sha256_bytes(staged_path.read_bytes()),
                    "rendered_rgb_width": record.rendered_rgb_width,
                    "rendered_rgb_height": record.rendered_rgb_height,
                    "rendered_rgb_sha256": record.rendered_rgb_sha256,
                }
            )
        unsigned: dict[str, object] = {
            "schema_version": TASK8_SMOKE_SCHEMA_VERSION,
            "status": "sealed-fixed-pages",
            "runtime_commit": runtime_commit,
            "fixed_page_fixture_path": str(fixture_path),
            "fixed_page_fixture_sha256": fixture_sha256,
            "qid": qid,
            "global_index_loaded": False,
            "pages": pages,
        }
        manifest = {**unsigned, "manifest_sha256": _sha256_bytes(_canonical_bytes(unsigned))}
        (stage / "smoke-input-manifest.json").write_bytes(_canonical_bytes(manifest))
        _publish_staged_directory(stage, root)
        return load_task8_smoke_inputs(root / "smoke-input-manifest.json")
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise


__all__ = [
    "TASK8_SMOKE_SCHEMA_VERSION",
    "load_task8_smoke_inputs",
    "seal_task8_smoke_inputs",
]
