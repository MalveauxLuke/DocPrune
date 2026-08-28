"""Deterministic visual audit of Task 8 regions and deep-token masks."""

from __future__ import annotations

import errno
import hashlib
import io
import json
import os
import secrets
import stat
from ctypes import CDLL, c_char_p, c_int, c_uint, get_errno
from pathlib import Path

from PIL import Image, ImageDraw

from docprune.segmentation import (
    RegionTokenMapping,
    load_region_mapping,
    region_mapping_json_bytes,
)

OVERLAY_SCHEMA_VERSION = "docprune-task8-region-overlay-v1"
OVERLAY_CONTRACT = (
    "left=source-boxes;right=post-qtp-token-masks;"
    "color=sha256-source-id;separator=8px-black;native-sealed-rgb"
)
SEPARATOR_WIDTH = 8
RENAME_NOREPLACE = 1


def _read_regular(path: Path, *, label: str) -> bytes:
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
        raise ValueError(f"{label} is missing or unsafe") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


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


def _stage_output_directory(path: Path) -> tuple[int, int, str, str]:
    destination = Path(path)
    if not destination.is_absolute() or destination.name in {"", ".", ".."}:
        raise ValueError("overlay output directory must be an absolute child path")
    parent_descriptor = _open_directory_nofollow(destination.parent)
    try:
        for _attempt in range(32):
            stage_name = f".{destination.name}.stage-{secrets.token_hex(8)}"
            try:
                os.mkdir(stage_name, mode=0o750, dir_fd=parent_descriptor)
            except FileExistsError:
                continue
            stage_descriptor = os.open(
                stage_name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=parent_descriptor,
            )
            return parent_descriptor, stage_descriptor, stage_name, destination.name
        raise FileExistsError("could not claim a unique overlay staging directory")
    except BaseException:
        os.close(parent_descriptor)
        raise


def _rename_directory_noreplace(
    parent_descriptor: int, stage_name: str, destination_name: str
) -> None:
    libc = CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOSYS, "renameat2 is unavailable", destination_name)
    renameat2.argtypes = [c_int, c_char_p, c_int, c_char_p, c_uint]
    renameat2.restype = c_int
    if (
        renameat2(
            parent_descriptor,
            os.fsencode(stage_name),
            parent_descriptor,
            os.fsencode(destination_name),
            RENAME_NOREPLACE,
        )
        != 0
    ):
        error_number = get_errno()
        raise OSError(error_number, os.strerror(error_number), destination_name)


def _publish_staged_directory_by_links(
    parent_descriptor: int,
    stage_descriptor: int,
    stage_name: str,
    destination_name: str,
    filenames: tuple[str, ...],
) -> None:
    """Publish regular files with the completion manifest linked last."""

    os.mkdir(destination_name, mode=0o750, dir_fd=parent_descriptor)
    destination_descriptor = os.open(
        destination_name,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        dir_fd=parent_descriptor,
    )
    linked: list[str] = []
    complete = False
    try:
        for filename in filenames:
            os.link(
                filename,
                filename,
                src_dir_fd=stage_descriptor,
                dst_dir_fd=destination_descriptor,
                follow_symlinks=False,
            )
            linked.append(filename)
        os.fsync(destination_descriptor)
        complete = True
    finally:
        if not complete:
            for filename in reversed(linked):
                try:
                    os.unlink(filename, dir_fd=destination_descriptor)
                except FileNotFoundError:
                    pass
        os.close(destination_descriptor)
        if not complete:
            os.rmdir(destination_name, dir_fd=parent_descriptor)
    os.fsync(parent_descriptor)
    for filename in filenames:
        try:
            os.unlink(filename, dir_fd=stage_descriptor)
        except FileNotFoundError:
            pass
    try:
        os.rmdir(stage_name, dir_fd=parent_descriptor)
    except OSError:
        pass


def _write_new(directory_descriptor: int, filename: str, content: bytes) -> None:
    descriptor = os.open(
        filename,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o640,
        dir_fd=directory_descriptor,
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _source_color(source_id: str) -> tuple[int, int, int]:
    digest = hashlib.sha256(source_id.encode("utf-8")).digest()
    return tuple(48 + byte % 160 for byte in digest[:3])


def _pixel_box(
    bbox: tuple[float, float, float, float],
    page_size: tuple[float, float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    page_width, page_height = page_size
    x0, y0, x1, y1 = bbox
    left = max(0, min(width - 1, round(width * x0 / page_width)))
    top = max(0, min(height - 1, round(height * y0 / page_height)))
    right = max(left, min(width - 1, round(width * x1 / page_width) - 1))
    bottom = max(top, min(height - 1, round(height * y1 / page_height) - 1))
    return left, top, right, bottom


def _render_page(
    mapping: RegionTokenMapping, input_page_index: int
) -> tuple[bytes, dict[str, object]]:
    artifact = next(
        item for item in mapping.artifacts if item.pages[0].input_page_index == input_page_index
    )
    page = artifact.pages[0]
    raw_image = _read_regular(artifact.mineru_input_path, label="Task 8 overlay input image")
    if hashlib.sha256(raw_image).hexdigest() != artifact.mineru_input_sha256:
        raise ValueError("Task 8 overlay input image SHA-256 mismatch")
    try:
        with Image.open(io.BytesIO(raw_image)) as opened:
            image = opened.convert("RGB")
    except Exception as error:
        raise ValueError("Task 8 overlay input image is invalid") from error
    if image.size != (page.rendered_rgb_width, page.rendered_rgb_height) or (
        hashlib.sha256(image.tobytes()).hexdigest() != page.rendered_rgb_sha256
    ):
        raise ValueError("Task 8 overlay input decoded RGB identity mismatch")

    width, height = image.size
    left = image.convert("RGBA")
    right = image.convert("RGBA")
    sources = tuple(
        source for source in mapping.sources if source.input_page_index == input_page_index
    )
    by_id = {source.source_id: source for source in sources}
    left_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    left_draw = ImageDraw.Draw(left_layer)
    for source in (source for source in sources if source.source_kind == "residual-grid"):
        color = _source_color(source.source_id)
        box = _pixel_box(source.bbox, source.page_size, width, height)
        left_draw.rectangle(box, fill=(*color, 20), outline=(*color, 255), width=1)
    audited_regions = tuple(
        region for region in mapping.audited_regions if region.input_page_index == input_page_index
    )
    for region in audited_regions:
        color = _source_color(region.source_id())
        box = _pixel_box(region.bbox, region.page_size, width, height)
        left_draw.rectangle(box, fill=(*color, 42), outline=(*color, 255), width=1)
    left = Image.alpha_composite(left, left_layer)

    right_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    right_draw = ImageDraw.Draw(right_layer)
    token_count = 0
    for token_id, token in enumerate(mapping.geometry):
        if token.page_index != input_page_index:
            continue
        source = by_id[mapping.token_to_source[token_id]]
        color = _source_color(source.source_id)
        box = (
            round(width * token.column / token.width),
            round(height * token.row / token.height),
            max(0, round(width * (token.column + 1) / token.width) - 1),
            max(0, round(height * (token.row + 1) / token.height) - 1),
        )
        right_draw.rectangle(box, fill=(*color, 112), outline=(*color, 224), width=1)
        token_count += 1
    right = Image.alpha_composite(right, right_layer)

    canvas = Image.new("RGB", (2 * width + SEPARATOR_WIDTH, height), color=(0, 0, 0))
    canvas.paste(left.convert("RGB"), (0, 0))
    canvas.paste(right.convert("RGB"), (width + SEPARATOR_WIDTH, 0))
    output = io.BytesIO()
    canvas.save(output, format="PNG", optimize=False, compress_level=9)
    content = output.getvalue()
    filename = f"page-{input_page_index:02d}-regions-and-token-masks.png"
    return content, {
        "input_page_index": input_page_index,
        "document_id": page.document_id,
        "source_page_index": page.source_page_index,
        "input_path": str(artifact.mineru_input_path),
        "input_sha256": artifact.mineru_input_sha256,
        "rendered_rgb_sha256": page.rendered_rgb_sha256,
        "width": width,
        "height": height,
        "token_count": token_count,
        "output_filename": filename,
        "output_sha256": hashlib.sha256(content).hexdigest(),
        "sources": [
            {
                "source_id": source.source_id,
                "source_kind": source.source_kind,
                "region_type": source.region_type,
                "color_rgb": list(_source_color(source.source_id)),
                "token_cost": source.token_cost,
            }
            for source in sources
        ],
        "empty_regions": [
            {
                "source_id": region.source_id(),
                "region_type": region.region_type,
                "color_rgb": list(_source_color(region.source_id())),
                "token_cost": 0,
            }
            for region in audited_regions
            if region.source_id() in mapping.empty_region_source_ids
        ],
    }


def publish_task8_region_overlays(mapping_path: Path, output_dir: Path) -> dict[str, object]:
    """Publish one replayable overlay set from an authenticated cached mapping."""

    mapping_file = Path(mapping_path)
    mapping_bytes = _read_regular(mapping_file, label="Task 8 region mapping")
    mapping = load_region_mapping(mapping_file)
    if not isinstance(mapping, RegionTokenMapping):
        raise TypeError("Task 8 region mapping loader returned an invalid object")
    mapping_bytes_after_load = _read_regular(mapping_file, label="Task 8 region mapping")
    if not (mapping_bytes == mapping_bytes_after_load == region_mapping_json_bytes(mapping)):
        raise ValueError("Task 8 region mapping changed during authentication")
    page_indices = tuple(
        artifact.pages[0].input_page_index
        for artifact in sorted(mapping.artifacts, key=lambda item: item.pages[0].input_page_index)
    )
    if page_indices != tuple(range(len(page_indices))):
        raise ValueError("Task 8 overlay pages are not in exact input order")

    rendered_pages = tuple(_render_page(mapping, page_index) for page_index in page_indices)
    destination = Path(output_dir)
    parent_descriptor, stage_descriptor, stage_name, destination_name = _stage_output_directory(
        destination
    )
    written_names: list[str] = []
    published = False
    try:
        page_rows: list[dict[str, object]] = []
        for png_bytes, row in rendered_pages:
            filename = str(row["output_filename"])
            written_names.append(filename)
            _write_new(stage_descriptor, filename, png_bytes)
            page_rows.append(row)
        manifest: dict[str, object] = {
            "schema_version": OVERLAY_SCHEMA_VERSION,
            "overlay_contract": OVERLAY_CONTRACT,
            "mapping_path": str(mapping_file),
            "mapping_file_sha256": hashlib.sha256(mapping_bytes).hexdigest(),
            "mapping_sha256": mapping.sha256,
            "geometry_count": mapping.geometry_count,
            "page_count": len(page_rows),
            "pages": page_rows,
        }
        manifest_bytes = (
            json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
        ).encode()
        written_names.append("manifest.json")
        _write_new(stage_descriptor, "manifest.json", manifest_bytes)
        os.fsync(stage_descriptor)
        try:
            _rename_directory_noreplace(parent_descriptor, stage_name, destination_name)
        except OSError as error:
            if error.errno not in {errno.EINVAL, errno.ENOSYS, errno.ENOTSUP, errno.EOPNOTSUPP}:
                raise
            _publish_staged_directory_by_links(
                parent_descriptor,
                stage_descriptor,
                stage_name,
                destination_name,
                tuple(written_names),
            )
        published = True
        os.fsync(parent_descriptor)
    finally:
        if not published:
            for filename in written_names:
                try:
                    os.unlink(filename, dir_fd=stage_descriptor)
                except FileNotFoundError:
                    pass
        os.close(stage_descriptor)
        if not published:
            try:
                os.rmdir(stage_name, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass
        os.close(parent_descriptor)
    return manifest


__all__ = ["publish_task8_region_overlays"]
