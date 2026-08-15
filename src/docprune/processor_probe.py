"""Shape-only probe for the Qwen2-VL and ColPali processor contract."""

from __future__ import annotations

import json
import math
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from PIL import Image

COMMIT_PATTERN = re.compile(r"^[0-9a-fA-F]{40}$")


def require_immutable_revision(value: str, *, name: str) -> str:
    if not COMMIT_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must be a 40-character hexadecimal commit hash")
    return value.lower()


def _value(container: object, name: str) -> Any:
    if isinstance(container, Mapping):
        if name not in container:
            raise ValueError(f"processor output is missing {name}")
        return container[name]
    if not hasattr(container, name):
        raise ValueError(f"processor output is missing {name}")
    return getattr(container, name)


def _configuration_value(container: object, name: str) -> int:
    if isinstance(container, Mapping):
        value = container.get(name)
    else:
        value = getattr(container, name, None)
    if value is None:
        raise ValueError(f"Qwen vision configuration is missing {name}")
    return int(value)


def _image_token_id(processor: object) -> int | None:
    for owner in (processor, getattr(processor, "tokenizer", None)):
        if owner is None:
            continue
        for name in ("image_token_id", "image_token_index"):
            value = getattr(owner, name, None)
            if value is not None:
                return int(value)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "convert_tokens_to_ids"):
        candidate = tokenizer.convert_tokens_to_ids("<image>")
        unknown = getattr(tokenizer, "unk_token_id", None)
        if candidate is not None and candidate != unknown and int(candidate) >= 0:
            return int(candidate)
    return None


def _perfect_square_grid(token_count: int) -> list[int] | None:
    side = math.isqrt(token_count)
    return [side, side] if side * side == token_count and token_count > 0 else None


def collect_processor_contract(
    *,
    image: Image.Image,
    qwen_processor: object,
    qwen_config: object,
    qwen_model: str,
    qwen_revision: str,
    colpali_processor: object,
    colpali_model: str,
    colpali_revision: str,
) -> dict[str, object]:
    """Collect only structural metadata needed to connect QTP to Qwen2-VL."""

    qwen_revision = require_immutable_revision(qwen_revision, name="qwen_revision")
    colpali_revision = require_immutable_revision(colpali_revision, name="colpali_revision")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "processor-contract-probe"},
                {"type": "text", "text": "probe"},
            ],
        }
    ]
    prompt = qwen_processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    qwen_batch = qwen_processor(
        text=[prompt],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    colpali_batch = colpali_processor.process_images([image])

    qwen_pixels = torch.as_tensor(_value(qwen_batch, "pixel_values"))
    qwen_grid = torch.as_tensor(_value(qwen_batch, "image_grid_thw"), dtype=torch.long)
    if qwen_grid.shape != (1, 3):
        raise ValueError("one-page Qwen image_grid_thw must have shape [1, 3]")
    temporal, grid_height, grid_width = (int(value) for value in qwen_grid[0])
    vision_config = getattr(qwen_config, "vision_config", None)
    if vision_config is None:
        raise ValueError("Qwen configuration is missing vision_config")
    patch_size = _configuration_value(vision_config, "patch_size")
    temporal_patch_size = _configuration_value(vision_config, "temporal_patch_size")
    merge_size = _configuration_value(vision_config, "spatial_merge_size")
    fine_tokens = temporal * grid_height * grid_width
    merge_area = merge_size**2
    merge_valid = fine_tokens % merge_area == 0

    colpali_ids = torch.as_tensor(_value(colpali_batch, "input_ids"), dtype=torch.long)
    colpali_attention = torch.as_tensor(_value(colpali_batch, "attention_mask"))
    if colpali_ids.ndim != 2 or colpali_ids.shape[0] != 1:
        raise ValueError("one-page ColPali input_ids must have shape [1, sequence]")
    if colpali_attention.shape != colpali_ids.shape:
        raise ValueError("ColPali attention_mask must match input_ids")
    image_token_id = _image_token_id(colpali_processor)
    positions = (
        []
        if image_token_id is None
        else (colpali_ids[0] == image_token_id).nonzero(as_tuple=False).flatten().tolist()
    )
    inferred_grid = _perfect_square_grid(len(positions))
    unresolved: list[str] = [
        "ColPali raster order requires review against the pinned processor implementation."
    ]
    if image_token_id is None:
        unresolved.append("ColPali image token ID could not be detected.")
    elif not positions:
        unresolved.append("No ColPali image-token positions were detected.")
    if inferred_grid is None:
        unresolved.append("ColPali visual tokens do not form an inferred square grid.")
    if not merge_valid:
        unresolved.append("Qwen fine-token count is not divisible by its merge area.")

    return {
        "schema_version": 1,
        "resources": {
            "qwen": {"model": qwen_model, "revision": qwen_revision},
            "colpali": {"model": colpali_model, "revision": colpali_revision},
        },
        "page": {"raw_size_wh": [int(image.width), int(image.height)]},
        "qwen": {
            "grid_thw": [temporal, grid_height, grid_width],
            "merged_visual_token_count": fine_tokens // merge_area if merge_valid else None,
            "patch_size": patch_size,
            "pixel_values_shape": list(qwen_pixels.shape),
            "resized_size_hw": [grid_height * patch_size, grid_width * patch_size],
            "spatial_merge_size": merge_size,
            "temporal_patch_size": temporal_patch_size,
        },
        "colpali": {
            "attention_token_count": int(colpali_attention.sum().item()),
            "candidate_visual_token_count": len(positions),
            "image_token_id": image_token_id,
            "image_token_positions": positions,
            "inferred_visual_grid_hw": inferred_grid,
            "pixel_values_shape": list(torch.as_tensor(_value(colpali_batch, "pixel_values")).shape),
            "sequence_length": int(colpali_ids.shape[1]),
        },
        "mapping_checks": {
            "colpali_visual_grid_inferred": inferred_grid is not None,
            "qwen_merge_groups_valid": merge_valid,
            "raster_order_verified": False,
        },
        "unresolved": unresolved,
    }


def write_processor_contract(path: Path, payload: Mapping[str, object]) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"processor contract already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def run_processor_probe(
    *,
    page_image: Path,
    qwen_model: str,
    qwen_revision: str,
    colpali_model: str,
    colpali_revision: str,
    output: Path,
) -> dict[str, object]:
    """Load pinned processors, collect their shape contract, and write one report."""

    qwen_revision = require_immutable_revision(qwen_revision, name="qwen_revision")
    colpali_revision = require_immutable_revision(colpali_revision, name="colpali_revision")
    if Path(output).exists():
        raise FileExistsError(f"processor contract already exists: {output}")

    from colpali_engine.models import ColPaliProcessor
    from transformers import AutoConfig, AutoProcessor

    qwen_processor = AutoProcessor.from_pretrained(qwen_model, revision=qwen_revision)
    qwen_config = AutoConfig.from_pretrained(qwen_model, revision=qwen_revision)
    colpali_processor = ColPaliProcessor.from_pretrained(
        colpali_model,
        revision=colpali_revision,
    )
    with Image.open(page_image) as opened:
        image = opened.convert("RGB").copy()
    payload = collect_processor_contract(
        image=image,
        qwen_processor=qwen_processor,
        qwen_config=qwen_config,
        qwen_model=qwen_model,
        qwen_revision=qwen_revision,
        colpali_processor=colpali_processor,
        colpali_model=colpali_model,
        colpali_revision=colpali_revision,
    )
    write_processor_contract(output, payload)
    return payload
