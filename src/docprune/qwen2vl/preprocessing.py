"""Pinned Qwen2-VL page preprocessing and merge-group bookkeeping."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image


@dataclass(frozen=True)
class PreparedQwenPage:
    """Exact processor output for one retained uint8 page raster."""

    raster: torch.Tensor
    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor
    patch_size: int
    temporal_patch_size: int
    merge_size: int
    placeholder_count: int
    placeholder_raster_indices: torch.Tensor


def prepared_raster_image(page: PreparedQwenPage) -> Image.Image:
    """Convert the retained channels-first uint8 raster back to a PIL page."""

    raster = torch.as_tensor(page.raster)
    if raster.shape != (1, 3, raster.shape[-2], raster.shape[-1]) or raster.dtype != torch.uint8:
        raise ValueError("prepared page raster must be [1, 3, height, width] uint8")
    array = raster[0].permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(np.ascontiguousarray(array), mode="RGB")


def _value(container: object, name: str) -> object:
    if isinstance(container, Mapping):
        if name not in container:
            raise ValueError(f"Qwen processor output is missing {name}")
        return container[name]
    if not hasattr(container, name):
        raise ValueError(f"Qwen processor output is missing {name}")
    return getattr(container, name)


def _uint8_rgb(image: object) -> tuple[torch.Tensor, Image.Image]:
    if isinstance(image, Image.Image):
        rgb = image.convert("RGB")
        array = np.asarray(rgb, dtype=np.uint8).copy()
        raster = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        return raster, rgb

    raster = torch.as_tensor(image)
    if raster.ndim == 3:
        if raster.shape[0] in (1, 3):
            raster = raster.unsqueeze(0)
        elif raster.shape[-1] in (1, 3):
            raster = raster.permute(2, 0, 1).unsqueeze(0)
    if raster.ndim != 4 or raster.shape[0] != 1:
        raise ValueError("Qwen page raster must have shape [3, height, width] or [1, 3, height, width]")
    if raster.shape[1] != 3:
        raise ValueError("Qwen page raster must contain three RGB channels")
    if raster.dtype != torch.uint8:
        if not raster.is_floating_point() or not bool(torch.isfinite(raster).all()):
            raise ValueError("Qwen page raster must contain finite uint8-equivalent pixels")
        rounded = raster.round()
        if not torch.equal(raster, rounded) or float(rounded.min()) < 0 or float(rounded.max()) > 255:
            raise ValueError("Qwen page raster must contain uint8-equivalent pixels")
        raster = rounded.to(torch.uint8)
    array = raster[0].permute(1, 2, 0).cpu().numpy()
    return raster, Image.fromarray(np.ascontiguousarray(array), mode="RGB")


def prepare_qwen_page(processor: object, image: object) -> PreparedQwenPage:
    """Run the pinned Qwen image processor on an exact retained uint8 raster.

    The processor owns smart-resize geometry and patch ordering.  We validate
    its returned grid instead of reconstructing dimensions from model defaults;
    this keeps placeholder order exactly ``(t, h_group, w_group, dy, dx)``.
    """

    image_processor = getattr(processor, "image_processor", processor)
    raster, pil_image = _uint8_rgb(image)
    resized_image = pil_image
    if bool(getattr(image_processor, "do_resize", True)):
        try:
            from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

            resized_h, resized_w = smart_resize(
                pil_image.height,
                pil_image.width,
                factor=int(getattr(image_processor, "patch_size", 14))
                * int(getattr(image_processor, "merge_size", 2)),
                min_pixels=int(getattr(image_processor, "min_pixels", 56 * 56)),
                max_pixels=int(getattr(image_processor, "max_pixels", 28 * 28 * 1280)),
            )
            resample = getattr(image_processor, "resample", Image.Resampling.BICUBIC)
            resized_image = pil_image.resize((resized_w, resized_h), resample=resample)
        except ImportError:
            # A pinned real processor always takes the branch above.  Small
            # test doubles may own their own geometry and need no resize.
            resized_image = pil_image
    try:
        output = image_processor(images=[resized_image], return_tensors="pt")
    except TypeError:
        output = image_processor(resized_image, return_tensors="pt")
    pixels = torch.as_tensor(_value(output, "pixel_values"))
    grid = torch.as_tensor(_value(output, "image_grid_thw"), dtype=torch.long)
    if pixels.ndim != 2:
        raise ValueError("Qwen pixel_values must have shape [fine_tokens, channels*patch_area]")
    if grid.shape != (1, 3):
        raise ValueError("one Qwen page must produce image_grid_thw with shape [1, 3]")
    temporal, height, width = (int(value) for value in grid[0].tolist())
    patch_size = int(getattr(image_processor, "patch_size", 14))
    temporal_patch_size = int(getattr(image_processor, "temporal_patch_size", 2))
    merge_size = int(getattr(image_processor, "merge_size", 2))
    if min(patch_size, temporal_patch_size, merge_size) <= 0:
        raise ValueError("Qwen processor patch and merge geometry must be positive")
    if temporal != 1:
        raise ValueError("document pages require temporal grid size one")
    if height % merge_size or width % merge_size:
        raise ValueError("Qwen image_grid_thw height and width must be merge divisible")
    fine_count = temporal * height * width
    if pixels.shape[0] != fine_count:
        raise ValueError(
            f"Qwen pixel_values rows ({pixels.shape[0]}) do not match image_grid_thw ({fine_count})"
        )
    channels_times_patch = pixels.shape[1]
    if channels_times_patch % (temporal_patch_size * patch_size * patch_size):
        raise ValueError("Qwen pixel_values width does not match processor patch geometry")
    groups = fine_count // (merge_size**2)
    if groups <= 0:
        raise ValueError("Qwen processor must produce at least one visual merge group")
    raster_array = np.asarray(resized_image, dtype=np.uint8).copy()
    retained_raster = torch.from_numpy(raster_array).permute(2, 0, 1).unsqueeze(0)
    if retained_raster.shape[-2:] != (height * patch_size, width * patch_size):
        raise ValueError("retained raster dimensions do not match Qwen image_grid_thw")
    return PreparedQwenPage(
        raster=retained_raster,
        pixel_values=pixels,
        image_grid_thw=grid,
        patch_size=patch_size,
        temporal_patch_size=temporal_patch_size,
        merge_size=merge_size,
        placeholder_count=groups,
        placeholder_raster_indices=torch.arange(groups, dtype=torch.long),
    )
