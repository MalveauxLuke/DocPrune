"""Sparse, merge-safe execution of the pinned Qwen2-VL vision encoder."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional


@dataclass(frozen=True)
class CompactVisionBatch:
    image_embeds: torch.Tensor
    fine_position_embeddings: torch.Tensor
    page_fine_counts: tuple[int, ...]


def compact_vision_batch(
    visual_model: object,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    group_keep_mask: torch.Tensor,
) -> CompactVisionBatch:
    """Run only selected group-contiguous fine patches through the vision encoder."""

    grid = torch.as_tensor(image_grid_thw, dtype=torch.long)
    keep = torch.as_tensor(group_keep_mask, dtype=torch.bool, device=pixel_values.device)
    if grid.ndim != 2 or grid.shape[1] != 3:
        raise ValueError("image_grid_thw must have shape [pages, 3]")
    if torch.any(grid[:, 0] != 1):
        raise ValueError("the document reproduction supports one temporal grid per page")
    merge_size = int(visual_model.spatial_merge_size)
    merge_area = merge_size**2
    fine_counts = tuple(int(t * h * w) for t, h, w in grid.tolist())
    if any(count % merge_area for count in fine_counts):
        raise ValueError("fine-token counts must be divisible by the spatial merge area")
    group_counts = tuple(count // merge_area for count in fine_counts)
    if keep.ndim != 1 or keep.numel() != sum(group_counts):
        raise ValueError(f"group_keep_mask must contain {sum(group_counts)} values")
    page_group_offsets = [0]
    for count in group_counts:
        page_group_offsets.append(page_group_offsets[-1] + count)
    kept_groups_per_page = tuple(
        int(keep[start:end].sum().item())
        for start, end in zip(page_group_offsets[:-1], page_group_offsets[1:])
    )
    if any(count == 0 for count in kept_groups_per_page):
        raise ValueError("each page must retain at least one complete visual token group")
    if pixel_values.shape[0] != sum(fine_counts):
        raise ValueError(f"pixel_values must contain {sum(fine_counts)} fine patches")

    fine_keep = keep.repeat_interleave(merge_area)
    selected_pixels = pixel_values[fine_keep]
    hidden_states = visual_model.patch_embed(selected_pixels)
    full_positions = visual_model.rot_pos_emb(grid.to(pixel_values.device))
    rotary_positions = full_positions[fine_keep]
    page_fine_counts = tuple(count * merge_area for count in kept_groups_per_page)
    cumulative = torch.tensor(page_fine_counts, dtype=torch.int32, device=hidden_states.device).cumsum(
        dim=0, dtype=torch.int32
    )
    cumulative = functional.pad(cumulative, (1, 0), value=0)
    for block in visual_model.blocks:
        hidden_states = block(
            hidden_states,
            cu_seqlens=cumulative,
            rotary_pos_emb=rotary_positions,
        )
    image_embeds = visual_model.merger(hidden_states)
    return CompactVisionBatch(image_embeds, rotary_positions, page_fine_counts)
