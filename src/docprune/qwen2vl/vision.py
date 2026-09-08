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


def _cumulative_lengths(lengths: tuple[int, ...], *, device: torch.device) -> torch.Tensor:
    cumulative = torch.tensor(lengths, dtype=torch.int32, device=device).cumsum(
        dim=0, dtype=torch.int32
    )
    return functional.pad(cumulative, (1, 0), value=0)


def _compact_qwen25_vision(
    visual_model: object,
    selected_pixels: torch.Tensor,
    grid: torch.Tensor,
    keep: torch.Tensor,
    fine_keep: torch.Tensor,
    page_fine_counts: tuple[int, ...],
    merge_area: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Preserve Qwen2.5-VL's window order and attention boundaries after deletion."""

    hidden_states = visual_model.patch_embed(selected_pixels)
    full_rotary = visual_model.rot_pos_emb(grid.to(selected_pixels.device))
    selected_rotary = full_rotary[fine_keep]

    full_window_index, full_window_cumulative = visual_model.get_window_index(
        grid.to(selected_pixels.device)
    )
    full_window_index = torch.as_tensor(
        full_window_index, dtype=torch.long, device=hidden_states.device
    )
    if full_window_index.shape != keep.shape:
        raise ValueError("Qwen2.5-VL window index does not match visual-token groups")

    compact_group_ids = keep.to(torch.long).cumsum(dim=0) - 1
    retained_window_groups = full_window_index[keep[full_window_index]]
    compact_window_index = compact_group_ids[retained_window_groups]

    retained_groups = int(keep.sum().item())
    hidden_states = hidden_states.reshape(retained_groups, merge_area, -1)
    hidden_states = hidden_states[compact_window_index].reshape(retained_groups * merge_area, -1)
    rotary_positions = selected_rotary.reshape(retained_groups, merge_area, -1)
    rotary_positions = rotary_positions[compact_window_index].reshape(
        retained_groups * merge_area, -1
    )
    rotary_pair = torch.cat((rotary_positions, rotary_positions), dim=-1)
    position_embeddings = (rotary_pair.cos(), rotary_pair.sin())

    full_window_cumulative = tuple(int(value) for value in full_window_cumulative)
    window_fine_counts = []
    for start, end in zip(full_window_cumulative[:-1], full_window_cumulative[1:]):
        if start % merge_area or end % merge_area:
            raise ValueError("Qwen2.5-VL window boundary splits a visual-token group")
        original_groups = full_window_index[start // merge_area : end // merge_area]
        window_fine_counts.append(int(keep[original_groups].sum().item()) * merge_area)
    window_cumulative = _cumulative_lengths(
        tuple(window_fine_counts), device=hidden_states.device
    )
    window_cumulative = torch.unique_consecutive(window_cumulative)
    page_cumulative = _cumulative_lengths(page_fine_counts, device=hidden_states.device)

    full_attention_blocks = {int(value) for value in visual_model.fullatt_block_indexes}
    for layer_index, block in enumerate(visual_model.blocks):
        hidden_states = block(
            hidden_states,
            cu_seqlens=(
                page_cumulative if layer_index in full_attention_blocks else window_cumulative
            ),
            position_embeddings=position_embeddings,
        )
    image_embeds = visual_model.merger(hidden_states)
    reverse_indices = torch.argsort(compact_window_index)
    return image_embeds[reverse_indices], selected_rotary


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
    page_fine_counts = tuple(count * merge_area for count in kept_groups_per_page)

    if all(
        hasattr(visual_model, attribute)
        for attribute in ("get_window_index", "fullatt_block_indexes", "spatial_merge_unit")
    ):
        spatial_merge_unit = int(visual_model.spatial_merge_unit)
        if spatial_merge_unit != merge_area:
            raise ValueError("Qwen2.5-VL spatial merge unit differs from its merge area")
        image_embeds, rotary_positions = _compact_qwen25_vision(
            visual_model,
            selected_pixels,
            grid,
            keep,
            fine_keep,
            page_fine_counts,
            merge_area,
        )
        return CompactVisionBatch(image_embeds, rotary_positions, page_fine_counts)

    hidden_states = visual_model.patch_embed(selected_pixels)
    full_positions = visual_model.rot_pos_emb(grid.to(pixel_values.device))
    rotary_positions = full_positions[fine_keep]
    cumulative = _cumulative_lengths(page_fine_counts, device=hidden_states.device)
    for block in visual_model.blocks:
        hidden_states = block(
            hidden_states,
            cu_seqlens=cumulative,
            rotary_pos_emb=rotary_positions,
        )
    image_embeds = visual_model.merger(hidden_states)
    return CompactVisionBatch(image_embeds, rotary_positions, page_fine_counts)
