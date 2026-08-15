"""Compact image placeholders while retaining original M-ROPE positions."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CompactSequence:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    visual_indices: torch.Tensor
    original_keep_indices: torch.Tensor


def compact_multimodal_sequence(
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    image_token_id: int,
    group_keep_mask: torch.Tensor,
) -> CompactSequence:
    """Remove rejected image placeholders and select their original positions."""

    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("DocPrune Qwen2-VL integration requires batch size one")
    if attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must match input_ids")
    if position_ids.shape != (3, 1, input_ids.shape[1]):
        raise ValueError("position_ids must have shape [3, 1, sequence]")
    group_keep = torch.as_tensor(group_keep_mask, dtype=torch.bool, device=input_ids.device)
    placeholder_indices = (input_ids[0] == image_token_id).nonzero(as_tuple=False).flatten()
    if placeholder_indices.numel() != group_keep.numel():
        raise ValueError(
            f"image placeholders ({placeholder_indices.numel()}) do not match visual groups "
            f"({group_keep.numel()})"
        )
    sequence_keep = torch.ones(input_ids.shape[1], dtype=torch.bool, device=input_ids.device)
    sequence_keep[placeholder_indices] = group_keep
    original_keep_indices = sequence_keep.nonzero(as_tuple=False).flatten()
    compact_ids = input_ids[:, original_keep_indices]
    compact_attention = attention_mask[:, original_keep_indices]
    compact_positions = position_ids[:, :, original_keep_indices]
    visual_indices = (compact_ids[0] == image_token_id).nonzero(as_tuple=False).flatten()
    return CompactSequence(
        compact_ids,
        compact_attention,
        compact_positions,
        visual_indices,
        original_keep_indices,
    )
