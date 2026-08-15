"""Comprehension-aware Token Pruning from DocPrune equations 8 and 9."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CTPDecision:
    layer_index: int
    comprehension_norm: float
    original_visual_tokens: int
    retained_visual_indices: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "layer_index": self.layer_index,
            "comprehension_norm": self.comprehension_norm,
            "original_visual_tokens": self.original_visual_tokens,
            "retained_visual_tokens": len(self.retained_visual_indices),
            "retained_visual_indices": list(self.retained_visual_indices),
        }


class ComprehensionController:
    """Select exactly the first layer crossing the paper's L2 threshold."""

    def __init__(self, threshold: float) -> None:
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError("threshold must be finite and nonnegative")
        self.threshold = threshold
        self.selected_layer: int | None = None
        self.selected_norm: float | None = None

    def observe(self, layer_index: int, last_hidden_state: torch.Tensor) -> bool:
        hidden = torch.as_tensor(last_hidden_state)
        if hidden.ndim != 2 or hidden.shape[0] != 1:
            raise ValueError("comprehension observation requires batch size one and shape [1, dim]")
        if self.selected_layer is not None:
            return False
        norm = float(torch.linalg.vector_norm(hidden[0]).item())
        if norm < self.threshold:
            return False
        self.selected_layer = layer_index
        self.selected_norm = norm
        return True


def visual_attention_scores(
    attention: torch.Tensor,
    visual_indices: torch.Tensor,
    *,
    head_aggregation: str = "mean",
) -> torch.Tensor:
    """Aggregate last-query attention to the current visual tokens."""

    weights = torch.as_tensor(attention)
    indices = torch.as_tensor(visual_indices, dtype=torch.long, device=weights.device)
    if weights.ndim != 4 or weights.shape[0] != 1:
        raise ValueError("attention must have shape [1, heads, queries, keys]")
    if indices.ndim != 1:
        raise ValueError("visual_indices must be one-dimensional")
    if indices.numel() and (indices.min() < 0 or indices.max() >= weights.shape[-1]):
        raise ValueError("visual_indices are outside the attention key range")
    selected = weights[0, :, -1, indices]
    if head_aggregation == "mean":
        return selected.mean(dim=0)
    if head_aggregation == "max":
        return selected.max(dim=0).values
    raise ValueError("head_aggregation must be mean or max")


def ctp_keep_indices(
    *,
    token_count: int,
    visual_indices: torch.Tensor,
    visual_scores: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Retain all nonvisual tokens and visual tokens satisfying paper equation 9."""

    if token_count <= 0:
        raise ValueError("token_count must be positive")
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    indices = torch.as_tensor(visual_indices, dtype=torch.long)
    scores = torch.as_tensor(visual_scores)
    if indices.ndim != 1 or scores.ndim != 1 or indices.numel() != scores.numel():
        raise ValueError("visual_indices and visual_scores must be equal-length vectors")
    if torch.unique(indices).numel() != indices.numel():
        raise ValueError("visual_indices must be unique")
    if indices.numel() and (indices.min() < 0 or indices.max() >= token_count):
        raise ValueError("visual_indices must be in range")
    indices = indices.to(scores.device)
    keep = torch.ones(token_count, dtype=torch.bool, device=scores.device)
    keep[indices] = False
    keep[indices[scores >= threshold]] = True
    return keep.nonzero(as_tuple=False).flatten()
