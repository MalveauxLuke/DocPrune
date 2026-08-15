"""Background Token Pruning from DocPrune equations 3 and 4."""

from __future__ import annotations

import torch

from docprune.layout import VisualLayout


def rgb_to_bt601_grayscale(images: torch.Tensor) -> torch.Tensor:
    """Convert BCHW uint8 RGB images using the documented reconstruction default."""

    images = torch.as_tensor(images)
    if images.ndim != 4 or images.shape[1] not in {1, 3}:
        raise ValueError("images must have shape [batch, 1 or 3, height, width]")
    if images.dtype != torch.uint8:
        raise ValueError("images must use uint8 pixel intensities")
    if images.shape[1] == 1:
        return images[:, 0]
    weights = torch.tensor([0.299, 0.587, 0.114], device=images.device)
    return (images.to(torch.float32) * weights.view(1, 3, 1, 1)).sum(dim=1).round().to(torch.uint8)


def background_scores(
    images: torch.Tensor,
    *,
    patch_size: int,
    error_tolerance: float,
) -> torch.Tensor:
    """Return the near-modal-intensity pixel ratio for every non-overlapping patch."""

    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if error_tolerance < 0:
        raise ValueError("error_tolerance must be nonnegative")
    grayscale = rgb_to_bt601_grayscale(images)
    _, height, width = grayscale.shape
    if height % patch_size or width % patch_size:
        raise ValueError("image height and width must be divisible by patch_size")
    modes = torch.mode(grayscale.flatten(1), dim=1).values.to(torch.float32)
    patches = (
        grayscale.unfold(1, patch_size, patch_size)
        .unfold(2, patch_size, patch_size)
        .contiguous()
        .view(grayscale.shape[0], -1, patch_size * patch_size)
        .to(torch.float32)
    )
    close_to_background = (patches - modes[:, None, None]).abs() < error_tolerance
    return close_to_background.to(torch.float32).mean(dim=-1)


def threshold_keep_mask(scores: torch.Tensor, threshold: float) -> torch.Tensor:
    """Apply paper equation 4: equality remains content; only greater ratios prune."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    return torch.as_tensor(scores) <= threshold


def background_keep_mask(
    fine_scores: torch.Tensor,
    *,
    threshold: float,
    layout: VisualLayout,
    retention: str = "any",
) -> torch.Tensor:
    """Convert fine-patch BTP scores into complete spatial-merge group decisions."""

    scores = layout.flatten_page_values(fine_scores)
    indices = layout.group_fine_indices().to(scores.device)
    grouped_scores = scores[indices]
    if retention == "mean":
        return grouped_scores.mean(dim=-1) <= threshold
    fine_keep = threshold_keep_mask(grouped_scores, threshold)
    if retention == "any":
        return fine_keep.any(dim=-1)
    if retention == "all":
        return fine_keep.all(dim=-1)
    raise ValueError("retention must be any, all, or mean")
