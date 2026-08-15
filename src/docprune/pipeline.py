"""Compose BTP and QTP inputs while keeping ambiguous processor details explicit."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from docprune.btp import background_keep_mask, background_scores
from docprune.config import PagePruningConfig, ReconstructionDefaults
from docprune.layout import VisualLayout
from docprune.qtp import question_keep_mask
from docprune.qwen2vl.model import VisionPruningMasks


def prepare_qa_pruning_masks(
    *,
    resized_images: Sequence[torch.Tensor],
    image_grid_thw: torch.Tensor,
    document_tokens: Sequence[torch.Tensor],
    document_source_hw: Sequence[tuple[int, int]],
    question_tokens: torch.Tensor,
    patch_size: int,
    page_config: PagePruningConfig,
    reconstruction: ReconstructionDefaults,
) -> VisionPruningMasks:
    """Prepare merge-safe per-page masks from explicit visual-only ColPali tokens.

    The caller supplies images after the same resize used by Qwen2-VL and removes
    ColPali special/non-image tokens explicitly. Those processor-specific choices
    are intentionally not guessed here because the paper does not define them.
    """

    grid = torch.as_tensor(image_grid_thw, dtype=torch.long)
    page_count = grid.shape[0] if grid.ndim == 2 else 0
    if grid.ndim != 2 or grid.shape[1] != 3:
        raise ValueError("image_grid_thw must have shape [pages, 3]")
    if not (
        len(resized_images)
        == len(document_tokens)
        == len(document_source_hw)
        == page_count
    ):
        raise ValueError("images, document tokens, source grids, and Qwen grids must align by page")
    if torch.as_tensor(question_tokens).ndim != 2:
        raise ValueError("question_tokens must have shape [tokens, dim]")

    background_masks: list[torch.Tensor] = []
    question_masks: list[torch.Tensor] = []
    for page_index in range(page_count):
        temporal, target_height, target_width = (int(value) for value in grid[page_index])
        if temporal != 1:
            raise ValueError("document pages require temporal grid size one")
        image = torch.as_tensor(resized_images[page_index])
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError("each resized image must have shape [1, channels, height, width]")
        if image.shape[-2:] != (target_height * patch_size, target_width * patch_size):
            raise ValueError("resized image dimensions must match its Qwen patch grid")
        source_height, source_width = document_source_hw[page_index]
        document = torch.as_tensor(document_tokens[page_index])
        if document.ndim != 2 or document.shape[0] != source_height * source_width:
            raise ValueError(
                "document_tokens must be visual-only; explicitly slice ColPali special tokens "
                "to match document_source_hw"
            )
        page_layout = VisualLayout(grid[page_index : page_index + 1], spatial_merge_size=2)
        scores = background_scores(
            image,
            patch_size=patch_size,
            error_tolerance=page_config.background_error_tolerance,
        )
        background_masks.append(
            background_keep_mask(
                scores,
                threshold=page_config.qa_background_threshold,
                layout=page_layout,
                retention=reconstruction.group_retention,
            )
        )
        question_masks.append(
            question_keep_mask(
                document.unsqueeze(0),
                torch.as_tensor(question_tokens).unsqueeze(0),
                source_hw=(source_height, source_width),
                target_hw=(target_height, target_width),
                layout=page_layout,
                threshold=page_config.question_threshold,
                sigma=reconstruction.gaussian_sigma,
                retention=reconstruction.group_retention,
            )
        )
    return VisionPruningMasks(
        background_keep=torch.cat(background_masks),
        question_keep=torch.cat(question_masks),
    )
