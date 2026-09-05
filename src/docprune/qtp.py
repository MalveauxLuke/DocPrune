"""Question-aware Token Pruning from DocPrune equations 5 through 7."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as functional

from docprune.layout import VisualLayout


def cosine_sum_relevance(
    document_tokens: torch.Tensor,
    question_tokens: torch.Tensor,
) -> torch.Tensor:
    """Sum cosine similarity from every document token to every question token."""

    documents = torch.as_tensor(document_tokens)
    questions = torch.as_tensor(question_tokens, device=documents.device, dtype=documents.dtype)
    if documents.ndim != 3 or questions.ndim != 3:
        raise ValueError("document_tokens and question_tokens must have shape [batch, tokens, dim]")
    if documents.shape[-1] != questions.shape[-1]:
        raise ValueError("document and question embedding dimensions must match")
    if questions.shape[0] == 1 and documents.shape[0] > 1:
        questions = questions.expand(documents.shape[0], -1, -1)
    if documents.shape[0] != questions.shape[0]:
        raise ValueError("question batch must be one or match the document batch")
    normalized_documents = functional.normalize(documents, p=2, dim=-1)
    normalized_questions = functional.normalize(questions, p=2, dim=-1)
    pairwise = torch.matmul(normalized_documents, normalized_questions.transpose(-1, -2))
    return pairwise.sum(dim=-1)


def dense_relevance_from_sparse_tokens(
    tokens: torch.Tensor,
    raster_indices: torch.Tensor,
    source_hw: tuple[int, int],
    question_tokens: torch.Tensor,
) -> torch.Tensor:
    """Scatter sparse ColPali visual rows back to their original raster grid.

    ``encode_colpali_page`` removes BTP-rejected image placeholders, so the
    returned visual rows are no longer a dense 32-by-32 raster.  QTP must score
    those rows at their original positions; compacting them into the upper-left
    corner changes the spatial evidence.  Missing cells are represented by
    ``-inf`` and are deliberately carried as an explicit mask by the QTP
    pipeline rather than being treated as zero-valued evidence.
    """

    sparse = torch.as_tensor(tokens)
    indices = torch.as_tensor(raster_indices, dtype=torch.long, device=sparse.device)
    if sparse.ndim == 2:
        sparse = sparse.unsqueeze(0)
    if sparse.ndim != 3:
        raise ValueError("tokens must have shape [tokens, dim] or [batch, tokens, dim]")
    if indices.ndim == 1:
        indices = indices.unsqueeze(0).expand(sparse.shape[0], -1)
    if indices.ndim != 2 or indices.shape != sparse.shape[:2]:
        raise ValueError("raster_indices must have one index for every sparse token")
    if len(source_hw) != 2 or min(source_hw) <= 0:
        raise ValueError("source_hw must contain positive height and width")
    raster_count = int(source_hw[0]) * int(source_hw[1])
    if bool(((indices < 0) | (indices >= raster_count)).any()):
        raise ValueError("raster_indices must lie within source_hw")
    if any(torch.unique(row).numel() != row.numel() for row in indices):
        raise ValueError("raster_indices must be unique within each batch row")

    question = torch.as_tensor(question_tokens, device=sparse.device, dtype=sparse.dtype)
    if question.ndim == 2:
        question = question.unsqueeze(0)
    relevance = cosine_sum_relevance(sparse, question)
    dense = torch.full(
        (sparse.shape[0], raster_count),
        -torch.inf,
        dtype=relevance.dtype,
        device=relevance.device,
    )
    dense.scatter_(1, indices.to(relevance.device), relevance)
    return dense.view(sparse.shape[0], *source_hw)


def resize_relevance_map(scores: torch.Tensor, *, target_hw: tuple[int, int]) -> torch.Tensor:
    """Bilinearly resize a batch of retrieval relevance maps."""

    maps = torch.as_tensor(scores)
    if maps.ndim != 3:
        raise ValueError("scores must have shape [batch, height, width]")
    if len(target_hw) != 2 or min(target_hw) <= 0:
        raise ValueError("target_hw must contain positive height and width")
    return functional.interpolate(
        maps.unsqueeze(1),
        size=target_hw,
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)


def gaussian_smooth_2d(
    scores: torch.Tensor,
    *,
    sigma: float,
    truncate: float = 3.0,
) -> torch.Tensor:
    """Apply normalized Gaussian smoothing with replicated border padding."""

    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    if not math.isfinite(truncate) or truncate <= 0:
        raise ValueError("truncate must be finite and positive")
    maps = torch.as_tensor(scores)
    if maps.ndim != 3 or not maps.is_floating_point():
        raise ValueError("scores must be a floating tensor with shape [batch, height, width]")
    radius = max(1, math.ceil(truncate * sigma))
    coordinates = torch.arange(-radius, radius + 1, dtype=maps.dtype, device=maps.device)
    one_dimensional = torch.exp(-(coordinates**2) / (2 * sigma**2))
    one_dimensional = one_dimensional / one_dimensional.sum()
    kernel = torch.outer(one_dimensional, one_dimensional)
    padded = functional.pad(maps.unsqueeze(1), (radius,) * 4, mode="replicate")
    return functional.conv2d(padded, kernel.view(1, 1, *kernel.shape)).squeeze(1)


def threshold_relevance_keep_mask(scores: torch.Tensor, threshold: float) -> torch.Tensor:
    """Apply paper equation 7 with an inclusive relevance boundary."""

    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    return torch.as_tensor(scores) >= threshold


def group_relevance_keep_mask(
    fine_scores: torch.Tensor,
    *,
    threshold: float,
    layout: VisualLayout,
    retention: str = "any",
) -> torch.Tensor:
    """Convert fine-token relevance into complete spatial-merge group decisions."""

    scores = layout.flatten_page_values(fine_scores)
    indices = layout.group_fine_indices().to(scores.device)
    grouped_scores = scores[indices]
    if retention == "mean":
        return grouped_scores.mean(dim=-1) >= threshold
    fine_keep = threshold_relevance_keep_mask(grouped_scores, threshold)
    if retention == "any":
        return fine_keep.any(dim=-1)
    if retention == "all":
        return fine_keep.all(dim=-1)
    raise ValueError("retention must be any, all, or mean")


def question_keep_mask(
    document_tokens: torch.Tensor,
    question_tokens: torch.Tensor,
    *,
    source_hw: tuple[int, int],
    target_hw: tuple[int, int],
    layout: VisualLayout,
    threshold: float,
    sigma: float,
    retention: str = "any",
    raster_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Execute DocPrune QTP cosine, resize, smoothing, and threshold stages."""

    expected_source_tokens = source_hw[0] * source_hw[1]
    if raster_indices is None:
        relevance = cosine_sum_relevance(document_tokens, question_tokens)
        if relevance.shape[1] != expected_source_tokens:
            raise ValueError(
                f"source_hw describes {expected_source_tokens} tokens, got {relevance.shape[1]}"
            )
        maps = relevance.view(relevance.shape[0], *source_hw)
        resized = resize_relevance_map(maps, target_hw=target_hw)
        smoothed = gaussian_smooth_2d(resized, sigma=sigma)
        return group_relevance_keep_mask(
            smoothed.flatten(1),
            threshold=threshold,
            layout=layout,
            retention=retention,
        )

    maps = dense_relevance_from_sparse_tokens(
        document_tokens,
        raster_indices,
        source_hw,
        question_tokens,
    )
    valid = torch.isfinite(maps)
    # Interpolation and convolution do not define useful semantics for -inf.
    # Score only finite cells, while nearest-neighbor validity keeps a rejected
    # raster cell from becoming evidence through a neighboring cell.
    finite_maps = torch.where(valid, maps, torch.zeros_like(maps))
    resized = resize_relevance_map(finite_maps, target_hw=target_hw)
    resized_valid = functional.interpolate(
        valid.to(resized.dtype).unsqueeze(1),
        size=target_hw,
        mode="nearest",
    ).squeeze(1).bool()
    smoothed = gaussian_smooth_2d(resized, sigma=sigma)
    keep = group_relevance_keep_mask(
        smoothed.flatten(1),
        threshold=threshold,
        layout=layout,
        retention=retention,
    )
    target_fine_valid = resized_valid.flatten(1)
    group_indices = layout.group_fine_indices().to(target_fine_valid.device)
    group_valid = target_fine_valid.reshape(-1)[group_indices].all(dim=-1)
    if group_valid.numel() != keep.numel():
        raise ValueError("masked QTP batch and visual layout page counts do not match")
    return keep & group_valid.flatten()
