"""Deterministic token ownership from semantic memberships and local fallbacks."""

import torch

from .contracts import RegionLayout


def partition_tokens(
    page, coordinates, semantic_memberships, *, tile_size=8, max_fallback=128
):
    """Earlier supplied regions win overlaps; uncovered tokens get page-local tiles.

    The semantic priority order and coordinate units must be frozen by Stage 1.
    No layout/segmentation model is downloaded or silently substituted here.
    """
    if tile_size <= 0 or max_fallback <= 0:
        raise ValueError("Fallback limits must be positive")
    n = page.numel()
    owner = torch.full((n,), -1, dtype=torch.long, device=page.device)
    ids = []
    for name, positions in semantic_memberships:
        positions = torch.as_tensor(positions, dtype=torch.long, device=page.device)
        if positions.ndim != 1 or (positions < 0).any() or (positions >= n).any():
            raise ValueError("Invalid semantic membership")
        positions = positions.unique(sorted=True)
        positions = positions[owner[positions] < 0]
        if positions.numel():
            if torch.unique(page[positions]).numel() != 1:
                raise ValueError("Semantic region spans pages")
            owner[positions] = len(ids)
            ids.append(str(name))
    buckets = {}
    for idx in torch.nonzero(owner < 0, as_tuple=True)[0].tolist():
        key = (
            int(page[idx]),
            *torch.floor(coordinates[idx] / tile_size).long().tolist(),
        )
        buckets.setdefault(key, []).append(idx)
    for key, positions in sorted(buckets.items()):
        for offset in range(0, len(positions), max_fallback):
            owner[positions[offset : offset + max_fallback]] = len(ids)
            ids.append(f"fallback:{key}:{offset}")
    # Geometry is a default schema, replaceable by explicitly versioned metadata.
    metadata = []
    for i in range(len(ids)):
        ix = owner == i
        pts = coordinates[ix].float()
        metadata.append(
            torch.cat(
                (
                    pts.min(0).values,
                    pts.max(0).values,
                    page[ix][:1].float(),
                    ix.sum().reshape(1).float(),
                )
            )
        )
    return RegionLayout(
        tuple(ids), owner, page, coordinates, torch.stack(metadata)
    ).validate()
