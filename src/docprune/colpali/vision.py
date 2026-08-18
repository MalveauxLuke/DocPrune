"""Sparse SigLIP execution that preserves original raster positions."""

from __future__ import annotations

import torch


def sparse_siglip_features(
    vision_tower: object,
    pixel_values: torch.Tensor,
    patch_keep_mask: torch.Tensor,
) -> torch.Tensor:
    """Encode selected patches after adding their original positional embeddings."""

    pixels = torch.as_tensor(pixel_values)
    if pixels.ndim != 4:
        raise ValueError("pixel_values must have shape [batch, channels, height, width]")
    vision_model = getattr(vision_tower, "vision_model", None)
    if vision_model is None:
        raise ValueError("vision_tower must expose vision_model")

    embeddings = vision_model.embeddings(pixels)
    batch_size, patch_count, hidden_size = embeddings.shape
    keep = torch.as_tensor(patch_keep_mask, dtype=torch.bool, device=embeddings.device)
    if keep.ndim == 1:
        keep = keep.unsqueeze(0).expand(batch_size, -1)
    if keep.shape != (batch_size, patch_count):
        raise ValueError(f"patch_keep_mask must have shape [{batch_size}, {patch_count}]")
    kept_counts = keep.sum(dim=1)
    if bool((kept_counts == 0).any()):
        raise ValueError("patch_keep_mask must retain at least one patch per image")
    if not bool((kept_counts == kept_counts[0]).all()):
        raise ValueError("patch_keep_mask must retain the same count for every image")

    if bool(keep.all()):
        return vision_tower(pixels).last_hidden_state

    compact = embeddings[keep].reshape(batch_size, int(kept_counts[0].item()), hidden_size)
    encoded = vision_model.encoder(inputs_embeds=compact, return_dict=True).last_hidden_state
    return vision_model.post_layernorm(encoded)
