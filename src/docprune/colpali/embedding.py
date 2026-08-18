"""ColPali page encoding with sparse visual placeholders."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from docprune.colpali.vision import sparse_siglip_features
from docprune.processor_probe import ColPaliVisualMapping


@dataclass(frozen=True)
class ColPaliPageEmbedding:
    embeddings: torch.Tensor
    visual_embeddings: torch.Tensor
    raster_indices: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


def _batch_tensor(batch: Mapping[str, object], name: str) -> torch.Tensor:
    if name not in batch:
        raise ValueError(f"ColPali batch is missing {name}")
    return torch.as_tensor(batch[name])


def _model_pixel_values(model: object, pixel_values: torch.Tensor) -> torch.Tensor:
    """Match processor pixels to the model's vision parameter device and dtype."""

    vision_tower = getattr(getattr(model, "model", model), "vision_tower", None)
    parameters = getattr(vision_tower, "parameters", None)
    if parameters is None:
        parameters = getattr(model, "parameters", None)
    if parameters is None:
        return pixel_values
    try:
        parameter = next(parameters())
    except StopIteration:
        return pixel_values
    if not parameter.dtype.is_floating_point:
        raise ValueError("ColPali parameters must use a floating dtype")
    return pixel_values.to(device=parameter.device, dtype=parameter.dtype)


def encode_colpali_page(
    model: object,
    batch: Mapping[str, object],
    mapping: ColPaliVisualMapping,
    patch_keep_mask: torch.Tensor,
) -> ColPaliPageEmbedding:
    """Encode one page while removing only rejected PaliGemma image placeholders."""

    input_ids = _batch_tensor(batch, "input_ids")
    attention_mask = _batch_tensor(batch, "attention_mask")
    pixel_values = _batch_tensor(batch, "pixel_values")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("ColPali page encoding requires input_ids with batch size one")
    if attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must match input_ids")
    visual_count = mapping.visual_stop - mapping.visual_start
    if mapping.raster_indices != tuple(range(visual_count)):
        raise ValueError("mapping must preserve unique row-major raster indices")
    visual_ids = input_ids[0, mapping.visual_start : mapping.visual_stop]
    if visual_ids.numel() != visual_count or not bool((visual_ids == mapping.image_token_id).all()):
        raise ValueError("mapping does not match the page image placeholders")

    keep = torch.as_tensor(patch_keep_mask, dtype=torch.bool, device=input_ids.device)
    if keep.ndim != 1 or keep.numel() != visual_count:
        raise ValueError(f"patch_keep_mask must contain {visual_count} values")
    if not bool(keep.any()):
        raise ValueError("patch_keep_mask must retain at least one visual token")

    raster_indices = torch.arange(visual_count, device=input_ids.device, dtype=torch.long)[keep]
    model_pixel_values = _model_pixel_values(model, pixel_values)
    if bool(keep.all()):
        model_batch = dict(batch)
        model_batch["pixel_values"] = model_pixel_values
        embeddings = model(**model_batch)
        visual_embeddings = embeddings[:, mapping.visual_start : mapping.visual_stop]
        return ColPaliPageEmbedding(
            embeddings=embeddings,
            visual_embeddings=visual_embeddings,
            raster_indices=raster_indices,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    sequence_keep = torch.ones(input_ids.shape[1], dtype=torch.bool, device=input_ids.device)
    sequence_keep[mapping.visual_start : mapping.visual_stop] = keep
    compact_ids = input_ids[:, sequence_keep]
    compact_attention = attention_mask[:, sequence_keep]
    compact_visual_start = mapping.visual_start
    compact_visual_stop = compact_visual_start + int(keep.sum().item())

    vision_features = sparse_siglip_features(
        model.model.vision_tower,
        model_pixel_values,
        keep,
    )
    image_features = model.model.multi_modal_projector(vision_features)
    image_features = image_features / (model.model.config.hidden_size**0.5)
    inputs_embeds = model.model.get_input_embeddings()(compact_ids)
    inputs_embeds = inputs_embeds.clone()
    inputs_embeds[:, compact_visual_start:compact_visual_stop] = image_features.to(
        device=inputs_embeds.device,
        dtype=inputs_embeds.dtype,
    )

    outputs = model.model(
        inputs_embeds=inputs_embeds,
        attention_mask=compact_attention,
        output_hidden_states=True,
        return_dict=True,
    )
    projected = model.custom_text_proj(outputs.hidden_states[-1])
    projected = projected / projected.norm(dim=-1, keepdim=True)
    projected = projected * compact_attention.unsqueeze(-1)
    return ColPaliPageEmbedding(
        embeddings=projected,
        visual_embeddings=projected[:, compact_visual_start:compact_visual_stop],
        raster_indices=raster_indices,
        input_ids=compact_ids,
        attention_mask=compact_attention,
    )
