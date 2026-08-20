"""Decoder execution with one-time comprehension-aware visual-token pruning."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from docprune.ctp import (
    ComprehensionController,
    CTPDecision,
    ctp_keep_indices,
    visual_attention_scores,
)


@dataclass(frozen=True)
class PrefillResult:
    hidden_states: torch.Tensor
    cache: object
    position_ids: torch.Tensor
    keep_indices: torch.Tensor
    decision: CTPDecision | None


def _causal_mask(sequence_length: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    minimum = torch.finfo(dtype).min
    future = torch.triu(
        torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=device), diagonal=1
    )
    mask = torch.zeros(sequence_length, sequence_length, dtype=dtype, device=device)
    return mask.masked_fill(future, minimum).view(1, 1, sequence_length, sequence_length)


def _prefill_attention_mask(
    decoder_model: object,
    sequence_length: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor | None:
    """Use FlashAttention's causal varlen path for an unpadded batch-one prefill."""

    implementation = getattr(decoder_model.config, "_attn_implementation", "eager")
    if implementation == "flash_attention_2":
        return None
    return _causal_mask(sequence_length, dtype=dtype, device=device)


def _last_query_attention(
    decoder_layer: object,
    layer_input: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    causal_mask: torch.Tensor | None,
) -> torch.Tensor:
    from transformers.models.qwen2_vl.modeling_qwen2_vl import (
        apply_multimodal_rotary_pos_emb,
        repeat_kv,
    )

    attention = decoder_layer.self_attn
    hidden = decoder_layer.input_layernorm(layer_input)
    batch, sequence_length, _ = hidden.shape
    # CTP only needs the final prompt query.  Projecting the whole sequence
    # here needlessly allocates an [batch, heads, sequence, head_dim] tensor.
    query = attention.q_proj(hidden[:, -1:, :])
    key = attention.k_proj(hidden)
    query = query.view(batch, 1, attention.num_heads, attention.head_dim).transpose(1, 2)
    key = key.view(batch, sequence_length, attention.num_key_value_heads, attention.head_dim).transpose(1, 2)
    cosine, sine = position_embeddings
    query, _ = apply_multimodal_rotary_pos_emb(
        query,
        query,
        cosine[..., -1:, :],
        sine[..., -1:, :],
        attention.rope_scaling["mrope_section"],
    )
    _, key = apply_multimodal_rotary_pos_emb(
        key,
        key,
        cosine,
        sine,
        attention.rope_scaling["mrope_section"],
    )
    key = repeat_kv(key, attention.num_key_value_groups)
    scores = torch.matmul(query[:, :, -1:, :], key.transpose(2, 3)) / math.sqrt(attention.head_dim)
    if causal_mask is not None:
        scores = scores + causal_mask[:, :, -1:, :]
    return torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)


def prefill_with_ctp(
    decoder_model: object,
    inputs_embeds: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    visual_indices: torch.Tensor,
    comprehension_threshold: float,
    attention_threshold: float,
    head_aggregation: str = "mean",
) -> PrefillResult:
    """Prefill heterogeneous per-layer caches and prune only after the selected layer."""

    from transformers.cache_utils import DynamicCache

    if inputs_embeds.ndim != 3 or inputs_embeds.shape[0] != 1:
        raise ValueError("decoder prefill requires embeddings with batch size one")
    if position_ids.shape != (3, 1, inputs_embeds.shape[1]):
        raise ValueError("position_ids must have shape [3, 1, sequence]")
    current_visual = torch.as_tensor(visual_indices, dtype=torch.long, device=inputs_embeds.device)
    if current_visual.numel() and (
        current_visual.min() < 0 or current_visual.max() >= inputs_embeds.shape[1]
    ):
        raise ValueError("visual_indices must reference the prefill sequence")

    cache = DynamicCache()
    controller = ComprehensionController(comprehension_threshold)
    hidden = inputs_embeds
    positions = position_ids
    original_keep = torch.arange(hidden.shape[1], device=hidden.device)
    decision: CTPDecision | None = None

    for layer_index, decoder_layer in enumerate(decoder_model.layers):
        layer_input = hidden
        position_embeddings = decoder_model.rotary_emb(layer_input, positions)
        layer_attention_mask = _prefill_attention_mask(
            decoder_model, hidden.shape[1], hidden.dtype, hidden.device
        )
        causal_mask = (
            None
            if layer_attention_mask is None
            else _causal_mask(hidden.shape[1], dtype=hidden.dtype, device=hidden.device)
        )
        layer_outputs = decoder_layer(
            hidden,
            attention_mask=layer_attention_mask,
            position_ids=positions,
            past_key_value=cache,
            output_attentions=False,
            use_cache=True,
            cache_position=torch.arange(hidden.shape[1], device=hidden.device),
            position_embeddings=position_embeddings,
        )
        hidden = layer_outputs[0]
        if not controller.observe(layer_index, hidden[:, -1, :]):
            continue

        attention = _last_query_attention(decoder_layer, layer_input, position_embeddings, causal_mask)
        visual_scores = visual_attention_scores(
            attention,
            current_visual,
            head_aggregation=head_aggregation,
        )
        local_keep = ctp_keep_indices(
            token_count=hidden.shape[1],
            visual_indices=current_visual,
            visual_scores=visual_scores,
            threshold=attention_threshold,
        ).to(hidden.device)
        visual_boolean = torch.zeros(hidden.shape[1], dtype=torch.bool, device=hidden.device)
        visual_boolean[current_visual] = True
        retained_visual_original = original_keep[local_keep[visual_boolean[local_keep]]]
        decision = CTPDecision(
            layer_index=layer_index,
            comprehension_norm=float(controller.selected_norm),
            original_visual_tokens=current_visual.numel(),
            retained_visual_indices=tuple(int(value) for value in retained_visual_original.tolist()),
        )
        hidden = hidden[:, local_keep]
        positions = positions[:, :, local_keep]
        original_keep = original_keep[local_keep]
        current_visual = visual_boolean[local_keep].nonzero(as_tuple=False).flatten()

    hidden = decoder_model.norm(hidden)
    return PrefillResult(hidden, cache, positions, original_keep, decision)


def decode_one_token(
    decoder_model: object,
    token_embedding: torch.Tensor,
    position_ids: torch.Tensor,
    cache: object,
) -> torch.Tensor:
    """Decode one token using each layer's independently sized dynamic cache."""

    hidden = token_embedding
    for layer_index, decoder_layer in enumerate(decoder_model.layers):
        position_embeddings = decoder_model.rotary_emb(hidden, position_ids)
        cached_length = cache.key_cache[layer_index].shape[-2]
        outputs = decoder_layer(
            hidden,
            attention_mask=None,
            position_ids=position_ids,
            past_key_value=cache,
            output_attentions=False,
            use_cache=True,
            cache_position=torch.tensor([cached_length], device=hidden.device),
            position_embeddings=position_embeddings,
        )
        hidden = outputs[0]
    return decoder_model.norm(hidden)
