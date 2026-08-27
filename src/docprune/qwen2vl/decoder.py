"""Decoder execution with one-time comprehension-aware visual-token pruning."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace

import torch

from docprune.ctp import (
    ComprehensionController,
    CTPDecision,
    boundary_score_vectors_from_logits,
    ctp_keep_indices,
    visual_attention_scores,
)
from docprune.ctp_policy import (
    CTPPolicy,
    CTPSelectionRecord,
    PolicySelectionContext,
    no_crossing_selection,
    select_boundary_policy,
)


@dataclass(frozen=True)
class PrefillResult:
    hidden_states: torch.Tensor
    cache: object
    position_ids: torch.Tensor
    keep_indices: torch.Tensor
    decision: CTPDecision | None
    forced: ForcedInterventionRecord | None = None
    selection: CTPSelectionRecord | None = None


@dataclass(frozen=True)
class ForcedVisualIntervention:
    """A preselected visual-token intervention at an explicit decoder boundary."""

    boundary: str | int
    mode: str
    retained_visual_ids: tuple[int, ...]


@dataclass(frozen=True)
class ForcedInterventionRecord:
    """Small replay record for a forced intervention, separate from native CTP."""

    boundary: str
    mode: str
    selection_kind: str
    visual_population: int
    requested_budget: int
    achieved_budget: int
    retained_visual_ids: tuple[int, ...]
    logical_retained_sequence_ids: tuple[int, ...]
    prefill_cache_lengths: tuple[int, ...]
    retained_mrope_position_shape: tuple[int, int, int]
    retained_mrope_position_sha256: str

    def to_dict(self) -> dict[str, object]:
        """Return the small durable forced-intervention record without decoder tensors."""

        return {
            "boundary": self.boundary,
            "mode": self.mode,
            "selection_kind": self.selection_kind,
            "visual_population": self.visual_population,
            "requested_budget": self.requested_budget,
            "achieved_budget": self.achieved_budget,
            "retained_visual_ids": list(self.retained_visual_ids),
            "logical_retained_sequence_ids": list(self.logical_retained_sequence_ids),
            "prefill_cache_lengths": list(self.prefill_cache_lengths),
            "retained_mrope_position_shape": list(self.retained_mrope_position_shape),
            "retained_mrope_position_sha256": self.retained_mrope_position_sha256,
        }


@dataclass(frozen=True)
class _ForcedInterventionApplication:
    """Boundary-local state exposed only to focused CPU contract tests."""

    hidden_states: torch.Tensor
    position_ids: torch.Tensor
    original_keep_indices: torch.Tensor
    current_visual_indices: torch.Tensor
    record: ForcedInterventionRecord


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


def _last_query_attention_logits(
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
    key = key.view(
        batch, sequence_length, attention.num_key_value_heads, attention.head_dim
    ).transpose(1, 2)
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
    return scores


def _last_query_attention(
    decoder_layer: object,
    layer_input: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    causal_mask: torch.Tensor | None,
) -> torch.Tensor:
    scores = _last_query_attention_logits(
        decoder_layer, layer_input, position_embeddings, causal_mask
    )
    return torch.softmax(scores, dim=-1, dtype=torch.float32).to(scores.dtype)


def _forced_boundary_name(boundary: str | int, *, layer_count: int) -> str:
    if boundary == "input":
        return "B_input"
    if type(boundary) is not int or not 0 <= boundary < layer_count:
        raise ValueError("forced intervention boundary must be input or an in-range block index")
    return f"B_{boundary}"


def _validate_forced_intervention(
    intervention: ForcedVisualIntervention,
    *,
    layer_count: int,
    visual_population: int,
) -> tuple[str, tuple[int, ...]]:
    boundary = _forced_boundary_name(intervention.boundary, layer_count=layer_count)
    if intervention.mode not in {"physical_delete", "zero_mask"}:
        raise ValueError("forced intervention mode must be physical_delete or zero_mask")
    retained = tuple(intervention.retained_visual_ids)
    if any(type(value) is not int for value in retained):
        raise ValueError("forced retained visual IDs must be integers")
    if len(set(retained)) != len(retained):
        raise ValueError("forced retained visual IDs must be unique")
    if any(value < 0 or value >= visual_population for value in retained):
        raise ValueError("forced retained visual IDs are outside the visual population range")
    return boundary, tuple(sorted(retained))


def _retained_mrope_position_identity(
    position_ids: torch.Tensor,
) -> tuple[tuple[int, int, int], str]:
    """Return a dtype-independent digest of the actual post-boundary M-RoPE positions."""

    canonical = position_ids.detach().to(device="cpu", dtype=torch.int64).contiguous()
    return tuple(int(value) for value in canonical.shape), hashlib.sha256(
        canonical.numpy().tobytes()
    ).hexdigest()


def _apply_forced_visual_intervention(
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    original_keep_indices: torch.Tensor,
    current_visual_indices: torch.Tensor,
    intervention: ForcedVisualIntervention,
) -> _ForcedInterventionApplication:
    """Apply a validated forced operation to the boundary-local decoder state."""

    visual_population = current_visual_indices.numel()
    if intervention.mode not in {"physical_delete", "zero_mask"}:
        raise ValueError("forced intervention mode must be physical_delete or zero_mask")
    retained = tuple(intervention.retained_visual_ids)
    if any(type(value) is not int for value in retained):
        raise ValueError("forced retained visual IDs must be integers")
    if len(set(retained)) != len(retained):
        raise ValueError("forced retained visual IDs must be unique")
    if any(value < 0 or value >= visual_population for value in retained):
        raise ValueError("forced retained visual IDs are outside the visual population range")
    retained = tuple(sorted(retained))

    def record_for(
        applied_positions: torch.Tensor, applied_original_keep: torch.Tensor
    ) -> ForcedInterventionRecord:
        position_shape, position_sha256 = _retained_mrope_position_identity(applied_positions)
        return ForcedInterventionRecord(
            boundary="B_input"
            if intervention.boundary == "input"
            else f"B_{intervention.boundary}",
            mode=intervention.mode,
            selection_kind="forced",
            visual_population=visual_population,
            requested_budget=len(retained),
            achieved_budget=len(retained),
            retained_visual_ids=retained,
            logical_retained_sequence_ids=tuple(
                int(value) for value in applied_original_keep.tolist()
            ),
            prefill_cache_lengths=(),
            retained_mrope_position_shape=position_shape,
            retained_mrope_position_sha256=position_sha256,
        )

    if not visual_population or len(retained) == visual_population:
        return _ForcedInterventionApplication(
            hidden_states,
            position_ids,
            original_keep_indices,
            current_visual_indices,
            record_for(position_ids, original_keep_indices),
        )

    selected = torch.zeros(visual_population, dtype=torch.bool, device=hidden_states.device)
    selected[list(retained)] = True
    if intervention.mode == "zero_mask":
        zeroed = hidden_states.clone()
        zeroed[:, current_visual_indices[~selected]] = 0
        return _ForcedInterventionApplication(
            zeroed,
            position_ids,
            original_keep_indices,
            current_visual_indices,
            record_for(position_ids, original_keep_indices),
        )

    visual_boolean = torch.zeros(
        hidden_states.shape[1], dtype=torch.bool, device=hidden_states.device
    )
    visual_boolean[current_visual_indices] = True
    keep = ~visual_boolean
    keep[current_visual_indices[selected]] = True
    keep_indices = keep.nonzero(as_tuple=False).flatten()
    compact_original = original_keep_indices[keep_indices]
    return _ForcedInterventionApplication(
        hidden_states[:, keep_indices],
        position_ids[:, :, keep_indices],
        compact_original,
        visual_boolean[keep_indices].nonzero(as_tuple=False).flatten(),
        record_for(position_ids[:, :, keep_indices], compact_original),
    )


def prefill_with_ctp(
    decoder_model: object,
    inputs_embeds: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    visual_indices: torch.Tensor,
    comprehension_threshold: float,
    attention_threshold: float,
    head_aggregation: str = "mean",
    forced_intervention: ForcedVisualIntervention | None = None,
    ctp_policy: CTPPolicy | None = None,
    selection_context: PolicySelectionContext | None = None,
) -> PrefillResult:
    """Prefill heterogeneous per-layer caches and prune only after the selected layer."""

    from transformers.cache_utils import DynamicCache

    if inputs_embeds.ndim != 3 or inputs_embeds.shape[0] != 1:
        raise ValueError("decoder prefill requires embeddings with batch size one")
    if position_ids.shape != (3, 1, inputs_embeds.shape[1]):
        raise ValueError("position_ids must have shape [3, 1, sequence]")
    current_visual = torch.as_tensor(visual_indices, dtype=torch.long, device=inputs_embeds.device)
    if current_visual.ndim != 1:
        raise ValueError("visual_indices must be a rank-one vector")
    if current_visual.numel() and (
        current_visual.min() < 0 or current_visual.max() >= inputs_embeds.shape[1]
    ):
        raise ValueError("visual_indices must reference the prefill sequence")
    if current_visual.numel() > 1 and torch.any(current_visual[1:] <= current_visual[:-1]):
        raise ValueError("visual_indices must be unique and strictly increasing")

    forced_boundary: str | None = None
    forced_retained: tuple[int, ...] | None = None
    if forced_intervention is not None and ctp_policy is not None:
        raise ValueError("forced intervention and corrected CTP policy cannot be combined")
    if forced_intervention is not None:
        forced_boundary, forced_retained = _validate_forced_intervention(
            forced_intervention,
            layer_count=len(decoder_model.layers),
            visual_population=current_visual.numel(),
        )

    cache = DynamicCache()
    controller = (
        None
        if forced_intervention is not None
        or (ctp_policy is not None and ctp_policy.family == "no-ctp")
        else ComprehensionController(comprehension_threshold)
    )
    hidden = inputs_embeds
    positions = position_ids
    original_keep = torch.arange(hidden.shape[1], device=hidden.device)
    decision: CTPDecision | None = None
    forced: ForcedInterventionRecord | None = None
    selection: CTPSelectionRecord | None = None

    if forced_intervention is not None and forced_boundary == "B_input":
        applied = _apply_forced_visual_intervention(
            hidden,
            positions,
            original_keep,
            current_visual,
            ForcedVisualIntervention(
                boundary="input",
                mode=forced_intervention.mode,
                retained_visual_ids=forced_retained or (),
            ),
        )
        hidden = applied.hidden_states
        positions = applied.position_ids
        original_keep = applied.original_keep_indices
        current_visual = applied.current_visual_indices
        forced = applied.record

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
        if forced_intervention is not None:
            if forced_boundary == f"B_{layer_index}":
                applied = _apply_forced_visual_intervention(
                    hidden,
                    positions,
                    original_keep,
                    current_visual,
                    ForcedVisualIntervention(
                        boundary=layer_index,
                        mode=forced_intervention.mode,
                        retained_visual_ids=forced_retained or (),
                    ),
                )
                hidden = applied.hidden_states
                positions = applied.position_ids
                original_keep = applied.original_keep_indices
                current_visual = applied.current_visual_indices
                forced = applied.record
            continue
        if controller is None:
            continue
        if not controller.observe(layer_index, hidden[:, -1, :]):
            continue

        if ctp_policy is None:
            attention = _last_query_attention(
                decoder_layer, layer_input, position_embeddings, causal_mask
            )
            visual_scores = visual_attention_scores(
                attention,
                current_visual,
                head_aggregation=head_aggregation,
            )
        else:
            raw_logits = _last_query_attention_logits(
                decoder_layer, layer_input, position_embeddings, causal_mask
            )
            literal_scores, aggregate_scores = boundary_score_vectors_from_logits(
                raw_logits, current_visual
            )
            selection = select_boundary_policy(
                ctp_policy,
                literal_scores=tuple(float(value) for value in literal_scores.tolist()),
                aggregate_scores=tuple(float(value) for value in aggregate_scores.tolist()),
                attention_threshold=attention_threshold,
                boundary=f"B_{layer_index}",
                native_layer=layer_index,
                selection_context=selection_context,
            )
            visual_scores = torch.full(
                (current_visual.numel(),),
                float("-inf"),
                dtype=raw_logits.dtype,
                device=raw_logits.device,
            )
            visual_scores[list(selection.retained_visual_ids)] = 0.0
            attention_threshold = 0.0
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
            retained_visual_indices=tuple(
                int(value) for value in retained_visual_original.tolist()
            ),
        )
        hidden = hidden[:, local_keep]
        positions = positions[:, :, local_keep]
        original_keep = original_keep[local_keep]
        current_visual = visual_boolean[local_keep].nonzero(as_tuple=False).flatten()

    hidden = decoder_model.norm(hidden)
    if forced is not None:
        forced = replace(
            forced,
            prefill_cache_lengths=tuple(item.shape[-2] for item in cache.key_cache),
        )
    if ctp_policy is not None and selection is None:
        selection = no_crossing_selection(
            ctp_policy,
            int(visual_indices.numel()),
            selection_context=selection_context,
        )
    if selection is not None:
        position_shape, position_sha256 = _retained_mrope_position_identity(positions)
        selection = replace(
            selection,
            prefill_cache_lengths=tuple(item.shape[-2] for item in cache.key_cache),
            retained_mrope_position_shape=position_shape,
            retained_mrope_position_sha256=position_sha256,
        )
    return PrefillResult(hidden, cache, positions, original_keep, decision, forced, selection)


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
