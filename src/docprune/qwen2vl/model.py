"""End-to-end greedy DocPrune generation for the pinned Qwen2-VL model."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from docprune.ctp_policy import CTPPolicy, CTPSelectionRecord, PolicySelectionContext
from docprune.qwen2vl.compat import assert_supported_qwen2vl
from docprune.qwen2vl.decoder import (
    ForcedInterventionRecord,
    ForcedVisualIntervention,
    PrefillResult,
    decode_one_token,
    prefill_with_ctp,
)
from docprune.qwen2vl.sequence import compact_multimodal_sequence
from docprune.qwen2vl.vision import compact_vision_batch


@dataclass(frozen=True)
class VisionPruningMasks:
    background_keep: torch.Tensor
    question_keep: torch.Tensor

    def combined(self) -> torch.Tensor:
        background = torch.as_tensor(self.background_keep, dtype=torch.bool)
        question = torch.as_tensor(self.question_keep, dtype=torch.bool, device=background.device)
        if background.ndim != 1 or question.shape != background.shape:
            raise ValueError("background_keep and question_keep must be equal-length vectors")
        return background & question


@dataclass(frozen=True)
class PruningTrace:
    original_visual_tokens: int
    post_btp_visual_tokens: int
    post_qtp_visual_tokens: int
    post_ctp_visual_tokens: int
    ctp_layer: int | None

    def to_dict(self) -> dict[str, int | None]:
        return {
            "original_visual_tokens": self.original_visual_tokens,
            "post_btp_visual_tokens": self.post_btp_visual_tokens,
            "post_qtp_visual_tokens": self.post_qtp_visual_tokens,
            "post_ctp_visual_tokens": self.post_ctp_visual_tokens,
            "ctp_layer": self.ctp_layer,
        }


@dataclass(frozen=True)
class GenerationResult:
    generated_ids: torch.Tensor
    trace: PruningTrace
    first_step_logits: torch.Tensor | None = None
    encoder_seconds: float = 0.0
    decoder_seconds: float = 0.0
    forced_intervention: ForcedInterventionRecord | None = None
    policy_selection: CTPSelectionRecord | None = None
    teacher_forced_loglikelihoods: tuple[float, ...] | None = None


def _clone_dynamic_cache(cache: object) -> object:
    """Clone the pinned DynamicCache without retaining target-continuation state."""

    from transformers.cache_utils import DynamicCache

    if not isinstance(cache, DynamicCache):
        raise TypeError("teacher-forced likelihood requires a DynamicCache")
    cloned = DynamicCache()
    cloned.key_cache = [value.clone() for value in cache.key_cache]
    cloned.value_cache = [value.clone() for value in cache.value_cache]
    cloned._seen_tokens = cache._seen_tokens
    return cloned


def teacher_forced_sequence_loglikelihoods(
    decoder_model: object,
    lm_head: object,
    prefill: PrefillResult,
    target_token_ids: tuple[tuple[int, ...], ...],
) -> tuple[float, ...]:
    """Return mean full-answer log likelihood for each nonempty token sequence.

    The first target token is scored from the prompt prefill.  Every later
    token is scored after feeding only its gold predecessor.  EOS is not added;
    normalization is over the supplied answer tokens exactly.
    """

    targets = tuple(tuple(target) for target in target_token_ids)
    if not targets or any(not target for target in targets):
        raise ValueError("teacher-forced targets must be nonempty token sequences")
    first_logits = lm_head(prefill.hidden_states[:, -1, :]).float()
    if first_logits.ndim != 2 or first_logits.shape[0] != 1:
        raise ValueError("teacher-forced likelihood requires batch-one vocabulary logits")
    vocabulary_size = int(first_logits.shape[-1])
    if any(
        type(token) is not int or not 0 <= token < vocabulary_size
        for target in targets
        for token in target
    ):
        raise ValueError("teacher-forced target token is outside the model vocabulary")
    first_log_probs = torch.log_softmax(first_logits, dim=-1)
    next_position = int(prefill.position_ids.max().item()) + 1
    values: list[float] = []
    for target in targets:
        cache = _clone_dynamic_cache(prefill.cache)
        total = first_log_probs[0, target[0]]
        for offset, (previous_token, current_token) in enumerate(
            zip(target, target[1:], strict=False)
        ):
            token = torch.tensor(
                [[previous_token]],
                dtype=torch.long,
                device=prefill.hidden_states.device,
            )
            token_embedding = decoder_model.embed_tokens(token)
            step_positions = torch.full(
                (3, 1, 1),
                next_position + offset,
                dtype=prefill.position_ids.dtype,
                device=token_embedding.device,
            )
            hidden = decode_one_token(decoder_model, token_embedding, step_positions, cache)
            log_probs = torch.log_softmax(lm_head(hidden[:, -1, :]).float(), dim=-1)
            total = total + log_probs[0, current_token]
        values.append(float((total / len(target)).item()))
    return tuple(values)


def _model_device(model: object) -> torch.device | None:
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return None
    try:
        return next(parameters()).device
    except StopIteration:
        return None


def _begin_synchronized_timer(model: object) -> tuple[float, torch.device | None]:
    device = _model_device(model)
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    return time.perf_counter(), device


def _end_synchronized_timer(started: float, device: torch.device | None) -> float:
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    return max(time.perf_counter() - started, 1e-12)


def _module_timer_hooks(
    model: object, modules: list[object]
) -> tuple[list[float | tuple[object, object]], list[object]]:
    """Record module execution intervals without synchronizing every hook."""

    elapsed: list[float | tuple[object, object]] = []
    handles: list[object] = []
    device = _model_device(model)
    use_cuda_events = device is not None and device.type == "cuda" and torch.cuda.is_available()
    for module in modules:
        register_pre = getattr(module, "register_forward_pre_hook", None)
        register_post = getattr(module, "register_forward_hook", None)
        if not callable(register_pre) or not callable(register_post):
            continue

        def callbacks() -> tuple[object, object]:
            started: list[tuple[float, torch.device | None] | object] = []

            def before(*_args: object, **_kwargs: object) -> None:
                if use_cuda_events:
                    event = torch.cuda.Event(enable_timing=True)
                    event.record()
                    started.append(event)
                else:
                    started.append(_begin_synchronized_timer(model))

            def after(*_args: object, **_kwargs: object) -> None:
                if started:
                    start = started.pop()
                    if use_cuda_events:
                        end = torch.cuda.Event(enable_timing=True)
                        end.record()
                        elapsed.append((start, end))
                    else:
                        started_at, started_device = start
                        elapsed.append(_end_synchronized_timer(started_at, started_device))

            return before, after

        before, after = callbacks()
        handles.extend([register_pre(before), register_post(after)])
    return elapsed, handles


def _remove_module_timer_hooks(handles: list[object]) -> None:
    for handle in handles:
        remove = getattr(handle, "remove", None)
        if callable(remove):
            remove()


def _resolve_module_timer_groups(
    model: object, groups: tuple[list[float | tuple[object, object]], ...]
) -> tuple[float, ...]:
    """Resolve hook timings, synchronizing CUDA once for all stage groups."""

    device = _model_device(model)
    event_groups = [[sample for sample in group if isinstance(sample, tuple)] for group in groups]
    if any(event_groups):
        if device is None or device.type != "cuda":
            raise RuntimeError("CUDA timer events require a CUDA model device")
        torch.cuda.synchronize(device)
    return tuple(
        sum(
            sample
            if isinstance(sample, int | float)
            else sample[0].elapsed_time(sample[1]) / 1000.0
            for sample in group
        )
        for group in groups
    )


def _resolve_module_timer_samples(
    model: object, samples: list[float | tuple[object, object]]
) -> float:
    return _resolve_module_timer_groups(model, (samples,))[0]


class DocPruneQwen2VL:
    def __init__(self, model: object) -> None:
        self.compatibility = assert_supported_qwen2vl(model)
        self.model = model

    def generate_with_trace(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        pruning_masks: VisionPruningMasks,
        comprehension_threshold: float,
        attention_threshold: float,
        max_new_tokens: int,
        eos_token_ids: tuple[int, ...],
        head_aggregation: str = "mean",
        forced_intervention: ForcedVisualIntervention | None = None,
        ctp_policy: CTPPolicy | None = None,
        selection_context: PolicySelectionContext | None = None,
        teacher_forced_target_token_ids: tuple[tuple[int, ...], ...] | None = None,
    ) -> GenerationResult:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("DocPrune generation requires batch size one")
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids")
        if not bool(torch.as_tensor(attention_mask, dtype=torch.bool).all()):
            raise ValueError(
                "DocPrune Qwen2-VL manual cache compaction does not support padding in attention_mask"
            )
        full_positions, _ = self.model.get_rope_index(
            input_ids,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
        )
        background = torch.as_tensor(pruning_masks.background_keep, dtype=torch.bool)
        combined = pruning_masks.combined().to(pixel_values.device)
        vision_dtype = getattr(self.model.visual, "get_dtype", lambda: pixel_values.dtype)()
        vision_device = getattr(self.model.visual, "get_device", lambda: pixel_values.device)()
        pixel_values = pixel_values.to(device=vision_device, dtype=vision_dtype)
        encoder_started, encoder_device = _begin_synchronized_timer(self.model)
        vision = compact_vision_batch(
            self.model.visual,
            pixel_values,
            image_grid_thw,
            combined,
        )
        encoder_elapsed = _end_synchronized_timer(encoder_started, encoder_device)
        encoder_seconds = encoder_elapsed
        compact = compact_multimodal_sequence(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=full_positions,
            image_token_id=self.model.config.image_token_id,
            group_keep_mask=combined,
        )
        decoder_started, decoder_device = _begin_synchronized_timer(self.model)
        try:
            embeddings = self.model.model.embed_tokens(compact.input_ids)
            if compact.visual_indices.numel() != vision.image_embeds.shape[0]:
                raise ValueError(
                    "compacted image placeholders and sparse vision features do not match"
                )
            embeddings = embeddings.clone()
            embeddings[0, compact.visual_indices] = vision.image_embeds.to(
                device=embeddings.device,
                dtype=embeddings.dtype,
            )
            prefill = prefill_with_ctp(
                self.model.model,
                embeddings,
                compact.position_ids,
                visual_indices=compact.visual_indices,
                comprehension_threshold=comprehension_threshold,
                attention_threshold=attention_threshold,
                head_aggregation=head_aggregation,
                forced_intervention=forced_intervention,
                ctp_policy=ctp_policy,
                selection_context=selection_context,
            )
            teacher_forced_values = (
                None
                if teacher_forced_target_token_ids is None
                else teacher_forced_sequence_loglikelihoods(
                    self.model.model,
                    self.model.lm_head,
                    prefill,
                    teacher_forced_target_token_ids,
                )
            )
            generated: list[torch.Tensor] = []
            logits = self.model.lm_head(prefill.hidden_states[:, -1, :])
            first_step_logits = logits.detach()
            next_token = logits.argmax(dim=-1)
            next_position = int(prefill.position_ids.max().item()) + 1
            eos = set(eos_token_ids)
            for token_index in range(max_new_tokens):
                generated.append(next_token)
                if int(next_token.item()) in eos or token_index + 1 == max_new_tokens:
                    break
                token_embedding = self.model.model.embed_tokens(next_token[:, None])
                step_positions = torch.full(
                    (3, 1, 1),
                    next_position,
                    dtype=compact.position_ids.dtype,
                    device=token_embedding.device,
                )
                hidden = decode_one_token(
                    self.model.model,
                    token_embedding,
                    step_positions,
                    prefill.cache,
                )
                next_token = self.model.lm_head(hidden[:, -1, :]).argmax(dim=-1)
                next_position += 1
            generated_ids = torch.stack(generated, dim=1)
        finally:
            decoder_elapsed = _end_synchronized_timer(decoder_started, decoder_device)
        decoder_seconds = decoder_elapsed
        if prefill.forced is not None:
            post_ctp = (
                prefill.forced.achieved_budget
                if prefill.forced.mode == "physical_delete"
                else int(combined.sum().item())
            )
        else:
            post_ctp = (
                len(prefill.decision.retained_visual_indices)
                if prefill.decision is not None
                else int(combined.sum().item())
            )
        trace = PruningTrace(
            original_visual_tokens=background.numel(),
            post_btp_visual_tokens=int(background.sum().item()),
            post_qtp_visual_tokens=int(combined.sum().item()),
            post_ctp_visual_tokens=post_ctp,
            ctp_layer=prefill.decision.layer_index if prefill.decision is not None else None,
        )
        return GenerationResult(
            generated_ids,
            trace,
            first_step_logits,
            encoder_seconds,
            decoder_seconds,
            prefill.forced,
            prefill.selection,
            teacher_forced_values,
        )
