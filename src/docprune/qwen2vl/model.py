"""End-to-end greedy DocPrune generation for the pinned Qwen2-VL model."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from docprune.qwen2vl.compat import assert_supported_qwen2vl
from docprune.qwen2vl.decoder import decode_one_token, prefill_with_ctp
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
        vision = compact_vision_batch(
            self.model.visual,
            pixel_values,
            image_grid_thw,
            combined,
        )
        compact = compact_multimodal_sequence(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=full_positions,
            image_token_id=self.model.config.image_token_id,
            group_keep_mask=combined,
        )
        embeddings = self.model.model.embed_tokens(compact.input_ids)
        if compact.visual_indices.numel() != vision.image_embeds.shape[0]:
            raise ValueError("compacted image placeholders and sparse vision features do not match")
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
        return GenerationResult(generated_ids, trace, first_step_logits)
