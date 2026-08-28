import hashlib
import json
import os
from pathlib import Path

import pytest
import torch
from PIL import Image

from docprune.qwen2vl.decoder import ForcedVisualIntervention
from docprune.qwen2vl.model import (
    DocPruneQwen2VL,
    VisionPruningMasks,
    _module_timer_hooks,
    _remove_module_timer_hooks,
    _resolve_module_timer_samples,
    teacher_forced_sequence_loglikelihoods,
)


def test_module_timer_hooks_use_cuda_events_without_per_module_synchronization(monkeypatch) -> None:
    import docprune.qwen2vl.model as model_module

    class FakeEvent:
        pairs = []

        def __init__(self, **kwargs):
            del kwargs
            self.recorded = False

        def record(self):
            self.recorded = True

        def elapsed_time(self, other):
            assert self.recorded and other.recorded
            return 2.0

    synchronize_calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_args: synchronize_calls.append(True))
    monkeypatch.setattr(model_module, "_model_device", lambda _model: torch.device("cuda"))

    module = torch.nn.Identity()
    samples, handles = _module_timer_hooks(module, [module])
    module(torch.ones(1))
    _remove_module_timer_hooks(handles)
    assert synchronize_calls == []
    assert _resolve_module_timer_samples(module, samples) == pytest.approx(0.002)
    assert len(synchronize_calls) == 1


def test_tiny_model_generates_with_monotonic_pruning_trace(tiny_qwen2vl) -> None:
    torch.manual_seed(11)
    adapter = DocPruneQwen2VL(tiny_qwen2vl)
    input_ids = torch.tensor([[10, 102, 100, 100, 100, 100, 103, 11]])
    attention_mask = torch.ones_like(input_ids)
    pixel_values = torch.randn(16, 24)
    grid = torch.tensor([[1, 4, 4]])
    masks = VisionPruningMasks(
        background_keep=torch.tensor([True, True, False, True]),
        question_keep=torch.tensor([True, False, True, True]),
    )

    with torch.no_grad():
        got = adapter.generate_with_trace(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=grid,
            pruning_masks=masks,
            comprehension_threshold=0.0,
            attention_threshold=1.0,
            max_new_tokens=2,
            eos_token_ids=(),
        )

    assert got.generated_ids.shape == (1, 2)
    assert got.encoder_seconds > 0
    assert got.decoder_seconds > 0
    assert got.trace.original_visual_tokens == 4
    assert got.trace.post_btp_visual_tokens == 3
    assert got.trace.post_qtp_visual_tokens == 2
    assert got.trace.post_ctp_visual_tokens == 0
    assert got.trace.ctp_layer == 0


def test_teacher_forced_likelihood_scores_every_answer_token_without_eos(
    tiny_qwen2vl,
) -> None:
    """Catch scoring only the first token, shifting targets, or adding EOS."""

    from docprune.qwen2vl.decoder import decode_one_token, prefill_with_ctp

    input_ids = torch.tensor([[10, 102, 100, 100, 100, 100, 103, 11]])
    attention_mask = torch.ones_like(input_ids)
    positions, _ = tiny_qwen2vl.get_rope_index(
        input_ids,
        image_grid_thw=torch.tensor([[1, 4, 4]]),
        attention_mask=attention_mask,
    )
    hidden = tiny_qwen2vl.model.embed_tokens(input_ids)
    with torch.no_grad():
        prefill = prefill_with_ctp(
            tiny_qwen2vl.model,
            hidden,
            positions,
            visual_indices=torch.tensor([2, 3, 4, 5]),
            comprehension_threshold=1e9,
            attention_threshold=0.0,
        )
        first_logits = tiny_qwen2vl.lm_head(prefill.hidden_states[:, -1, :]).float()
        first_log_probs = torch.log_softmax(first_logits, dim=-1)
        first_target = 12
        second_target = 13
        step_hidden = decode_one_token(
            tiny_qwen2vl.model,
            tiny_qwen2vl.model.embed_tokens(torch.tensor([[first_target]])),
            torch.full(
                (3, 1, 1),
                int(prefill.position_ids.max().item()) + 1,
                dtype=prefill.position_ids.dtype,
            ),
            prefill.cache,
        )
        second_log_probs = torch.log_softmax(
            tiny_qwen2vl.lm_head(step_hidden[:, -1, :]).float(), dim=-1
        )
        expected = float(
            (first_log_probs[0, first_target] + second_log_probs[0, second_target]).item() / 2
        )

        # Rebuild because the hand-derived continuation above intentionally
        # advanced its cache.
        fresh = prefill_with_ctp(
            tiny_qwen2vl.model,
            hidden,
            positions,
            visual_indices=torch.tensor([2, 3, 4, 5]),
            comprehension_threshold=1e9,
            attention_threshold=0.0,
        )
        before = tuple(cache.shape[-2] for cache in fresh.cache.key_cache)
        scores = teacher_forced_sequence_loglikelihoods(
            tiny_qwen2vl.model,
            tiny_qwen2vl.lm_head,
            fresh,
            ((first_target, second_target), (second_target,)),
        )

    assert scores[0] == pytest.approx(expected)
    assert scores[1] == pytest.approx(float(first_log_probs[0, second_target].item()))
    assert tuple(cache.shape[-2] for cache in fresh.cache.key_cache) == before


@pytest.mark.parametrize("targets", [(), ((),), ((-1,),), ((10**9,),)])
def test_teacher_forced_likelihood_rejects_missing_or_invalid_targets(
    tiny_qwen2vl, targets: tuple[tuple[int, ...], ...]
) -> None:
    """Catch an empty or invalid gold target producing a meaningless finite score."""

    from docprune.qwen2vl.decoder import prefill_with_ctp

    input_ids = torch.tensor([[10, 102, 100, 100, 100, 100, 103, 11]])
    positions = torch.arange(8).view(1, 1, 8).expand(3, 1, 8).clone()
    with torch.no_grad():
        prefill = prefill_with_ctp(
            tiny_qwen2vl.model,
            tiny_qwen2vl.model.embed_tokens(input_ids),
            positions,
            visual_indices=torch.tensor([2, 3, 4, 5]),
            comprehension_threshold=1e9,
            attention_threshold=0.0,
        )
        with pytest.raises(ValueError, match="teacher-forced target"):
            teacher_forced_sequence_loglikelihoods(
                tiny_qwen2vl.model,
                tiny_qwen2vl.lm_head,
                prefill,
                targets,
            )


def test_all_kept_adapter_matches_stock_qwen_generation(tiny_qwen2vl) -> None:
    torch.manual_seed(19)
    adapter = DocPruneQwen2VL(tiny_qwen2vl)
    input_ids = torch.tensor([[10, 102, 100, 100, 100, 100, 103, 11]])
    attention_mask = torch.ones_like(input_ids)
    pixel_values = torch.randn(16, 24)
    grid = torch.tensor([[1, 4, 4]])
    masks = VisionPruningMasks(
        background_keep=torch.ones(4, dtype=torch.bool),
        question_keep=torch.ones(4, dtype=torch.bool),
    )

    with torch.no_grad():
        stock_positions, _ = tiny_qwen2vl.get_rope_index(
            input_ids,
            image_grid_thw=grid,
            attention_mask=attention_mask,
        )
        stock_forward = tiny_qwen2vl(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=stock_positions,
            pixel_values=pixel_values,
            image_grid_thw=grid,
        )
        stock = tiny_qwen2vl.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=grid,
            max_new_tokens=2,
            do_sample=False,
            num_beams=1,
        )
        got = adapter.generate_with_trace(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=grid,
            pruning_masks=masks,
            comprehension_threshold=1e9,
            attention_threshold=1.0,
            max_new_tokens=2,
            eos_token_ids=(),
        )

    assert got.generated_ids.tolist() == stock[:, -2:].tolist()
    assert got.first_step_logits is not None
    torch.testing.assert_close(
        got.first_step_logits, stock_forward.logits[:, -1], rtol=1e-5, atol=1e-5
    )


def test_adapter_exposes_forced_record_without_relabeling_native_ctp(tiny_qwen2vl) -> None:
    """Catch forced execution disappearing at the adapter boundary or becoming native CTP."""

    adapter = DocPruneQwen2VL(tiny_qwen2vl)
    with torch.no_grad():
        got = adapter.generate_with_trace(
            input_ids=torch.tensor([[10, 102, 100, 100, 100, 100, 103, 11]]),
            attention_mask=torch.ones((1, 8), dtype=torch.long),
            pixel_values=torch.randn(16, 24),
            image_grid_thw=torch.tensor([[1, 4, 4]]),
            pruning_masks=VisionPruningMasks(
                background_keep=torch.ones(4, dtype=torch.bool),
                question_keep=torch.ones(4, dtype=torch.bool),
            ),
            comprehension_threshold=0.0,
            attention_threshold=0.0,
            max_new_tokens=1,
            eos_token_ids=(),
            forced_intervention=ForcedVisualIntervention(0, "physical_delete", (1, 3)),
        )

    assert got.forced_intervention is not None
    assert got.forced_intervention.boundary == "B_0"
    assert got.forced_intervention.retained_visual_ids == (1, 3)
    assert got.trace.ctp_layer is None
    assert got.trace.post_ctp_visual_tokens == 2


@pytest.mark.parametrize("mode", ("physical_delete", "zero_mask"))
@pytest.mark.parametrize("boundary", ("input", 0, 1, 2, 3))
def test_adapter_all_kept_forced_boundaries_match_native_no_crossing(
    tiny_qwen2vl, boundary: str | int, mode: str
) -> None:
    """Catch a forced no-op boundary changing adapter-visible generation or first logits."""

    torch.manual_seed(29)
    adapter = DocPruneQwen2VL(tiny_qwen2vl)
    input_ids = torch.tensor([[10, 102, 100, 100, 100, 100, 103, 11]])
    attention_mask = torch.ones_like(input_ids)
    pixel_values = torch.randn(16, 24)
    grid = torch.tensor([[1, 4, 4]])
    masks = VisionPruningMasks(
        background_keep=torch.ones(4, dtype=torch.bool),
        question_keep=torch.ones(4, dtype=torch.bool),
    )
    with torch.no_grad():
        native = adapter.generate_with_trace(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=grid,
            pruning_masks=masks,
            comprehension_threshold=1e9,
            attention_threshold=1.0,
            max_new_tokens=2,
            eos_token_ids=(),
        )
        forced = adapter.generate_with_trace(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=grid,
            pruning_masks=masks,
            comprehension_threshold=0.0,
            attention_threshold=0.0,
            max_new_tokens=2,
            eos_token_ids=(),
            forced_intervention=ForcedVisualIntervention(boundary, mode, (0, 1, 2, 3)),
        )

    assert forced.forced_intervention is not None
    assert forced.generated_ids.tolist() == native.generated_ids.tolist()
    assert forced.first_step_logits is not None and native.first_step_logits is not None
    torch.testing.assert_close(
        forced.first_step_logits, native.first_step_logits, rtol=1e-5, atol=1e-5
    )


def test_adapter_rejects_padded_compact_attention_until_mask_support_exists(tiny_qwen2vl) -> None:
    adapter = DocPruneQwen2VL(tiny_qwen2vl)
    input_ids = torch.tensor([[10, 102, 100, 100, 100, 100, 103, 11]])
    masks = VisionPruningMasks(
        background_keep=torch.ones(4, dtype=torch.bool),
        question_keep=torch.ones(4, dtype=torch.bool),
    )

    try:
        adapter.generate_with_trace(
            input_ids=input_ids,
            attention_mask=torch.tensor([[1, 1, 1, 1, 1, 1, 1, 0]]),
            pixel_values=torch.randn(16, 24),
            image_grid_thw=torch.tensor([[1, 4, 4]]),
            pruning_masks=masks,
            comprehension_threshold=1e9,
            attention_threshold=0.0,
            max_new_tokens=1,
            eos_token_ids=(),
        )
    except ValueError as error:
        assert "padding" in str(error)
    else:
        raise AssertionError("padded compact attention must fail closed")


def test_adapter_casts_processor_pixels_to_vision_dtype(tiny_qwen2vl) -> None:
    model = tiny_qwen2vl.half()
    adapter = DocPruneQwen2VL(model)
    input_ids = torch.tensor([[10, 102, 100, 100, 100, 100, 103, 11]])
    masks = VisionPruningMasks(
        background_keep=torch.ones(4, dtype=torch.bool),
        question_keep=torch.ones(4, dtype=torch.bool),
    )

    with torch.no_grad():
        result = adapter.generate_with_trace(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            pixel_values=torch.randn(16, 24),
            image_grid_thw=torch.tensor([[1, 4, 4]]),
            pruning_masks=masks,
            comprehension_threshold=1e9,
            attention_threshold=0.0,
            max_new_tokens=1,
            eos_token_ids=(),
        )

    assert result.first_step_logits is not None
    assert result.first_step_logits.dtype == torch.float16


def test_real_model_all_kept_matches_stock_first_step_logits_without_download() -> None:
    if os.environ.get("DOCPRUNE_REAL_MODEL_TEST") != "1":
        pytest.skip("set DOCPRUNE_REAL_MODEL_TEST=1 to run the cached real-model probe")
    if not torch.cuda.is_available():
        pytest.skip("real-model probe requires CUDA; CPU tests never load the 7B checkpoint")
    model_root = os.environ.get("DOCPRUNE_QWEN_MODEL_PATH")
    probe_path = os.environ.get("DOCPRUNE_QWEN_PROBE_PAGE")
    revision = os.environ.get("DOCPRUNE_QWEN_REVISION")
    from docprune.benchmark_config import QWEN_REVISION

    if not model_root or not probe_path or revision != QWEN_REVISION:
        pytest.skip(
            "DOCPRUNE_QWEN_MODEL_PATH, DOCPRUNE_QWEN_PROBE_PAGE, and the pinned "
            "DOCPRUNE_QWEN_REVISION are required"
        )
    if not Path(model_root).is_dir() or not Path(probe_path).is_file():
        pytest.skip("cached exact Qwen model and fixed probe page are unavailable")

    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    from docprune.answerers import _resolved_eos_token_ids
    from docprune.qwen2vl.preprocessing import prepare_qwen_page, prepared_raster_image

    processor = AutoProcessor.from_pretrained(model_root, revision=revision, local_files_only=True)
    model = (
        Qwen2VLForConditionalGeneration.from_pretrained(
            model_root,
            revision=revision,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        .to("cuda")
        .eval()
    )
    with Image.open(probe_path) as opened:
        page = opened.convert("RGB").copy()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "dummy_content"},
                {"type": "text", "text": "question: probe\noutput only answer."},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[page], padding=True, return_tensors="pt")
    prepared = prepare_qwen_page(processor, page)
    prepared_inputs = processor(
        text=[prompt],
        images=[prepared_raster_image(prepared)],
        padding=True,
        return_tensors="pt",
    )
    for name in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw"):
        torch.testing.assert_close(prepared_inputs[name], inputs[name], rtol=0.0, atol=0.0)
    inputs = {
        name: value.to("cuda") if isinstance(value, torch.Tensor) else value
        for name, value in inputs.items()
    }
    input_ids = inputs["input_ids"]
    masks = VisionPruningMasks(
        background_keep=torch.ones(prepared.placeholder_count, dtype=torch.bool, device="cuda"),
        question_keep=torch.ones(prepared.placeholder_count, dtype=torch.bool, device="cuda"),
    )
    adapter = DocPruneQwen2VL(model)
    eos_token_ids = _resolved_eos_token_ids(model)
    with torch.no_grad():
        stock_positions, _ = model.get_rope_index(
            input_ids,
            image_grid_thw=inputs["image_grid_thw"],
            attention_mask=inputs["attention_mask"],
        )
        stock_forward = model(**inputs, position_ids=stock_positions, use_cache=True)
        stock_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            num_beams=1,
            eos_token_id=list(eos_token_ids),
        )
        adapted = adapter.generate_with_trace(
            input_ids=input_ids,
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs["pixel_values"],
            image_grid_thw=inputs["image_grid_thw"],
            pruning_masks=masks,
            comprehension_threshold=1e9,
            attention_threshold=0.0,
            max_new_tokens=128,
            eos_token_ids=eos_token_ids,
        )
    assert adapted.first_step_logits is not None
    assert adapted.generated_ids.tolist() == stock_ids[:, input_ids.shape[1] :].tolist()
    # Equivalent BF16 FlashAttention paths differ by up to one rounding bin on
    # L40S even when the complete greedy generation is exactly identical.
    torch.testing.assert_close(
        adapted.first_step_logits, stock_forward.logits[:, -1], rtol=2e-2, atol=7e-2
    )


def test_real_model_forced_all_kept_boundaries_match_stock_without_download() -> None:
    """Opt-in live admission: each forced all-kept boundary must remain stock-equivalent."""

    if os.environ.get("DOCPRUNE_REAL_MODEL_TEST") != "1":
        pytest.skip("set DOCPRUNE_REAL_MODEL_TEST=1 to run the cached real-model probe")
    if not torch.cuda.is_available():
        pytest.skip("real-model probe requires CUDA; CPU tests never load the 7B checkpoint")
    artifact_value = os.environ.get("DOCPRUNE_FORCED_BOUNDARY_ARTIFACT")
    if not artifact_value:
        pytest.fail(
            "live forced-boundary parity requires DOCPRUNE_FORCED_BOUNDARY_ARTIFACT as a fresh path"
        )
    artifact_path = Path(artifact_value)
    if artifact_path.exists() or not artifact_path.parent.is_dir():
        pytest.fail(
            "live forced-boundary artifact path must be fresh with an existing parent directory"
        )
    model_root = os.environ.get("DOCPRUNE_QWEN_MODEL_PATH")
    probe_path = os.environ.get("DOCPRUNE_QWEN_PROBE_PAGE")
    revision = os.environ.get("DOCPRUNE_QWEN_REVISION")
    from docprune.benchmark_config import QWEN_REVISION

    if not model_root or not probe_path or revision != QWEN_REVISION:
        pytest.skip(
            "DOCPRUNE_QWEN_MODEL_PATH, DOCPRUNE_QWEN_PROBE_PAGE, and the pinned "
            "DOCPRUNE_QWEN_REVISION are required"
        )
    if not Path(model_root).is_dir() or not Path(probe_path).is_file():
        pytest.skip("cached exact Qwen model and fixed probe page are unavailable")

    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    from docprune.answerers import _resolved_eos_token_ids
    from docprune.qwen2vl.preprocessing import prepare_qwen_page, prepared_raster_image

    processor = AutoProcessor.from_pretrained(model_root, revision=revision, local_files_only=True)
    model = (
        Qwen2VLForConditionalGeneration.from_pretrained(
            model_root,
            revision=revision,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        .to("cuda")
        .eval()
    )
    with Image.open(probe_path) as opened:
        page = opened.convert("RGB").copy()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "dummy_content"},
                {"type": "text", "text": "question: probe\noutput only answer."},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[page], padding=True, return_tensors="pt")
    prepared = prepare_qwen_page(processor, page)
    prepared_inputs = processor(
        text=[prompt],
        images=[prepared_raster_image(prepared)],
        padding=True,
        return_tensors="pt",
    )
    for name in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw"):
        torch.testing.assert_close(prepared_inputs[name], inputs[name], rtol=0.0, atol=0.0)
    inputs = {
        name: value.to("cuda") if isinstance(value, torch.Tensor) else value
        for name, value in inputs.items()
    }
    input_ids = inputs["input_ids"]
    masks = VisionPruningMasks(
        background_keep=torch.ones(prepared.placeholder_count, dtype=torch.bool, device="cuda"),
        question_keep=torch.ones(prepared.placeholder_count, dtype=torch.bool, device="cuda"),
    )
    adapter = DocPruneQwen2VL(model)
    eos_token_ids = _resolved_eos_token_ids(model)
    with torch.no_grad():
        stock_positions, _ = model.get_rope_index(
            input_ids,
            image_grid_thw=inputs["image_grid_thw"],
            attention_mask=inputs["attention_mask"],
        )
    expected_positions = stock_positions.detach().to(device="cpu", dtype=torch.int64).contiguous()
    expected_position_shape = tuple(int(value) for value in expected_positions.shape)
    expected_position_sha256 = hashlib.sha256(expected_positions.numpy().tobytes()).hexdigest()
    expected_logical_order = tuple(range(int(input_ids.shape[1])))
    expected_cache_lengths = tuple([int(input_ids.shape[1])] * len(model.model.layers))
    boundary_artifacts: list[dict[str, object]] = []
    with torch.no_grad():
        stock_forward = model(**inputs, position_ids=stock_positions, use_cache=True)
        stock_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            num_beams=1,
            eos_token_id=list(eos_token_ids),
        )
        for boundary in ("input", 0, 6, 13, 20, 23, 26):
            adapted = adapter.generate_with_trace(
                input_ids=input_ids,
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs["pixel_values"],
                image_grid_thw=inputs["image_grid_thw"],
                pruning_masks=masks,
                comprehension_threshold=0.0,
                attention_threshold=0.0,
                max_new_tokens=128,
                eos_token_ids=eos_token_ids,
                forced_intervention=ForcedVisualIntervention(
                    boundary=boundary,
                    mode="physical_delete",
                    retained_visual_ids=tuple(range(prepared.placeholder_count)),
                ),
            )
            assert adapted.forced_intervention is not None
            expected_boundary = "B_input" if boundary == "input" else f"B_{boundary}"
            assert adapted.first_step_logits is not None
            record = adapted.forced_intervention
            logits = adapted.first_step_logits.float()
            stock_logits = stock_forward.logits[:, -1].float()
            absolute_difference = (logits - stock_logits).abs()
            boundary_artifacts.append(
                {
                    "boundary": record.boundary,
                    "mode": record.mode,
                    "selection_kind": record.selection_kind,
                    "visual_population": record.visual_population,
                    "requested_budget": record.requested_budget,
                    "achieved_budget": record.achieved_budget,
                    "retained_visual_ids": list(record.retained_visual_ids),
                    "logical_retained_sequence_ids": list(record.logical_retained_sequence_ids),
                    "prefill_cache_lengths": list(record.prefill_cache_lengths),
                    "retained_mrope_position_shape": list(record.retained_mrope_position_shape),
                    "retained_mrope_position_sha256": record.retained_mrope_position_sha256,
                    "stock_all_kept_cache_lengths": list(expected_cache_lengths),
                    "stock_all_kept_logical_sequence_ids": list(expected_logical_order),
                    "stock_all_kept_mrope_position_shape": list(expected_position_shape),
                    "stock_all_kept_mrope_position_sha256": expected_position_sha256,
                    "exact_suffix_verdict": adapted.generated_ids.tolist()
                    == stock_ids[:, input_ids.shape[1] :].tolist(),
                    "logit_tolerance_verdict": bool(
                        torch.allclose(logits, stock_logits, rtol=2e-2, atol=7e-2)
                    ),
                    "max_abs_logit_difference": float(absolute_difference.max().item()),
                    "mean_abs_logit_difference": float(absolute_difference.mean().item()),
                    "max_relative_logit_difference": float(
                        (absolute_difference / stock_logits.abs().clamp_min(1e-12)).max().item()
                    ),
                    "boundary_identity_verdict": record.boundary == expected_boundary,
                    "cache_vector_verdict": record.prefill_cache_lengths == expected_cache_lengths,
                    "logical_order_verdict": record.logical_retained_sequence_ids
                    == expected_logical_order,
                    "position_identity_verdict": record.retained_mrope_position_shape
                    == expected_position_shape
                    and record.retained_mrope_position_sha256 == expected_position_sha256,
                }
            )
    with artifact_path.open("x", encoding="utf-8") as artifact_file:
        artifact_file.write(
            json.dumps(
                {"schema_version": 1, "boundaries": boundary_artifacts}, indent=2, sort_keys=True
            )
            + "\n"
        )
    assert all(
        entry["boundary_identity_verdict"]
        and entry["cache_vector_verdict"]
        and entry["logical_order_verdict"]
        and entry["position_identity_verdict"]
        and entry["exact_suffix_verdict"]
        and entry["logit_tolerance_verdict"]
        for entry in boundary_artifacts
    )
