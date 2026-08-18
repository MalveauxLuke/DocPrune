import os
from pathlib import Path

import pytest
import torch
from PIL import Image

from docprune.qwen2vl.model import DocPruneQwen2VL, VisionPruningMasks


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
    assert got.trace.original_visual_tokens == 4
    assert got.trace.post_btp_visual_tokens == 3
    assert got.trace.post_qtp_visual_tokens == 2
    assert got.trace.post_ctp_visual_tokens == 0
    assert got.trace.ctp_layer == 0


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
    torch.testing.assert_close(got.first_step_logits, stock_forward.logits[:, -1], rtol=1e-5, atol=1e-5)


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

    from docprune.qwen2vl.preprocessing import prepare_qwen_page

    processor = AutoProcessor.from_pretrained(model_root, revision=revision, local_files_only=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_root,
        revision=revision,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    ).to("cuda").eval()
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
    with torch.no_grad():
        stock_positions, _ = model.get_rope_index(
            input_ids,
            image_grid_thw=inputs["image_grid_thw"],
            attention_mask=inputs["attention_mask"],
        )
        stock_forward = model(**inputs, position_ids=stock_positions, use_cache=True)
        stock_ids = model.generate(**inputs, max_new_tokens=2, do_sample=False, num_beams=1)
        adapted = adapter.generate_with_trace(
            input_ids=input_ids,
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs["pixel_values"],
            image_grid_thw=inputs["image_grid_thw"],
            pruning_masks=masks,
            comprehension_threshold=1e9,
            attention_threshold=0.0,
            max_new_tokens=2,
            eos_token_ids=(),
        )
    assert adapted.first_step_logits is not None
    torch.testing.assert_close(adapted.first_step_logits, stock_forward.logits[:, -1], rtol=2e-2, atol=2e-2)
    assert adapted.generated_ids.tolist() == stock_ids[:, -2:].tolist()
