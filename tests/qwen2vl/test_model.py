import torch

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
