import torch

from docprune.qwen2vl.vision import compact_vision_batch


class _RecordingVisionBlock:
    def __init__(self) -> None:
        self.cu_seqlens: list[torch.Tensor] = []

    def __call__(self, hidden_states, *, cu_seqlens, rotary_pos_emb):
        del rotary_pos_emb
        self.cu_seqlens.append(cu_seqlens.detach().clone())
        return hidden_states


class _RecordingVisionModel:
    spatial_merge_size = 2

    def __init__(self) -> None:
        self.blocks = [_RecordingVisionBlock(), _RecordingVisionBlock()]

    def patch_embed(self, pixel_values):
        return pixel_values

    def rot_pos_emb(self, image_grid_thw):
        fine_count = int(torch.prod(image_grid_thw, dim=1).sum().item())
        return torch.arange(fine_count, dtype=torch.float32).unsqueeze(1)

    def merger(self, hidden_states):
        return hidden_states


def test_all_kept_sparse_vision_matches_official_forward(tiny_qwen2vl) -> None:
    torch.manual_seed(7)
    pixel_values = torch.randn(16, 24)
    grid = torch.tensor([[1, 4, 4]])

    with torch.no_grad():
        expected = tiny_qwen2vl.visual(pixel_values, grid_thw=grid)
        got = compact_vision_batch(
            tiny_qwen2vl.visual,
            pixel_values,
            grid,
            torch.tensor([True, True, True, True]),
        )

    assert torch.allclose(got.image_embeds, expected, atol=1e-6)
    assert got.page_fine_counts == (16,)


def test_sparse_vision_keeps_complete_groups_and_original_positions(tiny_qwen2vl) -> None:
    torch.manual_seed(7)
    pixel_values = torch.randn(16, 24)
    grid = torch.tensor([[1, 4, 4]])
    group_mask = torch.tensor([True, False, False, True])

    with torch.no_grad():
        got = compact_vision_batch(tiny_qwen2vl.visual, pixel_values, grid, group_mask)

    full_positions = tiny_qwen2vl.visual.rot_pos_emb(grid)
    assert got.image_embeds.shape == (2, 32)
    assert got.fine_position_embeddings.shape == (8, 4)
    assert torch.equal(
        got.fine_position_embeddings,
        full_positions[torch.tensor([0, 1, 2, 3, 12, 13, 14, 15])],
    )
    assert got.page_fine_counts == (8,)


def test_sparse_vision_passes_int32_page_boundaries_to_every_block() -> None:
    visual_model = _RecordingVisionModel()
    pixel_values = torch.arange(16 * 3, dtype=torch.float32).reshape(16, 3)
    grid = torch.tensor([[1, 2, 4], [1, 4, 2]])
    group_mask = torch.tensor([False, True, True, False])

    got = compact_vision_batch(visual_model, pixel_values, grid, group_mask)

    expected_pixel_indices = torch.tensor([4, 5, 6, 7, 8, 9, 10, 11])
    assert torch.equal(got.image_embeds, pixel_values[expected_pixel_indices])
    assert got.page_fine_counts == (4, 4)
    expected_cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.int32)
    for block in visual_model.blocks:
        assert len(block.cu_seqlens) == 1
        assert block.cu_seqlens[0].dtype == torch.int32
        assert torch.equal(block.cu_seqlens[0], expected_cu_seqlens)


def test_sparse_vision_rejects_a_page_with_no_tokens(tiny_qwen2vl) -> None:
    pixel_values = torch.randn(16, 24)
    grid = torch.tensor([[1, 4, 4]])

    try:
        compact_vision_batch(
            tiny_qwen2vl.visual,
            pixel_values,
            grid,
            torch.tensor([False, False, False, False]),
        )
    except ValueError as error:
        assert "at least one" in str(error)
    else:
        raise AssertionError("empty sparse page should be rejected")
