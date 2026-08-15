import torch

from docprune.qwen2vl.vision import compact_vision_batch


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
