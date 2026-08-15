import pytest
import torch

from docprune.layout import VisualLayout


def test_four_by_four_grid_is_grouped_in_spatial_order() -> None:
    layout = VisualLayout(torch.tensor([[1, 4, 4]]), spatial_merge_size=2)

    assert layout.group_fine_indices().tolist() == [
        [0, 1, 4, 5],
        [2, 3, 6, 7],
        [8, 9, 12, 13],
        [10, 11, 14, 15],
    ]
    assert layout.page_group_offsets == (0, 4)


def test_multiple_pages_have_global_offsets_without_reordering() -> None:
    layout = VisualLayout(torch.tensor([[1, 2, 2], [1, 2, 4]]), spatial_merge_size=2)

    assert layout.group_fine_indices().tolist() == [
        [0, 1, 2, 3],
        [4, 5, 8, 9],
        [6, 7, 10, 11],
    ]
    assert layout.page_fine_offsets == (0, 4, 12)
    assert layout.page_group_offsets == (0, 1, 3)


def test_group_mask_expands_to_fine_token_mask() -> None:
    layout = VisualLayout(torch.tensor([[1, 4, 4]]), spatial_merge_size=2)

    fine_mask = layout.expand_group_mask(torch.tensor([True, False, False, True]))

    assert fine_mask.nonzero(as_tuple=False).flatten().tolist() == [0, 1, 4, 5, 10, 11, 14, 15]


def test_grid_must_be_divisible_by_spatial_merge_size() -> None:
    with pytest.raises(ValueError, match="divisible"):
        VisualLayout(torch.tensor([[1, 3, 4]]), spatial_merge_size=2)
