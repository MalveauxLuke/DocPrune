import pytest
import torch

from docprune.btp import (
    background_keep_mask,
    background_scores,
    rgb_to_bt601_grayscale,
    threshold_keep_mask,
)
from docprune.layout import VisualLayout


def test_background_ratio_uses_strict_error_boundary() -> None:
    image = torch.tensor([[[[100, 100], [101, 130]]]], dtype=torch.uint8)

    got = background_scores(image, patch_size=2, error_tolerance=1)

    assert got.tolist() == [[0.5]]


def test_threshold_keeps_equal_ratio_and_prunes_greater_ratio() -> None:
    scores = torch.tensor([[0.8, 0.81]])

    assert threshold_keep_mask(scores, 0.8).tolist() == [[True, False]]


def test_bt601_color_conversion_is_rounded_to_uint8() -> None:
    red_green_blue = torch.tensor(
        [
            [
                [[255, 0, 0]],
                [[0, 255, 0]],
                [[0, 0, 255]],
            ]
        ],
        dtype=torch.uint8,
    )

    got = rgb_to_bt601_grayscale(red_green_blue)

    assert got.tolist() == [[[76, 150, 29]]]
    assert got.dtype == torch.uint8


def test_group_is_retained_when_any_fine_patch_contains_content() -> None:
    layout = VisualLayout(torch.tensor([[1, 2, 2]]), spatial_merge_size=2)
    fine_scores = torch.tensor([[1.0, 1.0, 1.0, 0.5]])

    got = background_keep_mask(fine_scores, threshold=0.9, layout=layout, retention="any")

    assert got.tolist() == [True]


def test_all_background_image_produces_no_retained_groups() -> None:
    image = torch.full((1, 1, 4, 4), 255, dtype=torch.uint8)
    layout = VisualLayout(torch.tensor([[1, 2, 2]]), spatial_merge_size=2)

    scores = background_scores(image, patch_size=2, error_tolerance=1)
    got = background_keep_mask(scores, threshold=0.9, layout=layout, retention="any")

    assert scores.tolist() == [[1.0, 1.0, 1.0, 1.0]]
    assert got.tolist() == [False]


def test_image_dimensions_must_match_patch_layout() -> None:
    image = torch.zeros((1, 1, 3, 4), dtype=torch.uint8)

    with pytest.raises(ValueError, match="divisible"):
        background_scores(image, patch_size=2, error_tolerance=1)
