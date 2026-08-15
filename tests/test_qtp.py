import math

import pytest
import torch

from docprune.layout import VisualLayout
from docprune.qtp import (
    cosine_sum_relevance,
    gaussian_smooth_2d,
    group_relevance_keep_mask,
    question_keep_mask,
    resize_relevance_map,
    threshold_relevance_keep_mask,
)


def test_cosine_sum_aggregates_every_question_token() -> None:
    documents = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    questions = torch.tensor([[[1.0, 0.0], [1.0, 1.0]]])

    got = cosine_sum_relevance(documents, questions)

    assert torch.allclose(got, torch.tensor([[1.7071068, 0.7071068]]), atol=1e-6)


def test_zero_vector_has_zero_cosine_contribution() -> None:
    documents = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]])
    questions = torch.tensor([[[1.0, 0.0]]])

    got = cosine_sum_relevance(documents, questions)

    assert got.tolist() == [[0.0, 1.0]]


def test_single_question_batch_is_shared_across_document_pages() -> None:
    documents = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    questions = torch.tensor([[[1.0, 0.0]]])

    got = cosine_sum_relevance(documents, questions)

    assert got.tolist() == [[1.0], [0.0]]


def test_bilinear_resize_preserves_shape_and_corner_values() -> None:
    scores = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])

    got = resize_relevance_map(scores, target_hw=(4, 4))

    assert got.shape == (1, 4, 4)
    assert got[0, 0, 0].item() == 1.0
    assert got[0, -1, -1].item() == 4.0


def test_gaussian_impulse_response_matches_literal_kernel() -> None:
    impulse = torch.zeros((1, 3, 3))
    impulse[0, 1, 1] = 1.0
    one_dimensional = torch.tensor([0.27406862, 0.45186276, 0.27406862])
    expected = torch.outer(one_dimensional, one_dimensional).unsqueeze(0)

    got = gaussian_smooth_2d(impulse, sigma=1.0, truncate=1.0)

    assert torch.allclose(got, expected, atol=1e-6)


def test_relevance_threshold_retains_equality() -> None:
    scores = torch.tensor([0.29, 0.3, 0.31])

    got = threshold_relevance_keep_mask(scores, threshold=0.3)

    assert got.tolist() == [False, True, True]


def test_group_is_retained_when_any_fine_token_is_relevant() -> None:
    layout = VisualLayout(torch.tensor([[1, 2, 2]]), spatial_merge_size=2)
    scores = torch.tensor([0.1, 0.1, 0.4, 0.1])

    got = group_relevance_keep_mask(scores, threshold=0.3, layout=layout, retention="any")

    assert got.tolist() == [True]


def test_question_keep_mask_runs_equation_resize_smooth_threshold_pipeline() -> None:
    documents = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]]
    )
    questions = torch.tensor([[[1.0, 0.0]]])
    layout = VisualLayout(torch.tensor([[1, 2, 2]]), spatial_merge_size=2)

    got = question_keep_mask(
        documents,
        questions,
        source_hw=(2, 2),
        target_hw=(2, 2),
        layout=layout,
        threshold=0.25,
        sigma=1.0,
        retention="any",
    )

    assert got.shape == (1,)
    assert got.item() is True


@pytest.mark.parametrize("sigma", [0.0, -1.0, math.inf])
def test_gaussian_sigma_must_be_finite_and_positive(sigma: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        gaussian_smooth_2d(torch.zeros((1, 3, 3)), sigma=sigma)
