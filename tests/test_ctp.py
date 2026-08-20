import pytest
import torch

from docprune.ctp import (
    ComprehensionController,
    CTPDecision,
    ctp_keep_indices,
    visual_attention_scores,
)


def test_trigger_selects_only_the_first_crossing() -> None:
    controller = ComprehensionController(threshold=5.0)

    assert controller.observe(3, torch.tensor([[3.0, 0.0]])) is False
    assert controller.observe(4, torch.tensor([[3.0, 4.0]])) is True
    assert controller.observe(5, torch.tensor([[6.0, 8.0]])) is False
    assert controller.selected_layer == 4
    assert controller.selected_norm == 5.0


def test_no_threshold_crossing_leaves_layer_unset() -> None:
    controller = ComprehensionController(threshold=5.0)

    assert controller.observe(1, torch.tensor([[1.0, 2.0]])) is False
    assert controller.observe(2, torch.tensor([[2.0, 2.0]])) is False
    assert controller.selected_layer is None
    assert controller.selected_norm is None


def test_comprehension_requires_batch_size_one() -> None:
    controller = ComprehensionController(threshold=5.0)

    with pytest.raises(ValueError, match="batch size one"):
        controller.observe(1, torch.ones((2, 4)))


def test_visual_attention_scores_average_heads_and_scales_by_visual_count() -> None:
    attention = torch.tensor(
        [
            [
                [[0.0, 0.2, 0.0, 0.8, 0.0]],
                [[0.0, 0.4, 0.0, 0.2, 0.0]],
            ]
        ]
    )

    got = visual_attention_scores(attention, torch.tensor([1, 3]), head_aggregation="mean")

    assert torch.allclose(got, torch.tensor([0.6, 1.0]), atol=1e-7)


def test_visual_attention_scores_maximizes_heads_and_scales_by_visual_count() -> None:
    attention = torch.tensor(
        [
            [
                [[0.0, 0.2, 0.0, 0.8, 0.0]],
                [[0.0, 0.4, 0.0, 0.2, 0.0]],
            ]
        ]
    )

    got = visual_attention_scores(attention, torch.tensor([1, 3]), head_aggregation="max")

    assert torch.allclose(got, torch.tensor([0.8, 1.6]))


def test_visual_attention_scores_preserves_empty_visual_vector() -> None:
    attention = torch.tensor(
        [
            [
                [[0.0, 0.2, 0.0, 0.8, 0.0]],
                [[0.0, 0.4, 0.0, 0.2, 0.0]],
            ]
        ]
    )

    got = visual_attention_scores(attention, torch.tensor([], dtype=torch.long))

    assert got.shape == (0,)
    assert got.dtype == attention.dtype


def test_ctp_retains_nonvisual_tokens_and_attention_threshold_equality() -> None:
    got = ctp_keep_indices(
        token_count=6,
        visual_indices=torch.tensor([1, 3, 4]),
        visual_scores=torch.tensor([0.3, 0.2, 0.5]),
        threshold=0.3,
    )

    assert got.tolist() == [0, 1, 2, 4, 5]


def test_ctp_preserves_original_order() -> None:
    got = ctp_keep_indices(
        token_count=7,
        visual_indices=torch.tensor([5, 1, 3]),
        visual_scores=torch.tensor([0.9, 0.1, 0.8]),
        threshold=0.5,
    )

    assert got.tolist() == [0, 2, 3, 4, 5, 6]


def test_ctp_decision_is_json_serializable() -> None:
    decision = CTPDecision(
        layer_index=20,
        comprehension_norm=65.5,
        original_visual_tokens=4,
        retained_visual_indices=(2, 4),
    )

    assert decision.to_dict() == {
        "layer_index": 20,
        "comprehension_norm": 65.5,
        "original_visual_tokens": 4,
        "retained_visual_tokens": 2,
        "retained_visual_indices": [2, 4],
    }


def test_visual_indices_must_be_unique_and_in_range() -> None:
    with pytest.raises(ValueError, match="unique"):
        ctp_keep_indices(
            token_count=5,
            visual_indices=torch.tensor([1, 1]),
            visual_scores=torch.tensor([0.5, 0.5]),
            threshold=0.3,
        )
    with pytest.raises(ValueError, match="range"):
        ctp_keep_indices(
            token_count=5,
            visual_indices=torch.tensor([5]),
            visual_scores=torch.tensor([0.5]),
            threshold=0.3,
        )
