import torch

from docprune.qwen2vl.sequence import compact_multimodal_sequence


def test_only_rejected_image_placeholders_are_removed() -> None:
    input_ids = torch.tensor([[10, 102, 100, 100, 100, 100, 103, 11]])
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(8).view(1, 1, 8).expand(3, 1, 8).clone()

    got = compact_multimodal_sequence(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        image_token_id=100,
        group_keep_mask=torch.tensor([True, False, False, True]),
    )

    assert got.input_ids.tolist() == [[10, 102, 100, 100, 103, 11]]
    assert got.attention_mask.tolist() == [[1, 1, 1, 1, 1, 1]]
    assert got.position_ids[0, 0].tolist() == [0, 1, 2, 5, 6, 7]
    assert got.visual_indices.tolist() == [2, 3]
    assert got.original_keep_indices.tolist() == [0, 1, 2, 5, 6, 7]


def test_placeholder_count_must_match_group_mask() -> None:
    input_ids = torch.tensor([[10, 100, 100, 11]])
    positions = torch.arange(4).view(1, 1, 4).expand(3, 1, 4)

    try:
        compact_multimodal_sequence(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            position_ids=positions,
            image_token_id=100,
            group_keep_mask=torch.tensor([True]),
        )
    except ValueError as error:
        assert "placeholders" in str(error)
    else:
        raise AssertionError("mismatched placeholder count should be rejected")
