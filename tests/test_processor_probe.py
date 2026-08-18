from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from docprune.processor_probe import (
    collect_processor_contract,
    require_immutable_revision,
    resolve_colpali_visual_mapping,
    validate_processor_contract,
    write_processor_contract,
)

REVISION_A = "a" * 40
REVISION_B = "b" * 40


class FakeQwenProcessor:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert messages[0]["content"][0]["type"] == "image"
        assert tokenize is False
        assert add_generation_prompt is True
        return "qwen probe"

    def __call__(self, *, text, images, padding, return_tensors):
        assert text == ["qwen probe"]
        assert len(images) == 1
        assert padding is True
        assert return_tensors == "pt"
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "pixel_values": torch.zeros((16, 24)),
            "image_grid_thw": torch.tensor([[1, 4, 4]]),
        }


class FakeColPaliProcessor:
    image_token_id = 99
    image_seq_length = 9

    def process_images(self, images):
        assert len(images) == 1
        return {
            "input_ids": torch.tensor([[1] + [99] * 9 + [2]]),
            "attention_mask": torch.ones((1, 11), dtype=torch.long),
            "pixel_values": torch.zeros((1, 3, 32, 32)),
        }


def test_contract_contains_shapes_and_layout_but_no_raw_content() -> None:
    image = Image.new("RGB", (100, 80), color=(17, 23, 41))
    qwen_config = SimpleNamespace(
        vision_config=SimpleNamespace(
            patch_size=14,
            temporal_patch_size=2,
            spatial_merge_size=2,
        )
    )

    got = collect_processor_contract(
        image=image,
        qwen_processor=FakeQwenProcessor(),
        qwen_config=qwen_config,
        qwen_model="Qwen/Qwen2-VL-7B-Instruct",
        qwen_revision=REVISION_A,
        colpali_processor=FakeColPaliProcessor(),
        colpali_model="vidore/colpali-v1.2",
        colpali_revision=REVISION_B,
    )

    assert got["schema_version"] == 2
    assert got["resources"]["qwen"]["revision"] == REVISION_A
    assert got["resources"]["colpali"] == {
        "model": "vidore/colpali-v1.2",
        "revision": REVISION_B,
    }
    assert got["resources"]["colpali_backbone"] == {
        "model": "vidore/colpaligemma-3b-pt-448-base",
        "revision": "30ab955d073de4a91dc5a288e8c97226647e3e5a",
    }
    assert got["page"]["raw_size_wh"] == [100, 80]
    assert got["qwen"] == {
        "grid_thw": [1, 4, 4],
        "merged_visual_token_count": 4,
        "patch_size": 14,
        "pixel_values_shape": [16, 24],
        "resized_size_hw": [56, 56],
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
    }
    assert got["colpali"]["image_token_id"] == 99
    assert got["colpali"]["image_token_positions"] == list(range(1, 10))
    assert got["colpali"]["candidate_visual_token_count"] == 9
    assert got["colpali"]["inferred_visual_grid_hw"] == [3, 3]
    assert got["colpali"]["raster_indices"] == list(range(9))
    assert got["colpali"]["visual_start"] == 1
    assert got["colpali"]["visual_stop"] == 10
    assert got["mapping_checks"] == {
        "colpali_visual_grid_inferred": True,
        "qwen_merge_groups_valid": True,
        "raster_order_verified": True,
    }
    serialized = str(got)
    assert "17, 23, 41" not in serialized
    assert "input_ids" not in serialized


def test_colpali_mapping_proves_contiguous_row_major_visual_span() -> None:
    mapping = resolve_colpali_visual_mapping(
        input_ids=torch.tensor([[7, 42, 42, 42, 42, 9]]),
        attention_mask=torch.ones(1, 6),
        image_token_id=42,
        image_seq_length=4,
    )

    assert mapping.image_token_id == 42
    assert mapping.visual_start == 1
    assert mapping.visual_stop == 5
    assert mapping.grid_hw == (2, 2)
    assert mapping.raster_indices == (0, 1, 2, 3)


@pytest.mark.parametrize(
    ("input_ids", "attention_mask", "message"),
    [
        (torch.tensor([[7, 42, 42, 9]]), torch.ones(1, 4), "exactly 4"),
        (torch.tensor([[7, 42, 42, 42, 42, 42, 9]]), torch.ones(1, 7), "exactly 4"),
        (torch.tensor([[7, 42, 9, 42, 42, 42]]), torch.ones(1, 6), "contiguous"),
        (torch.tensor([[7, 42, 42, 42, 42, 9]]), torch.tensor([[1, 1, 1, 1, 0, 0]]), "padded"),
    ],
)
def test_colpali_mapping_rejects_invalid_visual_positions(
    input_ids: torch.Tensor, attention_mask: torch.Tensor, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_colpali_visual_mapping(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_token_id=42,
            image_seq_length=4,
        )


def test_colpali_mapping_rejects_nonsquare_visual_grid() -> None:
    with pytest.raises(ValueError, match="square"):
        resolve_colpali_visual_mapping(
            input_ids=torch.tensor([[7, 42, 42, 42, 9]]),
            attention_mask=torch.ones(1, 5),
            image_token_id=42,
            image_seq_length=3,
        )


def test_revision_must_be_full_commit_hash() -> None:
    with pytest.raises(ValueError, match="40-character"):
        require_immutable_revision("main", name="qwen_revision")


def test_writer_refuses_to_replace_existing_report(tmp_path) -> None:
    path = tmp_path / "processor-contract.json"
    path.write_text("existing\n")

    with pytest.raises(FileExistsError, match="already exists"):
        write_processor_contract(path, {"schema_version": 1})


def test_structural_failure_is_rejected_after_report_can_be_serialized() -> None:
    payload = {
        "mapping_checks": {
            "colpali_visual_grid_inferred": False,
            "qwen_merge_groups_valid": True,
            "raster_order_verified": False,
        },
        "unresolved": ["ColPali image token ID could not be detected."],
    }

    with pytest.raises(ValueError, match="ColPali image token ID"):
        validate_processor_contract(payload)


def test_raster_order_failure_is_rejected_after_report_can_be_serialized() -> None:
    payload = {
        "mapping_checks": {
            "colpali_visual_grid_inferred": True,
            "qwen_merge_groups_valid": True,
            "raster_order_verified": False,
        },
        "unresolved": ["ColPali visual positions are padded."],
    }

    with pytest.raises(ValueError, match="raster_order_verified"):
        validate_processor_contract(payload)
