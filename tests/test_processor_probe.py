from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from docprune.processor_probe import (
    collect_processor_contract,
    require_immutable_revision,
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
        colpali_model="vidore/colpali-v1",
        colpali_revision=REVISION_B,
    )

    assert got["schema_version"] == 1
    assert got["resources"]["qwen"]["revision"] == REVISION_A
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
    assert got["mapping_checks"] == {
        "colpali_visual_grid_inferred": True,
        "qwen_merge_groups_valid": True,
        "raster_order_verified": False,
    }
    serialized = str(got)
    assert "17, 23, 41" not in serialized
    assert "input_ids" not in serialized


def test_revision_must_be_full_commit_hash() -> None:
    with pytest.raises(ValueError, match="40-character"):
        require_immutable_revision("main", name="qwen_revision")


def test_writer_refuses_to_replace_existing_report(tmp_path) -> None:
    path = tmp_path / "processor-contract.json"
    path.write_text("existing\n")

    with pytest.raises(FileExistsError, match="already exists"):
        write_processor_contract(path, {"schema_version": 1})
