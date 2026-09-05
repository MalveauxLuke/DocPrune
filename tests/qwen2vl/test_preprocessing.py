from __future__ import annotations

import torch
from PIL import Image

from docprune.qwen2vl.preprocessing import prepare_qwen_page


def test_prepare_qwen_page_uses_processor_grid_and_page_order() -> None:
    from transformers import Qwen2VLImageProcessor

    processor = Qwen2VLImageProcessor(min_pixels=56 * 56, max_pixels=28 * 28 * 1280)
    image = Image.new("RGB", (112, 56), color=(12, 34, 56))

    prepared = prepare_qwen_page(processor, image)

    assert prepared.image_grid_thw.tolist() == [[1, 4, 8]]
    assert prepared.pixel_values.shape == (32, 3 * 2 * 14 * 14)
    assert prepared.merge_size == 2
    assert prepared.placeholder_count == 8
    assert prepared.placeholder_raster_indices.tolist() == list(range(8))


def test_prepare_qwen_page_rejects_non_merge_safe_processor_grid() -> None:
    class BrokenImageProcessor:
        patch_size = 14
        temporal_patch_size = 2
        merge_size = 2

        def __call__(self, **_: object) -> dict[str, torch.Tensor]:
            return {
                "pixel_values": torch.zeros((6, 3 * 2 * 14 * 14)),
                "image_grid_thw": torch.tensor([[1, 2, 3]]),
            }

    class BrokenProcessor:
        image_processor = BrokenImageProcessor()

    try:
        prepare_qwen_page(BrokenProcessor(), Image.new("RGB", (56, 56)))
    except ValueError as error:
        assert "merge" in str(error)
    else:
        raise AssertionError("a non-merge-safe grid must fail closed")
