import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "examples" / "m3docvqa"))
from gate_dtype import prepare_stock_colpali_batch  # noqa: E402


def _fake_colpali(*, dtype: torch.dtype = torch.bfloat16, with_parameter: bool = True):
    vision_tower = torch.nn.Linear(2, 2)
    if with_parameter:
        vision_tower = vision_tower.to(dtype=dtype)
    else:
        vision_tower = torch.nn.Module()
    return SimpleNamespace(model=SimpleNamespace(vision_tower=vision_tower))


def _fake_colpali_with_integer_parameter():
    vision_tower = torch.nn.Module()
    vision_tower.register_parameter(
        "weight",
        torch.nn.Parameter(torch.ones((2, 2), dtype=torch.int64), requires_grad=False),
    )
    return SimpleNamespace(model=SimpleNamespace(vision_tower=vision_tower))


def test_stock_batch_casts_only_pixels_to_exact_vision_parameter_spec() -> None:
    model = _fake_colpali()
    batch = {
        "pixel_values": torch.ones((1, 3, 2, 2), dtype=torch.float32),
        "input_ids": torch.tensor([[11, 127, 23]], dtype=torch.int64),
        "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.int32),
    }

    prepared = prepare_stock_colpali_batch(model, batch)

    parameter = next(model.model.vision_tower.parameters())
    assert prepared["pixel_values"].device == parameter.device
    assert prepared["pixel_values"].dtype == parameter.dtype
    assert prepared["input_ids"].dtype == batch["input_ids"].dtype
    assert prepared["attention_mask"].dtype == batch["attention_mask"].dtype
    torch.testing.assert_close(prepared["input_ids"], batch["input_ids"])
    torch.testing.assert_close(prepared["attention_mask"], batch["attention_mask"])
    assert batch["pixel_values"].dtype == torch.float32


@pytest.mark.parametrize(
    "model",
    [
        _fake_colpali(with_parameter=False),
        _fake_colpali_with_integer_parameter(),
    ],
)
def test_stock_batch_fails_closed_without_floating_vision_parameters(model) -> None:
    with pytest.raises(ValueError, match="vision parameters"):
        prepare_stock_colpali_batch(
            model,
            {"pixel_values": torch.ones((1, 3, 2, 2), dtype=torch.float32)},
        )
