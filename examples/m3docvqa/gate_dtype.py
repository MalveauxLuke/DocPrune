"""Input preparation for the gate's independent stock ColPali check."""

from __future__ import annotations

from collections.abc import Mapping

import torch


def _vision_parameter(model: object) -> torch.Tensor:
    model_core = getattr(model, "model", None)
    vision_tower = getattr(model_core, "vision_tower", None)
    parameters = getattr(vision_tower, "parameters", None)
    if not callable(parameters):
        raise ValueError("ColPali model has no vision parameters")
    try:
        parameter = next(parameters())
    except StopIteration as error:
        raise ValueError("ColPali model has no vision parameters") from error
    if not parameter.dtype.is_floating_point:
        raise ValueError("ColPali model requires floating vision parameters")
    return parameter


def prepare_stock_colpali_batch(
    model: object, batch: Mapping[str, object]
) -> dict[str, object]:
    """Match processor pixels to vision weights without changing token dtypes."""

    if not isinstance(batch, Mapping) or "pixel_values" not in batch:
        raise ValueError("ColPali batch is missing pixel_values")
    parameter = _vision_parameter(model)
    prepared = dict(batch)
    prepared["pixel_values"] = torch.as_tensor(batch["pixel_values"]).to(
        device=parameter.device,
        dtype=parameter.dtype,
    )
    return prepared
