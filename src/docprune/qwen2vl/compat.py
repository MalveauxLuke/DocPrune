"""Fail-fast structural checks for the pinned Qwen2-VL implementation."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any

SUPPORTED_TRANSFORMERS_VERSION = "4.46.3"
SUPPORTED_QWEN25_TRANSFORMERS_VERSION = "4.49.0"


class QwenCompatibilityError(RuntimeError):
    """The supplied model cannot safely run the version-specific adapter."""


@dataclass(frozen=True)
class QwenCompatibility:
    transformers_version: str
    spatial_merge_size: int
    decoder_layers: int


def _require_path(root: object, path: str) -> Any:
    value: Any = root
    traversed: list[str] = []
    for component in path.split("."):
        traversed.append(component)
        if not hasattr(value, component):
            raise QwenCompatibilityError(f"Qwen2-VL compatibility requires {'.'.join(traversed)}")
        value = getattr(value, component)
    return value


def assert_supported_qwen2vl(model: object) -> QwenCompatibility:
    try:
        installed_version = version("transformers")
    except PackageNotFoundError as exc:
        raise QwenCompatibilityError("transformers is not installed") from exc
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    expected_version = (
        SUPPORTED_QWEN25_TRANSFORMERS_VERSION
        if model_type == "qwen2_5_vl"
        else SUPPORTED_TRANSFORMERS_VERSION
    )
    if installed_version != expected_version:
        raise QwenCompatibilityError(
            f"transformers {expected_version} is required for {model_type or 'Qwen2-VL'}, "
            f"got {installed_version}"
        )
    _require_path(model, "config.vision_config.spatial_merge_size")
    _require_path(model, "visual.patch_embed")
    _require_path(model, "visual.blocks")
    _require_path(model, "visual.merger")
    _require_path(model, "visual.rot_pos_emb")
    layers = _require_path(model, "model.layers")
    _require_path(model, "model.rotary_emb")
    _require_path(model, "model.norm")
    _require_path(model, "model.embed_tokens")
    _require_path(model, "lm_head")
    if not layers:
        raise QwenCompatibilityError("Qwen2-VL compatibility requires at least one model.layers entry")
    merge_size = int(model.config.vision_config.spatial_merge_size)
    if merge_size != 2:
        raise QwenCompatibilityError(f"paper reproduction requires spatial_merge_size=2, got {merge_size}")
    return QwenCompatibility(installed_version, merge_size, len(layers))
