"""Fail-fast checks for the pinned ColPali/PaliGemma implementation."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any

SUPPORTED_COLPALI_ENGINE_VERSION = "0.3.1"
SUPPORTED_TRANSFORMERS_VERSION = "4.46.3"
IMAGE_SIZE = 448
PATCH_SIZE = 14
GRID_HW = (32, 32)
IMAGE_SEQ_LENGTH = 1024
PROJECTION_DIM = 128


class ColPaliCompatibilityError(RuntimeError):
    """The supplied objects cannot safely run the version-specific adapter."""


@dataclass(frozen=True)
class ColPaliCompatibility:
    colpali_engine_version: str
    transformers_version: str
    image_size: int
    patch_size: int
    grid_hw: tuple[int, int]
    projection_dim: int
    image_token_id: int


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError as error:
        raise ColPaliCompatibilityError(f"{name} is not installed") from error


def _require_path(root: object, path: str) -> Any:
    value: Any = root
    traversed: list[str] = []
    for component in path.split("."):
        traversed.append(component)
        if not hasattr(value, component):
            raise ColPaliCompatibilityError(f"ColPali compatibility requires {'.'.join(traversed)}")
        value = getattr(value, component)
    return value


def _processor_image_size(processor: object) -> tuple[int, int] | None:
    size = getattr(getattr(processor, "image_processor", None), "size", None)
    if isinstance(size, dict) and "height" in size and "width" in size:
        return int(size["height"]), int(size["width"])
    return None


def assert_supported_colpali(model: object, processor: object) -> ColPaliCompatibility:
    """Require the exact package types, component paths, and 448/14 geometry."""

    engine_version = _package_version("colpali-engine")
    transformers_version = _package_version("transformers")
    if engine_version != SUPPORTED_COLPALI_ENGINE_VERSION:
        raise ColPaliCompatibilityError(
            f"colpali-engine {SUPPORTED_COLPALI_ENGINE_VERSION} is required, got {engine_version}"
        )
    if transformers_version != SUPPORTED_TRANSFORMERS_VERSION:
        raise ColPaliCompatibilityError(
            f"transformers {SUPPORTED_TRANSFORMERS_VERSION} is required, got {transformers_version}"
        )

    from colpali_engine.models import ColPali, ColPaliProcessor
    from transformers import SiglipImageProcessor
    from transformers.models.paligemma.modeling_paligemma import (
        PaliGemmaForConditionalGeneration,
        PaliGemmaMultiModalProjector,
    )
    from transformers.models.siglip.modeling_siglip import SiglipVisionModel

    if type(model) is not ColPali:
        raise ColPaliCompatibilityError("compatibility requires the exact ColPali wrapper")
    if type(processor) is not ColPaliProcessor:
        raise ColPaliCompatibilityError("compatibility requires the exact ColPaliProcessor")
    if type(getattr(processor, "image_processor", None)) is not SiglipImageProcessor:
        raise ColPaliCompatibilityError("ColPaliProcessor must use the exact SiglipImageProcessor")

    paligemma = _require_path(model, "model")
    vision_tower = _require_path(model, "model.vision_tower")
    projector = _require_path(model, "model.multi_modal_projector")
    _require_path(model, "model.language_model")
    patch_embedding = _require_path(
        model, "model.vision_tower.vision_model.embeddings.patch_embedding"
    )
    position_embedding = _require_path(
        model, "model.vision_tower.vision_model.embeddings.position_embedding"
    )
    _require_path(model, "model.vision_tower.vision_model.encoder")
    _require_path(model, "model.vision_tower.vision_model.post_layernorm")
    text_projection = _require_path(model, "custom_text_proj")

    if type(paligemma) is not PaliGemmaForConditionalGeneration:
        raise ColPaliCompatibilityError("model.model must be the exact PaliGemma component")
    if type(vision_tower) is not SiglipVisionModel:
        raise ColPaliCompatibilityError(
            "model.model.vision_tower must be the exact SigLIP component"
        )
    if type(projector) is not PaliGemmaMultiModalProjector:
        raise ColPaliCompatibilityError("model.model.multi_modal_projector has an unsupported type")

    vision_config = _require_path(model, "model.config.vision_config")
    image_size = int(getattr(vision_config, "image_size", -1))
    patch_size = int(getattr(vision_config, "patch_size", -1))
    if image_size != IMAGE_SIZE or _processor_image_size(processor) != (IMAGE_SIZE, IMAGE_SIZE):
        raise ColPaliCompatibilityError("ColPali compatibility requires 448-by-448 image input")
    if patch_size != PATCH_SIZE or tuple(patch_embedding.kernel_size) != (PATCH_SIZE, PATCH_SIZE):
        raise ColPaliCompatibilityError("ColPali compatibility requires 14-pixel SigLIP patches")
    if int(getattr(processor, "image_seq_length", -1)) != IMAGE_SEQ_LENGTH:
        raise ColPaliCompatibilityError("ColPali compatibility requires 1024 image placeholders")
    if int(position_embedding.num_embeddings) != IMAGE_SEQ_LENGTH:
        raise ColPaliCompatibilityError("SigLIP must expose 1024 original raster positions")
    if int(getattr(text_projection, "out_features", -1)) != PROJECTION_DIM:
        raise ColPaliCompatibilityError("ColPali projected output width must equal 128")

    return ColPaliCompatibility(
        colpali_engine_version=engine_version,
        transformers_version=transformers_version,
        image_size=image_size,
        patch_size=patch_size,
        grid_hw=GRID_HW,
        projection_dim=PROJECTION_DIM,
        image_token_id=int(_require_path(model, "model.config.image_token_index")),
    )
