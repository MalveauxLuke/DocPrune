from __future__ import annotations

import pytest
from colpali_engine.models import ColPali

from docprune.colpali.compat import ColPaliCompatibilityError, assert_supported_colpali

from .conftest import make_processor, make_tiny_colpali


def test_supported_contract_requires_pinned_versions_geometry_and_component_paths() -> None:
    model = make_tiny_colpali(image_size=448, patch_size=14)
    processor = make_processor(image_size=448, image_seq_length=1024)

    compatibility = assert_supported_colpali(model, processor)

    assert compatibility.colpali_engine_version == "0.3.1"
    assert compatibility.transformers_version == "4.46.3"
    assert compatibility.image_size == 448
    assert compatibility.patch_size == 14
    assert compatibility.grid_hw == (32, 32)
    assert compatibility.projection_dim == 128
    assert compatibility.image_token_id == 127
    assert model.model.vision_tower.vision_model.embeddings.patch_embedding.kernel_size == (
        14,
        14,
    )
    assert (
        model.model.vision_tower.vision_model.embeddings.position_embedding.num_embeddings == 1024
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda model, processor: setattr(model.model.config.vision_config, "image_size", 224),
            "448",
        ),
        (
            lambda model, processor: setattr(model.model.config.vision_config, "patch_size", 16),
            "14",
        ),
        (lambda model, processor: setattr(processor, "image_seq_length", 256), "1024"),
        (lambda model, processor: setattr(model.custom_text_proj, "out_features", 127), "128"),
    ],
)
def test_supported_contract_rejects_geometry_or_projection_drift(mutation, message: str) -> None:
    model = make_tiny_colpali(image_size=448, patch_size=14)
    processor = make_processor(image_size=448, image_seq_length=1024)
    mutation(model, processor)

    with pytest.raises(ColPaliCompatibilityError, match=message):
        assert_supported_colpali(model, processor)


def test_supported_contract_rejects_non_colpali_wrapper() -> None:
    model = make_tiny_colpali(image_size=448, patch_size=14)
    processor = make_processor(image_size=448, image_seq_length=1024)

    with pytest.raises(ColPaliCompatibilityError, match="exact ColPali"):
        assert_supported_colpali(model.model, processor)

    assert type(model) is ColPali
