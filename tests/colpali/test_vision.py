from __future__ import annotations

import torch
from transformers import SiglipVisionConfig, SiglipVisionModel

from docprune.colpali.vision import sparse_siglip_features


def make_vision_tower(*, layers: int = 1) -> SiglipVisionModel:
    torch.manual_seed(11)
    config = SiglipVisionConfig(
        image_size=4,
        patch_size=2,
        num_channels=3,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=layers,
        num_attention_heads=2,
        vision_use_head=False,
        hidden_act="gelu_pytorch_tanh",
        _attn_implementation="eager",
    )
    return SiglipVisionModel(config).eval()


def test_all_true_sparse_siglip_matches_stock_vision_tower() -> None:
    tower = make_vision_tower()
    pixels = torch.arange(48, dtype=torch.float32).reshape(1, 3, 4, 4) / 47

    with torch.no_grad():
        stock = tower(pixels).last_hidden_state
        sparse = sparse_siglip_features(tower, pixels, torch.ones(4, dtype=torch.bool))

    torch.testing.assert_close(sparse, stock, rtol=0, atol=0)


def test_sparse_siglip_selects_original_position_embeddings_in_raster_order() -> None:
    tower = make_vision_tower(layers=0)
    pixels = torch.arange(48, dtype=torch.float32).reshape(1, 3, 4, 4) / 47
    keep = torch.tensor([False, True, False, True])
    vision_model = tower.vision_model

    with torch.no_grad():
        full_embeddings = vision_model.embeddings(pixels)
        expected = vision_model.post_layernorm(full_embeddings[:, [1, 3]])
        sparse = sparse_siglip_features(tower, pixels, keep)

    assert sparse.shape == (1, 2, 8)
    torch.testing.assert_close(sparse, expected, rtol=0, atol=0)


def test_sparse_siglip_rejects_empty_or_wrong_length_masks() -> None:
    tower = make_vision_tower()
    pixels = torch.zeros(1, 3, 4, 4)

    for keep in (torch.zeros(4, dtype=torch.bool), torch.ones(3, dtype=torch.bool)):
        try:
            sparse_siglip_features(tower, pixels, keep)
        except ValueError as error:
            assert "patch_keep_mask" in str(error)
        else:
            raise AssertionError("invalid sparse masks must fail closed")
