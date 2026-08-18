from __future__ import annotations

import pytest
from colpali_engine.models import ColPali, ColPaliProcessor
from transformers import PaliGemmaConfig, SiglipImageProcessor


def make_tiny_colpali(*, image_size: int = 28, patch_size: int = 14) -> ColPali:
    config = PaliGemmaConfig(
        vision_config={
            "image_size": image_size,
            "patch_size": patch_size,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "projection_dim": 16,
            "vision_use_head": False,
            "hidden_act": "gelu_pytorch_tanh",
        },
        text_config={
            "vocab_size": 128,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "hidden_activation": "gelu_pytorch_tanh",
        },
        hidden_size=16,
        projection_dim=16,
        image_token_index=127,
        vocab_size=128,
    )
    return ColPali(config).eval()


def make_processor(*, image_size: int, image_seq_length: int) -> ColPaliProcessor:
    processor = object.__new__(ColPaliProcessor)
    processor.image_processor = SiglipImageProcessor(
        size={"height": image_size, "width": image_size}
    )
    processor.image_seq_length = image_seq_length
    return processor


@pytest.fixture
def tiny_colpali() -> ColPali:
    return make_tiny_colpali()
