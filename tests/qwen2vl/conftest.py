import pytest


@pytest.fixture
def tiny_qwen2vl():
    from transformers import Qwen2VLConfig, Qwen2VLForConditionalGeneration

    config = Qwen2VLConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        vision_config={
            "depth": 2,
            "embed_dim": 32,
            "hidden_size": 32,
            "hidden_act": "quick_gelu",
            "mlp_ratio": 2,
            "num_heads": 4,
            "in_channels": 3,
            "patch_size": 2,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        },
        image_token_id=100,
        video_token_id=101,
        vision_start_token_id=102,
        vision_end_token_id=103,
        rope_scaling={"type": "mrope", "mrope_section": [1, 1, 2]},
        _attn_implementation="eager",
    )
    model = Qwen2VLForConditionalGeneration(config).eval()
    return model
