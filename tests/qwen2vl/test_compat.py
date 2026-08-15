from types import SimpleNamespace

import pytest

from docprune.qwen2vl.compat import QwenCompatibilityError, assert_supported_qwen2vl


def test_installed_tiny_qwen_has_supported_contract(tiny_qwen2vl) -> None:
    compatibility = assert_supported_qwen2vl(tiny_qwen2vl)

    assert compatibility.transformers_version == "4.46.3"
    assert compatibility.spatial_merge_size == 2
    assert compatibility.decoder_layers == 4


def test_missing_visual_merger_is_named_before_generation() -> None:
    fake = SimpleNamespace(
        config=SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2)),
        visual=SimpleNamespace(patch_embed=object(), blocks=[]),
        model=SimpleNamespace(layers=[]),
    )

    with pytest.raises(QwenCompatibilityError, match="visual.merger"):
        assert_supported_qwen2vl(fake)
