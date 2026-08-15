import torch

from docprune.qwen2vl.decoder import prefill_with_ctp


def test_ctp_reduces_only_deeper_layer_cache_lengths(tiny_qwen2vl) -> None:
    input_ids = torch.tensor([[10, 102, 100, 100, 103, 11]])
    hidden = tiny_qwen2vl.model.embed_tokens(input_ids)
    positions = torch.arange(6).view(1, 1, 6).expand(3, 1, 6)

    with torch.no_grad():
        got = prefill_with_ctp(
            tiny_qwen2vl.model,
            hidden,
            positions,
            visual_indices=torch.tensor([2, 3]),
            comprehension_threshold=0.0,
            attention_threshold=1.0,
        )

    cache_lengths = tuple(cache.shape[-2] for cache in got.cache.key_cache)
    assert got.decision.layer_index == 0
    assert got.decision.original_visual_tokens == 2
    assert got.decision.retained_visual_indices == ()
    assert got.hidden_states.shape == (1, 4, 32)
    assert cache_lengths == (6, 4, 4, 4)
    assert got.keep_indices.tolist() == [0, 1, 4, 5]


def test_no_crossing_preserves_full_sequence_for_every_layer(tiny_qwen2vl) -> None:
    input_ids = torch.tensor([[10, 102, 100, 100, 103, 11]])
    hidden = tiny_qwen2vl.model.embed_tokens(input_ids)
    positions = torch.arange(6).view(1, 1, 6).expand(3, 1, 6)

    with torch.no_grad():
        got = prefill_with_ctp(
            tiny_qwen2vl.model,
            hidden,
            positions,
            visual_indices=torch.tensor([2, 3]),
            comprehension_threshold=1e9,
            attention_threshold=1.0,
        )

    assert got.decision is None
    assert got.hidden_states.shape[1] == 6
    assert tuple(cache.shape[-2] for cache in got.cache.key_cache) == (6, 6, 6, 6)
