import torch

from docprune.qwen2vl.decoder import (
    _causal_mask,
    _last_query_attention,
    _prefill_attention_mask,
    decode_one_token,
    prefill_with_ctp,
)


def test_flash_prefill_uses_unpadded_varlen_mask_contract(tiny_qwen2vl) -> None:
    tiny_qwen2vl.model.config._attn_implementation = "flash_attention_2"

    assert (
        _prefill_attention_mask(tiny_qwen2vl.model, 6, torch.float32, torch.device("cpu")) is None
    )


def test_flash_ctp_recomputes_final_query_without_square_causal_mask(tiny_qwen2vl, monkeypatch) -> None:
    tiny_qwen2vl.model.config._attn_implementation = "flash_attention_2"

    def fail_if_square_mask(*args, **kwargs):
        raise AssertionError("FlashAttention CTP must not allocate a square causal mask")

    monkeypatch.setattr("docprune.qwen2vl.decoder._causal_mask", fail_if_square_mask)
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

    assert got.decision is not None


def test_ctp_attention_matches_literal_final_query_full_key_rotary_fixture(tiny_qwen2vl) -> None:
    from transformers.models.qwen2_vl.modeling_qwen2_vl import (
        apply_multimodal_rotary_pos_emb,
        repeat_kv,
    )

    decoder = tiny_qwen2vl.model
    input_ids = torch.tensor([[10, 102, 100, 100, 103, 11]])
    hidden = decoder.embed_tokens(input_ids)
    positions = torch.arange(6).view(1, 1, 6).expand(3, 1, 6)
    layer = decoder.layers[0]
    position_embeddings = decoder.rotary_emb(hidden, positions)
    attention = layer.self_attn
    normalized = layer.input_layernorm(hidden)
    query = attention.q_proj(normalized)
    key = attention.k_proj(normalized)
    query = query.view(1, 6, attention.num_heads, attention.head_dim).transpose(1, 2)
    key = key.view(1, 6, attention.num_key_value_heads, attention.head_dim).transpose(1, 2)
    cosine, sine = position_embeddings
    query, key = apply_multimodal_rotary_pos_emb(
        query,
        key,
        cosine,
        sine,
        attention.rope_scaling["mrope_section"],
    )
    key = repeat_kv(key, attention.num_key_value_groups)
    literal = torch.softmax(
        torch.matmul(query[:, :, -1:, :], key.transpose(2, 3))
        / attention.head_dim**0.5
        + _causal_mask(6, dtype=hidden.dtype, device=hidden.device)[:, :, -1:, :],
        dim=-1,
        dtype=torch.float32,
    ).to(query.dtype)

    got = _last_query_attention(
        layer,
        hidden,
        position_embeddings,
        _causal_mask(6, dtype=hidden.dtype, device=hidden.device),
    )

    torch.testing.assert_close(got, literal, rtol=1e-5, atol=1e-5)


def test_eager_prefill_uses_explicit_causal_mask(tiny_qwen2vl) -> None:
    mask = _prefill_attention_mask(tiny_qwen2vl.model, 3, torch.float32, torch.device("cpu"))

    assert mask.shape == (1, 1, 3, 3)
    assert mask[0, 0, 2].tolist() == [0.0, 0.0, 0.0]


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


def test_ctp_decode_keeps_trigger_layer_full_and_compacts_deeper_caches(tiny_qwen2vl) -> None:
    input_ids = torch.tensor([[10, 102, 100, 100, 103, 11]])
    hidden = tiny_qwen2vl.model.embed_tokens(input_ids)
    positions = torch.arange(6).view(1, 1, 6).expand(3, 1, 6)

    with torch.no_grad():
        prefill = prefill_with_ctp(
            tiny_qwen2vl.model,
            hidden,
            positions,
            visual_indices=torch.tensor([2, 3]),
            comprehension_threshold=0.0,
            attention_threshold=1.0,
        )
        token = tiny_qwen2vl.model.embed_tokens(torch.tensor([[12]]))
        decode_one_token(
            tiny_qwen2vl.model,
            token,
            torch.full((3, 1, 1), 6, dtype=torch.long),
            prefill.cache,
        )

    assert tuple(cache.shape[-2] for cache in prefill.cache.key_cache) == (7, 5, 5, 5)


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
