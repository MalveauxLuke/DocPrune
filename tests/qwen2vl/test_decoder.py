import hashlib

import pytest
import torch

from docprune.qwen2vl.decoder import (
    ForcedVisualIntervention,
    _apply_forced_visual_intervention,
    _causal_mask,
    _last_query_attention,
    _prefill_attention_mask,
    capture_forced_boundary_checkpoint,
    decode_one_token,
    prefill_with_ctp,
    resume_forced_boundary_checkpoint,
)


def _fixture_inputs(tiny_qwen2vl):
    input_ids = torch.tensor([[10, 102, 100, 100, 103, 11]])
    hidden = tiny_qwen2vl.model.embed_tokens(input_ids)
    positions = torch.arange(6).view(1, 1, 6).expand(3, 1, 6).clone()
    return hidden, positions


def test_cached_boundary_resume_matches_independent_forced_prefill(tiny_qwen2vl) -> None:
    """Catch changing intervention semantics when Task 9 reuses the full B_K state."""

    hidden, positions = _fixture_inputs(tiny_qwen2vl)
    intervention = ForcedVisualIntervention(1, "physical_delete", (1,))
    with torch.no_grad():
        independent = prefill_with_ctp(
            tiny_qwen2vl.model,
            hidden,
            positions,
            visual_indices=torch.tensor([2, 3]),
            comprehension_threshold=0.0,
            attention_threshold=0.0,
            forced_intervention=intervention,
        )
        checkpoint = capture_forced_boundary_checkpoint(
            tiny_qwen2vl.model,
            hidden,
            positions,
            visual_indices=torch.tensor([2, 3]),
            boundary=1,
        )
        resumed = resume_forced_boundary_checkpoint(
            tiny_qwen2vl.model,
            checkpoint,
            intervention,
        )

    assert checkpoint.boundary == "B_1"
    assert checkpoint.query_aggregate_attention_scores is not None
    assert len(checkpoint.query_aggregate_attention_scores) == 2
    assert all(torch.isfinite(torch.tensor(checkpoint.query_aggregate_attention_scores)))
    assert tuple(item.shape[-2] for item in checkpoint.cache.key_cache) == (6, 6)
    assert resumed.forced == independent.forced
    assert resumed.keep_indices.tolist() == independent.keep_indices.tolist()
    assert torch.equal(resumed.position_ids, independent.position_ids)
    assert torch.allclose(resumed.hidden_states, independent.hidden_states, atol=0, rtol=0)
    assert len(resumed.cache.key_cache) == len(independent.cache.key_cache)
    for cached, expected in zip(resumed.cache.key_cache, independent.cache.key_cache, strict=True):
        assert torch.allclose(cached, expected, atol=0, rtol=0)


def test_cached_boundary_is_reusable_and_rejects_cross_boundary_intervention(tiny_qwen2vl) -> None:
    """Catch one regional branch mutating the shared prefix cache or using another B_K."""

    hidden, positions = _fixture_inputs(tiny_qwen2vl)
    with torch.no_grad():
        checkpoint = capture_forced_boundary_checkpoint(
            tiny_qwen2vl.model,
            hidden,
            positions,
            visual_indices=torch.tensor([2, 3]),
            boundary=1,
        )
        prefix = tuple(
            (key.clone(), value.clone())
            for key, value in zip(
                checkpoint.cache.key_cache, checkpoint.cache.value_cache, strict=True
            )
        )
        first = resume_forced_boundary_checkpoint(
            tiny_qwen2vl.model,
            checkpoint,
            ForcedVisualIntervention(1, "physical_delete", (0,)),
        )
        second = resume_forced_boundary_checkpoint(
            tiny_qwen2vl.model,
            checkpoint,
            ForcedVisualIntervention(1, "physical_delete", (1,)),
        )

    assert first.keep_indices.tolist() == [0, 1, 2, 4, 5]
    assert second.keep_indices.tolist() == [0, 1, 3, 4, 5]
    assert len(checkpoint.cache.key_cache) == 2
    for (key, value), expected_key, expected_value in zip(
        prefix, checkpoint.cache.key_cache, checkpoint.cache.value_cache, strict=True
    ):
        assert torch.equal(key, expected_key)
        assert torch.equal(value, expected_value)
    with pytest.raises(ValueError, match="boundary"):
        resume_forced_boundary_checkpoint(
            tiny_qwen2vl.model,
            checkpoint,
            ForcedVisualIntervention(0, "physical_delete", (0,)),
        )


def test_forced_physical_delete_after_boundary_compacts_only_later_caches(tiny_qwen2vl) -> None:
    """Catch inserting B_K before its named layer or retroactively compacting its cache."""

    hidden, positions = _fixture_inputs(tiny_qwen2vl)
    intervention = ForcedVisualIntervention(
        boundary=1,
        mode="physical_delete",
        retained_visual_ids=(1,),
    )

    with torch.no_grad():
        prefill = prefill_with_ctp(
            tiny_qwen2vl.model,
            hidden,
            positions,
            visual_indices=torch.tensor([2, 3]),
            comprehension_threshold=0.0,
            attention_threshold=1.0,
            forced_intervention=intervention,
        )
        token = tiny_qwen2vl.model.embed_tokens(torch.tensor([[12]]))
        decode_one_token(
            tiny_qwen2vl.model,
            token,
            torch.full((3, 1, 1), 6, dtype=torch.long),
            prefill.cache,
        )

    assert prefill.decision is None
    assert prefill.forced is not None
    assert prefill.forced.boundary == "B_1"
    assert prefill.forced.mode == "physical_delete"
    assert prefill.forced.selection_kind == "forced"
    assert prefill.forced.retained_visual_ids == (1,)
    assert prefill.keep_indices.tolist() == [0, 1, 3, 4, 5]
    assert prefill.forced.prefill_cache_lengths == (6, 6, 5, 5)
    assert tuple(cache.shape[-2] for cache in prefill.cache.key_cache) == (7, 7, 6, 6)


def test_forced_physical_delete_at_block_zero_keeps_only_its_cache_full(tiny_qwen2vl) -> None:
    """Catch a forced B_0 compacting block 0's cache or delaying compaction to B_1."""

    hidden, positions = _fixture_inputs(tiny_qwen2vl)
    with torch.no_grad():
        got = prefill_with_ctp(
            tiny_qwen2vl.model,
            hidden,
            positions,
            visual_indices=torch.tensor([2, 3]),
            comprehension_threshold=0.0,
            attention_threshold=0.0,
            forced_intervention=ForcedVisualIntervention(0, "physical_delete", (0,)),
        )

    assert got.forced is not None
    assert got.keep_indices.tolist() == [0, 1, 2, 4, 5]
    assert got.forced.prefill_cache_lengths == (6, 5, 5, 5)


def test_forced_physical_delete_at_final_tiny_block_leaves_all_caches_full(tiny_qwen2vl) -> None:
    """Catch compacting a cache before the final boundary's block has executed."""

    hidden, positions = _fixture_inputs(tiny_qwen2vl)
    with torch.no_grad():
        got = prefill_with_ctp(
            tiny_qwen2vl.model,
            hidden,
            positions,
            visual_indices=torch.tensor([2, 3]),
            comprehension_threshold=0.0,
            attention_threshold=0.0,
            forced_intervention=ForcedVisualIntervention(3, "physical_delete", ()),
        )

    assert got.forced is not None
    assert got.forced.prefill_cache_lengths == (6, 6, 6, 6)
    assert got.hidden_states.shape[1] == 4


def test_forced_input_physical_delete_preserves_survivor_mrope_and_original_order(
    tiny_qwen2vl,
) -> None:
    """Catch B_input after layer 0 or a gather that reorders/renumbers M-RoPE."""

    hidden, positions = _fixture_inputs(tiny_qwen2vl)
    positions[1, 0] += 10
    positions[2, 0] += 20
    intervention = ForcedVisualIntervention(
        boundary="input",
        mode="physical_delete",
        retained_visual_ids=(1,),
    )

    with torch.no_grad():
        got = prefill_with_ctp(
            tiny_qwen2vl.model,
            hidden,
            positions,
            visual_indices=torch.tensor([2, 3]),
            comprehension_threshold=0.0,
            attention_threshold=1.0,
            forced_intervention=intervention,
        )

    assert got.forced is not None
    assert got.forced.boundary == "B_input"
    assert got.keep_indices.tolist() == [0, 1, 3, 4, 5]
    torch.testing.assert_close(got.position_ids, positions[:, :, [0, 1, 3, 4, 5]])
    assert got.position_ids.dtype == positions.dtype
    assert got.position_ids.device == positions.device
    assert got.forced.prefill_cache_lengths == (5, 5, 5, 5)


def test_forced_zero_mask_zeros_only_unretained_rows_without_compacting(tiny_qwen2vl) -> None:
    """Catch a purported zero mask that changes positions, rows, or cache topology."""

    hidden, positions = _fixture_inputs(tiny_qwen2vl)
    original_keep = torch.arange(hidden.shape[1])
    current_visual = torch.tensor([2, 3])
    intervention = ForcedVisualIntervention(
        boundary=1,
        mode="zero_mask",
        retained_visual_ids=(1,),
    )
    applied = _apply_forced_visual_intervention(
        hidden,
        positions,
        original_keep,
        current_visual,
        intervention,
    )

    assert applied.hidden_states.data_ptr() != hidden.data_ptr()
    assert applied.hidden_states[0, 2].tolist() == [0.0] * hidden.shape[-1]
    torch.testing.assert_close(applied.hidden_states[0, 3], hidden[0, 3])
    torch.testing.assert_close(applied.position_ids, positions)
    assert applied.original_keep_indices.tolist() == list(range(6))
    assert applied.current_visual_indices.tolist() == [2, 3]
    assert applied.record.retained_visual_ids == (1,)
    assert applied.record.logical_retained_sequence_ids == (0, 1, 2, 3, 4, 5)

    with torch.no_grad():
        prefill = prefill_with_ctp(
            tiny_qwen2vl.model,
            hidden,
            positions,
            visual_indices=current_visual,
            comprehension_threshold=0.0,
            attention_threshold=1.0,
            forced_intervention=intervention,
        )
        token = tiny_qwen2vl.model.embed_tokens(torch.tensor([[12]]))
        decode_one_token(
            tiny_qwen2vl.model,
            token,
            torch.full((3, 1, 1), 6, dtype=torch.long),
            prefill.cache,
        )

    assert prefill.forced is not None
    assert prefill.forced.prefill_cache_lengths == (6, 6, 6, 6)
    assert tuple(cache.shape[-2] for cache in prefill.cache.key_cache) == (7, 7, 7, 7)


def test_forced_record_hashes_actual_boundary_mrope_positions(tiny_qwen2vl) -> None:
    """Catch a forced record that cannot durably identify the actual retained M-RoPE state."""

    hidden, positions = _fixture_inputs(tiny_qwen2vl)
    positions[1, 0] += 10
    positions[2, 0] += 20
    with torch.no_grad():
        got = prefill_with_ctp(
            tiny_qwen2vl.model,
            hidden,
            positions,
            visual_indices=torch.tensor([2, 3]),
            comprehension_threshold=0.0,
            attention_threshold=0.0,
            forced_intervention=ForcedVisualIntervention("input", "physical_delete", (1,)),
        )

    assert got.forced is not None
    expected_positions = positions[:, :, [0, 1, 3, 4, 5]].to(torch.int64).contiguous().cpu()
    assert got.forced.retained_mrope_position_shape == (3, 1, 5)
    assert (
        got.forced.retained_mrope_position_sha256
        == hashlib.sha256(expected_positions.numpy().tobytes()).hexdigest()
    )


@pytest.mark.parametrize(
    "visual_indices",
    [torch.tensor([[2, 3]]), torch.tensor([2, 2]), torch.tensor([3, 2])],
)
def test_prefill_rejects_ambiguous_visual_indices_before_any_layer(
    tiny_qwen2vl, monkeypatch: pytest.MonkeyPatch, visual_indices: torch.Tensor
) -> None:
    """Catch malformed visual populations reaching a layer before stable ordinal IDs are validated."""

    hidden, positions = _fixture_inputs(tiny_qwen2vl)

    def layer_must_not_run(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("visual index validation must happen before layer execution")

    monkeypatch.setattr(tiny_qwen2vl.model.layers[0], "forward", layer_must_not_run)
    with pytest.raises(ValueError, match="visual_indices"):
        prefill_with_ctp(
            tiny_qwen2vl.model,
            hidden,
            positions,
            visual_indices=visual_indices,
            comprehension_threshold=0.0,
            attention_threshold=0.0,
            forced_intervention=ForcedVisualIntervention("input", "physical_delete", ()),
        )


def test_fixed_synthetic_deletion_and_zero_mask_remain_distinct_diagnostics(tiny_qwen2vl) -> None:
    """Catch serializing physical deletion and local zero mask as one intervention."""

    hidden, positions = _fixture_inputs(tiny_qwen2vl)
    kwargs = {
        "decoder_model": tiny_qwen2vl.model,
        "inputs_embeds": hidden,
        "position_ids": positions,
        "visual_indices": torch.tensor([2, 3]),
        "comprehension_threshold": 0.0,
        "attention_threshold": 0.0,
    }
    with torch.no_grad():
        deleted = prefill_with_ctp(
            **kwargs,
            forced_intervention=ForcedVisualIntervention(1, "physical_delete", (1,)),
        )
        zeroed = prefill_with_ctp(
            **kwargs,
            forced_intervention=ForcedVisualIntervention(1, "zero_mask", (1,)),
        )

    assert deleted.forced is not None and zeroed.forced is not None
    assert deleted.forced.mode == "physical_delete"
    assert zeroed.forced.mode == "zero_mask"
    assert deleted.forced.retained_visual_ids == zeroed.forced.retained_visual_ids == (1,)
    assert deleted.forced.prefill_cache_lengths == (6, 6, 5, 5)
    assert zeroed.forced.prefill_cache_lengths == (6, 6, 6, 6)
    assert deleted.hidden_states.shape[1] == 5
    assert zeroed.hidden_states.shape[1] == 6


def test_forced_noop_and_empty_population_do_not_mutate_boundary_state(tiny_qwen2vl) -> None:
    """Catch M=|V| or empty V allocating a gather/clone instead of a true no-op."""

    hidden, positions = _fixture_inputs(tiny_qwen2vl)
    original_keep = torch.arange(hidden.shape[1])
    full = _apply_forced_visual_intervention(
        hidden,
        positions,
        original_keep,
        torch.tensor([2, 3]),
        ForcedVisualIntervention("input", "physical_delete", (1, 0)),
    )
    empty = _apply_forced_visual_intervention(
        hidden,
        positions,
        original_keep,
        torch.empty(0, dtype=torch.long),
        ForcedVisualIntervention("input", "zero_mask", ()),
    )

    assert full.hidden_states.data_ptr() == hidden.data_ptr()
    assert full.position_ids.data_ptr() == positions.data_ptr()
    assert empty.hidden_states.data_ptr() == hidden.data_ptr()
    assert empty.position_ids.data_ptr() == positions.data_ptr()
    assert full.record.retained_visual_ids == (0, 1)
    assert empty.record.retained_visual_ids == ()


@pytest.mark.parametrize(
    ("mode", "current_visual", "retained_visual_ids"),
    [
        ("physical_delete", torch.tensor([2, 3]), (0, 1)),
        ("zero_mask", torch.tensor([2, 3]), (0, 1)),
        ("physical_delete", torch.empty(0, dtype=torch.long), ()),
        ("zero_mask", torch.empty(0, dtype=torch.long), ()),
    ],
)
def test_forced_noop_helper_preserves_pointer_identity_for_all_kept_or_empty_population(
    tiny_qwen2vl, mode: str, current_visual: torch.Tensor, retained_visual_ids: tuple[int, ...]
) -> None:
    """Catch no-op modes allocating a clone/gather despite retaining the whole local state."""

    hidden, positions = _fixture_inputs(tiny_qwen2vl)
    applied = _apply_forced_visual_intervention(
        hidden,
        positions,
        torch.arange(hidden.shape[1]),
        current_visual,
        ForcedVisualIntervention("input", mode, retained_visual_ids),
    )

    assert applied.hidden_states.data_ptr() == hidden.data_ptr()
    assert applied.position_ids.data_ptr() == positions.data_ptr()
    assert applied.original_keep_indices.tolist() == list(range(hidden.shape[1]))


@pytest.mark.parametrize(
    ("mode", "boundary"),
    [
        ("physical_delete", "input"),
        ("physical_delete", 1),
        ("zero_mask", "input"),
        ("zero_mask", 1),
    ],
)
@pytest.mark.parametrize(
    (
        "visual_indices",
        "retained_visual_ids",
        "expected_keep",
        "physical_input_cache",
        "physical_b1_cache",
    ),
    [
        (torch.tensor([2, 3]), (), [0, 1, 4, 5], (4, 4, 4, 4), (6, 6, 4, 4)),
        (torch.tensor([2, 3]), (0, 1), [0, 1, 2, 3, 4, 5], (6, 6, 6, 6), (6, 6, 6, 6)),
        (torch.empty(0, dtype=torch.long), (), [0, 1, 2, 3, 4, 5], (6, 6, 6, 6), (6, 6, 6, 6)),
    ],
)
def test_forced_edge_matrix_executes_prefill_with_expected_positions_order_and_cache(
    tiny_qwen2vl,
    mode: str,
    boundary: str | int,
    visual_indices: torch.Tensor,
    retained_visual_ids: tuple[int, ...],
    expected_keep: list[int],
    physical_input_cache: tuple[int, ...],
    physical_b1_cache: tuple[int, ...],
) -> None:
    """Catch an M=0/all-kept/empty edge changing boundary-local order or heterogeneous caches."""

    hidden, positions = _fixture_inputs(tiny_qwen2vl)
    with torch.no_grad():
        got = prefill_with_ctp(
            tiny_qwen2vl.model,
            hidden,
            positions,
            visual_indices=visual_indices,
            comprehension_threshold=0.0,
            attention_threshold=0.0,
            forced_intervention=ForcedVisualIntervention(boundary, mode, retained_visual_ids),
        )

    expected_cache = (
        (physical_input_cache if boundary == "input" else physical_b1_cache)
        if mode == "physical_delete"
        else (6, 6, 6, 6)
    )
    expected_order = expected_keep if mode == "physical_delete" else list(range(6))
    assert got.forced is not None
    assert got.keep_indices.tolist() == expected_order
    torch.testing.assert_close(got.position_ids, positions[:, :, expected_order])
    assert got.forced.prefill_cache_lengths == expected_cache
    assert tuple(item.shape[-2] for item in got.cache.key_cache) == expected_cache


@torch.no_grad()
def test_forced_all_kept_matches_native_no_crossing_on_tiny_model(tiny_qwen2vl) -> None:
    """Catch forced all-kept changing adapter-visible logits or generated IDs."""

    hidden, positions = _fixture_inputs(tiny_qwen2vl)
    native = prefill_with_ctp(
        tiny_qwen2vl.model,
        hidden,
        positions,
        visual_indices=torch.tensor([2, 3]),
        comprehension_threshold=1e9,
        attention_threshold=1.0,
    )
    forced = prefill_with_ctp(
        tiny_qwen2vl.model,
        hidden,
        positions,
        visual_indices=torch.tensor([2, 3]),
        comprehension_threshold=0.0,
        attention_threshold=1.0,
        forced_intervention=ForcedVisualIntervention("input", "physical_delete", (0, 1)),
    )

    torch.testing.assert_close(forced.hidden_states, native.hidden_states, rtol=1e-5, atol=1e-5)
    assert forced.forced is not None
    assert forced.forced.prefill_cache_lengths == (6, 6, 6, 6)


@torch.no_grad()
def test_forced_path_skips_native_comprehension_and_attention(tiny_qwen2vl, monkeypatch) -> None:
    """Catch forced execution accidentally querying native CTP semantics."""

    def fail(*_args, **_kwargs):
        raise AssertionError("native CTP must not execute for a forced intervention")

    monkeypatch.setattr("docprune.qwen2vl.decoder.ComprehensionController", fail)
    monkeypatch.setattr("docprune.qwen2vl.decoder._last_query_attention", fail)
    hidden, positions = _fixture_inputs(tiny_qwen2vl)
    result = prefill_with_ctp(
        tiny_qwen2vl.model,
        hidden,
        positions,
        visual_indices=torch.tensor([2, 3]),
        comprehension_threshold=0.0,
        attention_threshold=0.0,
        forced_intervention=ForcedVisualIntervention(3, "physical_delete", ()),
    )

    assert result.decision is None
    assert result.forced is not None


@torch.no_grad()
def test_forced_intervention_rejects_invalid_ids_and_boundaries_before_layers(tiny_qwen2vl) -> None:
    """Catch invalid forced identities leaking into decoder execution."""

    hidden, positions = _fixture_inputs(tiny_qwen2vl)
    for intervention, message in (
        (ForcedVisualIntervention(-1, "physical_delete", ()), "boundary"),
        (ForcedVisualIntervention(4, "physical_delete", ()), "boundary"),
        (ForcedVisualIntervention("input", "physical_delete", (0, 0)), "unique"),
        (ForcedVisualIntervention("input", "physical_delete", (2,)), "range"),
        (ForcedVisualIntervention("input", "wrong", ()), "mode"),
    ):
        try:
            prefill_with_ctp(
                tiny_qwen2vl.model,
                hidden,
                positions,
                visual_indices=torch.tensor([2, 3]),
                comprehension_threshold=0.0,
                attention_threshold=1.0,
                forced_intervention=intervention,
            )
        except ValueError as error:
            assert message in str(error)
        else:
            raise AssertionError("invalid forced intervention must fail before prefill")


def test_flash_prefill_uses_unpadded_varlen_mask_contract(tiny_qwen2vl) -> None:
    tiny_qwen2vl.model.config._attn_implementation = "flash_attention_2"

    assert (
        _prefill_attention_mask(tiny_qwen2vl.model, 6, torch.float32, torch.device("cpu")) is None
    )


def test_flash_ctp_recomputes_final_query_without_square_causal_mask(
    tiny_qwen2vl, monkeypatch
) -> None:
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
        torch.matmul(query[:, :, -1:, :], key.transpose(2, 3)) / attention.head_dim**0.5
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


def test_aggregate_native_policy_records_the_real_boundary_selection(tiny_qwen2vl) -> None:
    """Removing policy transport or relabeling native CTP as forced must fail here."""

    from docprune.ctp_policy import aggregate_native_threshold_policy

    input_ids = torch.tensor([[10, 102, 100, 100, 103, 11]])
    hidden = tiny_qwen2vl.model.embed_tokens(input_ids)
    positions = torch.arange(6).view(1, 1, 6).expand(3, 1, 6)

    with torch.no_grad():
        result = prefill_with_ctp(
            tiny_qwen2vl.model,
            hidden,
            positions,
            visual_indices=torch.tensor([2, 3]),
            comprehension_threshold=0.0,
            attention_threshold=0.0,
            ctp_policy=aggregate_native_threshold_policy(),
        )

    assert result.forced is None
    assert result.decision is not None
    assert result.selection is not None
    assert result.selection.policy.name == "aggregate-native-threshold"
    assert result.selection.policy.selection_kind == "native_threshold"
    assert result.selection.boundary == "B_0"
    assert result.selection.retained_visual_ids == (0, 1)
    assert result.selection.aggregate_native_reference_ids == (0, 1)
    assert result.selection.prefill_cache_lengths == tuple(
        item.shape[-2] for item in result.cache.key_cache
    )
    assert result.selection.retained_mrope_position_shape == tuple(result.position_ids.shape)
    assert (
        result.selection.retained_mrope_position_sha256
        == hashlib.sha256(
            result.position_ids.to(dtype=torch.int64).contiguous().numpy().tobytes()
        ).hexdigest()
    )


def test_corrected_policy_no_crossing_is_an_explicit_all_kept_no_op(tiny_qwen2vl) -> None:
    """A missing comprehension crossing must not masquerade as a real forced boundary."""

    from docprune.ctp_policy import aggregate_score_top_m_policy

    input_ids = torch.tensor([[10, 102, 100, 100, 103, 11]])
    hidden = tiny_qwen2vl.model.embed_tokens(input_ids)
    positions = torch.arange(6).view(1, 1, 6).expand(3, 1, 6)

    with torch.no_grad():
        result = prefill_with_ctp(
            tiny_qwen2vl.model,
            hidden,
            positions,
            visual_indices=torch.tensor([2, 3]),
            comprehension_threshold=1e9,
            attention_threshold=0.5,
            ctp_policy=aggregate_score_top_m_policy(),
        )

    assert result.decision is None
    assert result.forced is None
    assert result.selection is not None
    assert result.selection.boundary is None
    assert result.selection.native_layer is None
    assert result.selection.requested_budget == result.selection.achieved_budget == 2
    assert result.selection.retained_visual_ids == (0, 1)
    assert result.selection.prefill_cache_lengths == tuple(
        item.shape[-2] for item in result.cache.key_cache
    )


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
