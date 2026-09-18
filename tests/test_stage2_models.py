from dataclasses import replace

import pytest
import torch
from test_stage2_contracts import bank_for, example
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

from docprune.stage2.answerer import FrozenAnswerer
from docprune.stage2.contracts import RetrievalFeatures
from docprune.stage2.experiment import ExperimentConfig, Stage1Contract
from docprune.stage2.qwen import (
    PackedPrompt,
    ProxySelector,
    SharedVisionAdapter,
    apply_lora,
    assemble_prefill,
    extract_vision,
)
from docprune.stage2.readouts import CompactSelector, RegionReader
from docprune.stage2.training import (
    build_selector,
    objective,
    restore_checkpoint,
    save_checkpoint,
    train_epoch,
)


@pytest.fixture(autouse=True)
def cpu_only():
    torch.set_num_threads(1)
    torch.manual_seed(3)


def tiny_model(width=32):
    c = Qwen3VLConfig(
        text_config=dict(
            vocab_size=80,
            hidden_size=width,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=width // 8,
            num_key_value_heads=1,
            head_dim=8,
            rope_scaling={
                "rope_type": "default",
                "mrope_section": [1, 1, 2],
                "mrope_interleaved": True,
            },
        ),
        vision_config=dict(
            depth=2,
            hidden_size=32,
            intermediate_size=64,
            num_heads=4,
            out_hidden_size=width,
            patch_size=2,
            temporal_patch_size=1,
            spatial_merge_size=2,
            num_position_embeddings=16,
            deepstack_visual_indexes=[0, 1],
        ),
        image_token_id=70,
        video_token_id=71,
        vision_start_token_id=72,
        vision_end_token_id=73,
    )
    return Qwen3VLForConditionalGeneration(c).cpu().eval()


def prompt():
    return PackedPrompt(
        torch.tensor([[1, 2, 72, 70, 70, 70, 70, 73, 72, 70, 70, 70, 70, 73, 3]]),
        torch.tensor([0, 1]),
        torch.tensor([[1, 4, 4], [1, 4, 4]]),
        torch.randn(32, 12),
    )


def test_native_vision_and_manual_allkeep_prefill_match_real_transformers():
    model = tiny_model()
    p = prompt()
    memory = extract_vision(model, p, provenance="tiny-native")
    assert memory.merged.shape == (8, 32) and len(memory.deepstack) == 2
    packed, indices, _ = assemble_prefill(model, p, memory)
    with torch.no_grad():
        expected = model(
            **dict(
                input_ids=p.input_ids,
                attention_mask=torch.ones_like(p.input_ids),
                pixel_values=p.pixel_values,
                image_grid_thw=p.image_grid_thw,
            ),
            use_cache=False,
        ).logits
        actual = model.lm_head(
            model.model.language_model(**packed, use_cache=False).last_hidden_state
        )
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    assert indices.tolist() == list(range(15))


def test_sparse_all_streams_and_positions_preserved_and_cache_compacted():
    m = tiny_model()
    p = prompt()
    memory = extract_vision(m, p, provenance="t")
    full, _, _ = assemble_prefill(m, p, memory)
    retained = torch.tensor([0, 3, 5, 7])
    sparse, indices, nxt = assemble_prefill(m, p, memory, retained)
    assert int(sparse["visual_pos_masks"].sum()) == 4
    torch.testing.assert_close(
        sparse["position_ids"], full["position_ids"][:, :, indices]
    )
    for before, after in zip(memory.deepstack, sparse["deepstack_visual_embeds"]):
        torch.testing.assert_close(after, before[retained])
    assert sparse["cache_position"].tolist() == list(range(11))
    assert nxt == int(full["position_ids"].max()) + 1
    with pytest.raises(ValueError):
        assemble_prefill(m, p, memory, torch.tensor([3, 0]))


def test_frozen_answerer_likelihood_and_cached_generation():
    m = tiny_model()
    a = FrozenAnswerer(m)
    p = prompt()
    memory = a.vision(p, "test")
    retained = torch.tensor([0, 2, 4, 6])
    target = [7, 8, 9]
    result = a.likelihood(p, memory, retained, target)
    assert len(result["token_loglikelihoods"]) == 3 and result["mean"] < 0
    out = a.generate(p, memory, retained, max_new_tokens=3, eos_ids=[79])
    assert 1 <= len(out["token_ids"]) <= 3
    assert all(not v.requires_grad for v in a.parameters())
    assert all(v.grad is None for v in a.parameters())
    # Cached next-token path agrees with recomputing the full retained prefix.
    generated = out["token_ids"]
    packed, _, start = assemble_prefill(m, p, memory, retained)
    embeds = packed["inputs_embeds"]
    positions = packed["position_ids"]
    for step, token in enumerate(generated):
        if step:
            prev = torch.tensor([[generated[step - 1]]])
            embeds = torch.cat((embeds, m.model.get_input_embeddings()(prev)), 1)
            positions = torch.cat(
                (positions, torch.full((3, 1, 1), start + step - 1, dtype=torch.long)),
                -1,
            )
        length = embeds.shape[1]
        mask = torch.cat(
            (packed["visual_pos_masks"], torch.zeros(1, step, dtype=torch.bool)), 1
        )
        logits = m.lm_head(
            m.model.language_model(
                inputs_embeds=embeds,
                position_ids=positions,
                visual_pos_masks=mask,
                deepstack_visual_embeds=packed["deepstack_visual_embeds"],
                attention_mask=torch.ones(1, length, dtype=torch.long),
                use_cache=False,
            ).last_hidden_state[:, -1]
        )
        assert int(logits.argmax()) == token


@pytest.mark.parametrize("mode", ["rich", "pooled"])
@pytest.mark.parametrize("vision_mode", ["native", "shared_adapter"])
def test_proxy_real_peft_gradients_and_single_prefill(mode, vision_mode):
    m = apply_lora(tiny_model(), rank=2, alpha=4)
    adapter = (
        SharedVisionAdapter(32, 32, [0, 1], [0, 1])
        if vision_mode == "shared_adapter"
        else None
    )
    selector = ProxySelector(
        m,
        8,
        mode=mode,
        vision_mode=vision_mode,
        adapter=adapter,
        width=16,
        heads=2,
        slots=2,
    )
    x = example()
    calls = []
    handle = m.model.language_model.register_forward_hook(lambda *args: calls.append(1))
    scores = selector(x, prompt())
    scores.square().sum().backward()
    handle.remove()
    assert scores.shape == (4,) and len(calls) == 1
    grads = [p.grad for n, p in m.named_parameters() if "lora_B" in n]
    assert any(g is not None and bool(g.abs().sum() > 0) for g in grads)
    assert all(p.grad is None for n, p in m.named_parameters() if "lora_" not in n)
    if adapter is not None:
        assert adapter.merged.weight.grad is not None
        assert all(layer.weight.grad is not None for layer in adapter.deep)


def test_rich_rereads_same_memory_and_global_compare_is_bidirectional():
    x = example()
    reader = RegionReader(32, 16, 8, width=16, heads=2, slots=2, mode="rich")
    assert len(reader.reads) == 2
    seen = []
    hooks = [
        b.register_forward_pre_hook(
            lambda module, args: seen.append(args[2].data_ptr())
        )
        for b in reader.reads
    ]
    q = torch.randn(2, 16)
    a = reader(x.vision.merged, q, x.layout, 0.5)
    for h in hooks:
        h.remove()
    assert len(seen) == 2 and seen[0] == seen[1]
    memory = x.vision.merged.clone()
    memory[6:] += 4
    b = reader(memory, q, x.layout, 0.5)
    assert not torch.allclose(a[0], b[0])  # later page can affect earlier page region


def test_compact_optional_retrieval_and_loss_backward_without_training():
    x = example()
    c = ExperimentConfig(
        architecture="compact",
        vision_mode="shared_answerer",
        width=16,
        heads=2,
        slots=2,
        stage1=Stage1Contract(epsilon=0.1, margin=0.01),
    )
    selector = build_selector(c, answerer_config=tiny_model().config)
    loss = objective(selector, [x], [bank_for(x)], c)
    loss.backward()
    assert selector.head.net[0].weight.grad is not None
    assert x.vision.merged.grad is None
    features = RetrievalFeatures(
        torch.randn(4, 2, 3), "query-profile-v1", torch.ones(4, 2, dtype=torch.bool)
    )
    x.retrieval = features
    with pytest.raises(ValueError):
        selector(x)
    other = CompactSelector(32, 80, 8, width=16, heads=2, slots=2, retrieval_dim=3)
    assert other(x).shape == (4,)


def test_checkpoint_roundtrip_and_no_training_side_effects(tmp_path):
    m = CompactSelector(32, 80, 8, width=16, heads=2, slots=2)
    c = ExperimentConfig(architecture="compact", vision_mode="shared_answerer")
    path = tmp_path / "checkpoint.pt"
    before = {n: p.clone() for n, p in m.named_parameters()}
    save_checkpoint(path, m, c, step=0)
    with torch.no_grad():
        next(m.parameters()).zero_()
    assert restore_checkpoint(path, m, c) == 0
    assert all(torch.equal(p, before[n]) for n, p in m.named_parameters())
    with pytest.raises(ValueError):
        restore_checkpoint(path, m, replace(c, seed=1))
    with pytest.raises(PermissionError):
        train_epoch(m, [], None, c)


def test_question_first_and_geometry_fail_closed():
    p = prompt()
    m = tiny_model()
    p.question_positions = torch.tensor([14])
    with pytest.raises(ValueError, match="question-first"):
        p.validate(m.config)
    p = prompt()
    memory = example().vision
    memory.grid_thw = torch.tensor([[1, 2, 8], [1, 2, 8]])
    with pytest.raises(ValueError, match="geometry"):
        assemble_prefill(m, p, memory)


def test_shared_adapter_different_widths_and_no_source_gradient():
    x = example(40)
    x.vision.merged.requires_grad_(True)
    adapter = SharedVisionAdapter(40, 24, [8, 16], [5, 11])
    m = apply_lora(tiny_model(24), rank=2, alpha=4)
    selector = ProxySelector(
        m,
        8,
        mode="rich",
        vision_mode="shared_adapter",
        adapter=adapter,
        width=16,
        heads=2,
        slots=2,
    )
    scores = selector(x, prompt())
    scores.square().sum().backward()
    assert x.vision.merged.grad is None
    assert adapter.merged.weight.grad.abs().sum() > 0


def test_pooled_invariant_to_zero_mean_detail_change_but_rich_can_read_it():
    x = example()
    q = torch.randn(2, 16)
    pooled = RegionReader(32, 16, 8, width=16, heads=2, slots=2, mode="pooled").eval()
    rich = RegionReader(32, 16, 8, width=16, heads=2, slots=2, mode="rich").eval()
    changed = x.vision.merged.clone()
    changed[0] += 3
    changed[1] -= 3
    torch.testing.assert_close(
        pooled(x.vision.merged, q, x.layout, 0.5),
        pooled(changed, q, x.layout, 0.5),
        atol=1e-6,
        rtol=1e-5,
    )
    assert not torch.allclose(
        rich(x.vision.merged, q, x.layout, 0.5), rich(changed, q, x.layout, 0.5)
    )


def test_end_to_end_compact_selection_cost_and_original_token_output():
    from dataclasses import fields

    from docprune.stage2.evaluation import CostLedger, evaluate_selection

    x = example()
    answerer = FrozenAnswerer(tiny_model())
    p = prompt()
    x.vision = answerer.vision(p, "test")
    values = {f.name: "test-sealed" for f in fields(Stage1Contract)}
    values.update(
        answerer_revision=x.identity.answerer_revision,
        epsilon=0.1,
        margin=0.01,
        budget=4,
    )
    c = ExperimentConfig(
        architecture="compact",
        vision_mode="shared_answerer",
        width=16,
        heads=2,
        slots=2,
        stage1=Stage1Contract(**values),
    )
    selector = build_selector(c, answerer_config=answerer.model.config)
    ledger = CostLedger()
    out = evaluate_selection(
        selector,
        x,
        answerer,
        p,
        decode={"max_new_tokens": 2, "eos_ids": [79]},
        config=c,
        ledger=ledger,
    )
    assert out["budget"] == 4 and len(out["answer"]["retained_original_tokens"]) == 4
    assert "selector_readout" in out["cost"]["seconds"]
    assert (
        "answerer_vision" in out["cost"]["missing_stages"]
    )  # Cached bytes do not count as free compute.


def test_all_unordered_training_batch_has_no_learning_signal():
    from docprune.stage2.supervision import Outcome

    x = example()
    b = bank_for(x)
    b.outcomes = (Outcome(-1, -2),) * 3
    c = ExperimentConfig(
        architecture="compact",
        vision_mode="shared_answerer",
        width=16,
        heads=2,
        slots=2,
        stage1=Stage1Contract(epsilon=0.1, margin=0.01),
    )
    m = build_selector(c, answerer_config=tiny_model().config)
    assert objective(m, [x], [b], c) is None
    assert all(p.grad is None for p in m.parameters())
