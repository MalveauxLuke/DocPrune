"""Optional linked correction: synthetic CPU contracts, no optimizer steps."""

import json
from dataclasses import asdict, replace

import pytest
import torch
from test_stage2_contracts import bank_for, example
from test_stage2_models import prompt, tiny_model
from torch.nn import functional as F

from docprune.stage2.cost import CostLedger
from docprune.stage2.experiment import (
    ExperimentConfig,
    Stage1Contract,
    head2_comparison_configs,
    load_config,
    matched_configs,
)
from docprune.stage2.policy import SetCorrection, allocate, score_candidate_masks
from docprune.stage2.supervision import Outcome, question_loss
from docprune.stage2.training import (
    build_selector,
    objective,
    restore_checkpoint,
    save_checkpoint,
)


@pytest.fixture(autouse=True)
def cpu_only():
    torch.set_num_threads(1)
    torch.manual_seed(23)


def config(architecture="compact", vision_mode="shared_answerer", **kwargs):
    return ExperimentConfig(
        architecture=architecture,
        vision_mode=vision_mode,
        width=16,
        heads=2,
        slots=2,
        lora_rank=2,
        lora_alpha=4,
        stream_map=(0, 1),
        head2=True,
        stage1=Stage1Contract(epsilon=0.1, margin=0.01),
        **kwargs,
    )


def test_correction_permutation_invariance_and_empty_full_sets():
    correction = SetCorrection(8)
    regions = torch.randn(4, 8, requires_grad=True)
    masks = torch.tensor([[0, 0, 0, 0], [1, 1, 1, 1], [1, 0, 1, 0]]).bool()
    values = correction(regions, masks)
    order = torch.tensor([2, 0, 3, 1])
    torch.testing.assert_close(values, correction(regions[order], masks[:, order]))
    assert torch.isfinite(values).all()
    assert not torch.allclose(values[:1], values[1:2])
    values.square().sum().backward()
    assert regions.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="binary"):
        correction(regions, masks.float() + 0.5)
    with pytest.raises(ValueError, match="shape"):
        correction(regions, masks[:, :3])


def test_correction_can_reverse_same_exchange_across_backgrounds():
    # A fixed additive vector cannot prefer A over B with C, then B over A with D.
    # Construct one such function with this actual shallow GELU set model.
    correction = SetCorrection(2)
    regions = torch.tensor([[0.0, 0], [2.0, 0], [1.0, 0], [3.0, 0]])
    masks = torch.tensor(
        [[1, 0, 1, 0], [0, 1, 1, 0], [1, 0, 0, 1], [0, 1, 0, 1]]
    ).bool()
    with torch.no_grad():
        correction.elements[0].weight.copy_(torch.eye(2))
        correction.elements[0].bias.zero_()
        centers = masks.float() @ correction.elements(regions)[:, 0] / 2
        midpoint = centers.mean()
        first, last = correction.readout[0], correction.readout[2]
        first.weight.zero_()
        first.bias.copy_(torch.stack((-midpoint, midpoint)))
        first.weight[0, 0], first.weight[1, 0] = 1, -1
        last.weight.fill_(1)
        last.bias.zero_()
    values = correction(regions, masks)
    assert values[0] > values[1]
    assert values[2] < values[3]


@pytest.mark.parametrize("mode", ["g_only", "gold_aware", "pure_contrast"])
@pytest.mark.parametrize("baseline_correct", [False, True])
def test_joint_loss_uses_same_teacher_pairs_and_keeps_direct_term(
    mode, baseline_correct
):
    bank = bank_for(example())
    bank.baseline_correct = baseline_correct
    scores = torch.tensor([0.2, -0.7, 0.5, 0.1], requires_grad=True)
    correction = torch.tensor([0.9, -0.3, 0.2], requires_grad=True)
    pairs = bank.pairs(mode, 0.1, 0.01)
    plus, minus = torch.tensor(pairs).T
    direct = bank.masks.float() @ scores
    total = direct + correction
    expected = F.softplus(-(direct[plus] - direct[minus]) / 0.7).mean()
    expected += 0.3 * F.softplus(-(total[plus] - total[minus]) / 0.7).mean()
    kwargs = dict(mode=mode, epsilon=0.1, margin=0.01, temperature=0.7)
    actual = question_loss(
        scores, bank, correction_scores=correction, auxiliary_weight=0.3, **kwargs
    )
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert scores.grad.abs().sum() > 0 and correction.grad.abs().sum() > 0
    torch.testing.assert_close(
        question_loss(
            scores, bank, correction_scores=correction, auxiliary_weight=0, **kwargs
        ),
        question_loss(scores, bank, **kwargs),
    )
    bank.outcomes = (Outcome(-1, -1),) * len(bank.outcomes)
    assert question_loss(scores, bank, correction_scores=correction, **kwargs) is None


@pytest.mark.parametrize(
    "architecture,vision_mode",
    [
        ("compact", "shared_answerer"),
        ("rich", "native"),
        ("pooled", "native"),
        ("rich", "shared_adapter"),
        ("pooled", "shared_adapter"),
    ],
)
def test_optional_joint_objective_single_encoding_and_gradients(
    architecture, vision_mode
):
    x = example()
    c = config(architecture, vision_mode)
    backbone = tiny_model()
    selector = build_selector(
        c,
        answerer_config=backbone.config,
        selector_model=None if architecture == "compact" else backbone,
    )
    counts = []
    calls = selector.reader.register_forward_hook(lambda *args: counts.append("reader"))
    proxy_calls = []
    proxy = (
        None
        if architecture == "compact"
        else selector.backbone.model.language_model.register_forward_hook(
            lambda *args: proxy_calls.append(1)
        )
    )
    ledger = CostLedger()
    loss = objective(
        selector,
        [x],
        [bank_for(x)],
        c,
        proxy_prompts=None if architecture == "compact" else [prompt()],
        ledger=ledger,
    )
    loss.backward()
    calls.remove()
    if proxy is not None:
        proxy.remove()
    assert counts == ["reader"]
    assert len(proxy_calls) == (0 if architecture == "compact" else 1)
    for module in (selector.reader, selector.head, selector.correction):
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()
        )
    if architecture != "compact":
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for n, p in selector.backbone.named_parameters()
            if "lora_B" in n
        )
        assert all(
            p.grad is None
            for n, p in selector.backbone.named_parameters()
            if "lora_" not in n
        )
        if selector.adapter is not None:
            assert selector.adapter.merged.weight.grad.abs().sum() > 0
    assert x.vision.merged.grad is None
    assert ledger.calls["selector_head2"] == 1
    assert ledger.counters["head2_candidate_masks"] == 3


def test_auxiliary_gradients_reach_shared_encoder_even_with_head1_frozen():
    x, c = example(), config()
    model = build_selector(c, answerer_config=tiny_model().config)
    with torch.no_grad():
        for p in model.head.parameters():
            p.zero_()
    model.head.requires_grad_(False)
    loss = objective(model, [x], [bank_for(x)], c)
    loss.backward()
    assert model.reader.visual.weight.grad.abs().sum() > 0
    assert model.word.weight.grad.abs().sum() > 0


def test_default_deployment_never_calls_correction_and_initial_weights_are_matched():
    x, c = example(), config()
    answerer_config = tiny_model().config
    linked = build_selector(c, answerer_config=answerer_config).eval()
    direct = build_selector(
        replace(c, head2=False), answerer_config=answerer_config
    ).eval()
    assert direct.correction is None
    for name, parameter in direct.named_parameters():
        torch.testing.assert_close(
            parameter, dict(linked.named_parameters())[name], rtol=0, atol=0
        )

    def forbidden(*args):
        pytest.fail("Default inference must not execute Head 2")

    handle = linked.correction.register_forward_pre_hook(forbidden)
    before = linked(x)
    with torch.no_grad():
        for p in linked.correction.parameters():
            p.fill_(999)
    after = linked(x)
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    torch.testing.assert_close(before, direct(x), rtol=0, atol=0)
    assert torch.equal(
        allocate(before, x.layout.costs, x.budget)[0],
        allocate(after, x.layout.costs, x.budget)[0],
    )
    handle.remove()


def test_diagnostic_reuses_encoding_and_reports_direct_and_total_separately():
    x, c = example(), config()
    model = build_selector(c, answerer_config=tiny_model().config).eval()
    encoding = model(x, return_encoding=True)
    masks = bank_for(x).masks
    ledger = CostLedger()

    def forbidden(*args):
        pytest.fail("Candidate scoring must reuse the stored representation")

    handle = model.reader.register_forward_pre_hook(forbidden)
    values = score_candidate_masks(encoding, masks, model.correction, ledger=ledger)
    chunked = torch.cat(
        [
            score_candidate_masks(encoding, row[None], model.correction).total
            for row in masks
        ]
    )
    torch.testing.assert_close(values.total, values.direct + values.correction)
    torch.testing.assert_close(values.total, chunked)
    torch.testing.assert_close(values.direct, masks.float() @ encoding.scores)
    direct_only = score_candidate_masks(encoding, masks)
    torch.testing.assert_close(direct_only.total, direct_only.direct)
    assert ledger.calls["selector_head2"] == 1
    handle.remove()


def test_optional_configuration_is_separate_matched_serializable_and_still_gated(
    tmp_path,
):
    c = config(head2_weight=0.4)
    pairs = list(head2_comparison_configs(c).values())
    a, b = (asdict(x) for x in pairs)
    assert not a.pop("head2") and b.pop("head2")
    a.pop("name")
    b.pop("name")
    assert a == b
    assert all(not arm.head2 for arm in matched_configs(c).values())
    path = tmp_path / "optional.json"
    path.write_text(json.dumps(asdict(pairs[1])))
    assert load_config(path) == pairs[1]
    with pytest.raises(ValueError, match="unresolved"):
        pairs[1].validate(execution=True)
    for weight in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="weight"):
            replace(c, head2_weight=weight).validate()
    with pytest.raises(ValueError, match="boolean"):
        replace(c, head2="false").validate()


def test_optional_checkpoint_and_model_configuration_mismatch(tmp_path):
    x, c = example(), config()
    model = build_selector(c, answerer_config=tiny_model().config)
    expected = {n: p.clone() for n, p in model.named_parameters()}
    path = tmp_path / "head2.pt"
    save_checkpoint(path, model, c, step=0)
    with torch.no_grad():
        for p in model.correction.parameters():
            p.zero_()
    restore_checkpoint(path, model, c)
    for name, value in model.named_parameters():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
    with pytest.raises(ValueError, match="identity"):
        restore_checkpoint(path, model, replace(c, head2=False))
    with pytest.raises(ValueError, match="Head 2 mismatch"):
        objective(model, [x], [bank_for(x)], replace(c, head2=False))
    # Explicit zero weight reproduces direct loss and avoids correction work.
    ledger = CostLedger()
    zero = objective(
        model, [x], [bank_for(x)], replace(c, head2_weight=0), ledger=ledger
    )
    expected_loss = question_loss(
        model(x), bank_for(x), mode=c.supervision, epsilon=0.1, margin=0.01
    )
    torch.testing.assert_close(zero, expected_loss)
    assert "selector_head2" not in ledger.calls
