import itertools
import math
from dataclasses import replace

import pytest
import torch

from docprune.stage2.contracts import (
    InstanceIdentity,
    RegionLayout,
    RetrievalFeatures,
    SelectorInputs,
    VisualMemory,
)
from docprune.stage2.evaluation import (
    AnswerOutcome,
    CostLedger,
    answer_summary,
    exact_complete_answer,
    paired_family_bootstrap,
    random_region_mask,
)
from docprune.stage2.experiment import (
    matched_configs,
    verify_split_integrity,
)
from docprune.stage2.policy import achievable_budget, allocate
from docprune.stage2.regions import partition_tokens
from docprune.stage2.supervision import (
    Outcome,
    TeacherBank,
    grouped_loss,
    preference,
    question_loss,
)


def example(width=32):
    identity = InstanceIdentity(
        "test",
        "doc",
        "q",
        "question-hash",
        ("page1", "page2"),
        "render",
        "partition",
        "reader-revision",
        "retrieval",
        "scoring",
    )
    owner = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    page = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    coordinates = torch.tensor([[0, i // 2, i % 2] for i in range(4)] * 2).float()
    metadata = torch.randn(4, 8)
    layout = RegionLayout(("a", "b", "c", "d"), owner, page, coordinates, metadata)
    memory = VisualMemory(
        torch.randn(8, width),
        (torch.randn(8, width), torch.randn(8, width)),
        torch.tensor([[1, 4, 4], [1, 4, 4]]),
        "test-vision",
    )
    return SelectorInputs(identity, layout, memory, torch.tensor([1, 2]), 4)


def bank_for(x):
    return TeacherBank(
        x.identity.key,
        torch.tensor([[1, 1, 0, 0], [1, 0, 1, 0], [0, 0, 1, 1]], dtype=torch.bool),
        (Outcome(-1, -2), Outcome(-1.5, -3), Outcome(-2, -2)),
        Outcome(-1, -1),
        False,
        False,
        "doc",
    )


def test_partition_overlap_fallback_and_exact_coverage():
    p = torch.tensor([0] * 6 + [1] * 4)
    coords = torch.tensor(
        [[0, i, 0] for i in range(6)] + [[0, i, 0] for i in range(4)]
    ).float()
    layout = partition_tokens(
        p, coords, [("a", [0, 1, 2]), ("b", [2, 3])], tile_size=2, max_fallback=2
    )
    assert layout.owner[2] == 0 and layout.owner[3] == 1
    assert layout.costs.sum() == 10 and max(layout.costs[2:]) <= 2
    assert layout.retained_tokens(
        torch.ones(len(layout.region_ids), dtype=torch.bool)
    ).tolist() == list(range(10))
    with pytest.raises(ValueError):
        partition_tokens(p, coords, [("bad", [0, 9])])


@pytest.mark.parametrize(
    "costs,budget", [([2, 3, 4], 5), ([3, 5, 6], 8), ([2, 2, 2], 5), ([4, 7, 9, 2], 11)]
)
def test_allocator_matches_exhaustive_search_even_negative_utilities(costs, budget):
    for values in (
        [1.0, 3.0, 4.0, -9.0][: len(costs)],
        [-2.0, -1.0, -3.0, -4.0][: len(costs)],
    ):
        scores = torch.tensor(values)
        mask, target = allocate(scores, costs, budget)
        candidates = [
            z
            for z in itertools.product([0, 1], repeat=len(costs))
            if sum(c * k for c, k in zip(costs, z)) <= budget
        ]
        best_cost = max(sum(c * k for c, k in zip(costs, z)) for z in candidates)
        best = max(
            sum(v * k for v, k in zip(values, z))
            for z in candidates
            if sum(c * k for c, k in zip(costs, z)) == best_cost
        )
        assert target == best_cost
        assert math.isclose(float(scores[mask].sum()), best)
        assert allocate(scores, costs, budget)[0].tolist() == mask.tolist()


def test_allocator_rejects_invalid_inputs():
    for costs in ([0, 1], [-1, 2], [1.5, 2]):
        with pytest.raises(ValueError):
            achievable_budget(costs, 3)
    with pytest.raises(ValueError):
        allocate(torch.tensor([float("nan")]), [1], 1)
    with pytest.raises(ValueError):
        achievable_budget([9], 3)


def test_contract_rejects_uncovered_and_bad_streams():
    x = example()
    x.validate()
    x.layout.owner[0] = -1
    with pytest.raises(ValueError):
        x.validate()
    x = example()
    x.vision.deepstack = (torch.randn(7, 32),)
    with pytest.raises(ValueError):
        x.validate()
    with pytest.raises(ValueError):
        RetrievalFeatures(
            torch.randn(4, 2, 3), "s", torch.zeros(4, 2, dtype=torch.bool)
        ).validate(4)


def test_preferences_gold_guard_correct_stratum_and_unordered_margin():
    reference = Outcome(-1, -1)
    a, b = Outcome(-1.4, -4), Outcome(-1.02, -2)
    kwargs = dict(reference=reference, baseline_correct=False, epsilon=0.1, margin=0.01)
    assert preference(a, b, mode="pure_contrast", **kwargs) == 1
    assert preference(a, b, mode="gold_aware", **kwargs) == -1
    kwargs["baseline_correct"] = True
    assert preference(a, b, mode="pure_contrast", **kwargs) == -1
    assert preference(a, a, mode="g_only", **kwargs) == 0
    kwargs["baseline_correct"] = False
    assert (
        preference(
            Outcome(-1.099, -2), Outcome(-1.101, -5), mode="gold_aware", **kwargs
        )
        == 0
    )


def test_bank_identity_s_acquisition_and_budget_rejected():
    x = example()
    b = bank_for(x)
    b.validate(x)
    b.proposal_uses_s = True
    with pytest.raises(ValueError, match="independent"):
        b.validate(x)
    b.proposal_uses_s = False
    b.instance_key = "wrong"
    with pytest.raises(ValueError, match="identity"):
        b.validate(x)
    b.instance_key = x.identity.key
    b.masks[0, 2] = True
    with pytest.raises(ValueError, match="budget"):
        b.validate(x)


def test_question_weighting_does_not_count_pairs_as_independent_questions():
    losses = [torch.tensor(1.0), torch.tensor(3.0), torch.tensor(8.0)]
    assert grouped_loss(losses, ["a", "a", "b"], weighting="family") == 5
    assert grouped_loss(losses, ["a", "a", "b"], weighting="question") == 4
    x = example()
    b = bank_for(x)
    good = torch.tensor([2.0, 1.0, 0.0, -1.0], requires_grad=True)
    loss = question_loss(good, b, mode="g_only", epsilon=0.1, margin=0.01)
    assert loss < question_loss(-good, b, mode="g_only", epsilon=0.1, margin=0.01)
    loss.backward()
    assert good.grad is not None


def test_matched_configs_and_stage1_gate():
    configs = matched_configs()
    assert len(configs) == 7
    rich = configs["rich-gold_aware-shared"]
    pooled = configs["pooled-gold-aware-shared"]
    assert rich.selector == pooled.selector and rich.supervision == pooled.supervision
    assert not any(c.head2 for c in configs.values())
    with pytest.raises(ValueError, match="unresolved"):
        rich.validate(execution=True)
    assert replace(rich, head2=True).validate().head2


def test_splits_quarantine_shared_background_and_same_family():
    a = example().identity
    b = replace(a, question_id="other")
    with pytest.raises(ValueError):
        verify_split_integrity([a], [b], [])
    b = replace(b, document_family="other")
    with pytest.raises(ValueError):
        verify_split_integrity([a], [b], [])
    b = replace(b, ordered_pages=("new-page",))
    verify_split_integrity([a], [b], [])
    with pytest.raises(ValueError):
        verify_split_integrity([a], [], [], known_diagnostic_keys=[a.key])


def test_random_controls_cost_reproducibility_and_no_best_seed():
    costs = torch.tensor([2, 3, 4, 2])
    masks = [random_region_mask(costs, 6, seed=s) for s in range(30)]
    assert all(int(m.long() @ costs) == 6 for m in masks)
    assert torch.equal(masks[0], random_region_mask(costs, 6, seed=0))
    assert len({tuple(m.tolist()) for m in masks}) > 1


def test_complete_list_and_rescue_damage_metrics():
    assert not exact_complete_answer(["a"], [["a", "b"]])
    assert exact_complete_answer(["B", "A"], [["a", "b"]])
    rows = [
        AnswerOutcome("1", "d1", False, True),
        AnswerOutcome("2", "d2", True, False),
        AnswerOutcome("3", "d3", True, True),
    ]
    out = answer_summary(rows)
    assert (
        out["rescue_rate"] == 1
        and out["damage_rate"] == 0.5
        and out["net_accuracy_change"] == 0
    )
    assert paired_family_bootstrap(rows, rows, repeats=20)["ci95"] == [0, 0]
    with pytest.raises(ValueError):
        answer_summary(rows + rows[:1])
    ledger = CostLedger()
    ledger.add("selector", 1.2)
    assert not ledger.summary()["whole_system_complete"]
    with pytest.raises(ValueError):
        ledger.add("bad", float("nan"))


def test_sealed_data_roundtrip_and_tampering(tmp_path):
    from docprune.stage2.data import load_example, save_example

    x = example()
    b = bank_for(x)
    path = tmp_path / "sealed"
    save_example(path, x, b)
    actual, bank = load_example(path)
    assert actual.identity.key == x.identity.key
    torch.testing.assert_close(actual.vision.merged, x.vision.merged)
    assert bank.outcomes == b.outcomes
    with pytest.raises(FileExistsError):
        save_example(path, x, b)
    (path / "example.json").write_text("{}")
    with pytest.raises(ValueError, match="Changed sealed"):
        load_example(path)


def test_random_repetition_report_averages_instead_of_choosing_winner():
    from docprune.stage2.evaluation import repeated_random_summary

    a = [AnswerOutcome("q", "d", False, True)]
    b = [AnswerOutcome("q", "d", False, False)]
    assert repeated_random_summary([a, b])["rescue_rate"] == 0.5
