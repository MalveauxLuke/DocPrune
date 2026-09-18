"""Matched selection baselines, paired answer outcomes, and whole-system cost records."""

import random
from collections import Counter
from dataclasses import dataclass

import torch

from .cost import CostLedger, measure
from .policy import achievable_budget, allocate


def random_region_mask(costs, budget, *, seed):
    """Uniform over feasible exact-cost subsets; never choose a best seed using answers."""
    values = costs.cpu().tolist()
    target = achievable_budget(costs, budget)
    ways = [[0] * (target + 1) for _ in range(len(values) + 1)]
    ways[0][0] = 1
    for i, c in enumerate(values, 1):
        for b in range(target + 1):
            ways[i][b] = ways[i - 1][b] + (ways[i - 1][b - c] if b >= c else 0)
    rng = random.Random(seed)
    mask = torch.zeros(len(values), dtype=torch.bool, device=costs.device)
    b = target
    for i in range(len(values), 0, -1):
        c = values[i - 1]
        take_count = ways[i - 1][b - c] if b >= c else 0
        if rng.randrange(ways[i][b]) < take_count:
            mask[i - 1] = True
            b -= c
    return mask


def baseline_masks(layout, budget, *, random_seeds, retrieval_scores=None):
    result = {
        "unpruned": torch.ones(
            len(layout.region_ids), dtype=torch.bool, device=layout.owner.device
        )
    }
    for seed in random_seeds:
        result[f"random:{seed}"] = random_region_mask(layout.costs, budget, seed=seed)
    if retrieval_scores is not None:
        result["retrieval"], _ = allocate(retrieval_scores, layout.costs, budget)
    return result


def whole_page_mask(layout, budget, page_scores):
    pages = layout.page.unique(sorted=True)
    costs = torch.stack([(layout.page == p).sum() for p in pages])
    selected, actual = allocate(page_scores, costs, budget)
    return torch.isin(layout.page, pages[selected]), actual


def exact_complete_answer(prediction, accepted_alternatives):
    """Explicit exact baseline; lists mean complete component sets, not answer alternatives.

    Semantic adjudication and benchmark-specific number/unit rules are external callbacks.
    """

    def normalized(value):
        items = value if isinstance(value, list | tuple) else [value]
        return Counter(" ".join(str(x).casefold().split()) for x in items)

    return any(normalized(prediction) == normalized(a) for a in accepted_alternatives)


@dataclass(frozen=True)
class AnswerOutcome:
    instance_key: str
    family: str
    baseline_correct: bool
    selected_correct: bool
    stratum: str = "unknown"


def answer_summary(rows):
    if not rows:
        raise ValueError("No answer outcomes")
    if len({r.instance_key for r in rows}) != len(rows):
        raise ValueError(
            "Aggregate repeated random seeds within question before reporting"
        )
    wrong = [r for r in rows if not r.baseline_correct]
    right = [r for r in rows if r.baseline_correct]
    return {
        "questions": len(rows),
        "baseline_accuracy": sum(r.baseline_correct for r in rows) / len(rows),
        "selected_accuracy": sum(r.selected_correct for r in rows) / len(rows),
        "net_accuracy_change": sum(
            int(r.selected_correct) - int(r.baseline_correct) for r in rows
        )
        / len(rows),
        "rescue_rate": sum(r.selected_correct for r in wrong) / len(wrong)
        if wrong
        else None,
        "damage_rate": sum(not r.selected_correct for r in right) / len(right)
        if right
        else None,
        "strata": {
            s: len([r for r in rows if r.stratum == s])
            for s in sorted({r.stratum for r in rows})
        },
    }


def paired_family_bootstrap(first, second, *, repeats=2000, seed=0):
    a, b = {r.instance_key: r for r in first}, {r.instance_key: r for r in second}
    if (
        len(a) != len(first)
        or len(b) != len(second)
        or a.keys() != b.keys()
        or not a
        or repeats < 1
    ):
        raise ValueError("Paired evaluation requires identical unique questions")
    groups = {}
    for key in a:
        if (
            a[key].family != b[key].family
            or a[key].baseline_correct != b[key].baseline_correct
        ):
            raise ValueError("Paired evaluation metadata differs")
        groups.setdefault(a[key].family, []).append(
            int(a[key].selected_correct) - int(b[key].selected_correct)
        )
    rng = random.Random(seed)
    families = sorted(groups)
    draws = []
    for _ in range(repeats):
        values = [
            x
            for family in rng.choices(families, k=len(families))
            for x in groups[family]
        ]
        draws.append(sum(values) / len(values))
    draws.sort()
    return {
        "paired_difference": sum(x for g in groups.values() for x in g) / len(a),
        "ci95": [draws[int(0.025 * (repeats - 1))], draws[int(0.975 * (repeats - 1))]],
        "families": len(families),
        "seed": seed,
        "resamples": repeats,
    }


def evaluate_selection(
    model,
    inputs,
    answerer,
    answerer_prompt,
    *,
    decode,
    config,
    proxy_prompt=None,
    native_layout=None,
    ledger=None,
    decode_tokens=None,
    accepted_answers=None,
    correctness_metric=exact_complete_answer,
):
    """Future inference entry point; config admission prevents premature scientific execution."""
    config.validate(execution=True)
    if (decode_tokens is None) != (accepted_answers is None):
        raise ValueError(
            "Provide both a token decoder and complete accepted answers for quality evaluation"
        )
    if (
        inputs.identity.answerer_revision != config.stage1.answerer_revision
        or inputs.budget != config.stage1.budget
    ):
        raise ValueError("Inference input differs from frozen Stage 1 contract")
    ledger = ledger or CostLedger()
    model.eval()
    with torch.no_grad():
        scores = (
            model(inputs, ledger=ledger)
            if config.architecture == "compact"
            else model(inputs, proxy_prompt, native_layout=native_layout, ledger=ledger)
        )
    with measure(ledger, "allocation", inputs.vision.merged):
        mask, actual = allocate(scores, inputs.layout.costs, inputs.budget)
        retained = inputs.layout.retained_tokens(mask)
    with measure(ledger, "answerer_decode", inputs.vision.merged):
        answer = answerer.generate(answerer_prompt, inputs.vision, retained, **decode)
    decoded = None if decode_tokens is None else decode_tokens(answer["token_ids"])
    correct = (
        None if decoded is None else bool(correctness_metric(decoded, accepted_answers))
    )
    ledger.capture_process_memory(inputs.vision.merged)
    return {
        "decoded_answer": decoded,
        "complete_correct": correct,
        "instance_key": inputs.identity.key,
        "mask": mask.cpu().tolist(),
        "budget": actual,
        "answer": answer,
        "cost": ledger.summary(),
    }


def repeated_random_summary(repetitions):
    """Average outcomes across fixed repetitions, never choose each question's best seed."""
    if not repetitions:
        raise ValueError("No random repetitions")
    expected = {r.instance_key: r for r in repetitions[0]}
    summaries = []
    for rows in repetitions:
        if {r.instance_key for r in rows} != expected.keys():
            raise ValueError("Random repetitions must evaluate the same questions")
        if any(
            (r.family, r.baseline_correct)
            != (
                expected[r.instance_key].family,
                expected[r.instance_key].baseline_correct,
            )
            for r in rows
        ):
            raise ValueError("Random repetition metadata differs")
        summaries.append(answer_summary(rows))
    fields = ("selected_accuracy", "net_accuracy_change", "rescue_rate", "damage_rate")
    return {
        "repetitions": len(repetitions),
        "questions": len(expected),
        **{
            k: None
            if summaries[0][k] is None
            else sum(s[k] for s in summaries) / len(summaries)
            for k in fields
        },
    }
