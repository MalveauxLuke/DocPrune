import json
from dataclasses import asdict

import torch

from experiments.training_pilot.evaluate_checkpoints import (
    content_analysis,
    freeze_pairs,
    question_metrics,
    result_from_mask_scores,
    region_score_rows,
    summarize,
    tie_credit,
)
from experiments.training_pilot.train64 import ranking_metrics
from docprune.stage2.experiment import ExperimentConfig
from test_stage2_contracts import example
from docprune.stage2.supervision import Outcome, TeacherBank
from docprune.stage2.policy import PolicyEncoding


def bank_fixture():
    inputs = example()
    inputs.budget = inputs.layout.owner.numel()
    masks = torch.tensor([[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.bool)
    bank = TeacherBank(
        inputs.identity.key,
        masks,
        tuple(Outcome(None, value) for value in (-4., -2., -1., 0.)),
        Outcome(None, 0.),
        True,
        True,
        inputs.identity.document_family,
        "s",
        "sealed",
        "variable_pilot_v1",
        "hamming_families_v1",
    )
    return inputs, bank


def test_json_contract_normalizes_tuple_fields_to_lists():
    config = ExperimentConfig()
    sealed = json.loads(json.dumps(asdict(config)))
    assert sealed["lora_targets"] == list(config.lora_targets)
    assert sealed["stream_map"] == list(config.stream_map)


def test_pair_manifest_freezes_ties_weights_and_metric_contract():
    example, bank = bank_fixture()
    manifest = freeze_pairs("q", "train", example, bank, [{}] * len(bank.outcomes), epsilon=.1, margin=.05, temperature=1.)
    assert manifest["candidate_pair_count"] == 6
    assert manifest["strict_pair_count"] == len(bank.pairs("gold_aware", .1, .05))
    assert manifest["candidate_pair_count"] == manifest["strict_pair_count"] + manifest["tie_count"]
    assert abs(sum(row["within_question_weight"] for row in manifest["pairs"] if not row["tie"]) - 1.) < 1e-8
    scores = torch.tensor([-4., -2., -1., 0.])
    metrics, pairs = question_metrics(scores, manifest)
    assert metrics["accuracy"] == 1. and len(pairs) == manifest["strict_pair_count"]
    assert set(metrics["families"]) == {"single_flip", "two_flip", "broad"}


def test_question_metrics_exactly_reproduces_training_float32_reduction():
    example, bank = bank_fixture()
    manifest = freeze_pairs("q", "dev", example, bank, [{}] * len(bank.outcomes), epsilon=.1, margin=.05, temperature=1.)
    scores = torch.tensor([1234.125, -81.5, 0.03125, 999.75], dtype=torch.float32)
    expected = ranking_metrics(scores, bank)
    actual, _ = question_metrics(scores, manifest)
    assert actual["accuracy"] == expected["accuracy"]
    assert actual["loss"] == expected["loss"]
    assert actual["pairs"] == expected["pairs"]


def test_result_from_mask_scores_rebuilds_metrics_without_model_output():
    example, bank = bank_fixture()
    manifest = freeze_pairs("q", "train", example, bank, [{}] * len(bank.outcomes), epsilon=.1, margin=.05, temperature=1.)
    saved = dict(
        qid="q",
        split="train",
        baseline_correct=True,
        exposures_in_branch_path=0,
        masks=[
            dict(mask_index=i, retained_tokens=int((bank.masks[i] * example.layout.costs).sum()), head1=value, head2_correction=0., combined=value)
            for i, value in enumerate((-4., -2., -1., 0.))
        ],
    )
    rebuilt = result_from_mask_scores(saved, manifest)
    assert rebuilt["head1"]["accuracy"] == 1.
    assert rebuilt["combined"] == rebuilt["head1"]
    assert rebuilt["masks"] == saved["masks"]


def test_region_scores_preserve_identity_page_and_token_cost():
    inputs,_=bank_fixture()
    encoding=PolicyEncoding(torch.zeros(4,3),torch.tensor([.5,-1.,2.,.25]))
    rows=region_score_rows(encoding,inputs.layout)
    assert [row["region_id"] for row in rows]==list(inputs.layout.region_ids)
    assert [row["token_cost"] for row in rows]==inputs.layout.costs.tolist()
    assert [row["head1_score"] for row in rows]==[.5,-1.,2.,.25]
    assert all(isinstance(row["page_index"],int) for row in rows)


def test_summary_preserves_question_macro_and_reports_pooled_secondary():
    example, bank = bank_fixture()
    manifest = freeze_pairs("q", "dev", example, bank, [{}] * len(bank.outcomes), epsilon=.1, margin=.05, temperature=1.)
    first, first_pairs = question_metrics(torch.tensor([-4., -2., -1., 0.]), manifest)
    second, second_pairs = question_metrics(torch.tensor([4., 2., 1., 0.]), manifest)
    rows = [
        dict(baseline_correct=True, head1=first, head1_pairs=first_pairs, combined=first, combined_pairs=first_pairs, retention=first, retention_pairs=first_pairs),
        dict(baseline_correct=False, head1=second, head1_pairs=second_pairs, combined=second, combined_pairs=second_pairs, retention=second, retention_pairs=second_pairs),
    ]
    result = summarize(rows)
    assert result["head1"]["all"]["questions"] == 2
    assert result["head1"]["all"]["question_balanced_accuracy"] == .5
    assert result["head1"]["correct"]["question_balanced_accuracy"] == 1.
    assert result["head1"]["incorrect"]["question_balanced_accuracy"] == 0.


def test_content_analysis_uses_signed_teacher_cost_actual_distance_and_shared_ties():
    example, bank = bank_fixture()
    manifest = freeze_pairs("q", "dev", example, bank, [{}] * len(bank.outcomes), epsilon=.1, margin=.05, temperature=1.)
    masks = [dict(mask_index=i, retained_tokens=int((bank.masks[i] * example.layout.costs).sum()), head1=0., combined=0.) for i in range(len(bank.outcomes))]
    rows = [dict(qid="q", masks=masks)]
    result = content_analysis(rows, {"q": manifest})
    assert tie_credit(0.) == .5
    assert result["contract"]["local_definition"] == "actual changed-region count <= 2"
    for predictor in ("head1", "combined", "retention"):
        slices = result[predictor]["cost_direction"]
        assert sum(value["pairs"] for value in slices.values()) == manifest["strict_pair_count"]
        assert all(0 <= value["question_family_balanced_accuracy"] <= 1 for value in slices.values())
    # The baseline and selector are evaluated on exactly the same signed strata.
    assert set(result["head1"]["cost_direction"]) == set(result["retention"]["cost_direction"])
