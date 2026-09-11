"""Tests for the paired B13-versus-input Task 9 diagnostic analysis."""

import pytest

from archive.experiments.task6_9_2026_09_10.examples.analyze_task9_paired_diagnostics import _diagnosis, _prepare_target


def _summary(passed: bool, *, predictive_fidelity: bool = True) -> dict[str, object]:
    return {
        "one_question_reliability_pass": passed,
        "criteria": {
            "heldout_lds_at_least_0_5": predictive_fidelity,
            "heldout_rmse_beats_fit_mean_constant": predictive_fidelity,
            "minimum_five_refit_selection_jaccard_at_least_0_8": passed,
        },
    }


def test_paired_diagnosis_distinguishes_sample_count_from_boundary_failure() -> None:
    assert _diagnosis(_summary(True), _summary(True))["code"] == "both-pass-mask-count-likely"
    assert _diagnosis(_summary(False), _summary(True))["code"] == "input-only-pass-b13-mixing"
    assert (
        _diagnosis(_summary(False), _summary(False))["code"]
        == "both-predictive-fidelity-pass-selection-unstable"
    )
    assert (
        _diagnosis(
            _summary(False, predictive_fidelity=False),
            _summary(False, predictive_fidelity=False),
        )["code"]
        == "both-fail-regional-linearity"
    )
    assert _diagnosis(_summary(True), _summary(False))["code"] == "b13-only-pass-input-shift"


def test_prepare_target_keeps_accepted_answer_scores_on_their_recorded_scale() -> None:
    outcomes = [{"normalized_target": -1.25}]
    target = {
        "target_kind": "max-accepted-reference-mean-loglikelihood",
        "outcomes": outcomes,
    }

    prepared, metadata, target_scale = _prepare_target(
        target, target_mode="accepted-answer", generated_token_count=7
    )

    assert prepared is outcomes
    assert metadata == {
        "input_scale": "max-accepted-reference-mean-loglikelihood",
        "output_scale": "normalized-full-sequence-loglikelihood",
        "formula": "no-transform",
    }
    assert target_scale == "normalized-full-sequence-loglikelihood"


def test_prepare_target_rejects_the_wrong_artifact_for_accepted_answer() -> None:
    with pytest.raises(ValueError, match="accepted-answer target"):
        _prepare_target(
            {"target_kind": "unpruned-generated-response-mean-loglikelihood", "outcomes": []},
            target_mode="accepted-answer",
            generated_token_count=7,
        )
