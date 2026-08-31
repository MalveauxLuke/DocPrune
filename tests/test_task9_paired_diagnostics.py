"""Tests for the paired B13-versus-input Task 9 diagnostic analysis."""

from examples.analyze_task9_paired_diagnostics import _diagnosis


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
