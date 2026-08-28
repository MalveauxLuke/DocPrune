"""Aggregation contracts for Task 9 fidelity and refit-stability admission inputs."""

from __future__ import annotations

import math
from copy import deepcopy
from itertools import combinations

import pytest

from docprune import task9_attribution


def _identity(qid: str, *, boundary: str = "B_13") -> dict[str, object]:
    return task9_attribution._build_attribution_identity(
        question_id=qid,
        forced_boundary=boundary,
        mapping_artifact_sha256="a" * 64,
        prompt_input_sha256="b" * 64,
        target_kind="max-accepted-reference-mean-loglikelihood",
        reference_set_token_ids_sha256="c" * 64,
        generated_response_token_ids_sha256=None,
    )


def _fidelity(qid: str, lds: float | None, rmse: float, constant_rmse: float) -> dict[str, object]:
    identity = _identity(qid)
    value: dict[str, object] = {
        "method": "contextcite-per-question-heldout-fidelity",
        "mask_design_sha256": "d" * 64,
        "attribution_identity": identity,
        "attribution_identity_sha256": identity["attribution_identity_sha256"],
        "surrogate_sha256": "e" * 64,
        "fit_outcomes_sha256": "f" * 64,
        "fit_targets_sha256": "1" * 64,
        "holdout_mask_count": 32,
        "holdout_seed_start": 64,
        "holdout_seed_stop_exclusive": 96,
        "lds_definition": "spearman-rank-correlation-average-ties",
        "lds_spearman": lds,
        "lds_defined": lds is not None,
        "heldout_rmse": rmse,
        "constant_baseline": "fit-target-mean",
        "constant_prediction": 0.0,
        "constant_rmse": constant_rmse,
        "surrogate_beats_constant": rmse < constant_rmse,
        "holdout_masks_sha256": "2" * 64,
        "holdout_outcomes_sha256": "3" * 64,
        "holdout_targets_sha256": "4" * 64,
        "holdout_predictions_sha256": "5" * 64,
    }
    value["fidelity_sha256"] = task9_attribution._canonical_sha256(value)
    return value


def _summary(values: list[float | None]) -> dict[str, object]:
    defined = [value for value in values if value is not None]
    return {
        "pair_count": len(values),
        "defined_count": len(defined),
        "undefined_count": len(values) - len(defined),
        "all_defined": len(defined) == len(values),
        "minimum_defined": min(defined) if defined else None,
        "mean_defined": math.fsum(defined) / len(defined) if defined else None,
    }


def _stability(qid: str, jaccard: float | None = 0.85) -> dict[str, object]:
    identity = _identity(qid)
    refits = []
    for seed in range(5):
        indices = [seed] * 64
        coefficients = {"region-a": float(seed)}
        top_ids = ["region-a"]
        row: dict[str, object] = {
            "seed": seed,
            "resample_indices": indices,
            "resample_indices_sha256": task9_attribution._canonical_sha256(indices),
            "coefficients": coefficients,
            "coefficients_sha256": task9_attribution._canonical_sha256(coefficients),
            "intercept": 0.0,
            "top_source_ids": top_ids,
            "top_source_ids_sha256": task9_attribution._canonical_sha256(top_ids),
            "achieved_budget": 1,
        }
        row["refit_sha256"] = task9_attribution._canonical_sha256(row)
        refits.append(row)
    coefficient_pairs = []
    selection_pairs = []
    for left, right in combinations(range(5), 2):
        coefficient_pairs.append(
            {"left_seed": left, "right_seed": right, "spearman": 1.0, "defined": True}
        )
        selection_pairs.append(
            {
                "left_seed": left,
                "right_seed": right,
                "jaccard": jaccard,
                "defined": jaccard is not None,
            }
        )
    value: dict[str, object] = {
        "method": "contextcite-five-bootstrap-refit-stability",
        "bootstrap_rng": "numpy-legacy-randomstate-choice",
        "bootstrap_seed_start": 0,
        "bootstrap_seed_stop_exclusive": 5,
        "bootstrap_draw_count": 64,
        "bootstrap_replace": True,
        "mask_design_sha256": "d" * 64,
        "attribution_identity": identity,
        "attribution_identity_sha256": identity["attribution_identity_sha256"],
        "surrogate_sha256": "e" * 64,
        "fit_outcomes_sha256": "f" * 64,
        "fit_targets_sha256": "1" * 64,
        "regions_sha256": "6" * 64,
        "source_ids": ["region-a"],
        "requested_budget": 1,
        "achieved_budget": 1,
        "refits": refits,
        "coefficient_pairwise": coefficient_pairs,
        "coefficient_summary": _summary([1.0] * 10),
        "top_selection_pairwise": selection_pairs,
        "top_selection_summary": _summary([jaccard] * 10),
    }
    value["stability_sha256"] = task9_attribution._canonical_sha256(value)
    return value


def _artifacts() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    return (
        [_fidelity("q1", 0.6, 0.2, 0.5), _fidelity("q2", 0.8, 0.4, 0.6)],
        [_stability("q1", 0.9), _stability("q2", 0.85)],
    )


def test_aggregate_fidelity_uses_component_bootstrap_rmse_and_worst_stability() -> None:
    fidelities, stabilities = _artifacts()
    first = task9_attribution.aggregate_contextcite_admission_metrics(
        fidelities,
        stabilities,
        support_components=(("q1",), ("q2",)),
        draws=20,
        seed=3,
    )
    second = task9_attribution.aggregate_contextcite_admission_metrics(
        fidelities,
        stabilities,
        support_components=(("q1",), ("q2",)),
        draws=20,
        seed=3,
    )

    assert first == second
    assert first["qids"] == ["q1", "q2"]
    assert first["forced_boundary"] == "B_13"
    assert first["target_kind"] == "max-accepted-reference-mean-loglikelihood"
    assert first["lds"]["point_mean_per_question"] == pytest.approx(0.7)
    assert first["lds"]["all_defined"] is True
    assert first["lds"]["bootstrap"]["draw_count"] == 20
    assert first["lds"]["bootstrap"]["seed"] == 3
    assert len(first["lds"]["bootstrap"]["draws_sha256"]) == 64
    assert first["rmse"]["heldout"] == pytest.approx(math.sqrt(0.1))
    assert first["rmse"]["constant"] == pytest.approx(math.sqrt(0.305))
    assert first["stability"]["minimum_pairwise_jaccard_across_questions"] == 0.85
    assert first["threshold_inputs"] == {
        "lds_point_at_least_0_5": True,
        "lds_95_lower_bound_above_0_2": True,
        "aggregate_rmse_beats_constant": True,
        "all_selection_jaccards_defined": True,
        "minimum_selection_jaccard_at_least_0_8": True,
    }
    assert "final_verdict" not in first
    assert len(first["aggregate_metrics_sha256"]) == 64


def test_aggregate_fidelity_keeps_undefined_inputs_explicit_and_nonpassing() -> None:
    fidelities, stabilities = _artifacts()
    fidelities[1] = _fidelity("q2", None, 0.4, 0.6)
    stabilities[1] = _stability("q2", None)

    result = task9_attribution.aggregate_contextcite_admission_metrics(
        fidelities,
        stabilities,
        support_components=(("q1",), ("q2",)),
        draws=20,
        seed=3,
    )

    assert result["lds"]["point_mean_per_question"] is None
    assert result["lds"]["undefined_qids"] == ["q2"]
    assert result["lds"]["bootstrap"]["interval_95"] is None
    assert result["lds"]["bootstrap"]["draws_sha256"] is None
    assert result["stability"]["minimum_pairwise_jaccard_across_questions"] is None
    assert result["stability"]["undefined_qids"] == ["q2"]
    assert result["threshold_inputs"]["lds_point_at_least_0_5"] is False
    assert result["threshold_inputs"]["lds_95_lower_bound_above_0_2"] is False
    assert result["threshold_inputs"]["all_selection_jaccards_defined"] is False
    assert result["threshold_inputs"]["minimum_selection_jaccard_at_least_0_8"] is False


def test_aggregate_fidelity_rejects_rehashed_schema_scalar_identity_and_partition_drift() -> None:
    fidelities, stabilities = _artifacts()
    invalid: list[tuple[list[dict[str, object]], list[dict[str, object]], object]] = []

    changed_fidelity = deepcopy(fidelities)
    changed_fidelity[0]["unexpected"] = "self-rehashed schema drift"
    changed_fidelity[0]["fidelity_sha256"] = task9_attribution._canonical_sha256(
        {key: value for key, value in changed_fidelity[0].items() if key != "fidelity_sha256"}
    )
    invalid.append((changed_fidelity, stabilities, (("q1",), ("q2",))))

    bool_fidelity = deepcopy(fidelities)
    bool_fidelity[0]["heldout_rmse"] = True
    bool_fidelity[0]["fidelity_sha256"] = task9_attribution._canonical_sha256(
        {key: value for key, value in bool_fidelity[0].items() if key != "fidelity_sha256"}
    )
    invalid.append((bool_fidelity, stabilities, (("q1",), ("q2",))))

    nonfinite_fidelity = deepcopy(fidelities)
    nonfinite_fidelity[0]["heldout_rmse"] = float("inf")
    nonfinite_fidelity[0]["fidelity_sha256"] = task9_attribution._canonical_sha256(
        {key: value for key, value in nonfinite_fidelity[0].items() if key != "fidelity_sha256"}
    )
    invalid.append((nonfinite_fidelity, stabilities, (("q1",), ("q2",))))

    bool_summary_stability = deepcopy(stabilities)
    bool_summary_stability[0]["top_selection_summary"]["all_defined"] = 1
    bool_summary_stability[0]["stability_sha256"] = task9_attribution._canonical_sha256(
        {
            key: value
            for key, value in bool_summary_stability[0].items()
            if key != "stability_sha256"
        }
    )
    invalid.append((fidelities, bool_summary_stability, (("q1",), ("q2",))))

    negative_jaccard_stability = deepcopy(stabilities)
    negative_jaccard_stability[0]["top_selection_pairwise"][0]["jaccard"] = -0.1
    negative_jaccard_stability[0]["top_selection_summary"] = _summary([-0.1, *([0.9] * 9)])
    negative_jaccard_stability[0]["stability_sha256"] = task9_attribution._canonical_sha256(
        {
            key: value
            for key, value in negative_jaccard_stability[0].items()
            if key != "stability_sha256"
        }
    )
    invalid.append((fidelities, negative_jaccard_stability, (("q1",), ("q2",))))

    other_stability = deepcopy(stabilities)
    other_stability[1] = _stability("q2", 0.85)
    other_stability[1]["attribution_identity"] = _identity("q2", boundary="B_20")
    other_stability[1]["attribution_identity_sha256"] = other_stability[1]["attribution_identity"][
        "attribution_identity_sha256"
    ]
    other_stability[1]["stability_sha256"] = task9_attribution._canonical_sha256(
        {key: value for key, value in other_stability[1].items() if key != "stability_sha256"}
    )
    invalid.append((fidelities, other_stability, (("q1",), ("q2",))))
    invalid.append((fidelities, list(reversed(stabilities)), (("q1",), ("q2",))))
    invalid.append((fidelities, stabilities, (("q1", "q2"), ("q2",))))
    invalid.append((fidelities, stabilities, (("q1",),)))

    for candidate_fidelity, candidate_stability, components in invalid:
        with pytest.raises(ValueError):
            task9_attribution.aggregate_contextcite_admission_metrics(
                candidate_fidelity,
                candidate_stability,
                support_components=components,
                draws=20,
                seed=3,
            )
