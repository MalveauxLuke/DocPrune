"""Tests for paper-derived Task 9 regional mask and deployment contracts."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy

import pytest

from docprune.task9_attribution import (
    build_region_mask_design,
    evaluate_contextcite_holdout,
    fit_contextcite_lasso,
    whole_region_knapsack,
)


def _regions() -> list[dict[str, object]]:
    return [
        {"source_id": "region-a", "token_cost": 4},
        {"source_id": "region-b", "token_cost": 3},
        {"source_id": "region-c", "token_cost": 2},
        {"source_id": "region-d", "token_cost": 1},
        {"source_id": "empty-audit-region", "token_cost": 0},
    ]


_PRIMARY_TARGET_KIND = "max-accepted-reference-mean-loglikelihood"
_SECONDARY_TARGET_KIND = "unpruned-generated-response-mean-loglikelihood"


def _identity_kwargs(
    *,
    question_id: str = "question-001",
    forced_boundary: str = "B_13",
    target_kind: str = _PRIMARY_TARGET_KIND,
) -> dict[str, object]:
    identity: dict[str, object] = {
        "question_id": question_id,
        "forced_boundary": forced_boundary,
        "mapping_artifact_sha256": "1" * 64,
        "prompt_input_sha256": "2" * 64,
        "target_kind": target_kind,
        "reference_set_token_ids_sha256": None,
        "generated_response_token_ids_sha256": None,
    }
    if target_kind == _PRIMARY_TARGET_KIND:
        identity["reference_set_token_ids_sha256"] = "3" * 64
    elif target_kind == _SECONDARY_TARGET_KIND:
        identity["generated_response_token_ids_sha256"] = "4" * 64
    return identity


def _design(
    regions: list[dict[str, object]] | None = None,
    **identity_overrides: object,
) -> dict[str, object]:
    identity = _identity_kwargs(**identity_overrides)
    return build_region_mask_design(_regions() if regions is None else regions, **identity)


def _sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _fit_outcomes(
    design: dict[str, object],
    *,
    constant: float | None = None,
) -> list[dict[str, object]]:
    # For the nonconstant fixture, fit y=a*x+b so pinned alpha=0.01
    # shrinkage yields coefficient=1 and intercept=0 up to solver precision. The 64 canonical
    # one-source fit masks contain 31 retained and 33 deleted rows.
    scale = math.sqrt(31 * 33) / 64
    slope = 1.0 + 0.01 / scale
    offset = -0.01 * (31 / 64) / scale
    return [
        {
            "split": "fit",
            "seed": mask["seed"],
            "vector_sha256": mask["vector_sha256"],
            "attribution_identity_sha256": design["attribution_identity_sha256"],
            "normalized_target": (
                constant + (1e-6 if mask["seed"] % 2 == 0 else -1e-6)
                if constant is not None
                else offset + slope * float(mask["vector"][0])
            ),
        }
        for mask in design["fit_masks"]
    ]


def _holdout_outcomes(design: dict[str, object]) -> list[dict[str, object]]:
    outcomes = [
        {
            "split": "holdout",
            "seed": mask["seed"],
            "vector_sha256": mask["vector_sha256"],
            "attribution_identity_sha256": design["attribution_identity_sha256"],
            "normalized_target": float(mask["vector"][0]),
        }
        for mask in design["holdout_masks"]
    ]
    # One deterministic disagreement yields a nontrivial tie-aware Spearman
    # value while leaving every expected metric hand-computable.
    outcomes[0]["normalized_target"] = 1.0
    return outcomes


def test_region_mask_design_matches_pinned_contextcite_randomstate_schedule() -> None:
    """Catch replacing upstream RandomState masks or overlapping held-out seeds."""

    design = _design()

    assert design["method"] == "pinned-contextcite-bernoulli-source-ablation"
    assert design["upstream_commit"] == "c11f8ace6e68ba0121b2e2f1f5c896da9e4156f4"
    assert design["upstream_utils_sha256"] == (
        "d3825dc3292886e0fc9e4c4f0d397f45a84ce1a1ee387ce6985d3006e1c28a4b"
    )
    assert design["keep_probability"] == 0.5
    assert design["attribution_identity"] == {
        **_identity_kwargs(),
        "attribution_identity_sha256": design["attribution_identity_sha256"],
    }
    assert len(design["attribution_identity_sha256"]) == 64
    assert design["source_ids"] == ["region-a", "region-b", "region-c", "region-d"]
    assert design["excluded_zero_cost_source_ids"] == ["empty-audit-region"]
    assert len(design["fit_masks"]) == 64
    assert len(design["holdout_masks"]) == 32
    assert [row["seed"] for row in design["fit_masks"]] == list(range(64))
    assert [row["seed"] for row in design["holdout_masks"]] == list(range(64, 96))
    assert design["fit_masks"][0]["vector"] == [True, True, True, True]
    assert design["fit_masks"][1]["vector"] == [False, True, False, False]
    assert design["holdout_masks"][0]["vector"] == [False, True, True, False]
    assert design["fit_masks"][0]["retained_source_ids"] == [
        "region-a",
        "region-b",
        "region-c",
        "region-d",
    ]
    assert set(design["fit_masks"][0]) == {
        "split",
        "seed",
        "vector",
        "retained_source_ids",
        "vector_sha256",
    }
    assert len(design["source_order_sha256"]) == 64
    assert len(design["design_sha256"]) == 64
    assert len({row["vector_sha256"] for row in design["fit_masks"]}) > 1


def test_region_mask_design_is_replayable_and_rejects_ambiguous_sources() -> None:
    """Catch nondeterminism, duplicate IDs, negative costs, or inert fitted columns."""

    assert _design() == _design()

    for invalid in (
        [],
        [{"source_id": "", "token_cost": 1}],
        [
            {"source_id": "duplicate", "token_cost": 1},
            {"source_id": "duplicate", "token_cost": 2},
        ],
        [{"source_id": "negative", "token_cost": -1}],
        [{"source_id": "only-empty", "token_cost": 0}],
        [{"source_id": "extra", "token_cost": 1, "coefficient": 2.0}],
    ):
        with pytest.raises(ValueError, match="region"):
            _design(invalid)


def test_region_mask_design_rejects_cross_target_or_input_identity() -> None:
    """Catch ambiguous or noncanonical question/intervention/target identities."""

    valid_secondary = _design(target_kind=_SECONDARY_TARGET_KIND)
    assert valid_secondary["attribution_identity"]["reference_set_token_ids_sha256"] is None
    assert valid_secondary["attribution_identity"]["generated_response_token_ids_sha256"] == (
        "4" * 64
    )

    invalid_identities = [
        {**_identity_kwargs(), "question_id": ""},
        {**_identity_kwargs(), "forced_boundary": "13"},
        {**_identity_kwargs(), "mapping_artifact_sha256": "1" * 63},
        {**_identity_kwargs(), "prompt_input_sha256": "Z" * 64},
        {**_identity_kwargs(), "target_kind": "generic-target"},
        {**_identity_kwargs(), "target_kind": ["not", "hashable"]},
        {
            **_identity_kwargs(),
            "generated_response_token_ids_sha256": "4" * 64,
        },
        {
            **_identity_kwargs(target_kind=_SECONDARY_TARGET_KIND),
            "reference_set_token_ids_sha256": "3" * 64,
        },
    ]
    for identity in invalid_identities:
        with pytest.raises(ValueError, match="identity"):
            build_region_mask_design(_regions(), **identity)


def test_region_mask_design_uses_qwen_decoder_boundary_range() -> None:
    """Catch signing a boundary the frozen 28-block decoder cannot execute."""

    assert _design(forced_boundary="B_input")["attribution_identity"]["forced_boundary"] == (
        "B_input"
    )
    assert _design(forced_boundary="B_27")["attribution_identity"]["forced_boundary"] == "B_27"
    for invalid_boundary in ("B_28", "B_999"):
        with pytest.raises(ValueError, match="identity"):
            _design(forced_boundary=invalid_boundary)


def test_contextcite_mask_design_rejects_rehashed_nonboolean_vectors() -> None:
    """Catch Python bool/int equality admitting a noncanonical mask schedule."""

    design = _design([{"source_id": "region-a", "token_cost": 1}])
    altered = deepcopy(design)
    altered["fit_masks"][0]["vector"][0] = 1
    unsigned = dict(altered)
    unsigned.pop("design_sha256")
    altered["design_sha256"] = _sha256(unsigned)
    outcomes = _fit_outcomes(altered)

    with pytest.raises(ValueError, match="canonical seed schedule"):
        fit_contextcite_lasso(altered, outcomes)


def test_whole_region_knapsack_uses_max_attainable_cost_before_coefficient() -> None:
    """Catch greedy ranking, region splitting, or unequal top/reverse budgets."""

    result = whole_region_knapsack(
        _regions()[:-1],
        coefficients={"region-a": 5.0, "region-b": 4.0, "region-c": 1.0, "region-d": -5.0},
        requested_budget=5,
    )

    assert result == {
        "selection_kind": "whole-region-exact-attainable-cost",
        "requested_budget": 5,
        "achieved_budget": 5,
        "budget_gap": 0,
        "top_source_ids": ["region-b", "region-c"],
        "top_coefficient_sum": 5.0,
        "reverse_source_ids": ["region-a", "region-d"],
        "reverse_coefficient_sum": 0.0,
    }


def test_whole_region_knapsack_reports_gap_and_deterministic_region_id_ties() -> None:
    """Catch token-index tie breaks or claiming an unattainable requested budget."""

    regions = [
        {"source_id": "region-z", "token_cost": 4},
        {"source_id": "region-a", "token_cost": 4},
    ]
    result = whole_region_knapsack(
        regions,
        coefficients={"region-z": 1.0, "region-a": 1.0},
        requested_budget=5,
    )

    assert result["achieved_budget"] == 4
    assert result["budget_gap"] == 1
    assert result["top_source_ids"] == ["region-a"]
    assert result["reverse_source_ids"] == ["region-a"]

    for mutation in (
        ({"region-z": 1.0}, 5),
        ({"region-z": 1.0, "region-a": float("nan")}, 5),
        ({"region-z": 1.0, "region-a": 1.0}, -1),
    ):
        with pytest.raises(ValueError):
            whole_region_knapsack(regions, coefficients=mutation[0], requested_budget=mutation[1])

    zero_cost = deepcopy(regions)
    zero_cost[0]["token_cost"] = 0
    with pytest.raises(ValueError, match="positive-cost"):
        whole_region_knapsack(
            zero_cost,
            coefficients={"region-z": 1.0, "region-a": 1.0},
            requested_budget=5,
        )


def test_contextcite_lasso_rejects_noncanonical_fit_inputs_before_solver_import() -> None:
    """Catch fitting the paper-derived solver to the wrong split or target scale."""

    design = _design()
    outcomes = [
        {
            "split": "fit",
            "seed": mask["seed"],
            "vector_sha256": mask["vector_sha256"],
            "attribution_identity_sha256": design["attribution_identity_sha256"],
            "normalized_target": float(mask["seed"]) / 64,
        }
        for mask in design["fit_masks"]
    ]
    invalid_cases: list[tuple[dict[str, object], list[dict[str, object]]]] = []
    tampered_design = deepcopy(design)
    tampered_design["fit_masks"][0]["vector"][0] = not tampered_design["fit_masks"][0]["vector"][0]
    invalid_cases.append((tampered_design, outcomes))
    invalid_cases.extend(
        [
            (design, outcomes[:-1]),
            (design, list(reversed(outcomes))),
            (design, [{**outcomes[0], "split": "holdout"}, *outcomes[1:]]),
            (design, [{**outcomes[0], "seed": False}, *outcomes[1:]]),
            (design, [{**outcomes[0], "seed": 64}, *outcomes[1:]]),
            (design, [{**outcomes[0], "vector_sha256": "0" * 64}, *outcomes[1:]]),
            (design, [*outcomes[:-1], {**outcomes[-1], "normalized_target": float("nan")}]),
        ]
    )
    for candidate_design, candidate_outcomes in invalid_cases:
        with pytest.raises(ValueError):
            fit_contextcite_lasso(candidate_design, candidate_outcomes)


def test_contextcite_lasso_persists_fit_only_constant_baseline_mean() -> None:
    """Catch a held-out mean being substituted for the fit-target baseline."""

    pytest.importorskip("sklearn")
    design = _design([{"source_id": "region-a", "token_cost": 1}])
    outcomes = [
        {
            "split": "fit",
            "seed": mask["seed"],
            "vector_sha256": mask["vector_sha256"],
            "attribution_identity_sha256": design["attribution_identity_sha256"],
            "normalized_target": float(mask["seed"]) / 10.0,
        }
        for mask in design["fit_masks"]
    ]

    surrogate = fit_contextcite_lasso(design, outcomes)

    assert surrogate["fit_target_mean"] == pytest.approx(3.15)
    assert surrogate["attribution_identity"] == design["attribution_identity"]
    assert surrogate["attribution_identity_sha256"] == design["attribution_identity_sha256"]
    assert len(surrogate["fit_outcomes_sha256"]) == 64


def test_contextcite_holdout_fidelity_is_ordered_tie_aware_and_leakage_free() -> None:
    """Catch ordinal ranks, unordered masks, or a held-out constant baseline."""

    pytest.importorskip("sklearn")
    design = _design([{"source_id": "region-a", "token_cost": 1}])
    fit_outcomes = _fit_outcomes(design)
    surrogate = fit_contextcite_lasso(design, fit_outcomes)
    outcomes = _holdout_outcomes(design)

    result = evaluate_contextcite_holdout(design, fit_outcomes, surrogate, outcomes)

    assert result["method"] == "contextcite-per-question-heldout-fidelity"
    assert result["holdout_mask_count"] == 32
    assert result["holdout_seed_start"] == 64
    assert result["holdout_seed_stop_exclusive"] == 96
    assert result["lds_definition"] == "spearman-rank-correlation-average-ties"
    # Holdout bit counts are x=(13 kept, 19 deleted). After the one flipped
    # target they are y=(14 positive, 18 negative), so tie-aware Spearman is
    # the binary phi coefficient 234/sqrt(13*19*14*18).
    assert result["lds_spearman"] == pytest.approx(234 / math.sqrt(13 * 19 * 14 * 18), abs=1e-15)
    assert result["lds_defined"] is True
    assert result["heldout_rmse"] == pytest.approx(math.sqrt(1 / 32), abs=1e-9)
    assert result["constant_baseline"] == "fit-target-mean"
    assert result["constant_prediction"] == pytest.approx(31 / 64, abs=1e-15)
    expected_constant_rmse = math.sqrt((14 * (33 / 64) ** 2 + 18 * (31 / 64) ** 2) / 32)
    assert result["constant_rmse"] == pytest.approx(expected_constant_rmse, abs=1e-15)
    assert result["surrogate_beats_constant"] is True
    assert result["mask_design_sha256"] == design["design_sha256"]
    assert result["attribution_identity"] == design["attribution_identity"]
    assert result["attribution_identity_sha256"] == design["attribution_identity_sha256"]
    assert result["surrogate_sha256"] == surrogate["surrogate_sha256"]
    assert len(result["holdout_masks_sha256"]) == 64
    assert len(result["holdout_outcomes_sha256"]) == 64
    assert len(result["holdout_targets_sha256"]) == 64
    assert len(result["holdout_predictions_sha256"]) == 64
    assert len(result["fidelity_sha256"]) == 64


def test_contextcite_holdout_fidelity_records_undefined_constant_rank() -> None:
    """Catch silently coercing undefined Spearman correlation to zero."""

    pytest.importorskip("sklearn")
    design = _design([{"source_id": "region-a", "token_cost": 1}])
    fit_outcomes = _fit_outcomes(design, constant=0.5)
    surrogate = fit_contextcite_lasso(design, fit_outcomes)

    result = evaluate_contextcite_holdout(
        design,
        fit_outcomes,
        surrogate,
        _holdout_outcomes(design),
    )

    assert result["lds_spearman"] is None
    assert result["lds_defined"] is False
    assert result["surrogate_beats_constant"] is False


def test_contextcite_holdout_fidelity_fails_closed_on_identity_or_order_drift() -> None:
    """Catch evaluating a re-signed noncanonical design or mismatched holdout rows."""

    pytest.importorskip("sklearn")
    design = _design([{"source_id": "region-a", "token_cost": 1}])
    fit_outcomes = _fit_outcomes(design)
    surrogate = fit_contextcite_lasso(design, fit_outcomes)
    outcomes = _holdout_outcomes(design)

    altered_design = deepcopy(design)
    altered_design["adaptation"] = "locally changed after the design was frozen"
    unsigned_design = dict(altered_design)
    unsigned_design.pop("design_sha256")
    altered_design["design_sha256"] = _sha256(unsigned_design)

    altered_surrogate = deepcopy(surrogate)
    altered_surrogate["intercept"] = 2.0

    wrong_design_surrogate = deepcopy(surrogate)
    wrong_design_surrogate["mask_design_sha256"] = "b" * 64
    unsigned_surrogate = dict(wrong_design_surrogate)
    unsigned_surrogate.pop("surrogate_sha256")
    wrong_design_surrogate["surrogate_sha256"] = _sha256(unsigned_surrogate)

    bool_random_state_surrogate = deepcopy(surrogate)
    bool_random_state_surrogate["random_state"] = False
    unsigned_surrogate = dict(bool_random_state_surrogate)
    unsigned_surrogate.pop("surrogate_sha256")
    bool_random_state_surrogate["surrogate_sha256"] = _sha256(unsigned_surrogate)

    overflow_surrogate = deepcopy(surrogate)
    overflow_surrogate["coefficients"]["region-a"] = 1e308
    overflow_surrogate["intercept"] = 1e308
    unsigned_surrogate = dict(overflow_surrogate)
    unsigned_surrogate.pop("surrogate_sha256")
    overflow_surrogate["surrogate_sha256"] = _sha256(unsigned_surrogate)

    fully_rehashed_nonfit_surrogate = deepcopy(surrogate)
    fully_rehashed_nonfit_surrogate["coefficients"]["region-a"] = 123.0
    fully_rehashed_nonfit_surrogate["fit_target_mean"] = 0.75
    fully_rehashed_nonfit_surrogate["fit_targets_sha256"] = "e" * 64
    fully_rehashed_nonfit_surrogate["fit_outcomes_sha256"] = "f" * 64
    unsigned_surrogate = dict(fully_rehashed_nonfit_surrogate)
    unsigned_surrogate.pop("surrogate_sha256")
    fully_rehashed_nonfit_surrogate["surrogate_sha256"] = _sha256(unsigned_surrogate)

    changed_fit_outcomes = deepcopy(fit_outcomes)
    changed_fit_outcomes[0]["normalized_target"] += 1.0

    invalid_cases = [
        (altered_design, fit_outcomes, surrogate, outcomes),
        (design, fit_outcomes, altered_surrogate, outcomes),
        (design, fit_outcomes, wrong_design_surrogate, outcomes),
        (design, fit_outcomes, bool_random_state_surrogate, outcomes),
        (design, fit_outcomes, overflow_surrogate, outcomes),
        (design, fit_outcomes, fully_rehashed_nonfit_surrogate, outcomes),
        (design, changed_fit_outcomes, surrogate, outcomes),
        (design, fit_outcomes, surrogate, outcomes[:-1]),
        (design, fit_outcomes, surrogate, list(reversed(outcomes))),
        (design, fit_outcomes, surrogate, [{**outcomes[0], "split": "fit"}, *outcomes[1:]]),
        (design, fit_outcomes, surrogate, [{**outcomes[0], "seed": False}, *outcomes[1:]]),
        (design, fit_outcomes, surrogate, [{**outcomes[0], "seed": 65}, *outcomes[1:]]),
        (
            design,
            fit_outcomes,
            surrogate,
            [{**outcomes[0], "vector_sha256": "0" * 64}, *outcomes[1:]],
        ),
        (
            design,
            fit_outcomes,
            surrogate,
            [{**outcomes[0], "normalized_target": float("nan")}, *outcomes[1:]],
        ),
        (
            design,
            fit_outcomes,
            surrogate,
            [
                {**outcomes[0], "attribution_identity_sha256": "9" * 64},
                *outcomes[1:],
            ],
        ),
    ]
    for candidate_design, candidate_fit, candidate_surrogate, candidate_outcomes in invalid_cases:
        with pytest.raises(ValueError):
            evaluate_contextcite_holdout(
                candidate_design,
                candidate_fit,
                candidate_surrogate,
                candidate_outcomes,
            )


def test_contextcite_holdout_rejects_python_equal_noncanonical_surrogates() -> None:
    """Catch stale digests hidden by bool/int and signed-zero equality."""

    pytest.importorskip("sklearn")
    design = _design([{"source_id": "region-a", "token_cost": 1}])
    fit_outcomes = _fit_outcomes(design, constant=0.5)
    holdout_outcomes = [
        {
            "split": "holdout",
            "seed": mask["seed"],
            "vector_sha256": mask["vector_sha256"],
            "attribution_identity_sha256": design["attribution_identity_sha256"],
            "normalized_target": 0.0,
        }
        for mask in design["holdout_masks"]
    ]
    surrogate = fit_contextcite_lasso(design, fit_outcomes)

    altered_surrogates = []
    altered = deepcopy(surrogate)
    altered["random_state"] = False
    altered_surrogates.append(altered)
    altered = deepcopy(surrogate)
    altered["fit_intercept"] = 1
    altered_surrogates.append(altered)
    altered = deepcopy(surrogate)
    altered["coefficients"]["region-a"] = -0.0
    altered_surrogates.append(altered)

    for altered in altered_surrogates:
        assert altered == surrogate
        with pytest.raises(ValueError, match="canonical fit outcomes|identity"):
            evaluate_contextcite_holdout(
                design,
                fit_outcomes,
                altered,
                holdout_outcomes,
            )


def test_contextcite_holdout_fidelity_rejects_cross_question_or_target_mixing() -> None:
    """Catch fully canonical artifacts from different attribution identities being mixed."""

    pytest.importorskip("sklearn")
    regions = [{"source_id": "region-a", "token_cost": 1}]
    design = _design(regions)
    fit_outcomes = _fit_outcomes(design)
    surrogate = fit_contextcite_lasso(design, fit_outcomes)
    holdout = _holdout_outcomes(design)

    other_design = _design(
        regions,
        question_id="question-002",
        target_kind=_SECONDARY_TARGET_KIND,
    )
    other_fit = _fit_outcomes(other_design)
    other_surrogate = fit_contextcite_lasso(other_design, other_fit)
    other_holdout = _holdout_outcomes(other_design)

    for candidate_fit, candidate_surrogate, candidate_holdout in (
        (other_fit, surrogate, holdout),
        (fit_outcomes, other_surrogate, holdout),
        (fit_outcomes, surrogate, other_holdout),
    ):
        with pytest.raises(ValueError, match="identity|canonical fit"):
            evaluate_contextcite_holdout(
                design,
                candidate_fit,
                candidate_surrogate,
                candidate_holdout,
            )
