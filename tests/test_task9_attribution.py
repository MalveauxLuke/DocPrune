"""Tests for paper-derived Task 9 regional mask and deployment contracts."""

from __future__ import annotations

from copy import deepcopy

import pytest

from docprune.task9_attribution import (
    build_region_mask_design,
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


def test_region_mask_design_matches_pinned_contextcite_randomstate_schedule() -> None:
    """Catch replacing upstream RandomState masks or overlapping held-out seeds."""

    design = build_region_mask_design(_regions())

    assert design["method"] == "pinned-contextcite-bernoulli-source-ablation"
    assert design["upstream_commit"] == "c11f8ace6e68ba0121b2e2f1f5c896da9e4156f4"
    assert design["upstream_utils_sha256"] == (
        "d3825dc3292886e0fc9e4c4f0d397f45a84ce1a1ee387ce6985d3006e1c28a4b"
    )
    assert design["keep_probability"] == 0.5
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

    assert build_region_mask_design(_regions()) == build_region_mask_design(_regions())

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
            build_region_mask_design(invalid)


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

    design = build_region_mask_design(_regions())
    outcomes = [
        {
            "split": "fit",
            "seed": mask["seed"],
            "vector_sha256": mask["vector_sha256"],
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
