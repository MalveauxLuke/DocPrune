"""Paper-derived CPU contracts for Task 9 regional attribution."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence

import numpy as np

_CONTEXTCITE_REPOSITORY = "https://github.com/MadryLab/context-cite"
_CONTEXTCITE_COMMIT = "c11f8ace6e68ba0121b2e2f1f5c896da9e4156f4"
_CONTEXTCITE_UTILS_SHA256 = "d3825dc3292886e0fc9e4c4f0d397f45a84ce1a1ee387ce6985d3006e1c28a4b"
_CONTEXTCITE_SOLVER_SHA256 = "9c3de5c4b06b08a82245431105a58aecada0944a7bb38f506b6eb23f434fd37c"


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validated_regions(
    regions: Sequence[Mapping[str, object]], *, allow_zero_cost: bool
) -> tuple[tuple[str, int], ...]:
    if not isinstance(regions, Sequence) or isinstance(regions, str | bytes) or not regions:
        raise ValueError("regional attribution requires a nonempty region sequence")
    validated: list[tuple[str, int]] = []
    observed: set[str] = set()
    for region in regions:
        if not isinstance(region, Mapping) or set(region) != {"source_id", "token_cost"}:
            raise ValueError("each region must contain only source_id and token_cost")
        source_id = region["source_id"]
        token_cost = region["token_cost"]
        if not isinstance(source_id, str) or not source_id or source_id in observed:
            raise ValueError("region source IDs must be unique nonempty strings")
        if (
            type(token_cost) is not int
            or token_cost < 0
            or (not allow_zero_cost and token_cost == 0)
        ):
            qualifier = "nonnegative" if allow_zero_cost else "positive-cost"
            raise ValueError(f"regional attribution requires {qualifier} integer token costs")
        observed.add(source_id)
        validated.append((source_id, token_cost))
    return tuple(validated)


def _mask_record(source_ids: Sequence[str], *, split: str, seed: int) -> dict[str, object]:
    # This deliberately mirrors context_cite.utils._create_mask at the pinned
    # revision: legacy RandomState, per-mask seed, and np.choice([False, True]).
    vector = np.random.RandomState(seed).choice([False, True], size=len(source_ids), p=[0.5, 0.5])
    serialized = [bool(value) for value in vector.tolist()]
    return {
        "split": split,
        "seed": seed,
        "vector": serialized,
        "retained_source_ids": [
            source_id
            for source_id, retained in zip(source_ids, serialized, strict=True)
            if retained
        ],
        "vector_sha256": _canonical_sha256(serialized),
    }


def build_region_mask_design(
    regions: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Build the frozen 64-fit/32-held-out whole-region mask schedule.

    Zero-token audit regions remain in the mapping audit but are excluded from
    the regression design because toggling them cannot change a physical mask.
    """

    validated = _validated_regions(regions, allow_zero_cost=True)
    source_ids = [source_id for source_id, cost in validated if cost > 0]
    excluded = [source_id for source_id, cost in validated if cost == 0]
    if not source_ids:
        raise ValueError("regional attribution requires at least one positive-cost region")
    fit_masks = [_mask_record(source_ids, split="fit", seed=seed) for seed in range(64)]
    holdout_masks = [_mask_record(source_ids, split="holdout", seed=seed) for seed in range(64, 96)]
    for column in range(len(source_ids)):
        values = {record["vector"][column] for record in fit_masks}
        if values != {False, True}:
            raise ValueError("regional fit masks contain an inert source column")
    design: dict[str, object] = {
        "method": "pinned-contextcite-bernoulli-source-ablation",
        "upstream_repository": _CONTEXTCITE_REPOSITORY,
        "upstream_commit": _CONTEXTCITE_COMMIT,
        "upstream_utils_sha256": _CONTEXTCITE_UTILS_SHA256,
        "adaptation": (
            "whole MinerU-derived regions replace text chunks; seeds 64..95 are an added "
            "independent held-out split; zero-token audit regions are excluded"
        ),
        "keep_probability": 0.5,
        "source_ids": source_ids,
        "excluded_zero_cost_source_ids": excluded,
        "source_order_sha256": hashlib.sha256("\n".join(source_ids).encode("utf-8")).hexdigest(),
        "fit_masks": fit_masks,
        "holdout_masks": holdout_masks,
    }
    design["design_sha256"] = _canonical_sha256(design)
    return design


def _validated_mask_design(
    design: Mapping[str, object],
) -> tuple[list[str], list[dict[str, object]]]:
    required = {
        "method",
        "upstream_repository",
        "upstream_commit",
        "upstream_utils_sha256",
        "adaptation",
        "keep_probability",
        "source_ids",
        "excluded_zero_cost_source_ids",
        "source_order_sha256",
        "fit_masks",
        "holdout_masks",
        "design_sha256",
    }
    if not isinstance(design, Mapping) or set(design) != required:
        raise ValueError("ContextCite mask design schema is invalid")
    unsigned = dict(design)
    observed_digest = unsigned.pop("design_sha256")
    source_ids = design["source_ids"]
    excluded = design["excluded_zero_cost_source_ids"]
    if (
        design["method"] != "pinned-contextcite-bernoulli-source-ablation"
        or design["upstream_repository"] != _CONTEXTCITE_REPOSITORY
        or design["upstream_commit"] != _CONTEXTCITE_COMMIT
        or design["upstream_utils_sha256"] != _CONTEXTCITE_UTILS_SHA256
        or design["keep_probability"] != 0.5
        or observed_digest != _canonical_sha256(unsigned)
        or not isinstance(source_ids, list)
        or not source_ids
        or any(not isinstance(source_id, str) or not source_id for source_id in source_ids)
        or len(set(source_ids)) != len(source_ids)
        or not isinstance(excluded, list)
        or any(not isinstance(source_id, str) or not source_id for source_id in excluded)
        or len(set(excluded)) != len(excluded)
        or bool(set(source_ids).intersection(excluded))
        or design["source_order_sha256"]
        != hashlib.sha256("\n".join(source_ids).encode("utf-8")).hexdigest()
    ):
        raise ValueError("ContextCite mask design identity is invalid")
    expected_fit = [_mask_record(source_ids, split="fit", seed=seed) for seed in range(64)]
    expected_holdout = [
        _mask_record(source_ids, split="holdout", seed=seed) for seed in range(64, 96)
    ]
    if design["fit_masks"] != expected_fit or design["holdout_masks"] != expected_holdout:
        raise ValueError("ContextCite mask design is not the canonical seed schedule")
    return source_ids, expected_fit


def fit_contextcite_lasso(
    design: Mapping[str, object],
    fit_outcomes: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Fit the pinned ContextCite Lasso to already normalized sequence targets.

    Upstream divides summed token outputs by ``num_output_tokens`` and rescales
    the coefficients afterward. Task 9 supplies a mean log-likelihood target
    directly, so the faithful adapter uses the upstream solver with an
    effective token count of one and records that interface change.
    """

    source_ids, fit_masks = _validated_mask_design(design)
    if (
        not isinstance(fit_outcomes, Sequence)
        or isinstance(fit_outcomes, str | bytes)
        or len(fit_outcomes) != 64
    ):
        raise ValueError("ContextCite fitting requires exactly 64 keyed fit outcomes")
    checked_masks = [list(mask["vector"]) for mask in fit_masks]
    checked_targets: list[float] = []
    for index, (outcome, mask) in enumerate(zip(fit_outcomes, fit_masks, strict=True)):
        if (
            not isinstance(outcome, Mapping)
            or set(outcome) != {"split", "seed", "vector_sha256", "normalized_target"}
            or outcome["split"] != "fit"
            or type(outcome["seed"]) is not int
            or outcome["seed"] != index
            or outcome["vector_sha256"] != mask["vector_sha256"]
        ):
            raise ValueError("ContextCite fit outcomes are not bound to canonical masks")
        value = outcome["normalized_target"]
        if isinstance(value, bool):
            raise ValueError("ContextCite normalized targets must be finite numbers")
        try:
            target = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError("ContextCite normalized targets must be finite numbers") from error
        if not math.isfinite(target):
            raise ValueError("ContextCite normalized targets must be finite numbers")
        checked_targets.append(target)
    try:
        from sklearn.linear_model import Lasso
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as error:  # pragma: no cover - exercised in the isolated tool environment
        raise RuntimeError("pinned ContextCite scikit-learn environment is required") from error

    matrix = np.asarray(checked_masks, dtype=np.float32)
    targets = np.asarray(checked_targets, dtype=np.float64)
    scaler = StandardScaler()
    lasso = Lasso(alpha=0.01, random_state=0, fit_intercept=True)
    pipeline = make_pipeline(scaler, lasso)
    pipeline.fit(matrix, targets)
    coefficients = lasso.coef_ / scaler.scale_
    intercept = float(lasso.intercept_ - (scaler.mean_ / scaler.scale_) @ lasso.coef_.T)
    result: dict[str, object] = {
        "method": "pinned-contextcite-standardscaler-lasso",
        "upstream_repository": _CONTEXTCITE_REPOSITORY,
        "upstream_commit": _CONTEXTCITE_COMMIT,
        "upstream_solver_sha256": _CONTEXTCITE_SOLVER_SHA256,
        "lasso_alpha": 0.01,
        "random_state": 0,
        "fit_intercept": True,
        "fit_mask_count": 64,
        "target": "normalized-full-sequence-loglikelihood",
        "adaptation": "upstream num_output_tokens=1 because targets are already per-token means",
        "source_ids": list(source_ids),
        "mask_design_sha256": design["design_sha256"],
        "coefficients": {
            source_id: float(coefficient)
            for source_id, coefficient in zip(source_ids, coefficients, strict=True)
        },
        "intercept": intercept,
        "fit_masks_sha256": _canonical_sha256(checked_masks),
        "fit_targets_sha256": _canonical_sha256(checked_targets),
    }
    result["surrogate_sha256"] = _canonical_sha256(result)
    return result


def whole_region_knapsack(
    regions: Sequence[Mapping[str, object]],
    *,
    coefficients: Mapping[str, float],
    requested_budget: int,
) -> dict[str, object]:
    """Select top and reverse whole regions at one exact attainable cost.

    The first objective is the largest subset cost no greater than ``M``. Only
    then are coefficient sums maximized/minimized. Lexical region IDs resolve
    exact coefficient ties; decoder/raster token indices never participate.
    """

    validated = tuple(sorted(_validated_regions(regions, allow_zero_cost=False)))
    if type(requested_budget) is not int or requested_budget < 0:
        raise ValueError("requested_budget must be a nonnegative integer")
    source_ids = {source_id for source_id, _ in validated}
    if set(coefficients) != source_ids:
        raise ValueError("coefficient IDs must exactly match positive-cost region IDs")
    checked_coefficients: dict[str, float] = {}
    for source_id, value in coefficients.items():
        try:
            coefficient = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError("region coefficients must be numeric") from error
        if not math.isfinite(coefficient):
            raise ValueError("region coefficients must be finite")
        checked_coefficients[source_id] = coefficient

    # cost -> (best score, best IDs, worst score, worst IDs)
    states: dict[int, tuple[float, tuple[str, ...], float, tuple[str, ...]]] = {
        0: (0.0, (), 0.0, ())
    }
    for source_id, cost in validated:
        coefficient = checked_coefficients[source_id]
        updated = dict(states)
        for previous_cost, (best, best_ids, worst, worst_ids) in states.items():
            next_cost = previous_cost + cost
            if next_cost > requested_budget:
                continue
            best_candidate = (best + coefficient, (*best_ids, source_id))
            worst_candidate = (worst + coefficient, (*worst_ids, source_id))
            current = updated.get(next_cost)
            if current is None:
                updated[next_cost] = (
                    best_candidate[0],
                    best_candidate[1],
                    worst_candidate[0],
                    worst_candidate[1],
                )
                continue
            current_best, current_best_ids, current_worst, current_worst_ids = current
            if best_candidate[0] > current_best or (
                best_candidate[0] == current_best and best_candidate[1] < current_best_ids
            ):
                current_best, current_best_ids = best_candidate
            if worst_candidate[0] < current_worst or (
                worst_candidate[0] == current_worst and worst_candidate[1] < current_worst_ids
            ):
                current_worst, current_worst_ids = worst_candidate
            updated[next_cost] = (
                current_best,
                current_best_ids,
                current_worst,
                current_worst_ids,
            )
        states = updated

    achieved_budget = max(states)
    top_score, top_ids, reverse_score, reverse_ids = states[achieved_budget]
    return {
        "selection_kind": "whole-region-exact-attainable-cost",
        "requested_budget": requested_budget,
        "achieved_budget": achieved_budget,
        "budget_gap": requested_budget - achieved_budget,
        "top_source_ids": list(top_ids),
        "top_coefficient_sum": top_score,
        "reverse_source_ids": list(reverse_ids),
        "reverse_coefficient_sum": reverse_score,
    }


__all__ = ["build_region_mask_design", "fit_contextcite_lasso", "whole_region_knapsack"]
