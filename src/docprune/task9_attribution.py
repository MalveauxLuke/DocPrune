"""Paper-derived CPU contracts for Task 9 regional attribution."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from itertools import combinations

import numpy as np

from docprune.qwen2vl.decoder import _forced_boundary_name

_CONTEXTCITE_REPOSITORY = "https://github.com/MadryLab/context-cite"
_CONTEXTCITE_COMMIT = "c11f8ace6e68ba0121b2e2f1f5c896da9e4156f4"
_CONTEXTCITE_UTILS_SHA256 = "d3825dc3292886e0fc9e4c4f0d397f45a84ce1a1ee387ce6985d3006e1c28a4b"
_CONTEXTCITE_SOLVER_SHA256 = "9c3de5c4b06b08a82245431105a58aecada0944a7bb38f506b6eb23f434fd37c"
_MASK_DESIGN_ADAPTATION = (
    "whole MinerU-derived regions replace text chunks; seeds 64..95 are an added "
    "independent held-out split; zero-token audit regions are excluded"
)
_SURROGATE_ADAPTATION = "upstream num_output_tokens=1 because targets are already per-token means"
_PRIMARY_TARGET_KIND = "max-accepted-reference-mean-loglikelihood"
_SECONDARY_TARGET_KIND = "unpruned-generated-response-mean-loglikelihood"
_TARGET_KINDS = {_PRIMARY_TARGET_KIND, _SECONDARY_TARGET_KIND}
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_BOUNDARY_PATTERN = re.compile(r"B_(?:0|[1-9][0-9]*)")
_QWEN_DECODER_LAYER_COUNT = 28


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256_PATTERN.fullmatch(value) is not None


def _is_forced_boundary(value: object) -> bool:
    if value == "B_input":
        decoder_boundary: str | int = "input"
    elif isinstance(value, str) and _BOUNDARY_PATTERN.fullmatch(value) is not None:
        decoder_boundary = int(value.removeprefix("B_"))
    else:
        return False
    try:
        return (
            _forced_boundary_name(decoder_boundary, layer_count=_QWEN_DECODER_LAYER_COUNT) == value
        )
    except ValueError:
        return False


def _build_attribution_identity(
    *,
    question_id: object,
    forced_boundary: object,
    mapping_artifact_sha256: object,
    prompt_input_sha256: object,
    target_kind: object,
    reference_set_token_ids_sha256: object,
    generated_response_token_ids_sha256: object,
) -> dict[str, object]:
    if (
        not isinstance(question_id, str)
        or not question_id
        or question_id != question_id.strip()
        or not _is_forced_boundary(forced_boundary)
        or not _is_sha256(mapping_artifact_sha256)
        or not _is_sha256(prompt_input_sha256)
        or not isinstance(target_kind, str)
        or target_kind not in _TARGET_KINDS
    ):
        raise ValueError("regional attribution identity is invalid")
    if target_kind == _PRIMARY_TARGET_KIND:
        valid_target = _is_sha256(reference_set_token_ids_sha256) and (
            generated_response_token_ids_sha256 is None
        )
    else:
        valid_target = _is_sha256(generated_response_token_ids_sha256) and (
            reference_set_token_ids_sha256 is None
        )
    if not valid_target:
        raise ValueError("regional attribution target identity is invalid")
    identity: dict[str, object] = {
        "question_id": question_id,
        "forced_boundary": forced_boundary,
        "mapping_artifact_sha256": mapping_artifact_sha256,
        "prompt_input_sha256": prompt_input_sha256,
        "target_kind": target_kind,
        "reference_set_token_ids_sha256": reference_set_token_ids_sha256,
        "generated_response_token_ids_sha256": generated_response_token_ids_sha256,
    }
    identity["attribution_identity_sha256"] = _canonical_sha256(identity)
    return identity


def _validated_attribution_identity(
    value: object,
    *,
    expected_sha256: object,
) -> dict[str, object]:
    required = {
        "question_id",
        "forced_boundary",
        "mapping_artifact_sha256",
        "prompt_input_sha256",
        "target_kind",
        "reference_set_token_ids_sha256",
        "generated_response_token_ids_sha256",
        "attribution_identity_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("regional attribution identity schema is invalid")
    expected = _build_attribution_identity(
        question_id=value["question_id"],
        forced_boundary=value["forced_boundary"],
        mapping_artifact_sha256=value["mapping_artifact_sha256"],
        prompt_input_sha256=value["prompt_input_sha256"],
        target_kind=value["target_kind"],
        reference_set_token_ids_sha256=value["reference_set_token_ids_sha256"],
        generated_response_token_ids_sha256=value["generated_response_token_ids_sha256"],
    )
    if value != expected or expected_sha256 != expected["attribution_identity_sha256"]:
        raise ValueError("regional attribution identity digest is invalid")
    return expected


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
    *,
    question_id: str,
    forced_boundary: str,
    mapping_artifact_sha256: str,
    prompt_input_sha256: str,
    target_kind: str,
    reference_set_token_ids_sha256: str | None,
    generated_response_token_ids_sha256: str | None,
) -> dict[str, object]:
    """Build the frozen 64-fit/32-held-out whole-region mask schedule.

    Zero-token audit regions remain in the mapping audit but are excluded from
    the regression design because toggling them cannot change a physical mask.
    """

    attribution_identity = _build_attribution_identity(
        question_id=question_id,
        forced_boundary=forced_boundary,
        mapping_artifact_sha256=mapping_artifact_sha256,
        prompt_input_sha256=prompt_input_sha256,
        target_kind=target_kind,
        reference_set_token_ids_sha256=reference_set_token_ids_sha256,
        generated_response_token_ids_sha256=generated_response_token_ids_sha256,
    )
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
        "adaptation": _MASK_DESIGN_ADAPTATION,
        "keep_probability": 0.5,
        "attribution_identity": attribution_identity,
        "attribution_identity_sha256": attribution_identity["attribution_identity_sha256"],
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
) -> tuple[list[str], list[dict[str, object]], list[dict[str, object]]]:
    required = {
        "method",
        "upstream_repository",
        "upstream_commit",
        "upstream_utils_sha256",
        "adaptation",
        "keep_probability",
        "attribution_identity",
        "attribution_identity_sha256",
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
    _validated_attribution_identity(
        design["attribution_identity"],
        expected_sha256=design["attribution_identity_sha256"],
    )
    if (
        design["method"] != "pinned-contextcite-bernoulli-source-ablation"
        or design["upstream_repository"] != _CONTEXTCITE_REPOSITORY
        or design["upstream_commit"] != _CONTEXTCITE_COMMIT
        or design["upstream_utils_sha256"] != _CONTEXTCITE_UTILS_SHA256
        or design["adaptation"] != _MASK_DESIGN_ADAPTATION
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
    if _canonical_bytes(design["fit_masks"]) != _canonical_bytes(expected_fit) or _canonical_bytes(
        design["holdout_masks"]
    ) != _canonical_bytes(expected_holdout):
        raise ValueError("ContextCite mask design is not the canonical seed schedule")
    return source_ids, expected_fit, expected_holdout


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

    source_ids, fit_masks, _ = _validated_mask_design(design)
    if (
        not isinstance(fit_outcomes, Sequence)
        or isinstance(fit_outcomes, str | bytes)
        or len(fit_outcomes) != 64
    ):
        raise ValueError("ContextCite fitting requires exactly 64 keyed fit outcomes")
    checked_masks = [list(mask["vector"]) for mask in fit_masks]
    checked_outcomes: list[dict[str, object]] = []
    checked_targets: list[float] = []
    for index, (outcome, mask) in enumerate(zip(fit_outcomes, fit_masks, strict=True)):
        if (
            isinstance(outcome, Mapping)
            and outcome.get("attribution_identity_sha256") != design["attribution_identity_sha256"]
        ):
            raise ValueError("ContextCite fit outcome attribution identity is invalid")
        if (
            not isinstance(outcome, Mapping)
            or set(outcome)
            != {
                "split",
                "seed",
                "vector_sha256",
                "attribution_identity_sha256",
                "normalized_target",
            }
            or outcome["split"] != "fit"
            or type(outcome["seed"]) is not int
            or outcome["seed"] != index
            or outcome["vector_sha256"] != mask["vector_sha256"]
            or outcome["attribution_identity_sha256"] != design["attribution_identity_sha256"]
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
        checked_outcomes.append(
            {
                "split": "fit",
                "seed": mask["seed"],
                "vector_sha256": mask["vector_sha256"],
                "attribution_identity_sha256": design["attribution_identity_sha256"],
                "normalized_target": target,
            }
        )
        checked_targets.append(target)
    matrix = np.asarray(checked_masks, dtype=np.float32)
    targets = np.asarray(checked_targets, dtype=np.float64)
    coefficients, intercept = _fit_contextcite_solver(matrix, targets)
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
        "adaptation": _SURROGATE_ADAPTATION,
        "attribution_identity": dict(design["attribution_identity"]),
        "attribution_identity_sha256": design["attribution_identity_sha256"],
        "source_ids": list(source_ids),
        "mask_design_sha256": design["design_sha256"],
        "coefficients": {
            source_id: float(coefficient)
            for source_id, coefficient in zip(source_ids, coefficients, strict=True)
        },
        "intercept": intercept,
        "fit_masks_sha256": _canonical_sha256(checked_masks),
        "fit_outcomes_sha256": _canonical_sha256(checked_outcomes),
        "fit_targets_sha256": _canonical_sha256(checked_targets),
        "fit_target_mean": float(np.mean(targets)),
    }
    result["surrogate_sha256"] = _canonical_sha256(result)
    return result


def _fit_contextcite_solver(
    masks: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Run the exact pinned StandardScaler-plus-Lasso solver adapter."""

    try:
        from sklearn.linear_model import Lasso
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as error:  # pragma: no cover - exercised in the isolated tool environment
        raise RuntimeError("pinned ContextCite scikit-learn environment is required") from error

    matrix = np.asarray(masks, dtype=np.float32)
    output = np.asarray(targets, dtype=np.float64)
    scaler = StandardScaler()
    lasso = Lasso(alpha=0.01, random_state=0, fit_intercept=True)
    pipeline = make_pipeline(scaler, lasso)
    pipeline.fit(matrix, output)
    coefficients = lasso.coef_ / scaler.scale_
    intercept = float(lasso.intercept_ - (scaler.mean_ / scaler.scale_) @ lasso.coef_.T)
    return coefficients, intercept


def _finite_float(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        checked = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a finite number") from error
    if not math.isfinite(checked):
        raise ValueError(f"{label} must be a finite number")
    return checked


def _validated_surrogate(
    surrogate: Mapping[str, object],
    *,
    design: Mapping[str, object],
    source_ids: Sequence[str],
    fit_masks: Sequence[Mapping[str, object]],
) -> tuple[dict[str, float], float, float]:
    required = {
        "method",
        "upstream_repository",
        "upstream_commit",
        "upstream_solver_sha256",
        "lasso_alpha",
        "random_state",
        "fit_intercept",
        "fit_mask_count",
        "target",
        "adaptation",
        "attribution_identity",
        "attribution_identity_sha256",
        "source_ids",
        "mask_design_sha256",
        "coefficients",
        "intercept",
        "fit_masks_sha256",
        "fit_outcomes_sha256",
        "fit_targets_sha256",
        "fit_target_mean",
        "surrogate_sha256",
    }
    if not isinstance(surrogate, Mapping) or set(surrogate) != required:
        raise ValueError("ContextCite surrogate schema is invalid")
    unsigned = dict(surrogate)
    observed_digest = unsigned.pop("surrogate_sha256")
    coefficients = surrogate["coefficients"]
    expected_fit_masks_sha256 = _canonical_sha256([mask["vector"] for mask in fit_masks])
    if (
        surrogate["method"] != "pinned-contextcite-standardscaler-lasso"
        or surrogate["upstream_repository"] != _CONTEXTCITE_REPOSITORY
        or surrogate["upstream_commit"] != _CONTEXTCITE_COMMIT
        or surrogate["upstream_solver_sha256"] != _CONTEXTCITE_SOLVER_SHA256
        or surrogate["lasso_alpha"] != 0.01
        or type(surrogate["random_state"]) is not int
        or surrogate["random_state"] != 0
        or surrogate["fit_intercept"] is not True
        or type(surrogate["fit_mask_count"]) is not int
        or surrogate["fit_mask_count"] != 64
        or surrogate["target"] != "normalized-full-sequence-loglikelihood"
        or surrogate["adaptation"] != _SURROGATE_ADAPTATION
        or surrogate["attribution_identity"] != design["attribution_identity"]
        or surrogate["attribution_identity_sha256"] != design["attribution_identity_sha256"]
        or surrogate["source_ids"] != list(source_ids)
        or surrogate["mask_design_sha256"] != design["design_sha256"]
        or surrogate["fit_masks_sha256"] != expected_fit_masks_sha256
        or not _is_sha256(surrogate["fit_outcomes_sha256"])
        or not _is_sha256(surrogate["fit_targets_sha256"])
        or not _is_sha256(observed_digest)
        or observed_digest != _canonical_sha256(unsigned)
        or not isinstance(coefficients, Mapping)
        or set(coefficients) != set(source_ids)
    ):
        raise ValueError("ContextCite surrogate identity is invalid")
    checked_coefficients = {
        source_id: _finite_float(coefficients[source_id], label="ContextCite coefficient")
        for source_id in source_ids
    }
    intercept = _finite_float(surrogate["intercept"], label="ContextCite intercept")
    fit_target_mean = _finite_float(
        surrogate["fit_target_mean"], label="ContextCite fit-target mean"
    )
    return checked_coefficients, intercept, fit_target_mean


def _average_tie_ranks(values: Sequence[float]) -> np.ndarray:
    checked = np.asarray(values, dtype=np.float64)
    order = np.argsort(checked, kind="mergesort")
    ranks = np.empty(len(checked), dtype=np.float64)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and checked[order[stop]] == checked[order[start]]:
            stop += 1
        # Ranks are one-based; every exact tie gets the group's average rank.
        average_rank = ((start + 1) + stop) / 2.0
        ranks[order[start:stop]] = average_rank
        start = stop
    return ranks


def _spearman_rank_correlation(
    predictions: Sequence[float], targets: Sequence[float]
) -> float | None:
    prediction_ranks = _average_tie_ranks(predictions)
    target_ranks = _average_tie_ranks(targets)
    centered_predictions = prediction_ranks - prediction_ranks.mean()
    centered_targets = target_ranks - target_ranks.mean()
    denominator = float(np.linalg.norm(centered_predictions) * np.linalg.norm(centered_targets))
    if denominator == 0.0:
        return None
    correlation = float(centered_predictions @ centered_targets / denominator)
    return max(-1.0, min(1.0, correlation))


def _root_mean_square(errors: Sequence[float]) -> float:
    scale = math.sqrt(len(errors))
    if any(not math.isfinite(error) for error in errors):
        raise ValueError("ContextCite fidelity errors must be finite")
    result = math.hypot(*(error / scale for error in errors))
    if not math.isfinite(result):
        raise ValueError("ContextCite fidelity RMSE must be finite")
    return result


def evaluate_contextcite_holdout(
    design: Mapping[str, object],
    fit_outcomes: Sequence[Mapping[str, object]],
    surrogate: Mapping[str, object],
    holdout_outcomes: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Evaluate one question's surrogate on the frozen 32-mask holdout.

    LDS is exactly Spearman rank correlation with average ranks for ties. The
    constant comparator is the fit-target mean persisted when the surrogate is
    fit; held-out targets never determine that baseline.
    """

    source_ids, fit_masks, holdout_masks = _validated_mask_design(design)
    expected_surrogate = fit_contextcite_lasso(design, fit_outcomes)
    coefficients, intercept, fit_target_mean = _validated_surrogate(
        surrogate,
        design=design,
        source_ids=source_ids,
        fit_masks=fit_masks,
    )
    if _canonical_bytes(surrogate) != _canonical_bytes(expected_surrogate):
        raise ValueError("ContextCite surrogate does not match the canonical fit outcomes")
    if (
        not isinstance(holdout_outcomes, Sequence)
        or isinstance(holdout_outcomes, str | bytes)
        or len(holdout_outcomes) != 32
    ):
        raise ValueError("ContextCite fidelity requires exactly 32 keyed holdout outcomes")

    checked_outcomes: list[dict[str, object]] = []
    targets: list[float] = []
    predictions: list[float] = []
    for outcome, mask in zip(holdout_outcomes, holdout_masks, strict=True):
        if (
            isinstance(outcome, Mapping)
            and outcome.get("attribution_identity_sha256") != design["attribution_identity_sha256"]
        ):
            raise ValueError("ContextCite holdout outcome attribution identity is invalid")
        if (
            not isinstance(outcome, Mapping)
            or set(outcome)
            != {
                "split",
                "seed",
                "vector_sha256",
                "attribution_identity_sha256",
                "normalized_target",
            }
            or outcome["split"] != "holdout"
            or type(outcome["seed"]) is not int
            or outcome["seed"] != mask["seed"]
            or outcome["vector_sha256"] != mask["vector_sha256"]
            or outcome["attribution_identity_sha256"] != design["attribution_identity_sha256"]
        ):
            raise ValueError("ContextCite holdout outcomes are not bound to canonical masks")
        target = _finite_float(outcome["normalized_target"], label="ContextCite normalized target")
        prediction = intercept + sum(
            coefficients[source_id] * float(retained)
            for source_id, retained in zip(source_ids, mask["vector"], strict=True)
        )
        if not math.isfinite(prediction):
            raise ValueError("ContextCite holdout predictions must be finite")
        checked_outcomes.append(
            {
                "split": "holdout",
                "seed": mask["seed"],
                "vector_sha256": mask["vector_sha256"],
                "attribution_identity_sha256": design["attribution_identity_sha256"],
                "normalized_target": target,
            }
        )
        targets.append(target)
        predictions.append(prediction)

    lds = _spearman_rank_correlation(predictions, targets)
    heldout_rmse = _root_mean_square(
        [prediction - target for prediction, target in zip(predictions, targets)]
    )
    constant_rmse = _root_mean_square([fit_target_mean - target for target in targets])
    result: dict[str, object] = {
        "method": "contextcite-per-question-heldout-fidelity",
        "mask_design_sha256": design["design_sha256"],
        "attribution_identity": dict(design["attribution_identity"]),
        "attribution_identity_sha256": design["attribution_identity_sha256"],
        "surrogate_sha256": expected_surrogate["surrogate_sha256"],
        "fit_outcomes_sha256": expected_surrogate["fit_outcomes_sha256"],
        "fit_targets_sha256": expected_surrogate["fit_targets_sha256"],
        "holdout_mask_count": 32,
        "holdout_seed_start": 64,
        "holdout_seed_stop_exclusive": 96,
        "lds_definition": "spearman-rank-correlation-average-ties",
        "lds_spearman": lds,
        "lds_defined": lds is not None,
        "heldout_rmse": heldout_rmse,
        "constant_baseline": "fit-target-mean",
        "constant_prediction": fit_target_mean,
        "constant_rmse": constant_rmse,
        "surrogate_beats_constant": heldout_rmse < constant_rmse,
        "holdout_masks_sha256": _canonical_sha256([mask["vector"] for mask in holdout_masks]),
        "holdout_outcomes_sha256": _canonical_sha256(checked_outcomes),
        "holdout_targets_sha256": _canonical_sha256(targets),
        "holdout_predictions_sha256": _canonical_sha256(predictions),
    }
    result["fidelity_sha256"] = _canonical_sha256(result)
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


def evaluate_contextcite_refit_stability(
    design: Mapping[str, object],
    fit_outcomes: Sequence[Mapping[str, object]],
    surrogate: Mapping[str, object],
    regions: Sequence[Mapping[str, object]],
    *,
    requested_budget: int,
) -> dict[str, object]:
    """Evaluate deterministic five-refit coefficient and selection stability."""

    source_ids, fit_masks, _ = _validated_mask_design(design)
    expected_surrogate = fit_contextcite_lasso(design, fit_outcomes)
    _validated_surrogate(
        surrogate,
        design=design,
        source_ids=source_ids,
        fit_masks=fit_masks,
    )
    if _canonical_bytes(surrogate) != _canonical_bytes(expected_surrogate):
        raise ValueError("ContextCite surrogate does not match the canonical fit outcomes")
    validated_regions = _validated_regions(regions, allow_zero_cost=False)
    if [source_id for source_id, _ in validated_regions] != source_ids:
        raise ValueError("stability regions must exactly match canonical source order")
    if type(requested_budget) is not int or requested_budget < 0:
        raise ValueError("requested_budget must be a nonnegative integer")

    checked_regions = [
        {"source_id": source_id, "token_cost": token_cost}
        for source_id, token_cost in validated_regions
    ]
    matrix = np.asarray([mask["vector"] for mask in fit_masks], dtype=np.float32)
    targets = np.asarray(
        [
            _finite_float(row["normalized_target"], label="ContextCite normalized target")
            for row in fit_outcomes
        ],
        dtype=np.float64,
    )
    refits: list[dict[str, object]] = []
    achieved_budget: int | None = None
    for seed in range(5):
        sampled = np.random.RandomState(seed).choice(64, size=64, replace=True)
        indices = [int(index) for index in sampled.tolist()]
        coefficients, intercept = _fit_contextcite_solver(matrix[sampled], targets[sampled])
        coefficient_map = {
            source_id: float(coefficient)
            for source_id, coefficient in zip(source_ids, coefficients, strict=True)
        }
        if any(not math.isfinite(value) for value in coefficient_map.values()) or not math.isfinite(
            intercept
        ):
            raise ValueError("ContextCite bootstrap refit must be finite")
        selection = whole_region_knapsack(
            checked_regions,
            coefficients=coefficient_map,
            requested_budget=requested_budget,
        )
        current_budget = int(selection["achieved_budget"])
        if achieved_budget is None:
            achieved_budget = current_budget
        elif current_budget != achieved_budget:
            raise ValueError("bootstrap refits did not use one achieved whole-region budget")
        record: dict[str, object] = {
            "seed": seed,
            "resample_indices": indices,
            "resample_indices_sha256": _canonical_sha256(indices),
            "coefficients": coefficient_map,
            "coefficients_sha256": _canonical_sha256(coefficient_map),
            "intercept": intercept,
            "top_source_ids": selection["top_source_ids"],
            "top_source_ids_sha256": _canonical_sha256(selection["top_source_ids"]),
            "achieved_budget": current_budget,
        }
        record["refit_sha256"] = _canonical_sha256(record)
        refits.append(record)

    coefficient_pairs: list[dict[str, object]] = []
    selection_pairs: list[dict[str, object]] = []
    for left, right in combinations(range(5), 2):
        correlation = _spearman_rank_correlation(
            [float(refits[left]["coefficients"][source_id]) for source_id in source_ids],
            [float(refits[right]["coefficients"][source_id]) for source_id in source_ids],
        )
        coefficient_pairs.append(
            {
                "left_seed": left,
                "right_seed": right,
                "spearman": correlation,
                "defined": correlation is not None,
            }
        )
        left_selection = set(refits[left]["top_source_ids"])
        right_selection = set(refits[right]["top_source_ids"])
        union = left_selection | right_selection
        jaccard = len(left_selection & right_selection) / len(union) if union else None
        selection_pairs.append(
            {
                "left_seed": left,
                "right_seed": right,
                "jaccard": jaccard,
                "defined": jaccard is not None,
            }
        )

    result: dict[str, object] = {
        "method": "contextcite-five-bootstrap-refit-stability",
        "bootstrap_rng": "numpy-legacy-randomstate-choice",
        "bootstrap_seed_start": 0,
        "bootstrap_seed_stop_exclusive": 5,
        "bootstrap_draw_count": 64,
        "bootstrap_replace": True,
        "mask_design_sha256": design["design_sha256"],
        "attribution_identity": dict(design["attribution_identity"]),
        "attribution_identity_sha256": design["attribution_identity_sha256"],
        "surrogate_sha256": expected_surrogate["surrogate_sha256"],
        "fit_outcomes_sha256": expected_surrogate["fit_outcomes_sha256"],
        "fit_targets_sha256": expected_surrogate["fit_targets_sha256"],
        "regions_sha256": _canonical_sha256(checked_regions),
        "source_ids": list(source_ids),
        "requested_budget": requested_budget,
        "achieved_budget": achieved_budget,
        "refits": refits,
        "coefficient_pairwise": coefficient_pairs,
        "coefficient_summary": _defined_pairwise_summary(
            [row["spearman"] for row in coefficient_pairs]
        ),
        "top_selection_pairwise": selection_pairs,
        "top_selection_summary": _defined_pairwise_summary(
            [row["jaccard"] for row in selection_pairs]
        ),
    }
    result["stability_sha256"] = _canonical_sha256(result)
    return result


def _defined_pairwise_summary(values: Sequence[object]) -> dict[str, object]:
    defined = [
        _finite_float(value, label="stability value") for value in values if value is not None
    ]
    return {
        "pair_count": len(values),
        "defined_count": len(defined),
        "undefined_count": len(values) - len(defined),
        "all_defined": len(defined) == len(values),
        "minimum_defined": min(defined) if defined else None,
        "mean_defined": math.fsum(defined) / len(defined) if defined else None,
    }


__all__ = [
    "build_region_mask_design",
    "evaluate_contextcite_holdout",
    "evaluate_contextcite_refit_stability",
    "fit_contextcite_lasso",
    "whole_region_knapsack",
]
