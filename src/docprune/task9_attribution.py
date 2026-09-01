"""Paper-derived CPU contracts for Task 9 regional attribution."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections.abc import Mapping, Sequence
from itertools import combinations

import numpy as np

from docprune.qwen2vl.decoder import ForcedVisualIntervention, _forced_boundary_name
from docprune.segmentation import RegionTokenMapping

_CONTEXTCITE_REPOSITORY = "https://github.com/MadryLab/context-cite"
_CONTEXTCITE_COMMIT = "c11f8ace6e68ba0121b2e2f1f5c896da9e4156f4"
_CONTEXTCITE_UTILS_SHA256 = "d3825dc3292886e0fc9e4c4f0d397f45a84ce1a1ee387ce6985d3006e1c28a4b"
_CONTEXTCITE_SOLVER_SHA256 = "9c3de5c4b06b08a82245431105a58aecada0944a7bb38f506b6eb23f434fd37c"
_MASK_DESIGN_ADAPTATION = (
    "whole MinerU-derived regions replace text chunks; seeds 64..95 are an added "
    "independent held-out split; zero-token audit regions are excluded"
)
_SURROGATE_ADAPTATION = "upstream num_output_tokens=1 because targets are already per-token means"
_MEAN_LOGLIKELIHOOD_SCALE = "normalized-full-sequence-loglikelihood"
_CONTEXTCITE_LOGIT_SCALE = "contextcite-sequence-logit-per-generated-token"
_TARGET_SCALES = {_MEAN_LOGLIKELIHOOD_SCALE, _CONTEXTCITE_LOGIT_SCALE}
_PRIMARY_TARGET_KIND = "max-accepted-reference-mean-loglikelihood"
_SECONDARY_TARGET_KIND = "unpruned-generated-response-mean-loglikelihood"
_MARGIN_TARGET_KIND = "gold-minus-unpruned-generated-response-mean-loglikelihood"
_TARGET_KINDS = {_PRIMARY_TARGET_KIND, _SECONDARY_TARGET_KIND, _MARGIN_TARGET_KIND}
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_BOUNDARY_PATTERN = re.compile(r"B_(?:0|[1-9][0-9]*)")
_QWEN_DECODER_LAYER_COUNT = 28


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def contextcite_logit_per_token_from_mean_loglikelihood(
    mean_loglikelihood: float, token_count: int
) -> float:
    """Reconstruct ContextCite's sequence logit and normalize by response length."""

    mean_value = _finite_float(mean_loglikelihood, label="mean sequence log-likelihood")
    if type(token_count) is not int or token_count <= 0:
        raise ValueError("generated response token count must be a positive integer")
    sequence_log_probability = mean_value * token_count
    if sequence_log_probability >= 0.0:
        raise ValueError("sequence log-probability must be strictly negative")
    if sequence_log_probability < -math.log(2.0):
        log_one_minus_probability = math.log1p(-math.exp(sequence_log_probability))
    else:
        log_one_minus_probability = math.log(-math.expm1(sequence_log_probability))
    result = (sequence_log_probability - log_one_minus_probability) / token_count
    if not math.isfinite(result):
        raise ValueError("ContextCite sequence-logit target must be finite")
    return result


def _surrogate_adaptation(target_scale: str) -> str:
    if target_scale == _MEAN_LOGLIKELIHOOD_SCALE:
        return _SURROGATE_ADAPTATION
    if target_scale == _CONTEXTCITE_LOGIT_SCALE:
        return (
            "released aggregate_logit_probs sequence logit reconstructed from sealed mean "
            "log-likelihood and divided by generated non-EOS token count"
        )
    raise ValueError("ContextCite target scale is invalid")


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
    elif target_kind == _SECONDARY_TARGET_KIND:
        valid_target = _is_sha256(generated_response_token_ids_sha256) and (
            reference_set_token_ids_sha256 is None
        )
    else:
        valid_target = _is_sha256(reference_set_token_ids_sha256) and _is_sha256(
            generated_response_token_ids_sha256
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
    fit_mask_count: int = 64,
    holdout_mask_count: int = 32,
) -> dict[str, object]:
    """Build a frozen fit/held-out whole-region mask schedule.

    Zero-token audit regions remain in the mapping audit but are excluded from
    the regression design because toggling them cannot change a physical mask.
    """

    if (
        type(fit_mask_count) is not int
        or fit_mask_count <= 0
        or type(holdout_mask_count) is not int
        or holdout_mask_count <= 0
    ):
        raise ValueError("regional attribution mask counts must be positive integers")
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
    fit_masks = [_mask_record(source_ids, split="fit", seed=seed) for seed in range(fit_mask_count)]
    holdout_masks = [
        _mask_record(source_ids, split="holdout", seed=seed)
        for seed in range(fit_mask_count, fit_mask_count + holdout_mask_count)
    ]
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
        "fit_mask_count": fit_mask_count,
        "holdout_mask_count": holdout_mask_count,
        "fit_masks": fit_masks,
        "holdout_masks": holdout_masks,
    }
    design["design_sha256"] = _canonical_sha256(design)
    return design


def build_task9_preliminary_mask_design(
    regions: Sequence[Mapping[str, object]],
    *,
    question_id: str,
    forced_boundary: str,
    mapping_artifact_sha256: str,
    prompt_input_sha256: str,
    target_kind: str,
    reference_set_token_ids_sha256: str | None,
    generated_response_token_ids_sha256: str | None,
    primary_budget_fraction: float = 0.65,
    budget_local_tolerance_fraction: float = 0.05,
    fit_mask_count: int = 256,
    global_holdout_mask_count: int = 32,
    budget_local_holdout_mask_count: int = 32,
) -> dict[str, object]:
    """Build the random-48 pilot's global and primary-budget-local mask schedule."""

    for value, label in (
        (primary_budget_fraction, "primary budget fraction"),
        (budget_local_tolerance_fraction, "budget-local tolerance fraction"),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            or not 0 < float(value) < 1
        ):
            raise ValueError(f"{label} must be finite and strictly between zero and one")
    if type(budget_local_holdout_mask_count) is not int or budget_local_holdout_mask_count <= 0:
        raise ValueError("budget-local holdout count must be a positive integer")
    validated = _validated_regions(regions, allow_zero_cost=True)
    positive = [
        {"source_id": source_id, "token_cost": cost} for source_id, cost in validated if cost > 0
    ]
    global_design = build_region_mask_design(
        regions,
        question_id=question_id,
        forced_boundary=forced_boundary,
        mapping_artifact_sha256=mapping_artifact_sha256,
        prompt_input_sha256=prompt_input_sha256,
        target_kind=target_kind,
        reference_set_token_ids_sha256=reference_set_token_ids_sha256,
        generated_response_token_ids_sha256=generated_response_token_ids_sha256,
        fit_mask_count=fit_mask_count,
        holdout_mask_count=global_holdout_mask_count,
    )
    source_ids = list(global_design["source_ids"])
    costs = {row["source_id"]: row["token_cost"] for row in positive}
    total_cost = sum(costs.values())
    requested_budget = math.floor(float(primary_budget_fraction) * total_cost + 0.5)
    attainable = whole_region_knapsack(
        positive,
        coefficients={source_id: 0.0 for source_id in source_ids},
        requested_budget=requested_budget,
    )["achieved_budget"]
    tolerance = max(1, math.floor(float(budget_local_tolerance_fraction) * total_cost + 0.5))
    minimum = max(0, attainable - tolerance)
    maximum = min(total_cost, attainable + tolerance)
    observed = {
        row["vector_sha256"]
        for row in [*global_design["fit_masks"], *global_design["holdout_masks"]]
    }
    local: list[dict[str, object]] = []
    seed = fit_mask_count + global_holdout_mask_count
    stop = seed + 1_000_000
    while len(local) < budget_local_holdout_mask_count and seed < stop:
        row = _mask_record(source_ids, split="budget_local_holdout", seed=seed)
        retained_cost = sum(costs[source_id] for source_id in row["retained_source_ids"])
        if minimum <= retained_cost <= maximum and row["vector_sha256"] not in observed:
            row["retained_token_cost"] = retained_cost
            local.append(row)
            observed.add(row["vector_sha256"])
        seed += 1
    if len(local) != budget_local_holdout_mask_count:
        raise ValueError("unable to construct enough unique primary-budget-local holdout masks")
    design: dict[str, object] = {
        "method": "task9-preliminary-global-plus-primary-budget-local",
        "global_design": global_design,
        "primary_budget_fraction": float(primary_budget_fraction),
        "requested_primary_budget": requested_budget,
        "attainable_primary_budget": attainable,
        "total_region_token_cost": total_cost,
        "budget_local_tolerance_fraction": float(budget_local_tolerance_fraction),
        "budget_local_bounds": {"minimum": minimum, "maximum": maximum},
        "budget_local_holdout_mask_count": budget_local_holdout_mask_count,
        "budget_local_holdout_masks": local,
    }
    design["design_sha256"] = _canonical_sha256(design)
    return design


def build_regional_intervention_plan(
    mapping: RegionTokenMapping,
    design: Mapping[str, object],
    *,
    mapping_artifact_sha256: str,
    split: str,
) -> tuple[dict[str, object], ...]:
    """Translate whole-source masks into physical post-QTP token retention.

    ContextCite mask booleans mean that a source is retained. Each row therefore
    passes the exact union of that row's retained source memberships to the
    decoder's existing physical-delete intervention.
    """

    if not isinstance(mapping, RegionTokenMapping):
        raise TypeError("regional intervention mapping must be a RegionTokenMapping")
    if not _is_sha256(mapping_artifact_sha256):
        raise ValueError("mapping artifact SHA-256 is invalid")
    try:
        source_ids, fit_masks, holdout_masks = _validated_mask_design(design)
    except ValueError as error:
        raise ValueError("regional mask design is invalid") from error
    identity = design["attribution_identity"]
    if identity["mapping_artifact_sha256"] != mapping_artifact_sha256:
        raise ValueError("mapping artifact does not match the regional mask design")
    if split == "fit":
        masks = fit_masks
    elif split == "holdout":
        masks = holdout_masks
    else:
        raise ValueError("regional intervention split must be fit or holdout")

    population = mapping.geometry_count
    mapping_source_ids = [source.source_id for source in mapping.sources]
    membership = {source.source_id: source.token_ids for source in mapping.sources}
    flattened = [token_id for source in mapping.sources for token_id in source.token_ids]
    if (
        type(population) is not int
        or population <= 0
        or len(mapping.geometry) != population
        or len(mapping.token_to_source) != population
        or mapping_source_ids != source_ids
        or len(set(mapping_source_ids)) != len(mapping_source_ids)
        or sorted(flattened) != list(range(population))
        or any(
            mapping.token_to_source[token_id] != source.source_id
            for source in mapping.sources
            for token_id in source.token_ids
        )
    ):
        raise ValueError("regional mapping is not an exact post-QTP token partition")

    boundary_label = identity["forced_boundary"]
    boundary: str | int = (
        "input" if boundary_label == "B_input" else int(str(boundary_label).removeprefix("B_"))
    )
    plan: list[dict[str, object]] = []
    for mask in masks:
        retained_sources = list(mask["retained_source_ids"])
        retained_visual_ids = tuple(
            sorted(token_id for source_id in retained_sources for token_id in membership[source_id])
        )
        row: dict[str, object] = {
            "split": split,
            "seed": mask["seed"],
            "vector": list(mask["vector"]),
            "vector_sha256": mask["vector_sha256"],
            "attribution_identity_sha256": design["attribution_identity_sha256"],
            "retained_source_ids": retained_sources,
            "retained_visual_ids": list(retained_visual_ids),
            "retained_visual_count": len(retained_visual_ids),
            "visual_population": population,
            "forced_intervention": ForcedVisualIntervention(
                boundary=boundary,
                mode="physical_delete",
                retained_visual_ids=retained_visual_ids,
            ),
        }
        plan.append(row)
    return tuple(plan)


def build_task9_preliminary_intervention_plan(
    mapping: RegionTokenMapping,
    preliminary_design: Mapping[str, object],
    *,
    mapping_artifact_sha256: str,
) -> tuple[dict[str, object], ...]:
    """Append the preliminary pilot's budget-local masks to its global plan."""

    if not isinstance(preliminary_design, Mapping):
        raise ValueError("Task 9 preliminary design is invalid")
    unsigned = dict(preliminary_design)
    supplied_sha = unsigned.pop("design_sha256", None)
    if supplied_sha != _canonical_sha256(unsigned):
        raise ValueError("Task 9 preliminary design digest is invalid")
    global_design = preliminary_design.get("global_design")
    local_masks = preliminary_design.get("budget_local_holdout_masks")
    if not isinstance(global_design, Mapping) or not isinstance(local_masks, list):
        raise ValueError("Task 9 preliminary design schema is invalid")
    global_rows = (
        *build_regional_intervention_plan(
            mapping,
            global_design,
            mapping_artifact_sha256=mapping_artifact_sha256,
            split="fit",
        ),
        *build_regional_intervention_plan(
            mapping,
            global_design,
            mapping_artifact_sha256=mapping_artifact_sha256,
            split="holdout",
        ),
    )
    source_ids = list(global_design["source_ids"])
    membership = {source.source_id: source.token_ids for source in mapping.sources}
    costs = {source.source_id: len(source.token_ids) for source in mapping.sources}
    identity = global_design["attribution_identity"]
    boundary_label = identity["forced_boundary"]
    boundary: str | int = (
        "input" if boundary_label == "B_input" else int(str(boundary_label).removeprefix("B_"))
    )
    if len(local_masks) != preliminary_design.get("budget_local_holdout_mask_count"):
        raise ValueError("Task 9 preliminary local mask count is invalid")
    local_rows: list[dict[str, object]] = []
    for mask in local_masks:
        if (
            not isinstance(mask, Mapping)
            or mask.get("split") != "budget_local_holdout"
            or not isinstance(mask.get("vector"), list)
            or len(mask["vector"]) != len(source_ids)
            or mask.get("vector_sha256") != _canonical_sha256(mask["vector"])
        ):
            raise ValueError("Task 9 preliminary local mask is invalid")
        retained_sources = [
            source_id
            for source_id, retained in zip(source_ids, mask["vector"], strict=True)
            if retained is True
        ]
        if mask.get("retained_source_ids") != retained_sources or mask.get(
            "retained_token_cost"
        ) != sum(costs[source_id] for source_id in retained_sources):
            raise ValueError("Task 9 preliminary local mask membership is invalid")
        retained_visual_ids = tuple(
            sorted(token_id for source_id in retained_sources for token_id in membership[source_id])
        )
        local_rows.append(
            {
                "split": "budget_local_holdout",
                "seed": mask["seed"],
                "vector": list(mask["vector"]),
                "vector_sha256": mask["vector_sha256"],
                "attribution_identity_sha256": global_design["attribution_identity_sha256"],
                "retained_source_ids": retained_sources,
                "retained_visual_ids": list(retained_visual_ids),
                "retained_visual_count": len(retained_visual_ids),
                "visual_population": mapping.geometry_count,
                "forced_intervention": ForcedVisualIntervention(
                    boundary=boundary,
                    mode="physical_delete",
                    retained_visual_ids=retained_visual_ids,
                ),
            }
        )
    return (*global_rows, *local_rows)


def build_task9_selected_arm_plan(
    mapping: RegionTokenMapping,
    selections: Mapping[str, object],
    *,
    boundary: str,
) -> tuple[dict[str, object], ...]:
    """Translate selected whole-region pilot arms into physical interventions."""

    if not isinstance(mapping, RegionTokenMapping) or not _is_forced_boundary(boundary):
        raise ValueError("Task 9 selected-arm mapping or boundary is invalid")
    budgets = selections.get("budgets") if isinstance(selections, Mapping) else None
    if (
        not isinstance(selections.get("question_id"), str)
        or not selections["question_id"]
        or not isinstance(budgets, list)
        or not budgets
    ):
        raise ValueError("Task 9 selected-arm schema is invalid")
    membership = {source.source_id: source.token_ids for source in mapping.sources}
    flattened = [token_id for source in mapping.sources for token_id in source.token_ids]
    if (
        len(membership) != len(mapping.sources)
        or sorted(flattened) != list(range(mapping.geometry_count))
        or len(mapping.token_to_source) != mapping.geometry_count
    ):
        raise ValueError("Task 9 selected-arm mapping is not a token partition")
    decoder_boundary: str | int = (
        "input" if boundary == "B_input" else int(boundary.removeprefix("B_"))
    )
    plan: list[dict[str, object]] = []
    seen: set[tuple[float, str]] = set()
    for budget in budgets:
        arms = budget.get("arms") if isinstance(budget, Mapping) else None
        fraction = budget.get("retained_fraction") if isinstance(budget, Mapping) else None
        achieved = budget.get("achieved_token_count") if isinstance(budget, Mapping) else None
        if (
            isinstance(fraction, bool)
            or not isinstance(fraction, int | float)
            or not isinstance(achieved, int)
            or not isinstance(arms, list)
            or not arms
        ):
            raise ValueError("Task 9 selected-arm budget is invalid")
        for arm in arms:
            name = arm.get("arm") if isinstance(arm, Mapping) else None
            retained_sources = arm.get("retained_source_ids") if isinstance(arm, Mapping) else None
            key = (float(fraction), name) if isinstance(name, str) else None
            if (
                not isinstance(name, str)
                or not name
                or key in seen
                or not isinstance(retained_sources, list)
                or len(set(retained_sources)) != len(retained_sources)
                or any(source_id not in membership for source_id in retained_sources)
            ):
                raise ValueError("Task 9 selected arm is invalid")
            retained_visual_ids = tuple(
                sorted(
                    token_id for source_id in retained_sources for token_id in membership[source_id]
                )
            )
            if arm.get("achieved_token_count") != achieved or len(retained_visual_ids) != achieved:
                raise ValueError("Task 9 selected arm does not match its achieved budget")
            seen.add(key)
            plan.append(
                {
                    "retained_fraction": float(fraction),
                    "arm": name,
                    "retained_source_ids": list(retained_sources),
                    "retained_visual_ids": list(retained_visual_ids),
                    "retained_visual_count": len(retained_visual_ids),
                    "visual_population": mapping.geometry_count,
                    "forced_intervention": ForcedVisualIntervention(
                        boundary=decoder_boundary,
                        mode="physical_delete",
                        retained_visual_ids=retained_visual_ids,
                    ),
                }
            )
    return tuple(plan)


def build_regional_development_plan(
    mapping: RegionTokenMapping,
    primary_design: Mapping[str, object],
    secondary_design: Mapping[str, object],
    *,
    mapping_artifact_sha256: str,
) -> tuple[dict[str, object], ...]:
    """Build the canonical 64-fit then 32-holdout physical intervention plan."""

    primary_identity = primary_design.get("attribution_identity")
    secondary_identity = secondary_design.get("attribution_identity")
    if (
        not isinstance(primary_identity, Mapping)
        or not isinstance(secondary_identity, Mapping)
        or primary_identity.get("target_kind") != _PRIMARY_TARGET_KIND
        or secondary_identity.get("target_kind") != _SECONDARY_TARGET_KIND
    ):
        raise ValueError("Task 9 development designs must bind the two canonical targets")
    shared_identity_keys = {
        "question_id",
        "forced_boundary",
        "mapping_artifact_sha256",
        "prompt_input_sha256",
    }
    if any(primary_identity[key] != secondary_identity[key] for key in shared_identity_keys):
        raise ValueError("Task 9 development target identities do not share one intervention")

    primary_rows = (
        *build_regional_intervention_plan(
            mapping,
            primary_design,
            mapping_artifact_sha256=mapping_artifact_sha256,
            split="fit",
        ),
        *build_regional_intervention_plan(
            mapping,
            primary_design,
            mapping_artifact_sha256=mapping_artifact_sha256,
            split="holdout",
        ),
    )
    secondary_rows = (
        *build_regional_intervention_plan(
            mapping,
            secondary_design,
            mapping_artifact_sha256=mapping_artifact_sha256,
            split="fit",
        ),
        *build_regional_intervention_plan(
            mapping,
            secondary_design,
            mapping_artifact_sha256=mapping_artifact_sha256,
            split="holdout",
        ),
    )
    combined: list[dict[str, object]] = []
    for primary, secondary in zip(primary_rows, secondary_rows, strict=True):
        primary_shared = {
            key: value
            for key, value in primary.items()
            if key not in {"attribution_identity_sha256", "forced_intervention"}
        }
        secondary_shared = {
            key: value
            for key, value in secondary.items()
            if key not in {"attribution_identity_sha256", "forced_intervention"}
        }
        if (
            primary_shared != secondary_shared
            or primary["forced_intervention"] != secondary["forced_intervention"]
        ):
            raise ValueError("Task 9 development targets do not share one physical mask plan")
        combined.append(
            {
                **primary_shared,
                "primary_attribution_identity_sha256": primary_design[
                    "attribution_identity_sha256"
                ],
                "secondary_attribution_identity_sha256": secondary_design[
                    "attribution_identity_sha256"
                ],
                "forced_intervention": primary["forced_intervention"],
            }
        )
    expected_count = len(primary_rows)
    if [row["seed"] for row in combined] != list(range(expected_count)):
        raise ValueError("Task 9 development plan must use consecutive canonical seeds")
    return tuple(combined)


def build_regional_development_targets(
    primary_design: Mapping[str, object],
    secondary_design: Mapping[str, object],
    development_plan: Sequence[Mapping[str, object]],
    *,
    mean_sequence_loglikelihoods: Sequence[Sequence[float]],
    reference_sequence_count: int,
) -> dict[str, object]:
    """Slice one branch-scoring result into two authenticated targets."""

    if type(reference_sequence_count) is not int or reference_sequence_count <= 0:
        raise ValueError("Task 9 requires a positive accepted-reference sequence count")
    try:
        primary_sources, primary_fit, primary_holdout = _validated_mask_design(primary_design)
        secondary_sources, secondary_fit, secondary_holdout = _validated_mask_design(
            secondary_design
        )
    except ValueError as error:
        raise ValueError("Task 9 development target design is invalid") from error
    expected_count = len(primary_fit) + len(primary_holdout)
    if (
        len(development_plan) != expected_count
        or len(mean_sequence_loglikelihoods) != expected_count
    ):
        raise ValueError("Task 9 development target branch count does not match its design")
    if (
        primary_design["attribution_identity"]["target_kind"] != _PRIMARY_TARGET_KIND
        or secondary_design["attribution_identity"]["target_kind"] != _SECONDARY_TARGET_KIND
        or primary_sources != secondary_sources
        or primary_fit != secondary_fit
        or primary_holdout != secondary_holdout
    ):
        raise ValueError("Task 9 development target designs are not a canonical pair")

    primary_values: list[list[float]] = []
    secondary_values: list[list[float]] = []
    primary_outcomes: list[dict[str, object]] = []
    secondary_outcomes: list[dict[str, object]] = []
    for expected_seed, (plan, raw_values) in enumerate(
        zip(development_plan, mean_sequence_loglikelihoods, strict=True)
    ):
        expected_split = "fit" if expected_seed < len(primary_fit) else "holdout"
        if (
            not isinstance(plan, Mapping)
            or plan.get("seed") != expected_seed
            or plan.get("split") != expected_split
            or plan.get("primary_attribution_identity_sha256")
            != primary_design["attribution_identity_sha256"]
            or plan.get("secondary_attribution_identity_sha256")
            != secondary_design["attribution_identity_sha256"]
        ):
            raise ValueError("Task 9 development target plan seed or identity drifted")
        checked = list(raw_values)
        if len(checked) != reference_sequence_count + 1:
            raise ValueError("Task 9 raw likelihood row does not contain references plus response")
        primary_row = {
            **plan,
            "attribution_identity_sha256": primary_design["attribution_identity_sha256"],
        }
        secondary_row = {
            **plan,
            "attribution_identity_sha256": secondary_design["attribution_identity_sha256"],
        }
        primary_slice = checked[:reference_sequence_count]
        secondary_slice = checked[reference_sequence_count:]
        primary_values.append([float(value) for value in primary_slice])
        secondary_values.append([float(value) for value in secondary_slice])
        primary_outcomes.append(
            build_regional_target_outcome(
                primary_design,
                primary_row,
                mean_sequence_loglikelihoods=primary_slice,
            )
        )
        secondary_outcomes.append(
            build_regional_target_outcome(
                secondary_design,
                secondary_row,
                mean_sequence_loglikelihoods=secondary_slice,
            )
        )

    def dataset(
        design: Mapping[str, object],
        values: list[list[float]],
        outcomes: list[dict[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": 1,
            "target_kind": design["attribution_identity"]["target_kind"],
            "attribution_identity_sha256": design["attribution_identity_sha256"],
            "design": dict(design),
            "per_sequence_mean_loglikelihoods": values,
            "outcomes": outcomes,
        }
        result["target_dataset_sha256"] = _canonical_sha256(result)
        return result

    return {
        "schema_version": 1,
        "seed_order": list(range(expected_count)),
        "primary": dataset(primary_design, primary_values, primary_outcomes),
        "secondary": dataset(secondary_design, secondary_values, secondary_outcomes),
    }


def validate_regional_development_targets(
    primary_target: Mapping[str, object],
    secondary_target: Mapping[str, object],
    development_plan: Sequence[Mapping[str, object]],
    mean_sequence_loglikelihoods: Sequence[Sequence[float]],
    *,
    reference_sequence_count: int,
) -> dict[str, object]:
    """Replay both development targets from raw normalized sequence likelihoods."""

    for payload, label in (
        (primary_target, "primary"),
        (secondary_target, "secondary"),
    ):
        if not isinstance(payload, Mapping):
            raise ValueError(f"Task 9 {label} target dataset is invalid")
        unsigned = dict(payload)
        observed = unsigned.pop("target_dataset_sha256", None)
        if not _is_sha256(observed) or observed != _canonical_sha256(unsigned):
            raise ValueError(f"Task 9 {label} target dataset digest is invalid")
    primary_design = primary_target.get("design")
    secondary_design = secondary_target.get("design")
    if not isinstance(primary_design, Mapping) or not isinstance(secondary_design, Mapping):
        raise ValueError("Task 9 target datasets do not contain mask designs")
    expected = build_regional_development_targets(
        primary_design,
        secondary_design,
        development_plan,
        mean_sequence_loglikelihoods=mean_sequence_loglikelihoods,
        reference_sequence_count=reference_sequence_count,
    )
    if primary_target != expected["primary"] or secondary_target != expected["secondary"]:
        raise ValueError("Task 9 stored targets do not match the raw likelihood reconstruction")
    return expected


def build_regional_target_outcome(
    design: Mapping[str, object],
    intervention: Mapping[str, object],
    *,
    mean_sequence_loglikelihoods: Sequence[float],
) -> dict[str, object]:
    """Bind one physical mask to its already per-token-normalized target."""

    try:
        _, fit_masks, holdout_masks = _validated_mask_design(design)
    except ValueError as error:
        raise ValueError("regional mask design is invalid") from error
    split = intervention.get("split") if isinstance(intervention, Mapping) else None
    masks = fit_masks if split == "fit" else holdout_masks if split == "holdout" else None
    seed = intervention.get("seed") if isinstance(intervention, Mapping) else None
    expected = (
        next((mask for mask in masks if mask["seed"] == seed), None) if masks is not None else None
    )
    if (
        expected is None
        or intervention.get("vector_sha256") != expected["vector_sha256"]
        or intervention.get("attribution_identity_sha256") != design["attribution_identity_sha256"]
    ):
        raise ValueError("regional target outcome intervention identity is invalid")
    values = tuple(mean_sequence_loglikelihoods)
    if not values or any(
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        for value in values
    ):
        raise ValueError("regional target outcome requires finite normalized likelihoods")
    target_kind = design["attribution_identity"]["target_kind"]
    if target_kind == _SECONDARY_TARGET_KIND:
        if len(values) != 1:
            raise ValueError("generated-response attribution requires exactly one target sequence")
        target = float(values[0])
    else:
        target = max(float(value) for value in values)
    return {
        "split": split,
        "seed": seed,
        "vector_sha256": expected["vector_sha256"],
        "attribution_identity_sha256": design["attribution_identity_sha256"],
        "normalized_target": target,
    }


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
        "fit_mask_count",
        "holdout_mask_count",
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
    fit_mask_count = design["fit_mask_count"]
    holdout_mask_count = design["holdout_mask_count"]
    if (
        type(fit_mask_count) is not int
        or fit_mask_count <= 0
        or type(holdout_mask_count) is not int
        or holdout_mask_count <= 0
    ):
        raise ValueError("ContextCite mask design counts are invalid")
    expected_fit = [
        _mask_record(source_ids, split="fit", seed=seed) for seed in range(fit_mask_count)
    ]
    expected_holdout = [
        _mask_record(source_ids, split="holdout", seed=seed)
        for seed in range(fit_mask_count, fit_mask_count + holdout_mask_count)
    ]
    if _canonical_bytes(design["fit_masks"]) != _canonical_bytes(expected_fit) or _canonical_bytes(
        design["holdout_masks"]
    ) != _canonical_bytes(expected_holdout):
        raise ValueError("ContextCite mask design is not the canonical seed schedule")
    return source_ids, expected_fit, expected_holdout


def fit_contextcite_lasso(
    design: Mapping[str, object],
    fit_outcomes: Sequence[Mapping[str, object]],
    *,
    target_scale: str = _MEAN_LOGLIKELIHOOD_SCALE,
) -> dict[str, object]:
    """Fit the pinned ContextCite Lasso to already normalized sequence targets.

    Upstream divides summed token outputs by ``num_output_tokens`` and rescales
    the coefficients afterward. Task 9 supplies a mean log-likelihood target
    directly, so the faithful adapter uses the upstream solver with an
    effective token count of one and records that interface change.
    """

    source_ids, fit_masks, _ = _validated_mask_design(design)
    adaptation = _surrogate_adaptation(target_scale)
    fit_mask_count = len(fit_masks)
    if (
        not isinstance(fit_outcomes, Sequence)
        or isinstance(fit_outcomes, str | bytes)
        or len(fit_outcomes) != fit_mask_count
    ):
        raise ValueError("ContextCite fitting outcomes do not match the frozen fit masks")
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
        "fit_mask_count": fit_mask_count,
        "target": target_scale,
        "adaptation": adaptation,
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
    target_scale: str = _MEAN_LOGLIKELIHOOD_SCALE,
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
    adaptation = _surrogate_adaptation(target_scale)
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
        or surrogate["fit_mask_count"] != len(fit_masks)
        or surrogate["target"] != target_scale
        or surrogate["adaptation"] != adaptation
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


def evaluate_contextcite_explicit_mask_fidelity(
    *,
    source_ids: Sequence[str],
    coefficients: Mapping[str, float],
    intercept: float,
    constant_prediction: float,
    masks: Sequence[Sequence[bool]],
    targets: Sequence[float],
    split: str,
) -> dict[str, object]:
    """Evaluate a fitted surrogate on sealed masks outside the global schedule."""

    checked_ids = tuple(source_ids)
    if (
        not checked_ids
        or any(not isinstance(source_id, str) or not source_id for source_id in checked_ids)
        or len(set(checked_ids)) != len(checked_ids)
        or set(coefficients) != set(checked_ids)
        or not isinstance(split, str)
        or not split
        or not masks
        or len(masks) != len(targets)
    ):
        raise ValueError("explicit ContextCite fidelity inputs are invalid")
    checked_coefficients = {
        source_id: _finite_float(coefficients[source_id], label="ContextCite coefficient")
        for source_id in checked_ids
    }
    checked_intercept = _finite_float(intercept, label="ContextCite intercept")
    checked_constant = _finite_float(constant_prediction, label="ContextCite constant prediction")
    checked_masks: list[list[bool]] = []
    predictions: list[float] = []
    checked_targets: list[float] = []
    for raw_mask, raw_target in zip(masks, targets, strict=True):
        vector = list(raw_mask)
        if len(vector) != len(checked_ids) or any(type(value) is not bool for value in vector):
            raise ValueError("explicit ContextCite masks must be boolean source vectors")
        target = _finite_float(raw_target, label="ContextCite normalized target")
        prediction = checked_intercept + sum(
            checked_coefficients[source_id] * float(retained)
            for source_id, retained in zip(checked_ids, vector, strict=True)
        )
        checked_masks.append(vector)
        checked_targets.append(target)
        predictions.append(prediction)
    lds = _spearman_rank_correlation(predictions, checked_targets)
    heldout_rmse = _root_mean_square(
        [prediction - target for prediction, target in zip(predictions, checked_targets)]
    )
    constant_rmse = _root_mean_square([checked_constant - target for target in checked_targets])
    result: dict[str, object] = {
        "method": "contextcite-explicit-mask-fidelity",
        "split": split,
        "mask_count": len(checked_masks),
        "lds_definition": "spearman-rank-correlation-average-ties",
        "lds_spearman": lds,
        "lds_defined": lds is not None,
        "heldout_rmse": heldout_rmse,
        "constant_baseline": "fit-target-mean",
        "constant_prediction": checked_constant,
        "constant_rmse": constant_rmse,
        "surrogate_beats_constant": heldout_rmse < constant_rmse,
        "masks_sha256": _canonical_sha256(checked_masks),
        "targets_sha256": _canonical_sha256(checked_targets),
        "predictions_sha256": _canonical_sha256(predictions),
    }
    result["fidelity_sha256"] = _canonical_sha256(result)
    return result


def evaluate_contextcite_holdout(
    design: Mapping[str, object],
    fit_outcomes: Sequence[Mapping[str, object]],
    surrogate: Mapping[str, object],
    holdout_outcomes: Sequence[Mapping[str, object]],
    *,
    target_scale: str = _MEAN_LOGLIKELIHOOD_SCALE,
) -> dict[str, object]:
    """Evaluate one question's surrogate on its frozen holdout.

    LDS is exactly Spearman rank correlation with average ranks for ties. The
    constant comparator is the fit-target mean persisted when the surrogate is
    fit; held-out targets never determine that baseline.
    """

    source_ids, fit_masks, holdout_masks = _validated_mask_design(design)
    expected_surrogate = fit_contextcite_lasso(design, fit_outcomes, target_scale=target_scale)
    coefficients, intercept, fit_target_mean = _validated_surrogate(
        surrogate,
        design=design,
        source_ids=source_ids,
        fit_masks=fit_masks,
        target_scale=target_scale,
    )
    if _canonical_bytes(surrogate) != _canonical_bytes(expected_surrogate):
        raise ValueError("ContextCite surrogate does not match the canonical fit outcomes")
    if (
        not isinstance(holdout_outcomes, Sequence)
        or isinstance(holdout_outcomes, str | bytes)
        or len(holdout_outcomes) != len(holdout_masks)
    ):
        raise ValueError("ContextCite fidelity outcomes do not match the frozen holdout masks")

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
        "holdout_mask_count": len(holdout_masks),
        "holdout_seed_start": len(fit_masks),
        "holdout_seed_stop_exclusive": len(fit_masks) + len(holdout_masks),
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


def build_task9_preliminary_arm_selections(
    regions: Sequence[Mapping[str, object]],
    *,
    question_id: str,
    requested_token_count: int,
    gold_support_scores: Mapping[str, float],
    gold_margin_scores: Mapping[str, float] | None = None,
) -> dict[str, object]:
    """Select ContextCite/random arms at native DocPrune's token budget."""

    if not isinstance(question_id, str) or not question_id:
        raise ValueError("question_id must be a nonempty string")
    validated = _validated_regions(regions, allow_zero_cost=False)
    checked_regions = [
        {"source_id": source_id, "token_cost": cost} for source_id, cost in validated
    ]
    source_ids = [source_id for source_id, _ in validated]
    score_sets: list[tuple[str, Mapping[str, float]]] = [
        ("contextcite_gold_support", gold_support_scores),
    ]
    if gold_margin_scores is not None:
        score_sets.append(("contextcite_gold_margin", gold_margin_scores))
    random_scores = {
        source_id: int.from_bytes(
            hashlib.sha256(
                f"task9-preliminary-random-v1\0{question_id}\0{source_id}".encode()
            ).digest()[:8],
            "big",
        )
        / 2**64
        for source_id in source_ids
    }
    score_sets.append(("random_region_size_aware", random_scores))

    total_cost = sum(cost for _, cost in validated)
    if (
        type(requested_token_count) is not int
        or requested_token_count <= 0
        or requested_token_count > total_cost
    ):
        raise ValueError("requested token count must be within the regional token population")
    arms: list[dict[str, object]] = []
    for arm, scores in score_sets:
        selected = whole_region_knapsack(
            checked_regions,
            coefficients=scores,
            requested_budget=requested_token_count,
        )
        arms.append(
            {
                "arm": arm,
                "retained_source_ids": selected["top_source_ids"],
                "achieved_token_count": selected["achieved_budget"],
            }
        )
    budgets = [
        {
            "retained_fraction": requested_token_count / total_cost,
            "requested_token_count": requested_token_count,
            "achieved_token_count": arms[0]["achieved_token_count"],
            "arms": arms,
        }
    ]
    result: dict[str, object] = {
        "schema_version": "docprune-task9-preliminary-dynamic-arm-selections-v1",
        "question_id": question_id,
        "total_region_token_count": total_cost,
        "gold_margin_available": gold_margin_scores is not None,
        "budgets": budgets,
    }
    result["selections_sha256"] = _canonical_sha256(result)
    return result


def analyze_task9_preliminary_question(
    *,
    question_id: str,
    retained_fraction: float,
    arm_results: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Build one self-contained preliminary-pilot result for later aggregation."""

    required = {"unpruned", "native_docprune", "contextcite_gold_support"}
    if not isinstance(question_id, str) or not question_id:
        raise ValueError("question_id must be a nonempty string")
    if set(arm_results) < required:
        raise ValueError("arm_results is missing a required preliminary pilot arm")
    if (
        isinstance(retained_fraction, bool)
        or not isinstance(retained_fraction, int | float)
        or not math.isfinite(float(retained_fraction))
        or not 0 < float(retained_fraction) <= 1
    ):
        raise ValueError("retained_fraction must be finite and in (0, 1]")

    checked: dict[str, dict[str, object]] = {}
    for arm, raw in arm_results.items():
        f1 = _finite_float(raw.get("normalized_token_f1"), label=f"{arm} token F1")
        if not 0 <= f1 <= 1:
            raise ValueError(f"{arm} token F1 must be in [0, 1]")
        exact_match = raw.get("exact_match")
        if type(exact_match) is not bool:
            raise ValueError(f"{arm} exact_match must be boolean")
        gold = _finite_float(raw.get("gold_mean_loglikelihood"), label=f"{arm} gold likelihood")
        alternative_raw = raw.get("alternative_mean_loglikelihood")
        alternative = (
            None
            if alternative_raw is None
            else _finite_float(alternative_raw, label=f"{arm} alternative likelihood")
        )
        checked[arm] = {
            **dict(raw),
            "normalized_token_f1": f1,
            "exact_match": exact_match,
            "gold_mean_loglikelihood": gold,
            "alternative_mean_loglikelihood": alternative,
            "gold_vs_alternative_margin": None if alternative is None else gold - alternative,
        }

    baseline = checked["unpruned"]
    docprune = checked["native_docprune"]
    comparisons: dict[str, dict[str, object]] = {}
    for contextcite_arm in ("contextcite_gold_support", "contextcite_gold_margin"):
        if contextcite_arm not in checked:
            continue
        contextcite = checked[contextcite_arm]
        f1_difference = float(contextcite["normalized_token_f1"]) - float(
            docprune["normalized_token_f1"]
        )
        contextcite_margin = contextcite["gold_vs_alternative_margin"]
        docprune_margin = docprune["gold_vs_alternative_margin"]
        comparisons[f"{contextcite_arm}_vs_docprune"] = {
            "normalized_token_f1_difference": f1_difference,
            "exact_match_difference": int(contextcite["exact_match"])
            - int(docprune["exact_match"]),
            "win_tie_loss": "win" if f1_difference > 0 else "loss" if f1_difference < 0 else "tie",
            "rescue": bool(contextcite["exact_match"] and not docprune["exact_match"]),
            "preservation_advantage": bool(
                baseline["exact_match"]
                and contextcite["exact_match"]
                and not docprune["exact_match"]
            ),
            "gold_likelihood_difference": float(contextcite["gold_mean_loglikelihood"])
            - float(docprune["gold_mean_loglikelihood"]),
            "gold_likelihood_change_from_unpruned": float(contextcite["gold_mean_loglikelihood"])
            - float(baseline["gold_mean_loglikelihood"]),
            "gold_vs_alternative_margin_difference": (
                None
                if contextcite_margin is None or docprune_margin is None
                else float(contextcite_margin) - float(docprune_margin)
            ),
        }
    result: dict[str, object] = {
        "schema_version": "docprune-task9-preliminary-question-analysis-v1",
        "question_id": question_id,
        "retained_fraction": float(retained_fraction),
        "baseline_correct": bool(baseline["exact_match"]),
        "arm_results": checked,
        **comparisons,
    }
    result["analysis_sha256"] = _canonical_sha256(result)
    return result


def evaluate_contextcite_refit_stability(
    design: Mapping[str, object],
    fit_outcomes: Sequence[Mapping[str, object]],
    surrogate: Mapping[str, object],
    regions: Sequence[Mapping[str, object]],
    *,
    requested_budget: int,
    target_scale: str = _MEAN_LOGLIKELIHOOD_SCALE,
) -> dict[str, object]:
    """Evaluate deterministic five-refit coefficient and selection stability."""

    source_ids, fit_masks, _ = _validated_mask_design(design)
    expected_surrogate = fit_contextcite_lasso(design, fit_outcomes, target_scale=target_scale)
    _validated_surrogate(
        surrogate,
        design=design,
        source_ids=source_ids,
        fit_masks=fit_masks,
        target_scale=target_scale,
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
        sampled = np.random.RandomState(seed).choice(
            len(fit_masks), size=len(fit_masks), replace=True
        )
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
        "bootstrap_draw_count": len(fit_masks),
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


def analyze_contextcite_development_question(
    design: Mapping[str, object],
    outcomes: Sequence[Mapping[str, object]],
    regions: Sequence[Mapping[str, object]],
    *,
    requested_budget: int,
    target_scale: str = _MEAN_LOGLIKELIHOOD_SCALE,
) -> dict[str, object]:
    """Analyze one Task 9 question without fabricating a cross-question LDS interval."""

    _, fit_masks, holdout_masks = _validated_mask_design(design)
    if not isinstance(outcomes, Sequence) or isinstance(outcomes, str | bytes):
        raise ValueError("Task 9 one-question analysis requires ordered outcomes")
    fit_mask_count = len(fit_masks)
    total_mask_count = fit_mask_count + len(holdout_masks)
    fit_outcomes = list(outcomes[:fit_mask_count])
    holdout_outcomes = list(outcomes[fit_mask_count:])
    if (
        len(fit_outcomes) != len(fit_masks)
        or len(holdout_outcomes) != len(holdout_masks)
        or [row.get("seed") for row in outcomes] != list(range(total_mask_count))
    ):
        raise ValueError("Task 9 one-question analysis requires consecutive canonical seeds")
    surrogate = fit_contextcite_lasso(design, fit_outcomes, target_scale=target_scale)
    fidelity = evaluate_contextcite_holdout(
        design,
        fit_outcomes,
        surrogate,
        holdout_outcomes,
        target_scale=target_scale,
    )
    stability = evaluate_contextcite_refit_stability(
        design,
        fit_outcomes,
        surrogate,
        regions,
        requested_budget=requested_budget,
        target_scale=target_scale,
    )
    result: dict[str, object] = {
        "schema_version": 1,
        "status": "analyzed-task9-one-question-development",
        "scope": "one-question-development-feasibility",
        "cross_question_inference_status": "deferred-to-multi-question-development",
        "attribution_identity": dict(design["attribution_identity"]),
        "attribution_identity_sha256": design["attribution_identity_sha256"],
        "surrogate": surrogate,
        "fidelity": fidelity,
        "stability": stability,
    }
    result["analysis_sha256"] = _canonical_sha256(result)
    return result


def analyze_task9_preliminary_attribution(
    *,
    primary_design: Mapping[str, object],
    primary_outcomes: Sequence[Mapping[str, object]],
    secondary_design: Mapping[str, object],
    secondary_outcomes: Sequence[Mapping[str, object]],
    regions: Sequence[Mapping[str, object]],
    requested_token_count: int,
    budget_local_masks: Sequence[Sequence[bool]],
    budget_local_primary_targets: Sequence[float],
    budget_local_secondary_targets: Sequence[float],
    gold_margin_available: bool,
) -> dict[str, object]:
    """Fit the preliminary pilot targets and produce matched arm selections."""

    primary_identity = _validated_attribution_identity(
        primary_design.get("attribution_identity"),
        expected_sha256=primary_design.get("attribution_identity_sha256"),
    )
    secondary_identity = _validated_attribution_identity(
        secondary_design.get("attribution_identity"),
        expected_sha256=secondary_design.get("attribution_identity_sha256"),
    )
    identity_fields = (
        "question_id",
        "forced_boundary",
        "mapping_artifact_sha256",
        "prompt_input_sha256",
    )
    if (
        primary_identity["target_kind"] != _PRIMARY_TARGET_KIND
        or secondary_identity["target_kind"] != _SECONDARY_TARGET_KIND
        or any(primary_identity[key] != secondary_identity[key] for key in identity_fields)
        or type(gold_margin_available) is not bool
        or len(budget_local_masks) != len(budget_local_primary_targets)
        or len(budget_local_masks) != len(budget_local_secondary_targets)
    ):
        raise ValueError("preliminary attribution target identities are incompatible")
    validated_regions = _validated_regions(regions, allow_zero_cost=False)
    checked_regions = [
        {"source_id": source_id, "token_cost": cost} for source_id, cost in validated_regions
    ]
    source_ids = [source_id for source_id, _ in validated_regions]
    total_cost = sum(cost for _, cost in validated_regions)
    if (
        type(requested_token_count) is not int
        or requested_token_count <= 0
        or requested_token_count > total_cost
    ):
        raise ValueError("native DocPrune budget is outside the regional token population")
    requested_primary_budget = requested_token_count

    primary = analyze_contextcite_development_question(
        primary_design,
        primary_outcomes,
        checked_regions,
        requested_budget=requested_primary_budget,
    )
    primary_surrogate = primary["surrogate"]
    primary_local = evaluate_contextcite_explicit_mask_fidelity(
        source_ids=source_ids,
        coefficients=primary_surrogate["coefficients"],
        intercept=primary_surrogate["intercept"],
        constant_prediction=primary_surrogate["fit_target_mean"],
        masks=budget_local_masks,
        targets=budget_local_primary_targets,
        split="budget-local-holdout",
    )

    margin: dict[str, object]
    margin_scores: Mapping[str, float] | None = None
    if gold_margin_available:
        if len(primary_outcomes) != len(secondary_outcomes):
            raise ValueError("gold-margin targets require paired global outcomes")
        margin_design = build_region_mask_design(
            checked_regions,
            question_id=primary_identity["question_id"],
            forced_boundary=primary_identity["forced_boundary"],
            mapping_artifact_sha256=primary_identity["mapping_artifact_sha256"],
            prompt_input_sha256=primary_identity["prompt_input_sha256"],
            target_kind=_MARGIN_TARGET_KIND,
            reference_set_token_ids_sha256=primary_identity["reference_set_token_ids_sha256"],
            generated_response_token_ids_sha256=secondary_identity[
                "generated_response_token_ids_sha256"
            ],
            fit_mask_count=primary_design["fit_mask_count"],
            holdout_mask_count=primary_design["holdout_mask_count"],
        )
        masks = [*margin_design["fit_masks"], *margin_design["holdout_masks"]]
        margin_outcomes: list[dict[str, object]] = []
        for primary_row, secondary_row, mask in zip(
            primary_outcomes, secondary_outcomes, masks, strict=True
        ):
            if (
                primary_row.get("split") != secondary_row.get("split")
                or primary_row.get("seed") != secondary_row.get("seed")
                or primary_row.get("vector_sha256") != secondary_row.get("vector_sha256")
                or primary_row.get("vector_sha256") != mask["vector_sha256"]
            ):
                raise ValueError("gold-margin global outcomes are not paired by mask")
            margin_outcomes.append(
                {
                    "split": mask["split"],
                    "seed": mask["seed"],
                    "vector_sha256": mask["vector_sha256"],
                    "attribution_identity_sha256": margin_design["attribution_identity_sha256"],
                    "normalized_target": _finite_float(
                        primary_row.get("normalized_target"), label="gold target"
                    )
                    - _finite_float(
                        secondary_row.get("normalized_target"), label="alternative target"
                    ),
                }
            )
        margin_analysis = analyze_contextcite_development_question(
            margin_design,
            margin_outcomes,
            checked_regions,
            requested_budget=requested_primary_budget,
        )
        margin_surrogate = margin_analysis["surrogate"]
        margin_local_targets = [
            _finite_float(gold, label="local gold target")
            - _finite_float(alternative, label="local alternative target")
            for gold, alternative in zip(
                budget_local_primary_targets,
                budget_local_secondary_targets,
                strict=True,
            )
        ]
        margin = {
            "available": True,
            "global_fidelity": margin_analysis["fidelity"],
            "budget_local_fidelity": evaluate_contextcite_explicit_mask_fidelity(
                source_ids=source_ids,
                coefficients=margin_surrogate["coefficients"],
                intercept=margin_surrogate["intercept"],
                constant_prediction=margin_surrogate["fit_target_mean"],
                masks=budget_local_masks,
                targets=margin_local_targets,
                split="budget-local-holdout",
            ),
            "surrogate": margin_surrogate,
            "stability": margin_analysis["stability"],
        }
        margin_scores = margin_surrogate["coefficients"]
    else:
        margin = {
            "available": False,
            "reason": "unpruned response is not a distinct non-gold alternative",
        }

    selections = build_task9_preliminary_arm_selections(
        checked_regions,
        question_id=primary_identity["question_id"],
        requested_token_count=requested_primary_budget,
        gold_support_scores=primary_surrogate["coefficients"],
        gold_margin_scores=margin_scores,
    )
    result: dict[str, object] = {
        "schema_version": "docprune-task9-preliminary-attribution-analysis-v1",
        "question_id": primary_identity["question_id"],
        "forced_boundary": primary_identity["forced_boundary"],
        "gold_support": {
            "global_fidelity": primary["fidelity"],
            "budget_local_fidelity": primary_local,
            "surrogate": primary_surrogate,
            "stability": primary["stability"],
        },
        "gold_margin": margin,
        "selections": selections,
    }
    result["analysis_sha256"] = _canonical_sha256(result)
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


def _validated_aggregate_fidelity(value: object) -> dict[str, object]:
    required = {
        "method",
        "mask_design_sha256",
        "attribution_identity",
        "attribution_identity_sha256",
        "surrogate_sha256",
        "fit_outcomes_sha256",
        "fit_targets_sha256",
        "holdout_mask_count",
        "holdout_seed_start",
        "holdout_seed_stop_exclusive",
        "lds_definition",
        "lds_spearman",
        "lds_defined",
        "heldout_rmse",
        "constant_baseline",
        "constant_prediction",
        "constant_rmse",
        "surrogate_beats_constant",
        "holdout_masks_sha256",
        "holdout_outcomes_sha256",
        "holdout_targets_sha256",
        "holdout_predictions_sha256",
        "fidelity_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("per-question fidelity artifact schema is invalid")
    fidelity = dict(value)
    digest = fidelity.pop("fidelity_sha256")
    if not _is_sha256(digest) or digest != _canonical_sha256(fidelity):
        raise ValueError("per-question fidelity artifact digest is invalid")
    identity = _validated_attribution_identity(
        fidelity["attribution_identity"],
        expected_sha256=fidelity["attribution_identity_sha256"],
    )
    hashes = (
        "mask_design_sha256",
        "surrogate_sha256",
        "fit_outcomes_sha256",
        "fit_targets_sha256",
        "holdout_masks_sha256",
        "holdout_outcomes_sha256",
        "holdout_targets_sha256",
        "holdout_predictions_sha256",
    )
    if (
        fidelity["method"] != "contextcite-per-question-heldout-fidelity"
        or any(not _is_sha256(fidelity[key]) for key in hashes)
        or type(fidelity["holdout_mask_count"]) is not int
        or fidelity["holdout_mask_count"] != 32
        or type(fidelity["holdout_seed_start"]) is not int
        or fidelity["holdout_seed_start"] != 64
        or type(fidelity["holdout_seed_stop_exclusive"]) is not int
        or fidelity["holdout_seed_stop_exclusive"] != 96
        or fidelity["lds_definition"] != "spearman-rank-correlation-average-ties"
        or fidelity["constant_baseline"] != "fit-target-mean"
        or type(fidelity["lds_defined"]) is not bool
        or type(fidelity["surrogate_beats_constant"]) is not bool
    ):
        raise ValueError("per-question fidelity artifact contract is invalid")
    lds: float | None
    if fidelity["lds_defined"]:
        lds = _finite_float(fidelity["lds_spearman"], label="per-question LDS")
        if not -1.0 <= lds <= 1.0:
            raise ValueError("per-question LDS must be a correlation")
    elif fidelity["lds_spearman"] is None:
        lds = None
    else:
        raise ValueError("undefined per-question LDS must be explicit")
    heldout_rmse = _finite_float(fidelity["heldout_rmse"], label="heldout RMSE")
    constant_prediction = _finite_float(
        fidelity["constant_prediction"], label="constant prediction"
    )
    constant_rmse = _finite_float(fidelity["constant_rmse"], label="constant RMSE")
    if heldout_rmse < 0 or constant_rmse < 0:
        raise ValueError("per-question RMSE must be nonnegative")
    if fidelity["surrogate_beats_constant"] != (heldout_rmse < constant_rmse):
        raise ValueError("per-question RMSE comparison is inconsistent")
    return {
        **fidelity,
        "attribution_identity": identity,
        "lds_spearman": lds,
        "heldout_rmse": heldout_rmse,
        "constant_prediction": constant_prediction,
        "constant_rmse": constant_rmse,
        "fidelity_sha256": digest,
    }


def _validated_pairwise_rows(
    value: object, *, metric: str, label: str
) -> tuple[list[dict[str, object]], list[float | None]]:
    if not isinstance(value, list) or len(value) != 10:
        raise ValueError(f"{label} must contain the exact ten refit pairs")
    expected_pairs = tuple(combinations(range(5), 2))
    rows: list[dict[str, object]] = []
    values: list[float | None] = []
    for raw, (left, right) in zip(value, expected_pairs, strict=True):
        if not isinstance(raw, Mapping) or set(raw) != {
            "left_seed",
            "right_seed",
            metric,
            "defined",
        }:
            raise ValueError(f"{label} row schema is invalid")
        if (
            type(raw["left_seed"]) is not int
            or raw["left_seed"] != left
            or type(raw["right_seed"]) is not int
            or raw["right_seed"] != right
            or type(raw["defined"]) is not bool
        ):
            raise ValueError(f"{label} order or defined flag is invalid")
        raw_metric = raw[metric]
        if raw["defined"]:
            checked = _finite_float(raw_metric, label=label)
            lower_bound = 0.0 if metric == "jaccard" else -1.0
            if not lower_bound <= checked <= 1.0:
                raise ValueError(f"{label} value is outside its range")
        elif raw_metric is None:
            checked = None
        else:
            raise ValueError(f"undefined {label} must be explicit")
        rows.append(dict(raw, **{metric: checked}))
        values.append(checked)
    return rows, values


def _validated_aggregate_stability(value: object) -> dict[str, object]:
    required = {
        "method",
        "bootstrap_rng",
        "bootstrap_seed_start",
        "bootstrap_seed_stop_exclusive",
        "bootstrap_draw_count",
        "bootstrap_replace",
        "mask_design_sha256",
        "attribution_identity",
        "attribution_identity_sha256",
        "surrogate_sha256",
        "fit_outcomes_sha256",
        "fit_targets_sha256",
        "regions_sha256",
        "source_ids",
        "requested_budget",
        "achieved_budget",
        "refits",
        "coefficient_pairwise",
        "coefficient_summary",
        "top_selection_pairwise",
        "top_selection_summary",
        "stability_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("per-question stability artifact schema is invalid")
    stability = dict(value)
    digest = stability.pop("stability_sha256")
    if not _is_sha256(digest) or digest != _canonical_sha256(stability):
        raise ValueError("per-question stability artifact digest is invalid")
    identity = _validated_attribution_identity(
        stability["attribution_identity"],
        expected_sha256=stability["attribution_identity_sha256"],
    )
    source_ids = stability["source_ids"]
    if (
        stability["method"] != "contextcite-five-bootstrap-refit-stability"
        or stability["bootstrap_rng"] != "numpy-legacy-randomstate-choice"
        or type(stability["bootstrap_seed_start"]) is not int
        or stability["bootstrap_seed_start"] != 0
        or type(stability["bootstrap_seed_stop_exclusive"]) is not int
        or stability["bootstrap_seed_stop_exclusive"] != 5
        or type(stability["bootstrap_draw_count"]) is not int
        or stability["bootstrap_draw_count"] != 64
        or stability["bootstrap_replace"] is not True
        or any(
            not _is_sha256(stability[key])
            for key in (
                "mask_design_sha256",
                "surrogate_sha256",
                "fit_outcomes_sha256",
                "fit_targets_sha256",
                "regions_sha256",
            )
        )
        or not isinstance(source_ids, list)
        or not source_ids
        or any(not isinstance(source_id, str) or not source_id for source_id in source_ids)
        or len(set(source_ids)) != len(source_ids)
        or type(stability["requested_budget"]) is not int
        or stability["requested_budget"] < 0
        or type(stability["achieved_budget"]) is not int
        or not 0 <= stability["achieved_budget"] <= stability["requested_budget"]
    ):
        raise ValueError("per-question stability artifact contract is invalid")
    raw_refits = stability["refits"]
    if not isinstance(raw_refits, list) or len(raw_refits) != 5:
        raise ValueError("per-question stability requires five refits")
    refit_keys = {
        "seed",
        "resample_indices",
        "resample_indices_sha256",
        "coefficients",
        "coefficients_sha256",
        "intercept",
        "top_source_ids",
        "top_source_ids_sha256",
        "achieved_budget",
        "refit_sha256",
    }
    checked_coefficients: list[dict[str, float]] = []
    checked_top_source_ids: list[list[str]] = []
    for seed, raw_refit in enumerate(raw_refits):
        if not isinstance(raw_refit, Mapping) or set(raw_refit) != refit_keys:
            raise ValueError("stability refit schema is invalid")
        refit = dict(raw_refit)
        refit_digest = refit.pop("refit_sha256")
        indices = refit["resample_indices"]
        coefficients = refit["coefficients"]
        top_ids = refit["top_source_ids"]
        expected_indices = [
            int(index)
            for index in np.random.RandomState(seed).choice(64, size=64, replace=True).tolist()
        ]
        if (
            type(refit["seed"]) is not int
            or refit["seed"] != seed
            or not isinstance(indices, list)
            or indices != expected_indices
            or refit["resample_indices_sha256"] != _canonical_sha256(indices)
            or not isinstance(coefficients, Mapping)
            or set(coefficients) != set(source_ids)
            or any(
                not math.isfinite(_finite_float(coefficients[source_id], label="refit coefficient"))
                for source_id in source_ids
            )
            or refit["coefficients_sha256"] != _canonical_sha256(coefficients)
            or not math.isfinite(_finite_float(refit["intercept"], label="refit intercept"))
            or not isinstance(top_ids, list)
            or any(source_id not in source_ids for source_id in top_ids)
            or len(top_ids) != len(set(top_ids))
            or refit["top_source_ids_sha256"] != _canonical_sha256(top_ids)
            or type(refit["achieved_budget"]) is not int
            or refit["achieved_budget"] != stability["achieved_budget"]
            or not _is_sha256(refit_digest)
            or refit_digest != _canonical_sha256(refit)
        ):
            raise ValueError("stability refit identity is invalid")
        checked_coefficients.append(
            {
                source_id: _finite_float(coefficients[source_id], label="refit coefficient")
                for source_id in source_ids
            }
        )
        checked_top_source_ids.append(list(top_ids))
    coefficient_rows, _ = _validated_pairwise_rows(
        stability["coefficient_pairwise"], metric="spearman", label="coefficient stability"
    )
    selection_rows, _ = _validated_pairwise_rows(
        stability["top_selection_pairwise"], metric="jaccard", label="selection stability"
    )
    expected_coefficient_rows: list[dict[str, object]] = []
    expected_selection_rows: list[dict[str, object]] = []
    for left, right in combinations(range(5), 2):
        correlation = _spearman_rank_correlation(
            [checked_coefficients[left][source_id] for source_id in source_ids],
            [checked_coefficients[right][source_id] for source_id in source_ids],
        )
        expected_coefficient_rows.append(
            {
                "left_seed": left,
                "right_seed": right,
                "spearman": correlation,
                "defined": correlation is not None,
            }
        )
        left_selection = set(checked_top_source_ids[left])
        right_selection = set(checked_top_source_ids[right])
        union = left_selection | right_selection
        jaccard = len(left_selection & right_selection) / len(union) if union else None
        expected_selection_rows.append(
            {
                "left_seed": left,
                "right_seed": right,
                "jaccard": jaccard,
                "defined": jaccard is not None,
            }
        )
    expected_coefficient_values = [row["spearman"] for row in expected_coefficient_rows]
    expected_selection_values = [row["jaccard"] for row in expected_selection_rows]
    if _canonical_bytes(coefficient_rows) != _canonical_bytes(expected_coefficient_rows):
        raise ValueError("coefficient stability pairs are inconsistent with refits")
    if _canonical_bytes(selection_rows) != _canonical_bytes(expected_selection_rows):
        raise ValueError("selection stability pairs are inconsistent with refits")
    if _canonical_bytes(stability["coefficient_summary"]) != _canonical_bytes(
        _defined_pairwise_summary(expected_coefficient_values)
    ):
        raise ValueError("coefficient stability summary is inconsistent")
    if _canonical_bytes(stability["top_selection_summary"]) != _canonical_bytes(
        _defined_pairwise_summary(expected_selection_values)
    ):
        raise ValueError("selection stability summary is inconsistent")
    return {
        **stability,
        "attribution_identity": identity,
        "coefficient_pairwise": coefficient_rows,
        "top_selection_pairwise": selection_rows,
        "stability_sha256": digest,
    }


def _bootstrap_quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def aggregate_task9_preliminary_pilot(
    question_rows: Sequence[Mapping[str, object]],
    *,
    expected_qids: Sequence[str],
    natural_pool_weights: Mapping[str, float],
    draws: int = 10_000,
    seed: int = 20_260_901,
) -> dict[str, object]:
    """Aggregate the paired, correct/wrong-stratified Task 9 preliminary pilot."""

    strata = ("baseline_correct", "baseline_wrong")
    expected = tuple(expected_qids)
    observed = tuple(row.get("question_id") for row in question_rows)
    if (
        not expected
        or len(set(expected)) != len(expected)
        or len(set(observed)) != len(observed)
        or set(observed) != set(expected)
    ):
        raise ValueError("pilot aggregate has duplicate or missing question IDs")
    if type(draws) is not int or draws < 1 or type(seed) is not int:
        raise ValueError("pilot aggregate bootstrap draws/seed are invalid")
    if set(natural_pool_weights) != set(strata):
        raise ValueError("pilot aggregate natural weights are invalid")
    weights = {key: float(natural_pool_weights[key]) for key in strata}
    if any(
        not math.isfinite(value) or value <= 0 for value in weights.values()
    ) or not math.isclose(math.fsum(weights.values()), 1.0):
        raise ValueError("pilot aggregate natural weights are invalid")

    checked: list[dict[str, object]] = []
    required_arms = (
        "native_docprune",
        "contextcite_gold_support",
        "random_region_size_aware",
    )
    for row in question_rows:
        stratum = row.get("baseline_stratum")
        arms = row.get("arms")
        if (
            stratum not in strata
            or not isinstance(arms, Mapping)
            or any(arm not in arms for arm in required_arms)
        ):
            raise ValueError("pilot aggregate row schema is invalid")
        checked_arms: dict[str, dict[str, float | int]] = {}
        for arm, metrics in arms.items():
            if not isinstance(arm, str) or not isinstance(metrics, Mapping):
                raise ValueError("pilot aggregate arm schema is invalid")
            if set(metrics) != {"f1", "em", "gold", "margin"}:
                raise ValueError("pilot aggregate arm metrics are invalid")
            values = {key: float(metrics[key]) for key in ("f1", "gold", "margin")}
            em = metrics["em"]
            if (
                any(not math.isfinite(value) for value in values.values())
                or not 0.0 <= values["f1"] <= 1.0
                or type(em) is not int
                or em not in (0, 1)
            ):
                raise ValueError("pilot aggregate arm metrics are invalid")
            checked_arms[arm] = {**values, "em": em}
        checked.append(
            {
                "question_id": row["question_id"],
                "baseline_stratum": stratum,
                "arms": checked_arms,
            }
        )

    by_stratum = {
        stratum: [row for row in checked if row["baseline_stratum"] == stratum]
        for stratum in strata
    }
    if any(not rows for rows in by_stratum.values()):
        raise ValueError("pilot aggregate requires both baseline strata")

    def mean(values: Sequence[float]) -> float:
        return math.fsum(values) / len(values)

    def arm_summary(rows: Sequence[Mapping[str, object]], arm: str) -> dict[str, object] | None:
        available = [row for row in rows if arm in row["arms"]]
        if not available:
            return None
        return {
            "n": len(available),
            "mean_f1": mean([float(row["arms"][arm]["f1"]) for row in available]),
            "exact_match_rate": mean([float(row["arms"][arm]["em"]) for row in available]),
            "mean_gold_loglikelihood": mean([float(row["arms"][arm]["gold"]) for row in available]),
            "mean_gold_vs_alternative_margin": mean(
                [float(row["arms"][arm]["margin"]) for row in available]
            ),
        }

    def contrast(rows: Sequence[Mapping[str, object]], other: str) -> dict[str, object]:
        support = "contextcite_gold_support"
        deltas = {
            metric: [
                float(row["arms"][support][metric]) - float(row["arms"][other][metric])
                for row in rows
            ]
            for metric in ("f1", "em", "gold", "margin")
        }
        f1_deltas = deltas["f1"]
        return {
            "n": len(rows),
            "f1_delta": mean(f1_deltas),
            "exact_match_delta": mean(deltas["em"]),
            "gold_loglikelihood_delta": mean(deltas["gold"]),
            "gold_vs_alternative_margin_delta": mean(deltas["margin"]),
            "wins": sum(value > 0 for value in f1_deltas),
            "ties": sum(value == 0 for value in f1_deltas),
            "losses": sum(value < 0 for value in f1_deltas),
            "rescues": sum(
                row["arms"][support]["em"] == 1 and row["arms"][other]["em"] == 0 for row in rows
            ),
            "harms": sum(
                row["arms"][support]["em"] == 0 and row["arms"][other]["em"] == 1 for row in rows
            ),
        }

    def summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
        arms = sorted({arm for row in rows for arm in row["arms"]})
        return {
            "n": len(rows),
            "arms": {arm: arm_summary(rows, arm) for arm in arms},
            "contextcite_vs_native": contrast(rows, "native_docprune"),
            "contextcite_vs_regional_random": contrast(rows, "random_region_size_aware"),
        }

    stratum_summaries = {stratum: summary(rows) for stratum, rows in by_stratum.items()}

    def weighted_contrast(weight_map: Mapping[str, float], other: str) -> dict[str, float]:
        keys = {
            "f1_delta",
            "exact_match_delta",
            "gold_loglikelihood_delta",
            "gold_vs_alternative_margin_delta",
        }
        return {
            key: math.fsum(
                weight_map[stratum]
                * float(
                    stratum_summaries[stratum][
                        "contextcite_vs_native"
                        if other == "native_docprune"
                        else "contextcite_vs_regional_random"
                    ][key]
                )
                for stratum in strata
            )
            for key in keys
        }

    balanced_weights = {stratum: 0.5 for stratum in strata}
    aggregate_views: dict[str, dict[str, object]] = {}
    for label, weight_map in (("balanced", balanced_weights), ("natural_reweighted", weights)):
        aggregate_views[label] = {
            "stratum_weights": dict(weight_map),
            "contextcite_vs_native": weighted_contrast(weight_map, "native_docprune"),
            "contextcite_vs_regional_random": weighted_contrast(
                weight_map, "random_region_size_aware"
            ),
        }

    generator = random.Random(seed)
    bootstrap: dict[str, list[float]] = {
        f"{view}:{other}:{metric}": []
        for view in ("balanced", "natural_reweighted")
        for other in ("native_docprune", "random_region_size_aware")
        for metric in ("f1", "em")
    }
    for _ in range(draws):
        sampled = {
            stratum: [rows[generator.randrange(len(rows))] for _ in rows]
            for stratum, rows in by_stratum.items()
        }
        for view, weight_map in (("balanced", balanced_weights), ("natural_reweighted", weights)):
            for other in ("native_docprune", "random_region_size_aware"):
                for metric in ("f1", "em"):
                    support = "contextcite_gold_support"
                    value = math.fsum(
                        weight_map[stratum]
                        * mean(
                            [
                                float(row["arms"][support][metric])
                                - float(row["arms"][other][metric])
                                for row in sampled[stratum]
                            ]
                        )
                        for stratum in strata
                    )
                    bootstrap[f"{view}:{other}:{metric}"].append(value)
    intervals = {
        view: {
            other: {
                f"{metric}_delta_interval_95": [
                    _bootstrap_quantile(bootstrap[f"{view}:{other}:{metric}"], 0.025),
                    _bootstrap_quantile(bootstrap[f"{view}:{other}:{metric}"], 0.975),
                ]
                for metric in ("f1", "em")
            }
            for other in ("native_docprune", "random_region_size_aware")
        }
        for view in ("balanced", "natural_reweighted")
    }
    for view in aggregate_views:
        aggregate_views[view]["bootstrap_intervals"] = intervals[view]

    return {
        "method": "paired-question-stratified-task9-preliminary-pilot",
        "question_count": len(checked),
        "strata": stratum_summaries,
        **aggregate_views,
        "bootstrap": {
            "method": "within-stratum-question-nonparametric-percentile",
            "confidence_level": 0.95,
            "draw_count": draws,
            "seed": seed,
            "draws_sha256": _canonical_sha256(bootstrap),
        },
    }


def _replay_aggregate_raw_input(
    value: object,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    required = {
        "design",
        "fit_outcomes",
        "surrogate",
        "holdout_outcomes",
        "regions",
        "requested_budget",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("per-question aggregate raw-input schema is invalid")
    raw = dict(value)
    fidelity = evaluate_contextcite_holdout(
        raw["design"],
        raw["fit_outcomes"],
        raw["surrogate"],
        raw["holdout_outcomes"],
    )
    stability = evaluate_contextcite_refit_stability(
        raw["design"],
        raw["fit_outcomes"],
        raw["surrogate"],
        raw["regions"],
        requested_budget=raw["requested_budget"],
    )
    identity = fidelity["attribution_identity"]
    binding: dict[str, object] = {
        "question_id": identity["question_id"],
        "attribution_identity": dict(identity),
        "attribution_identity_sha256": fidelity["attribution_identity_sha256"],
        "design_sha256": fidelity["mask_design_sha256"],
        "fit_outcomes_sha256": fidelity["fit_outcomes_sha256"],
        "surrogate_sha256": fidelity["surrogate_sha256"],
        "holdout_outcomes_sha256": fidelity["holdout_outcomes_sha256"],
        "regions_sha256": stability["regions_sha256"],
        "requested_budget": stability["requested_budget"],
        "raw_input_sha256": _canonical_sha256(value),
    }
    return fidelity, stability, binding


def aggregate_contextcite_admission_metrics(
    fidelity_artifacts: Sequence[Mapping[str, object]],
    stability_artifacts: Sequence[Mapping[str, object]],
    *,
    raw_inputs: Sequence[Mapping[str, object]],
    support_components: Sequence[Sequence[str]],
    draws: int = 100_000,
    seed: int = 20_260_827,
) -> dict[str, object]:
    """Aggregate outcome-blind Task 9 fidelity and selection-stability gate inputs."""

    if type(draws) is not int or draws < 1 or type(seed) is not int:
        raise ValueError("aggregate fidelity bootstrap draws/seed are invalid")
    if (
        not isinstance(fidelity_artifacts, Sequence)
        or isinstance(fidelity_artifacts, str | bytes)
        or not isinstance(stability_artifacts, Sequence)
        or isinstance(stability_artifacts, str | bytes)
        or not isinstance(raw_inputs, Sequence)
        or isinstance(raw_inputs, str | bytes)
        or not fidelity_artifacts
        or len(fidelity_artifacts) != len(stability_artifacts)
        or len(fidelity_artifacts) != len(raw_inputs)
    ):
        raise ValueError("aggregate fidelity requires paired per-question artifacts and raw inputs")
    persisted_fidelities = tuple(
        _validated_aggregate_fidelity(value) for value in fidelity_artifacts
    )
    persisted_stabilities = tuple(
        _validated_aggregate_stability(value) for value in stability_artifacts
    )
    replayed = tuple(_replay_aggregate_raw_input(value) for value in raw_inputs)
    fidelities = tuple(row[0] for row in replayed)
    stabilities = tuple(row[1] for row in replayed)
    raw_input_bindings = tuple(row[2] for row in replayed)
    if any(
        _canonical_bytes(persisted_fidelity) != _canonical_bytes(fidelity)
        or _canonical_bytes(persisted_stability) != _canonical_bytes(stability)
        for persisted_fidelity, persisted_stability, fidelity, stability in zip(
            persisted_fidelities,
            persisted_stabilities,
            fidelities,
            stabilities,
            strict=True,
        )
    ):
        raise ValueError("persisted fidelity/stability artifacts do not match canonical raw inputs")
    fidelity_qids = tuple(row["attribution_identity"]["question_id"] for row in fidelities)
    stability_qids = tuple(row["attribution_identity"]["question_id"] for row in stabilities)
    if (
        fidelity_qids != stability_qids
        or len(set(fidelity_qids)) != len(fidelity_qids)
        or any(
            _canonical_bytes(fidelity["attribution_identity"])
            != _canonical_bytes(stability["attribution_identity"])
            or any(
                fidelity[key] != stability[key]
                for key in (
                    "mask_design_sha256",
                    "surrogate_sha256",
                    "fit_outcomes_sha256",
                    "fit_targets_sha256",
                )
            )
            for fidelity, stability in zip(fidelities, stabilities, strict=True)
        )
    ):
        raise ValueError("aggregate fidelity artifacts mix QID order or attribution identity")
    boundaries = {row["attribution_identity"]["forced_boundary"] for row in fidelities}
    targets = {row["attribution_identity"]["target_kind"] for row in fidelities}
    if len(boundaries) != 1 or len(targets) != 1:
        raise ValueError("aggregate fidelity requires one forced boundary and target kind")

    components = tuple(tuple(component) for component in support_components)
    flattened = [qid for component in components for qid in component]
    if (
        len(components) < 2
        or any(not component for component in components)
        or any(not isinstance(qid, str) or not qid for qid in flattened)
        or len(flattened) != len(set(flattened))
        or set(flattened) != set(fidelity_qids)
    ):
        raise ValueError("support-document components must partition aggregate fidelity QIDs")

    lds_by_qid = {
        qid: fidelity["lds_spearman"]
        for qid, fidelity in zip(fidelity_qids, fidelities, strict=True)
    }
    undefined_lds = [qid for qid in fidelity_qids if lds_by_qid[qid] is None]
    point_lds: float | None = None
    interval: list[float] | None = None
    lower_bound: float | None = None
    draws_sha256: str | None = None
    if not undefined_lds:
        point_lds = math.fsum(float(lds_by_qid[qid]) for qid in fidelity_qids) / len(fidelity_qids)
        generator = random.Random(seed)
        bootstrap_draws: list[float] = []
        for _ in range(draws):
            sampled_components = [
                components[generator.randrange(len(components))] for _ in range(len(components))
            ]
            sampled_qids = [qid for component in sampled_components for qid in component]
            bootstrap_draws.append(
                math.fsum(float(lds_by_qid[qid]) for qid in sampled_qids) / len(sampled_qids)
            )
        interval = [
            _bootstrap_quantile(bootstrap_draws, 0.025),
            _bootstrap_quantile(bootstrap_draws, 0.975),
        ]
        lower_bound = interval[0]
        draws_sha256 = _canonical_sha256(bootstrap_draws)

    aggregate_heldout_rmse = _root_mean_square(
        [float(fidelity["heldout_rmse"]) for fidelity in fidelities]
    )
    aggregate_constant_rmse = _root_mean_square(
        [float(fidelity["constant_rmse"]) for fidelity in fidelities]
    )
    undefined_stability = [
        qid
        for qid, stability in zip(fidelity_qids, stabilities, strict=True)
        if stability["top_selection_summary"]["all_defined"] is not True
    ]
    stability_minimum: float | None = None
    if not undefined_stability:
        stability_minimum = min(
            float(stability["top_selection_summary"]["minimum_defined"])
            for stability in stabilities
        )
    result: dict[str, object] = {
        "schema_version": 1,
        "method": "contextcite-aggregate-fidelity-and-five-refit-stability",
        "qids": list(fidelity_qids),
        "forced_boundary": next(iter(boundaries)),
        "target_kind": next(iter(targets)),
        "support_components": [list(component) for component in components],
        "fidelity_sha256s": [row["fidelity_sha256"] for row in fidelities],
        "stability_sha256s": [row["stability_sha256"] for row in stabilities],
        "raw_input_bindings": [dict(binding) for binding in raw_input_bindings],
        "lds": {
            "definition": "mean-per-question-spearman-lds",
            "undefined_policy": (
                "any undefined per-question LDS makes the aggregate point and interval "
                "unavailable and both LDS threshold inputs false"
            ),
            "defined_count": len(fidelity_qids) - len(undefined_lds),
            "undefined_count": len(undefined_lds),
            "all_defined": not undefined_lds,
            "undefined_qids": undefined_lds,
            "point_mean_per_question": point_lds,
            "bootstrap": {
                "method": "support-component-nonparametric-percentile",
                "confidence_level": 0.95,
                "interval_95": interval,
                "lower_bound_95": lower_bound,
                "draw_count": draws,
                "seed": seed,
                "draws_sha256": draws_sha256,
            },
        },
        "rmse": {
            "aggregation": "sqrt(mean(per-question-rmse-squared))",
            "heldout": aggregate_heldout_rmse,
            "constant": aggregate_constant_rmse,
        },
        "stability": {
            "aggregation": "minimum-pairwise-jaccard-across-all-questions",
            "undefined_policy": (
                "any undefined question-level pair makes the cross-question minimum "
                "unavailable and both stability threshold inputs false"
            ),
            "all_defined": not undefined_stability,
            "undefined_qids": undefined_stability,
            "minimum_pairwise_jaccard_across_questions": stability_minimum,
        },
        "thresholds": {
            "lds_point_minimum_inclusive": 0.5,
            "lds_95_lower_bound_minimum_strict": 0.2,
            "aggregate_rmse_must_be_below_constant": True,
            "all_selection_jaccards_must_be_defined": True,
            "minimum_selection_jaccard_inclusive": 0.8,
        },
        "threshold_inputs": {
            "lds_point_at_least_0_5": point_lds is not None and point_lds >= 0.5,
            "lds_95_lower_bound_above_0_2": lower_bound is not None and lower_bound > 0.2,
            "aggregate_rmse_beats_constant": aggregate_heldout_rmse < aggregate_constant_rmse,
            "all_selection_jaccards_defined": not undefined_stability,
            "minimum_selection_jaccard_at_least_0_8": stability_minimum is not None
            and stability_minimum >= 0.8,
        },
    }
    result["aggregate_metrics_sha256"] = _canonical_sha256(result)
    return result


__all__ = [
    "analyze_task9_preliminary_question",
    "analyze_task9_preliminary_attribution",
    "build_region_mask_design",
    "build_task9_preliminary_arm_selections",
    "build_task9_preliminary_intervention_plan",
    "build_task9_preliminary_mask_design",
    "build_task9_selected_arm_plan",
    "build_regional_development_plan",
    "build_regional_development_targets",
    "validate_regional_development_targets",
    "aggregate_contextcite_admission_metrics",
    "analyze_contextcite_development_question",
    "evaluate_contextcite_explicit_mask_fidelity",
    "evaluate_contextcite_holdout",
    "evaluate_contextcite_refit_stability",
    "fit_contextcite_lasso",
    "whole_region_knapsack",
]
