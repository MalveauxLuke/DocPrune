"""Physical whole-region intervention contracts for Task 9."""

from __future__ import annotations

from copy import deepcopy

import pytest

from docprune import task9_attribution
from docprune.ctp_controls import VisualTokenGeometry
from docprune.segmentation import RegionTokenMapping, RegionTokenSource


def _source(source_id: str, token_ids: tuple[int, ...]) -> RegionTokenSource:
    return RegionTokenSource(
        source_id=source_id,
        source_kind="residual-grid",
        input_page_index=0,
        mineru_page_index=0,
        document_id="doc",
        source_page_index=0,
        region_type="residual",
        bbox=(0.0, 0.0, 1.0, 1.0),
        page_size=(1.0, 1.0),
        reading_order=None,
        token_ids=token_ids,
    )


def _mapping() -> RegionTokenMapping:
    sources = (
        _source("region-a", (0, 2)),
        _source("region-b", (1, 4)),
        _source("region-c", (3, 5)),
    )
    token_to_source = (
        "region-a",
        "region-b",
        "region-a",
        "region-c",
        "region-b",
        "region-c",
    )
    geometry = tuple(VisualTokenGeometry(0, 0, index, 1, 6) for index in range(6))
    return RegionTokenMapping(
        artifacts=(),
        geometry=geometry,
        geometry_count=6,
        geometry_sha256="1" * 64,
        residual_grid_size=4,
        assignment_contract="test-partition",
        audited_regions=(),
        empty_region_source_ids=(),
        sources=sources,
        token_to_source=token_to_source,
        sha256="2" * 64,
    )


def _design(*, boundary: str = "B_13") -> dict[str, object]:
    return task9_attribution.build_region_mask_design(
        [
            {"source_id": "region-a", "token_cost": 2},
            {"source_id": "region-b", "token_cost": 2},
            {"source_id": "region-c", "token_cost": 2},
        ],
        question_id="qid",
        forced_boundary=boundary,
        mapping_artifact_sha256="a" * 64,
        prompt_input_sha256="b" * 64,
        target_kind="max-accepted-reference-mean-loglikelihood",
        reference_set_token_ids_sha256="c" * 64,
        generated_response_token_ids_sha256=None,
    )


def _secondary_design(*, boundary: str = "B_13") -> dict[str, object]:
    return task9_attribution.build_region_mask_design(
        [
            {"source_id": "region-a", "token_cost": 2},
            {"source_id": "region-b", "token_cost": 2},
            {"source_id": "region-c", "token_cost": 2},
        ],
        question_id="qid",
        forced_boundary=boundary,
        mapping_artifact_sha256="a" * 64,
        prompt_input_sha256="b" * 64,
        target_kind="unpruned-generated-response-mean-loglikelihood",
        reference_set_token_ids_sha256=None,
        generated_response_token_ids_sha256="d" * 64,
    )


def test_development_plan_uses_one_physical_mask_sequence_for_exact_seed_order() -> None:
    """Catch split GPU calls, reordered seeds, or target-specific physical masks."""

    builder = getattr(task9_attribution, "build_regional_development_plan", None)
    assert builder is not None, "Task 9 development-plan builder is missing"

    plan = builder(
        _mapping(),
        _design(),
        _secondary_design(),
        mapping_artifact_sha256="a" * 64,
    )

    assert len(plan) == 96
    assert [row["seed"] for row in plan] == list(range(96))
    assert [row["split"] for row in plan] == ["fit"] * 64 + ["holdout"] * 32
    assert all(
        row["primary_attribution_identity_sha256"] == _design()["attribution_identity_sha256"]
        for row in plan
    )
    assert all(
        row["secondary_attribution_identity_sha256"]
        == _secondary_design()["attribution_identity_sha256"]
        for row in plan
    )
    assert all(row["forced_intervention"].mode == "physical_delete" for row in plan)


def test_development_targets_slice_reference_and_generated_likelihoods() -> None:
    """Catch mixing the appended generated-response score into the reference maximum."""

    plan = task9_attribution.build_regional_development_plan(
        _mapping(),
        _design(),
        _secondary_design(),
        mapping_artifact_sha256="a" * 64,
    )
    builder = getattr(task9_attribution, "build_regional_development_targets", None)
    assert builder is not None, "Task 9 dual-target packager is missing"
    raw = [(-0.9, -0.2, -0.01 - seed / 1000) for seed in range(96)]

    targets = builder(
        _design(),
        _secondary_design(),
        plan,
        mean_sequence_loglikelihoods=raw,
        reference_sequence_count=2,
    )

    assert targets["seed_order"] == list(range(96))
    assert len(targets["primary"]["outcomes"]) == 96
    assert len(targets["secondary"]["outcomes"]) == 96
    assert targets["primary"]["per_sequence_mean_loglikelihoods"][0] == [-0.9, -0.2]
    assert targets["secondary"]["per_sequence_mean_loglikelihoods"][0] == [-0.01]
    assert targets["primary"]["outcomes"][0]["normalized_target"] == -0.2
    assert targets["secondary"]["outcomes"][0]["normalized_target"] == -0.01
    assert targets["secondary"]["outcomes"][-1]["normalized_target"] == -0.105
    assert (
        targets["primary"]["attribution_identity_sha256"]
        != targets["secondary"]["attribution_identity_sha256"]
    )


def test_development_target_validation_reconstructs_outcomes_from_raw_rows() -> None:
    """Catch trusting a coherently rehashed stored target instead of replaying raw scores."""

    plan = task9_attribution.build_regional_development_plan(
        _mapping(), _design(), _secondary_design(), mapping_artifact_sha256="a" * 64
    )
    raw = [(-0.9, -0.2, -0.01 - seed / 1000) for seed in range(96)]
    targets = task9_attribution.build_regional_development_targets(
        _design(),
        _secondary_design(),
        plan,
        mean_sequence_loglikelihoods=raw,
        reference_sequence_count=2,
    )
    validator = getattr(task9_attribution, "validate_regional_development_targets", None)
    assert validator is not None, "Task 9 raw-target validator is missing"
    validator(targets["primary"], targets["secondary"], plan, raw, reference_sequence_count=2)

    altered = deepcopy(targets["primary"])
    altered["outcomes"][0]["normalized_target"] = 99.0
    unsigned = dict(altered)
    unsigned.pop("target_dataset_sha256")
    altered["target_dataset_sha256"] = task9_attribution._canonical_sha256(unsigned)
    with pytest.raises(ValueError, match="raw likelihood"):
        validator(altered, targets["secondary"], plan, raw, reference_sequence_count=2)


def test_intervention_plan_retains_exact_union_selected_by_contextcite_mask() -> None:
    """Catch inverting ContextCite booleans or broadcasting a region score per token."""

    plan = task9_attribution.build_regional_intervention_plan(
        _mapping(),
        _design(),
        mapping_artifact_sha256="a" * 64,
        split="fit",
    )

    # Pinned mask seed 1 is [False, True, False], so only region-b's two
    # non-contiguous post-QTP visual ordinals are retained.
    row = plan[1]
    assert row["seed"] == 1
    assert row["vector"] == [False, True, False]
    assert row["retained_source_ids"] == ["region-b"]
    assert row["retained_visual_ids"] == [1, 4]
    assert row["retained_visual_count"] == 2
    assert row["visual_population"] == 6
    assert row["forced_intervention"].boundary == 13
    assert row["forced_intervention"].mode == "physical_delete"
    assert row["forced_intervention"].retained_visual_ids == (1, 4)
    assert len(plan) == 64


def test_selected_arm_plan_translates_region_ids_to_physical_visual_ids() -> None:
    """Catch sending region IDs rather than their post-QTP token union to generation."""

    builder = getattr(task9_attribution, "build_task9_selected_arm_plan", None)
    assert builder is not None, "Task 9 selected-arm intervention builder is missing"
    selections = {
        "question_id": "qid",
        "budgets": [
            {
                "retained_fraction": 0.65,
                "requested_token_count": 4,
                "achieved_token_count": 4,
                "arms": [
                    {
                        "arm": "docprune_query_attention",
                        "retained_source_ids": ["region-a", "region-c"],
                        "achieved_token_count": 4,
                    },
                    {
                        "arm": "contextcite_gold_support",
                        "retained_source_ids": ["region-b", "region-c"],
                        "achieved_token_count": 4,
                    },
                ],
            }
        ],
    }

    plan = builder(_mapping(), selections, boundary="B_13")

    assert [(row["arm"], row["retained_visual_ids"]) for row in plan] == [
        ("docprune_query_attention", [0, 2, 3, 5]),
        ("contextcite_gold_support", [1, 3, 4, 5]),
    ]
    assert all(row["forced_intervention"].boundary == 13 for row in plan)
    assert all(row["forced_intervention"].mode == "physical_delete" for row in plan)


def test_preliminary_intervention_plan_appends_budget_local_masks() -> None:
    sources = tuple(_source(f"region-{index:02d}", (index,)) for index in range(12))
    mapping = RegionTokenMapping(
        artifacts=(),
        geometry=tuple(VisualTokenGeometry(0, 0, index, 1, 12) for index in range(12)),
        geometry_count=12,
        geometry_sha256="1" * 64,
        residual_grid_size=4,
        assignment_contract="test-partition",
        audited_regions=(),
        empty_region_source_ids=(),
        sources=sources,
        token_to_source=tuple(source.source_id for source in sources),
        sha256="2" * 64,
    )
    design = task9_attribution.build_task9_preliminary_mask_design(
        [{"source_id": source.source_id, "token_cost": 1} for source in sources],
        question_id="qid",
        forced_boundary="B_13",
        mapping_artifact_sha256="a" * 64,
        prompt_input_sha256="b" * 64,
        target_kind="max-accepted-reference-mean-loglikelihood",
        reference_set_token_ids_sha256="c" * 64,
        generated_response_token_ids_sha256=None,
        fit_mask_count=8,
        global_holdout_mask_count=4,
        budget_local_holdout_mask_count=3,
    )

    plan = task9_attribution.build_task9_preliminary_intervention_plan(
        mapping, design, mapping_artifact_sha256="a" * 64
    )

    assert len(plan) == 15
    assert [row["split"] for row in plan] == ["fit"] * 8 + ["holdout"] * 4 + [
        "budget_local_holdout"
    ] * 3
    assert all(row["forced_intervention"].boundary == 13 for row in plan)


def test_intervention_plan_translates_input_boundary_without_changing_mask() -> None:
    """Catch passing the report label B_input to the decoder's input-boundary API."""

    plan = task9_attribution.build_regional_intervention_plan(
        _mapping(),
        _design(boundary="B_input"),
        mapping_artifact_sha256="a" * 64,
        split="holdout",
    )

    assert len(plan) == 32
    assert plan[0]["seed"] == 64
    assert plan[0]["forced_intervention"].boundary == "input"
    assert plan[0]["forced_intervention"].retained_visual_ids == (1, 3, 4, 5)


def test_intervention_plan_rejects_mapping_design_or_partition_drift() -> None:
    """Catch using masks with a different mapping or a non-partition token population."""

    wrong_digest = "d" * 64
    with pytest.raises(ValueError, match="mapping artifact"):
        task9_attribution.build_regional_intervention_plan(
            _mapping(),
            _design(),
            mapping_artifact_sha256=wrong_digest,
            split="fit",
        )

    changed_design = deepcopy(_design())
    changed_design["source_ids"] = list(reversed(changed_design["source_ids"]))
    with pytest.raises(ValueError, match="mask design"):
        task9_attribution.build_regional_intervention_plan(
            _mapping(),
            changed_design,
            mapping_artifact_sha256="a" * 64,
            split="fit",
        )

    mapping = _mapping()
    broken = RegionTokenMapping(
        artifacts=mapping.artifacts,
        geometry=mapping.geometry,
        geometry_count=mapping.geometry_count,
        geometry_sha256=mapping.geometry_sha256,
        residual_grid_size=mapping.residual_grid_size,
        assignment_contract=mapping.assignment_contract,
        audited_regions=mapping.audited_regions,
        empty_region_source_ids=mapping.empty_region_source_ids,
        sources=(mapping.sources[0], mapping.sources[1], _source("region-c", (3, 4, 5))),
        token_to_source=mapping.token_to_source,
        sha256=mapping.sha256,
    )
    with pytest.raises(ValueError, match="partition"):
        task9_attribution.build_regional_intervention_plan(
            broken,
            _design(),
            mapping_artifact_sha256="a" * 64,
            split="fit",
        )


def test_regional_outcome_uses_best_reference_mean_sequence_loglikelihood() -> None:
    """Catch summing token likelihoods, averaging references, or selecting the minimum."""

    design = _design()
    row = task9_attribution.build_regional_intervention_plan(
        _mapping(), design, mapping_artifact_sha256="a" * 64, split="fit"
    )[1]

    outcome = task9_attribution.build_regional_target_outcome(
        design,
        row,
        mean_sequence_loglikelihoods=(-1.5, -0.25, -0.7),
    )

    assert outcome == {
        "split": "fit",
        "seed": 1,
        "vector_sha256": row["vector_sha256"],
        "attribution_identity_sha256": design["attribution_identity_sha256"],
        "normalized_target": -0.25,
    }


def test_generated_response_outcome_requires_exactly_one_normalized_sequence() -> None:
    """Catch mixing the generated-response target with any accepted-reference target."""

    design = task9_attribution.build_region_mask_design(
        [
            {"source_id": "region-a", "token_cost": 2},
            {"source_id": "region-b", "token_cost": 2},
            {"source_id": "region-c", "token_cost": 2},
        ],
        question_id="qid",
        forced_boundary="B_13",
        mapping_artifact_sha256="a" * 64,
        prompt_input_sha256="b" * 64,
        target_kind="unpruned-generated-response-mean-loglikelihood",
        reference_set_token_ids_sha256=None,
        generated_response_token_ids_sha256="d" * 64,
    )
    row = task9_attribution.build_regional_intervention_plan(
        _mapping(), design, mapping_artifact_sha256="a" * 64, split="holdout"
    )[0]

    outcome = task9_attribution.build_regional_target_outcome(
        design,
        row,
        mean_sequence_loglikelihoods=(-0.4,),
    )
    assert outcome["normalized_target"] == -0.4
    with pytest.raises(ValueError, match="exactly one"):
        task9_attribution.build_regional_target_outcome(
            design,
            row,
            mean_sequence_loglikelihoods=(-0.4, -0.6),
        )
