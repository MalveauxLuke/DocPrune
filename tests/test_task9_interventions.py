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
