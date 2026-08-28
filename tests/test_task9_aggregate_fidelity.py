"""Aggregation contracts for Task 9 fidelity and refit-stability admission inputs."""

from __future__ import annotations

import json
import math
from copy import deepcopy

import pytest

from docprune import task9_attribution


def _raw_input(
    qid: str,
    *,
    target_scale: float = 1.0,
    constant_fit: bool = False,
    requested_budget: int = 2,
) -> dict[str, object]:
    regions = [
        {"source_id": "region-z", "token_cost": 1},
        {"source_id": "region-a", "token_cost": 1},
        {"source_id": "region-m", "token_cost": 1},
    ]
    design = task9_attribution.build_region_mask_design(
        regions,
        question_id=qid,
        forced_boundary="B_13",
        mapping_artifact_sha256="a" * 64,
        prompt_input_sha256="b" * 64,
        target_kind="max-accepted-reference-mean-loglikelihood",
        reference_set_token_ids_sha256="c" * 64,
        generated_response_token_ids_sha256=None,
    )
    fit_outcomes = [
        {
            "split": "fit",
            "seed": mask["seed"],
            "vector_sha256": mask["vector_sha256"],
            "attribution_identity_sha256": design["attribution_identity_sha256"],
            "normalized_target": (
                0.5 + (1e-6 if int(mask["seed"]) % 2 == 0 else -1e-6)
                if constant_fit
                else target_scale
                * sum(
                    weight * float(retained)
                    for weight, retained in zip((3.0, 2.0, 1.0), mask["vector"], strict=True)
                )
            ),
        }
        for mask in design["fit_masks"]
    ]
    holdout_outcomes = [
        {
            "split": "holdout",
            "seed": mask["seed"],
            "vector_sha256": mask["vector_sha256"],
            "attribution_identity_sha256": design["attribution_identity_sha256"],
            "normalized_target": target_scale
            * sum(
                weight * float(retained)
                for weight, retained in zip((3.0, 2.0, 1.0), mask["vector"], strict=True)
            ),
        }
        for mask in design["holdout_masks"]
    ]
    surrogate = task9_attribution.fit_contextcite_lasso(design, fit_outcomes)
    return {
        "design": design,
        "fit_outcomes": fit_outcomes,
        "surrogate": surrogate,
        "holdout_outcomes": holdout_outcomes,
        "regions": regions,
        "requested_budget": requested_budget,
    }


def _artifacts(
    *,
    second_constant: bool = False,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    raw_inputs = [
        _raw_input("q1"),
        _raw_input(
            "q2",
            target_scale=0.5,
            constant_fit=second_constant,
            requested_budget=0 if second_constant else 2,
        ),
    ]
    fidelities = [
        task9_attribution.evaluate_contextcite_holdout(
            raw["design"],
            raw["fit_outcomes"],
            raw["surrogate"],
            raw["holdout_outcomes"],
        )
        for raw in raw_inputs
    ]
    stabilities = [
        task9_attribution.evaluate_contextcite_refit_stability(
            raw["design"],
            raw["fit_outcomes"],
            raw["surrogate"],
            raw["regions"],
            requested_budget=raw["requested_budget"],
        )
        for raw in raw_inputs
    ]
    return fidelities, stabilities, raw_inputs


def _aggregate(
    fidelities: list[dict[str, object]],
    stabilities: list[dict[str, object]],
    raw_inputs: list[dict[str, object]],
) -> dict[str, object]:
    return task9_attribution.aggregate_contextcite_admission_metrics(
        fidelities,
        stabilities,
        raw_inputs=raw_inputs,
        support_components=(("q1",), ("q2",)),
        draws=20,
        seed=3,
    )


def test_aggregate_replays_raw_inputs_and_binds_component_metrics() -> None:
    """Catch aggregation from persisted summaries instead of canonical raw replay."""

    pytest.importorskip("sklearn")
    fidelities, stabilities, raw_inputs = _artifacts()

    first = _aggregate(fidelities, stabilities, raw_inputs)
    second = _aggregate(fidelities, stabilities, raw_inputs)

    assert first == second
    assert first["qids"] == ["q1", "q2"]
    assert first["forced_boundary"] == "B_13"
    assert first["target_kind"] == "max-accepted-reference-mean-loglikelihood"
    assert first["lds"]["point_mean_per_question"] == pytest.approx(
        math.fsum(float(row["lds_spearman"]) for row in fidelities) / 2
    )
    assert first["lds"]["all_defined"] is True
    assert first["lds"]["bootstrap"]["draw_count"] == 20
    assert first["lds"]["bootstrap"]["seed"] == 3
    assert len(first["lds"]["bootstrap"]["draws_sha256"]) == 64
    assert first["rmse"]["heldout"] == pytest.approx(
        math.sqrt(math.fsum(float(row["heldout_rmse"]) ** 2 for row in fidelities) / 2)
    )
    assert first["rmse"]["constant"] == pytest.approx(
        math.sqrt(math.fsum(float(row["constant_rmse"]) ** 2 for row in fidelities) / 2)
    )
    assert first["stability"]["minimum_pairwise_jaccard_across_questions"] == 1.0
    assert first["threshold_inputs"] == {
        "lds_point_at_least_0_5": True,
        "lds_95_lower_bound_above_0_2": True,
        "aggregate_rmse_beats_constant": True,
        "all_selection_jaccards_defined": True,
        "minimum_selection_jaccard_at_least_0_8": True,
    }
    assert "final_verdict" not in first
    assert len(first["aggregate_metrics_sha256"]) == 64
    assert [row["question_id"] for row in first["raw_input_bindings"]] == ["q1", "q2"]
    for binding, raw, fidelity, stability in zip(
        first["raw_input_bindings"], raw_inputs, fidelities, stabilities, strict=True
    ):
        assert binding == {
            "question_id": raw["design"]["attribution_identity"]["question_id"],
            "attribution_identity": raw["design"]["attribution_identity"],
            "attribution_identity_sha256": raw["design"]["attribution_identity_sha256"],
            "design_sha256": raw["design"]["design_sha256"],
            "fit_outcomes_sha256": fidelity["fit_outcomes_sha256"],
            "surrogate_sha256": raw["surrogate"]["surrogate_sha256"],
            "holdout_outcomes_sha256": fidelity["holdout_outcomes_sha256"],
            "regions_sha256": stability["regions_sha256"],
            "requested_budget": raw["requested_budget"],
            "raw_input_sha256": task9_attribution._canonical_sha256(raw),
        }


def test_aggregate_keeps_undefined_inputs_explicit_and_nonpassing() -> None:
    """Catch undefined LDS or empty selections being silently omitted from the gate."""

    pytest.importorskip("sklearn")
    fidelities, stabilities, raw_inputs = _artifacts(second_constant=True)

    result = _aggregate(fidelities, stabilities, raw_inputs)

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


def _rehash_fidelity(fidelity: dict[str, object]) -> None:
    fidelity["fidelity_sha256"] = task9_attribution._canonical_sha256(
        {key: value for key, value in fidelity.items() if key != "fidelity_sha256"}
    )


def _rehash_stability(stability: dict[str, object]) -> None:
    stability["stability_sha256"] = task9_attribution._canonical_sha256(
        {key: value for key, value in stability.items() if key != "stability_sha256"}
    )


def _rehash_refit(refit: dict[str, object]) -> None:
    refit["refit_sha256"] = task9_attribution._canonical_sha256(
        {key: value for key, value in refit.items() if key != "refit_sha256"}
    )


def test_aggregate_rejects_coherently_rehashed_fidelity_metrics() -> None:
    """Catch fabricated LDS/RMSE summaries whose internal digest is self-consistent."""

    pytest.importorskip("sklearn")
    fidelities, stabilities, raw_inputs = _artifacts()
    fabricated = deepcopy(fidelities)
    fabricated[0]["lds_spearman"] = 0.75
    fabricated[0]["heldout_rmse"] = 0.0
    fabricated[0]["constant_rmse"] = 1.0
    fabricated[0]["surrogate_beats_constant"] = True
    fabricated[0]["holdout_predictions_sha256"] = "9" * 64
    _rehash_fidelity(fabricated[0])

    with pytest.raises(ValueError, match="canonical raw inputs"):
        _aggregate(fabricated, stabilities, raw_inputs)


def test_aggregate_rejects_coherently_rehashed_refits_and_selections() -> None:
    """Catch fabricated refits/selections even when every derived row remains coherent."""

    pytest.importorskip("sklearn")
    fidelities, stabilities, raw_inputs = _artifacts()
    fabricated_coefficients = deepcopy(stabilities)
    for seed, refit in enumerate(fabricated_coefficients[0]["refits"]):
        refit["coefficients"] = {
            "region-z": -3.0 - seed / 100,
            "region-a": -2.0 - seed / 100,
            "region-m": -1.0 - seed / 100,
        }
        refit["coefficients_sha256"] = task9_attribution._canonical_sha256(refit["coefficients"])
        _rehash_refit(refit)
    _rehash_stability(fabricated_coefficients[0])

    fabricated_selections = deepcopy(stabilities)
    for refit in fabricated_selections[0]["refits"]:
        refit["top_source_ids"] = ["region-m"]
        refit["top_source_ids_sha256"] = task9_attribution._canonical_sha256(
            refit["top_source_ids"]
        )
        _rehash_refit(refit)
    # Every fabricated selection is identical, so all ten Jaccards and the
    # persisted summary remain internally coherent at 1.0.
    _rehash_stability(fabricated_selections[0])

    for fabricated in (fabricated_coefficients, fabricated_selections):
        with pytest.raises(ValueError, match="canonical raw inputs"):
            _aggregate(fidelities, fabricated, raw_inputs)


def test_aggregate_rejects_raw_order_identity_schema_and_partition_drift() -> None:
    """Catch a raw/artifact QID mismatch or an incomplete support-component partition."""

    pytest.importorskip("sklearn")
    fidelities, stabilities, raw_inputs = _artifacts()
    changed_schema = deepcopy(raw_inputs)
    changed_schema[0]["unexpected"] = "not canonical"
    changed_identity = deepcopy(raw_inputs)
    changed_identity[0]["holdout_outcomes"][0]["attribution_identity_sha256"] = "9" * 64

    invalid = (
        (fidelities, stabilities, list(reversed(raw_inputs)), (("q1",), ("q2",))),
        (fidelities, stabilities, changed_schema, (("q1",), ("q2",))),
        (fidelities, stabilities, changed_identity, (("q1",), ("q2",))),
        (fidelities, list(reversed(stabilities)), raw_inputs, (("q1",), ("q2",))),
        (fidelities, stabilities, raw_inputs, (("q1", "q2"), ("q2",))),
        (fidelities, stabilities, raw_inputs, (("q1",),)),
    )
    for candidate_fidelity, candidate_stability, candidate_raw, components in invalid:
        with pytest.raises(ValueError):
            task9_attribution.aggregate_contextcite_admission_metrics(
                candidate_fidelity,
                candidate_stability,
                raw_inputs=candidate_raw,
                support_components=components,
                draws=20,
                seed=3,
            )


def test_aggregate_accepts_canonical_json_key_reordering_of_every_input() -> None:
    """Catch dependence on in-memory mapping insertion order rather than canonical JSON."""

    pytest.importorskip("sklearn")
    fidelities, stabilities, raw_inputs = _artifacts()
    serialized_fidelities = json.loads(json.dumps(fidelities, sort_keys=True))
    serialized_stabilities = json.loads(json.dumps(stabilities, sort_keys=True))
    serialized_raw_inputs = json.loads(json.dumps(raw_inputs, sort_keys=True))

    expected = _aggregate(fidelities, stabilities, raw_inputs)
    actual = _aggregate(serialized_fidelities, serialized_stabilities, serialized_raw_inputs)

    assert actual == expected
