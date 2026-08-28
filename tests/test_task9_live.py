"""Admission tests for live Task 9 regional-likelihood artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from docprune.task9_attribution import build_region_mask_design
from docprune.task9_live import admit_task9_regional_smoke


def _sha(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write(path: Path, value: object) -> str:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(root: Path) -> None:
    root.mkdir()
    design = build_region_mask_design(
        [
            {"source_id": "region-a", "token_cost": 2},
            {"source_id": "region-b", "token_cost": 2},
            {"source_id": "empty", "token_cost": 0},
        ],
        question_id="qid",
        forced_boundary="B_1",
        mapping_artifact_sha256="a" * 64,
        prompt_input_sha256="b" * 64,
        target_kind="max-accepted-reference-mean-loglikelihood",
        reference_set_token_ids_sha256="c" * 64,
        generated_response_token_ids_sha256=None,
    )
    manifest = {
        "schema_version": 1,
        "status": "configured-task9-regional-smoke",
        "runtime_commit": "d" * 40,
        "qid": "qid",
        "boundary": "B_1",
        "split": "fit",
        "mask_count": 4,
        "selected_seeds": [0, 1, 2, 3],
        "fixed_page_fixture_path": "/fixed.json",
        "fixed_page_fixture_sha256": "e" * 64,
        "config_sha256": "f" * 64,
        "run_config_sha256": "1" * 64,
        "index_manifest_sha256": "2" * 64,
        "mapping_path": "/mapping.json",
        "mapping_artifact_sha256": "a" * 64,
        "mapping_internal_sha256": "3" * 64,
        "geometry_count": 4,
        "geometry_sha256": "4" * 64,
        "design_sha256": design["design_sha256"],
        "attribution_identity_sha256": design["attribution_identity_sha256"],
        "assistant_prompt_sha256": "5" * 64,
        "prefill_input_ids_shape": [1, 12],
        "prefill_input_ids_sha256": "b" * 64,
        "reference_set_token_ids_sha256": "c" * 64,
        "fixed_page_provenance": True,
        "global_index_loaded": False,
        "retrieval_search_run": False,
        "cached_retrieved_pages_reused": True,
        "independent_first_mask_parity_required": True,
    }
    manifest["run_manifest_sha256"] = _sha(manifest)
    manifest_file_sha = _write(root / "run-manifest.json", manifest)
    plans = []
    outcomes = []
    forced = []
    likelihoods = []
    for mask in design["fit_masks"][:4]:
        retained = [index for index, keep in enumerate(mask["vector"]) if keep]
        plan = {
            "split": "fit",
            "seed": mask["seed"],
            "vector": mask["vector"],
            "vector_sha256": mask["vector_sha256"],
            "attribution_identity_sha256": design["attribution_identity_sha256"],
            "retained_source_ids": mask["retained_source_ids"],
            "retained_visual_ids": retained,
            "retained_visual_count": len(retained),
            "visual_population": 4,
        }
        compact = 8 + len(retained)
        plans.append(plan)
        values = [-0.5 - mask["seed"] / 100, -0.7]
        likelihoods.append(values)
        outcomes.append(
            {
                "split": "fit",
                "seed": mask["seed"],
                "vector_sha256": mask["vector_sha256"],
                "attribution_identity_sha256": design["attribution_identity_sha256"],
                "normalized_target": max(values),
            }
        )
        forced.append(
            {
                "boundary": "B_1",
                "mode": "physical_delete",
                "selection_kind": "forced",
                "visual_population": 4,
                "requested_budget": len(retained),
                "achieved_budget": len(retained),
                "retained_visual_ids": retained,
                "logical_retained_sequence_ids": list(range(compact)),
                "prefill_cache_lengths": [12, 12, *([compact] * 2)],
                "retained_mrope_position_shape": [3, 1, compact],
                "retained_mrope_position_sha256": "6" * 64,
            }
        )
    result = {
        "schema_version": 1,
        "status": "completed-task9-regional-smoke",
        "run_manifest_sha256": manifest["run_manifest_sha256"],
        "design": design,
        "selected_plan": plans,
        "outcomes": outcomes,
        "forced_interventions": forced,
        "per_reference_mean_loglikelihoods": likelihoods,
        "independent_first_mask_mean_loglikelihoods": likelihoods[0],
        "shared_prefix_parity_max_abs_error": 0.0,
        "original_visual_tokens": 8,
        "post_btp_visual_tokens": 6,
        "post_qtp_visual_tokens": 4,
        "checkpoint_cache_lengths": [12, 12],
        "encoder_seconds": 1.0,
        "prefix_decoder_seconds": 2.0,
        "branch_decoder_seconds": [3.0, 3.1, 3.2, 3.3],
        "branch_decoder_seconds_mean": 3.15,
        "projected_96_branch_decoder_seconds": 302.4,
        "measured_scoring_and_parity_seconds": 20.0,
        "peak_allocated_gpu_bytes": 100,
        "cuda_device_name": "NVIDIA L40S",
    }
    result["result_sha256"] = _sha(result)
    result_file_sha = _write(root / "result.json", result)
    completion = {
        "schema_version": 1,
        "status": "complete",
        "run_manifest_sha256": manifest["run_manifest_sha256"],
        "result_sha256": result["result_sha256"],
        "run_manifest_file_sha256": manifest_file_sha,
        "result_file_sha256": result_file_sha,
    }
    completion["completion_sha256"] = _sha(completion)
    _write(root / "completion-manifest.json", completion)


def test_admit_task9_smoke_authenticates_masks_likelihoods_and_cache_topology(
    tmp_path: Path,
) -> None:
    root = tmp_path / "smoke"
    _artifact(root)

    admitted = admit_task9_regional_smoke(
        root,
        expected_runtime_commit="d" * 40,
        expected_qid="qid",
        expected_boundary="B_1",
        expected_split="fit",
        expected_mask_count=4,
        expected_fixture_sha256="e" * 64,
        expected_mapping_sha256="a" * 64,
        expected_mapping_internal_sha256="3" * 64,
        expected_geometry_count=4,
        expected_geometry_sha256="4" * 64,
        expected_trace=(8, 6, 4),
        expected_decoder_layer_count=4,
        expected_gpu_substring="L40S",
    )

    assert admitted["status"] == "admitted-task9-regional-smoke"
    assert admitted["selected_seeds"] == [0, 1, 2, 3]
    assert admitted["shared_prefix_parity_max_abs_error"] == 0.0
    assert admitted["projected_96_branch_decoder_seconds"] == pytest.approx(302.4)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("shared_prefix_parity_max_abs_error", 1e-4, "parity"),
        ("selected_plan.1.seed", 9, "seed"),
        ("forced_interventions.0.prefill_cache_lengths.0", 11, "cache"),
    ),
)
def test_admit_task9_smoke_rejects_rehashed_semantic_drift(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    root = tmp_path / "smoke"
    _artifact(root)
    result_path = root / "result.json"
    result = json.loads(result_path.read_text())
    target = result
    parts = field.split(".")
    for part in parts[:-1]:
        target = target[int(part)] if isinstance(target, list) else target[part]
    last = parts[-1]
    if isinstance(target, list):
        target[int(last)] = value
    else:
        target[last] = value
    result.pop("result_sha256")
    result["result_sha256"] = _sha(result)
    result_file_sha = _write(result_path, result)
    completion_path = root / "completion-manifest.json"
    completion = json.loads(completion_path.read_text())
    completion["result_sha256"] = result["result_sha256"]
    completion["result_file_sha256"] = result_file_sha
    completion.pop("completion_sha256")
    completion["completion_sha256"] = _sha(completion)
    _write(completion_path, completion)

    with pytest.raises(ValueError, match=message):
        admit_task9_regional_smoke(
            root,
            expected_runtime_commit="d" * 40,
            expected_qid="qid",
            expected_boundary="B_1",
            expected_split="fit",
            expected_mask_count=4,
            expected_fixture_sha256="e" * 64,
            expected_mapping_sha256="a" * 64,
            expected_mapping_internal_sha256="3" * 64,
            expected_geometry_count=4,
            expected_geometry_sha256="4" * 64,
            expected_trace=(8, 6, 4),
            expected_decoder_layer_count=4,
            expected_gpu_substring="L40S",
        )
