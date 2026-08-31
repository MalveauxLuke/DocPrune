"""Admission tests for live Task 9 regional-likelihood artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from docprune import task9_attribution, task9_live
from docprune.ctp_controls import VisualTokenGeometry
from docprune.segmentation import RegionTokenMapping, RegionTokenSource
from docprune.task9_attribution import build_region_mask_design
from docprune.task9_live import admit_task9_regional_smoke


def _sha(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write(path: Path, value: object) -> str:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _signed(payload: dict[str, object], key: str) -> dict[str, object]:
    result = dict(payload)
    result[key] = _sha(result)
    return result


def _development_mapping() -> RegionTokenMapping:
    sources = tuple(
        RegionTokenSource(
            source_id=f"region-{index}",
            source_kind="residual-grid",
            input_page_index=0,
            mineru_page_index=0,
            document_id="doc",
            source_page_index=0,
            region_type="residual",
            bbox=(0.0, 0.0, 1.0, 1.0),
            page_size=(1.0, 1.0),
            reading_order=None,
            token_ids=(index,),
        )
        for index in range(4)
    )
    return RegionTokenMapping(
        artifacts=(),
        geometry=tuple(VisualTokenGeometry(0, 0, index, 1, 4) for index in range(4)),
        geometry_count=4,
        geometry_sha256="4" * 64,
        residual_grid_size=4,
        assignment_contract="test-partition",
        audited_regions=(),
        empty_region_source_ids=(),
        sources=sources,
        token_to_source=tuple(source.source_id for source in sources),
        sha256="3" * 64,
    )


def test_development_publication_is_no_replace_and_completion_manifest_last(
    tmp_path: Path,
) -> None:
    """Catch overwriting an artifact or admitting it without all signed member bytes."""

    publisher = getattr(task9_live, "publish_task9_regional_development", None)
    assert publisher is not None, "Task 9 development publisher is missing"
    root = tmp_path / "development"
    manifest = _signed({"schema_version": 1, "status": "configured"}, "run_manifest_sha256")
    raw = _signed({"schema_version": 1, "status": "raw"}, "raw_result_sha256")
    primary = _signed({"schema_version": 1, "target_kind": "primary"}, "target_dataset_sha256")
    secondary = _signed({"schema_version": 1, "target_kind": "secondary"}, "target_dataset_sha256")

    completion = publisher(root, manifest, raw, primary, secondary)

    assert [path.name for path in sorted(root.iterdir())] == [
        "completion-manifest.json",
        "primary-target.json",
        "raw-result.json",
        "run-manifest.json",
        "secondary-target.json",
    ]
    assert json.loads((root / "completion-manifest.json").read_text()) == completion
    assert completion["status"] == "complete"
    with pytest.raises(FileExistsError):
        publisher(root, manifest, raw, primary, secondary)


def test_development_scoring_uses_one_call_for_all_96_interventions() -> None:
    """Catch splitting fit and holdout masks into separate prefix-scoring calls."""

    scorer = getattr(task9_live, "score_task9_regional_development_once", None)
    assert scorer is not None, "Task 9 one-call development scorer is missing"

    class Answerer:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def score_forced_intervention_likelihoods(self, *args, **kwargs):
            self.calls.append({"args": args, **kwargs})
            return "shared-result"

    answerer = Answerer()
    interventions = tuple(object() for _ in range(96))
    result = scorer(
        answerer,
        images=("page",),
        question="question",
        retrieval_output="fixed-retrieval",
        forced_interventions=interventions,
        reference_target_token_ids=((1, 2), (3,)),
    )

    assert result == "shared-result"
    assert len(answerer.calls) == 1
    assert answerer.calls[0]["forced_interventions"] == interventions
    assert answerer.calls[0]["teacher_forced_target_token_ids"] == ((1, 2), (3,))
    assert answerer.calls[0]["include_unpruned_generated_response"] is True


def test_development_scoring_supports_one_320_intervention_call() -> None:
    """Catch splitting or rejecting the frozen 256-fit/64-holdout diagnostic."""

    answerer = type(
        "Answerer",
        (),
        {"score_forced_intervention_likelihoods": lambda self, *args, **kwargs: kwargs},
    )()
    interventions = tuple(object() for _ in range(320))

    result = task9_live.score_task9_regional_development_once(
        answerer,
        images=("page",),
        question="question",
        retrieval_output="fixed-retrieval",
        forced_interventions=interventions,
        reference_target_token_ids=((1, 2),),
        expected_mask_count=320,
    )

    assert result["forced_interventions"] == interventions
    assert result["include_unpruned_generated_response"] is True


def test_b13_256_launcher_freezes_256_fit_and_64_holdout_masks() -> None:
    launcher = (
        Path(__file__).parents[1]
        / "examples"
        / "sbatch"
        / "40_docprune_task9_b13_256_diagnostic.sbatch"
    ).read_text()

    assert "#SBATCH --constraint=l40s" in launcher
    assert "#SBATCH --mem=24G" in launcher
    assert "#SBATCH --no-requeue" in launcher
    assert "--boundary 13" in launcher
    assert "--fit-mask-count 256" in launcher
    assert "--holdout-mask-count 64" in launcher


def test_input_256_launcher_freezes_matching_schedule_before_decoder_blocks() -> None:
    launcher = (
        Path(__file__).parents[1]
        / "examples"
        / "sbatch"
        / "41_docprune_task9_input_256_diagnostic.sbatch"
    ).read_text()

    assert "#SBATCH --constraint=l40s" in launcher
    assert "#SBATCH --mem=24G" in launcher
    assert "#SBATCH --no-requeue" in launcher
    assert "29e6ac2294d6b87075a1d45b8a8df175b214248a" in launcher
    assert "--boundary input" in launcher
    assert "--boundary B_input" in launcher
    assert "--fit-mask-count 256" in launcher
    assert "--holdout-mask-count 64" in launcher


def test_development_admission_requires_completion_authority(tmp_path: Path) -> None:
    """Catch admission of a partial development publication after an interrupted run."""

    validator = getattr(task9_live, "admit_task9_regional_development", None)
    assert validator is not None, "Task 9 development validator is missing"
    root = tmp_path / "partial"
    root.mkdir()
    with pytest.raises(ValueError, match="completion manifest"):
        validator(
            root,
            expected_runtime_commit="d" * 40,
            expected_qid="qid",
            expected_boundary="B_13",
            expected_fixture_sha256="e" * 64,
            expected_mapping_sha256="a" * 64,
            expected_mapping_internal_sha256="3" * 64,
            expected_geometry_count=4,
            expected_geometry_sha256="4" * 64,
            expected_trace=(8, 6, 4),
            expected_decoder_layer_count=28,
            expected_gpu_substring="L40S",
        )


def test_development_admission_replays_dual_targets_and_all_physical_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch accepting stored targets or fewer than 96 authenticated cache records."""

    mapping = _development_mapping()
    input_paths = {}
    for name in ("config", "fixture", "mapping", "index", "run-config"):
        path = tmp_path / f"{name}.json"
        input_paths[name] = (path, _write(path, {"name": name}))
    mapping_sha = input_paths["mapping"][1]
    regions = [
        {"source_id": source.source_id, "token_cost": len(source.token_ids)}
        for source in mapping.sources
    ]
    primary_design = build_region_mask_design(
        regions,
        question_id="qid",
        forced_boundary="B_13",
        mapping_artifact_sha256=mapping_sha,
        prompt_input_sha256="b" * 64,
        target_kind="max-accepted-reference-mean-loglikelihood",
        reference_set_token_ids_sha256="c" * 64,
        generated_response_token_ids_sha256=None,
    )
    generated_ids = [70, 71]
    generated_hash = _sha(generated_ids)
    secondary_design = build_region_mask_design(
        regions,
        question_id="qid",
        forced_boundary="B_13",
        mapping_artifact_sha256=mapping_sha,
        prompt_input_sha256="b" * 64,
        target_kind="unpruned-generated-response-mean-loglikelihood",
        reference_set_token_ids_sha256=None,
        generated_response_token_ids_sha256=generated_hash,
    )
    plan = task9_attribution.build_regional_development_plan(
        mapping,
        primary_design,
        secondary_design,
        mapping_artifact_sha256=mapping_sha,
    )
    likelihoods = [[-0.8, -0.2, -0.1 - seed / 1000] for seed in range(96)]
    targets = task9_attribution.build_regional_development_targets(
        primary_design,
        secondary_design,
        plan,
        mean_sequence_loglikelihoods=likelihoods,
        reference_sequence_count=2,
    )
    manifest = _signed(
        {
            "schema_version": 1,
            "status": "configured-task9-regional-development",
            "runtime_commit": "d" * 40,
            "qid": "qid",
            "boundary": "B_13",
            "mask_count": 96,
            "seed_order": list(range(96)),
            "config_path": str(input_paths["config"][0]),
            "fixed_page_fixture_path": str(input_paths["fixture"][0]),
            "mapping_path": str(input_paths["mapping"][0]),
            "index_manifest_path": str(input_paths["index"][0]),
            "run_config_path": str(input_paths["run-config"][0]),
            "config_sha256": input_paths["config"][1],
            "run_config_sha256": input_paths["run-config"][1],
            "index_manifest_sha256": input_paths["index"][1],
            "fixed_page_fixture_sha256": input_paths["fixture"][1],
            "mapping_artifact_sha256": mapping_sha,
            "mapping_internal_sha256": mapping.sha256,
            "geometry_count": 4,
            "geometry_sha256": mapping.geometry_sha256,
            "assistant_prompt_sha256": "5" * 64,
            "prefill_input_ids_shape": [1, 7],
            "prefill_input_ids_sha256": "b" * 64,
            "reference_set_token_ids_sha256": "c" * 64,
            "generated_response_token_ids_sha256": generated_hash,
            "primary_attribution_identity_sha256": primary_design["attribution_identity_sha256"],
            "secondary_attribution_identity_sha256": secondary_design[
                "attribution_identity_sha256"
            ],
            "primary_target_dataset_sha256": targets["primary"]["target_dataset_sha256"],
            "secondary_target_dataset_sha256": targets["secondary"]["target_dataset_sha256"],
            "fixed_page_provenance": True,
            "cached_retrieved_pages_reused": True,
            "global_index_loaded": False,
            "retrieval_search_run": False,
        },
        "run_manifest_sha256",
    )
    forced_rows = []
    for row in plan:
        retained = row["retained_visual_ids"]
        compact = 3 + len(retained)
        forced_rows.append(
            {
                "boundary": "B_13",
                "mode": "physical_delete",
                "selection_kind": "forced",
                "visual_population": 4,
                "requested_budget": len(retained),
                "achieved_budget": len(retained),
                "retained_visual_ids": retained,
                "logical_retained_sequence_ids": list(range(compact)),
                "prefill_cache_lengths": [7] * 14 + [compact] * 14,
                "retained_mrope_position_shape": [3, 1, compact],
                "retained_mrope_position_sha256": "6" * 64,
            }
        )
    raw = _signed(
        {
            "schema_version": 1,
            "status": "completed-task9-regional-development",
            "run_manifest_sha256": manifest["run_manifest_sha256"],
            "reference_sequence_count": 2,
            "raw_mean_sequence_loglikelihoods": likelihoods,
            "development_plan": [
                {key: value for key, value in row.items() if key != "forced_intervention"}
                for row in plan
            ],
            "forced_interventions": forced_rows,
            "generated_response_token_ids": generated_ids,
            "generated_response_token_ids_sha256": generated_hash,
            "terminal_eos_token_id": 151645,
            "unpruned_generation_trace": {
                "original_visual_tokens": 8,
                "post_btp_visual_tokens": 6,
                "post_qtp_visual_tokens": 4,
                "post_ctp_visual_tokens": 4,
                "ctp_layer": None,
            },
            "original_visual_tokens": 8,
            "post_btp_visual_tokens": 6,
            "post_qtp_visual_tokens": 4,
            "checkpoint_cache_lengths": [7] * 14,
            "encoder_seconds": 1.0,
            "prefix_decoder_seconds": 2.0,
            "branch_decoder_seconds": [0.1] * 96,
            "unpruned_generation_encoder_seconds": 1.0,
            "unpruned_generation_decoder_seconds": 2.0,
            "peak_allocated_gpu_bytes": 100,
            "cuda_device_name": "NVIDIA L40S",
        },
        "raw_result_sha256",
    )
    root = tmp_path / "development"
    task9_live.publish_task9_regional_development(
        root, manifest, raw, targets["primary"], targets["secondary"]
    )
    monkeypatch.setattr(task9_live, "load_region_mapping", lambda *args, **kwargs: mapping)

    admitted = task9_live.admit_task9_regional_development(
        root,
        expected_runtime_commit="d" * 40,
        expected_qid="qid",
        expected_boundary="B_13",
        expected_fixture_sha256=input_paths["fixture"][1],
        expected_mapping_sha256=mapping_sha,
        expected_mapping_internal_sha256=mapping.sha256,
        expected_geometry_count=4,
        expected_geometry_sha256=mapping.geometry_sha256,
        expected_trace=(8, 6, 4),
        expected_decoder_layer_count=28,
        expected_gpu_substring="L40S",
    )

    assert admitted["status"] == "admitted-task9-regional-development"
    assert admitted["mask_count"] == 96

    with pytest.raises(ValueError, match="mask split"):
        task9_live.admit_task9_regional_development(
            root,
            expected_runtime_commit="d" * 40,
            expected_qid="qid",
            expected_boundary="B_13",
            expected_fixture_sha256=input_paths["fixture"][1],
            expected_mapping_sha256=mapping_sha,
            expected_mapping_internal_sha256=mapping.sha256,
            expected_geometry_count=4,
            expected_geometry_sha256=mapping.geometry_sha256,
            expected_trace=(8, 6, 4),
            expected_decoder_layer_count=28,
            expected_gpu_substring="L40S",
            expected_fit_mask_count=65,
            expected_holdout_mask_count=31,
        )


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
