"""Fail-closed admission for live Task 9 regional-likelihood artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path

from docprune.task9_attribution import (
    _validated_mask_design,
    build_regional_target_outcome,
)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_json(path: Path, label: str) -> tuple[dict[str, object], str]:
    file_path = Path(path)
    if not file_path.is_absolute():
        raise ValueError(f"{label} must be an absolute regular file")
    descriptor: int | None = None
    try:
        descriptor = os.open(file_path, os.O_RDONLY | os.O_NOFOLLOW)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"{label} must be an absolute regular file")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
    except OSError as error:
        raise ValueError(f"{label} must be an absolute regular file") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    content = b"".join(chunks)
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"{label} must contain valid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return payload, hashlib.sha256(content).hexdigest()


def _finite(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Task 9 {label} must be finite")
    try:
        checked = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Task 9 {label} must be finite") from error
    if not math.isfinite(checked) or (positive and checked <= 0):
        raise ValueError(f"Task 9 {label} must be finite")
    return checked


def _unsigned_digest(value: Mapping[str, object], digest_key: str, label: str) -> str:
    unsigned = dict(value)
    observed = unsigned.pop(digest_key, None)
    if not _is_sha256(observed) or observed != _canonical_sha256(unsigned):
        raise ValueError(f"Task 9 {label} internal digest mismatch")
    return observed


def admit_task9_regional_smoke(
    root: Path,
    *,
    expected_runtime_commit: str,
    expected_qid: str,
    expected_boundary: str,
    expected_split: str,
    expected_mask_count: int,
    expected_fixture_sha256: str,
    expected_mapping_sha256: str,
    expected_mapping_internal_sha256: str,
    expected_geometry_count: int,
    expected_geometry_sha256: str,
    expected_trace: tuple[int, int, int],
    expected_decoder_layer_count: int,
    expected_gpu_substring: str,
) -> dict[str, object]:
    """Authenticate one completion-manifest-last Task 9 smoke without fitting."""

    output = Path(root)
    if not output.is_absolute() or output.is_symlink() or not output.is_dir():
        raise ValueError("Task 9 smoke root must be an absolute regular directory")
    if (
        len(expected_runtime_commit) != 40
        or expected_split not in {"fit", "holdout"}
        or type(expected_mask_count) is not int
        or expected_mask_count <= 0
        or type(expected_geometry_count) is not int
        or expected_geometry_count <= 0
        or type(expected_decoder_layer_count) is not int
        or expected_decoder_layer_count <= 0
        or not expected_gpu_substring
        or any(
            not _is_sha256(value)
            for value in (
                expected_fixture_sha256,
                expected_mapping_sha256,
                expected_mapping_internal_sha256,
                expected_geometry_sha256,
            )
        )
    ):
        raise ValueError("Task 9 expected smoke identity is invalid")
    manifest, manifest_file_sha = _read_json(output / "run-manifest.json", "run manifest")
    result, result_file_sha = _read_json(output / "result.json", "result")
    completion, completion_file_sha = _read_json(
        output / "completion-manifest.json", "completion manifest"
    )
    manifest_sha = _unsigned_digest(manifest, "run_manifest_sha256", "run manifest")
    result_sha = _unsigned_digest(result, "result_sha256", "result")
    completion_sha = _unsigned_digest(completion, "completion_sha256", "completion")
    if set(completion) != {
        "schema_version",
        "status",
        "run_manifest_sha256",
        "result_sha256",
        "run_manifest_file_sha256",
        "result_file_sha256",
        "completion_sha256",
    } or completion != {
        "schema_version": 1,
        "status": "complete",
        "run_manifest_sha256": manifest_sha,
        "result_sha256": result_sha,
        "run_manifest_file_sha256": manifest_file_sha,
        "result_file_sha256": result_file_sha,
        "completion_sha256": completion_sha,
    }:
        raise ValueError("Task 9 completion manifest identity mismatch")

    required_manifest = {
        "schema_version",
        "status",
        "runtime_commit",
        "qid",
        "boundary",
        "split",
        "mask_count",
        "selected_seeds",
        "fixed_page_fixture_path",
        "fixed_page_fixture_sha256",
        "config_sha256",
        "run_config_sha256",
        "index_manifest_sha256",
        "mapping_path",
        "mapping_artifact_sha256",
        "mapping_internal_sha256",
        "geometry_count",
        "geometry_sha256",
        "design_sha256",
        "attribution_identity_sha256",
        "assistant_prompt_sha256",
        "prefill_input_ids_shape",
        "prefill_input_ids_sha256",
        "reference_set_token_ids_sha256",
        "fixed_page_provenance",
        "global_index_loaded",
        "retrieval_search_run",
        "cached_retrieved_pages_reused",
        "independent_first_mask_parity_required",
        "run_manifest_sha256",
    }
    expected_seeds = list(
        range(0, expected_mask_count)
        if expected_split == "fit"
        else range(64, 64 + expected_mask_count)
    )
    if (
        set(manifest) != required_manifest
        or manifest["schema_version"] != 1
        or manifest["status"] != "configured-task9-regional-smoke"
        or manifest["runtime_commit"] != expected_runtime_commit
        or manifest["qid"] != expected_qid
        or manifest["boundary"] != expected_boundary
        or manifest["split"] != expected_split
        or manifest["mask_count"] != expected_mask_count
        or manifest["selected_seeds"] != expected_seeds
        or manifest["fixed_page_fixture_sha256"] != expected_fixture_sha256
        or manifest["mapping_artifact_sha256"] != expected_mapping_sha256
        or manifest["mapping_internal_sha256"] != expected_mapping_internal_sha256
        or manifest["geometry_count"] != expected_geometry_count
        or manifest["geometry_sha256"] != expected_geometry_sha256
        or manifest["fixed_page_provenance"] is not True
        or manifest["global_index_loaded"] is not False
        or manifest["retrieval_search_run"] is not False
        or manifest["cached_retrieved_pages_reused"] is not True
        or manifest["independent_first_mask_parity_required"] is not True
    ):
        raise ValueError("Task 9 run manifest identity mismatch")
    for key in (
        "config_sha256",
        "run_config_sha256",
        "index_manifest_sha256",
        "design_sha256",
        "attribution_identity_sha256",
        "assistant_prompt_sha256",
        "prefill_input_ids_sha256",
        "reference_set_token_ids_sha256",
    ):
        if not _is_sha256(manifest[key]):
            raise ValueError("Task 9 run manifest checksum is invalid")
    if (
        not isinstance(manifest["prefill_input_ids_shape"], list)
        or len(manifest["prefill_input_ids_shape"]) != 2
        or manifest["prefill_input_ids_shape"][0] != 1
        or type(manifest["prefill_input_ids_shape"][1]) is not int
        or manifest["prefill_input_ids_shape"][1] <= 0
    ):
        raise ValueError("Task 9 prefill input identity is invalid")

    required_result = {
        "schema_version",
        "status",
        "run_manifest_sha256",
        "design",
        "selected_plan",
        "outcomes",
        "forced_interventions",
        "per_reference_mean_loglikelihoods",
        "independent_first_mask_mean_loglikelihoods",
        "shared_prefix_parity_max_abs_error",
        "original_visual_tokens",
        "post_btp_visual_tokens",
        "post_qtp_visual_tokens",
        "checkpoint_cache_lengths",
        "encoder_seconds",
        "prefix_decoder_seconds",
        "branch_decoder_seconds",
        "branch_decoder_seconds_mean",
        "projected_96_branch_decoder_seconds",
        "measured_scoring_and_parity_seconds",
        "peak_allocated_gpu_bytes",
        "cuda_device_name",
        "result_sha256",
    }
    if (
        set(result) != required_result
        or result["schema_version"] != 1
        or result["status"] != "completed-task9-regional-smoke"
        or result["run_manifest_sha256"] != manifest_sha
        or result["cuda_device_name"] is None
        or expected_gpu_substring not in str(result["cuda_device_name"])
        or (
            result["original_visual_tokens"],
            result["post_btp_visual_tokens"],
            result["post_qtp_visual_tokens"],
        )
        != expected_trace
    ):
        raise ValueError("Task 9 result identity mismatch")
    design = result["design"]
    if not isinstance(design, Mapping):
        raise ValueError("Task 9 mask design is invalid")
    try:
        _, fit_masks, holdout_masks = _validated_mask_design(design)
    except ValueError as error:
        raise ValueError("Task 9 mask design is invalid") from error
    identity = design["attribution_identity"]
    if (
        design["design_sha256"] != manifest["design_sha256"]
        or design["attribution_identity_sha256"] != manifest["attribution_identity_sha256"]
        or identity["question_id"] != expected_qid
        or identity["forced_boundary"] != expected_boundary
        or identity["mapping_artifact_sha256"] != expected_mapping_sha256
        or identity["prompt_input_sha256"] != manifest["prefill_input_ids_sha256"]
        or identity["reference_set_token_ids_sha256"] != manifest["reference_set_token_ids_sha256"]
    ):
        raise ValueError("Task 9 mask design identity mismatch")
    masks = fit_masks if expected_split == "fit" else holdout_masks
    expected_masks = masks[:expected_mask_count]
    plans = result["selected_plan"]
    outcomes = result["outcomes"]
    forced_rows = result["forced_interventions"]
    values = result["per_reference_mean_loglikelihoods"]
    if any(
        not isinstance(rows, list) or len(rows) != expected_mask_count
        for rows in (plans, outcomes, forced_rows, values)
    ):
        raise ValueError("Task 9 smoke mask row count mismatch")
    for index, (plan, outcome, forced, likelihood, mask) in enumerate(
        zip(plans, outcomes, forced_rows, values, expected_masks, strict=True)
    ):
        if not isinstance(plan, Mapping) or plan.get("seed") != expected_seeds[index]:
            raise ValueError("Task 9 selected seed order mismatch")
        if (
            plan.get("split") != expected_split
            or plan.get("vector") != mask["vector"]
            or plan.get("vector_sha256") != mask["vector_sha256"]
            or plan.get("retained_source_ids") != mask["retained_source_ids"]
            or plan.get("attribution_identity_sha256") != design["attribution_identity_sha256"]
        ):
            raise ValueError("Task 9 selected mask identity mismatch")
        retained = plan.get("retained_visual_ids")
        if (
            not isinstance(retained, list)
            or retained != sorted(set(retained))
            or any(
                type(value) is not int or not 0 <= value < expected_geometry_count
                for value in retained
            )
            or plan.get("retained_visual_count") != len(retained)
            or plan.get("visual_population") != expected_geometry_count
        ):
            raise ValueError("Task 9 retained token identity mismatch")
        if not isinstance(likelihood, Sequence) or isinstance(likelihood, str | bytes):
            raise ValueError("Task 9 likelihood row is invalid")
        expected_outcome = build_regional_target_outcome(
            design,
            plan,
            mean_sequence_loglikelihoods=likelihood,
        )
        if outcome != expected_outcome:
            raise ValueError("Task 9 normalized likelihood outcome mismatch")
        if not isinstance(forced, Mapping):
            raise ValueError("Task 9 forced intervention record is invalid")
        cache = forced.get("prefill_cache_lengths")
        logical = forced.get("logical_retained_sequence_ids")
        position_shape = forced.get("retained_mrope_position_shape")
        if (
            forced.get("boundary") != expected_boundary
            or forced.get("mode") != "physical_delete"
            or forced.get("selection_kind") != "forced"
            or forced.get("visual_population") != expected_geometry_count
            or forced.get("requested_budget") != len(retained)
            or forced.get("achieved_budget") != len(retained)
            or forced.get("retained_visual_ids") != retained
            or not isinstance(logical, list)
            or logical != sorted(set(logical))
            or not isinstance(cache, list)
            or len(cache) != expected_decoder_layer_count
            or not isinstance(position_shape, list)
            or len(position_shape) != 3
            or position_shape[:2] != [3, 1]
            or position_shape[2] != len(logical)
            or any(
                length != len(logical)
                for length in cache[len(result["checkpoint_cache_lengths"]) :]
            )
        ):
            raise ValueError("Task 9 forced cache topology mismatch")

    prefix = result["checkpoint_cache_lengths"]
    expected_prefix = 0 if expected_boundary == "B_input" else int(expected_boundary[2:]) + 1
    if (
        not isinstance(prefix, list)
        or len(prefix) != expected_prefix
        or (prefix and any(length != prefix[0] for length in prefix))
        or any(
            forced["prefill_cache_lengths"][:expected_prefix] != prefix for forced in forced_rows
        )
    ):
        raise ValueError("Task 9 shared prefix cache topology mismatch")
    first_values = result["independent_first_mask_mean_loglikelihoods"]
    if not isinstance(first_values, list) or len(first_values) != len(values[0]):
        raise ValueError("Task 9 independent parity row is invalid")
    observed_error = max(
        abs(_finite(left, "parity") - _finite(right, "parity"))
        for left, right in zip(values[0], first_values, strict=True)
    )
    recorded_error = _finite(result["shared_prefix_parity_max_abs_error"], "parity")
    if abs(recorded_error - observed_error) > 1e-12 or recorded_error > 1e-6:
        raise ValueError("Task 9 shared-prefix parity failed")
    branch_times = result["branch_decoder_seconds"]
    if not isinstance(branch_times, list) or len(branch_times) != expected_mask_count:
        raise ValueError("Task 9 branch timing count mismatch")
    checked_times = [_finite(value, "branch timing", positive=True) for value in branch_times]
    mean = sum(checked_times) / len(checked_times)
    if (
        abs(_finite(result["branch_decoder_seconds_mean"], "branch timing") - mean) > 1e-9
        or abs(_finite(result["projected_96_branch_decoder_seconds"], "projection") - mean * 96)
        > 1e-7
        or type(result["peak_allocated_gpu_bytes"]) is not int
        or result["peak_allocated_gpu_bytes"] < 0
    ):
        raise ValueError("Task 9 timing projection is invalid")
    return {
        "schema_version": 1,
        "status": "admitted-task9-regional-smoke",
        "qid": expected_qid,
        "boundary": expected_boundary,
        "split": expected_split,
        "selected_seeds": expected_seeds,
        "cuda_device_name": result["cuda_device_name"],
        "shared_prefix_parity_max_abs_error": recorded_error,
        "branch_decoder_seconds_mean": mean,
        "projected_96_branch_decoder_seconds": mean * 96,
        "peak_allocated_gpu_bytes": result["peak_allocated_gpu_bytes"],
        "run_manifest_file_sha256": manifest_file_sha,
        "result_file_sha256": result_file_sha,
        "completion_manifest_file_sha256": completion_file_sha,
        "completion_sha256": completion_sha,
    }


__all__ = ["admit_task9_regional_smoke"]
