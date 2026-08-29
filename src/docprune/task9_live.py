"""Fail-closed admission for live Task 9 regional-likelihood artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path

from docprune.segmentation import load_region_mapping
from docprune.task9_attribution import (
    _validated_mask_design,
    build_regional_development_plan,
    build_regional_target_outcome,
    validate_regional_development_targets,
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


def _regular_file_sha256(path: Path, label: str) -> str:
    file_path = Path(path)
    if not file_path.is_absolute():
        raise ValueError(f"{label} must be an absolute regular file")
    descriptor: int | None = None
    try:
        descriptor = os.open(file_path, os.O_RDONLY | os.O_NOFOLLOW)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"{label} must be an absolute regular file")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        return digest.hexdigest()
    except OSError as error:
        raise ValueError(f"{label} must be an absolute regular file") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


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


def _publish_json_member(path: Path, payload: Mapping[str, object]) -> str:
    content = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o644,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    return hashlib.sha256(content).hexdigest()


def publish_task9_regional_development(
    root: Path,
    run_manifest: Mapping[str, object],
    raw_result: Mapping[str, object],
    primary_target: Mapping[str, object],
    secondary_target: Mapping[str, object],
) -> dict[str, object]:
    """Publish four signed members and the completion authority without replacement."""

    output = Path(root)
    if not output.is_absolute() or output.is_symlink() or not output.parent.is_dir():
        raise ValueError("Task 9 development root must have an absolute existing parent")
    _unsigned_digest(run_manifest, "run_manifest_sha256", "run manifest")
    _unsigned_digest(raw_result, "raw_result_sha256", "raw result")
    _unsigned_digest(primary_target, "target_dataset_sha256", "primary target")
    _unsigned_digest(secondary_target, "target_dataset_sha256", "secondary target")
    os.mkdir(output)
    members = {
        "run-manifest.json": run_manifest,
        "raw-result.json": raw_result,
        "primary-target.json": primary_target,
        "secondary-target.json": secondary_target,
    }
    file_hashes = {
        name: _publish_json_member(output / name, payload) for name, payload in members.items()
    }
    completion: dict[str, object] = {
        "schema_version": 1,
        "status": "complete",
        "member_file_sha256": file_hashes,
        "run_manifest_sha256": run_manifest["run_manifest_sha256"],
        "raw_result_sha256": raw_result["raw_result_sha256"],
        "primary_target_dataset_sha256": primary_target["target_dataset_sha256"],
        "secondary_target_dataset_sha256": secondary_target["target_dataset_sha256"],
    }
    completion["completion_sha256"] = _canonical_sha256(completion)
    _publish_json_member(output / "completion-manifest.json", completion)
    directory = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return completion


def score_task9_regional_development_once(
    answerer: object,
    *,
    images: Sequence[object],
    question: str,
    retrieval_output: object,
    forced_interventions: Sequence[object],
    reference_target_token_ids: Sequence[Sequence[int]],
) -> object:
    """Execute the canonical 96-mask call, including internal response capture."""

    interventions = tuple(forced_interventions)
    targets = tuple(tuple(row) for row in reference_target_token_ids)
    scoring = getattr(answerer, "score_forced_intervention_likelihoods", None)
    if len(interventions) != 96 or not targets or not callable(scoring):
        raise ValueError("Task 9 development scoring requires 96 masks and reference targets")
    return scoring(
        images,
        question,
        retrieval_output=retrieval_output,
        forced_interventions=interventions,
        teacher_forced_target_token_ids=targets,
        include_unpruned_generated_response=True,
    )


def admit_task9_regional_development(
    root: Path,
    *,
    expected_runtime_commit: str,
    expected_qid: str,
    expected_boundary: str,
    expected_fixture_sha256: str,
    expected_mapping_sha256: str,
    expected_mapping_internal_sha256: str,
    expected_geometry_count: int,
    expected_geometry_sha256: str,
    expected_trace: tuple[int, int, int],
    expected_decoder_layer_count: int,
    expected_gpu_substring: str,
) -> dict[str, object]:
    """Authenticate the terminal one-question 96-mask dual-target artifact."""

    output = Path(root)
    if not output.is_absolute() or output.is_symlink() or not output.is_dir():
        raise ValueError("Task 9 development root must be an absolute regular directory")
    completion, completion_file_sha = _read_json(
        output / "completion-manifest.json", "completion manifest"
    )
    manifest, manifest_file_sha = _read_json(output / "run-manifest.json", "run manifest")
    raw, raw_file_sha = _read_json(output / "raw-result.json", "raw result")
    primary, primary_file_sha = _read_json(output / "primary-target.json", "primary target")
    secondary, secondary_file_sha = _read_json(output / "secondary-target.json", "secondary target")
    completion_sha = _unsigned_digest(completion, "completion_sha256", "completion")
    manifest_sha = _unsigned_digest(manifest, "run_manifest_sha256", "run manifest")
    raw_sha = _unsigned_digest(raw, "raw_result_sha256", "raw result")
    primary_sha = _unsigned_digest(primary, "target_dataset_sha256", "primary target")
    secondary_sha = _unsigned_digest(secondary, "target_dataset_sha256", "secondary target")
    expected_member_hashes = {
        "run-manifest.json": manifest_file_sha,
        "raw-result.json": raw_file_sha,
        "primary-target.json": primary_file_sha,
        "secondary-target.json": secondary_file_sha,
    }
    if completion != {
        "schema_version": 1,
        "status": "complete",
        "member_file_sha256": expected_member_hashes,
        "run_manifest_sha256": manifest_sha,
        "raw_result_sha256": raw_sha,
        "primary_target_dataset_sha256": primary_sha,
        "secondary_target_dataset_sha256": secondary_sha,
        "completion_sha256": completion_sha,
    }:
        raise ValueError("Task 9 completion manifest identity mismatch")
    required_manifest = {
        "schema_version",
        "status",
        "runtime_commit",
        "qid",
        "boundary",
        "mask_count",
        "seed_order",
        "config_path",
        "fixed_page_fixture_path",
        "mapping_path",
        "index_manifest_path",
        "run_config_path",
        "config_sha256",
        "run_config_sha256",
        "index_manifest_sha256",
        "fixed_page_fixture_sha256",
        "mapping_artifact_sha256",
        "mapping_internal_sha256",
        "geometry_count",
        "geometry_sha256",
        "assistant_prompt_sha256",
        "prefill_input_ids_shape",
        "prefill_input_ids_sha256",
        "reference_set_token_ids_sha256",
        "generated_response_token_ids_sha256",
        "primary_attribution_identity_sha256",
        "secondary_attribution_identity_sha256",
        "primary_target_dataset_sha256",
        "secondary_target_dataset_sha256",
        "fixed_page_provenance",
        "cached_retrieved_pages_reused",
        "global_index_loaded",
        "retrieval_search_run",
        "run_manifest_sha256",
    }
    if (
        set(manifest) != required_manifest
        or manifest["schema_version"] != 1
        or manifest["status"] != "configured-task9-regional-development"
        or manifest["runtime_commit"] != expected_runtime_commit
        or manifest["qid"] != expected_qid
        or manifest["boundary"] != expected_boundary
        or manifest["mask_count"] != 96
        or manifest["seed_order"] != list(range(96))
        or manifest["fixed_page_fixture_sha256"] != expected_fixture_sha256
        or manifest["mapping_artifact_sha256"] != expected_mapping_sha256
        or manifest["mapping_internal_sha256"] != expected_mapping_internal_sha256
        or manifest["geometry_count"] != expected_geometry_count
        or manifest["geometry_sha256"] != expected_geometry_sha256
        or manifest["primary_target_dataset_sha256"] != primary_sha
        or manifest["secondary_target_dataset_sha256"] != secondary_sha
        or manifest["fixed_page_provenance"] is not True
        or manifest["cached_retrieved_pages_reused"] is not True
        or manifest["global_index_loaded"] is not False
        or manifest["retrieval_search_run"] is not False
    ):
        raise ValueError("Task 9 development run manifest identity mismatch")
    for key in (
        "config_sha256",
        "run_config_sha256",
        "index_manifest_sha256",
        "assistant_prompt_sha256",
        "prefill_input_ids_sha256",
        "reference_set_token_ids_sha256",
        "generated_response_token_ids_sha256",
        "primary_attribution_identity_sha256",
        "secondary_attribution_identity_sha256",
    ):
        if not _is_sha256(manifest[key]):
            raise ValueError("Task 9 development manifest checksum is invalid")
    shape = manifest["prefill_input_ids_shape"]
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or shape[0] != 1
        or type(shape[1]) is not int
        or shape[1] <= 0
    ):
        raise ValueError("Task 9 development prefill input identity is invalid")
    for path_key, hash_key in (
        ("config_path", "config_sha256"),
        ("fixed_page_fixture_path", "fixed_page_fixture_sha256"),
        ("mapping_path", "mapping_artifact_sha256"),
        ("index_manifest_path", "index_manifest_sha256"),
        ("run_config_path", "run_config_sha256"),
    ):
        path = Path(str(manifest[path_key]))
        if _regular_file_sha256(path, path_key) != manifest[hash_key]:
            raise ValueError(f"Task 9 {path_key} bytes drifted")
    mapping = load_region_mapping(Path(str(manifest["mapping_path"])), validate_raw_artifacts=True)
    if (
        mapping.sha256 != expected_mapping_internal_sha256
        or mapping.geometry_count != expected_geometry_count
        or mapping.geometry_sha256 != expected_geometry_sha256
    ):
        raise ValueError("Task 9 development mapping identity mismatch")
    if (
        primary.get("attribution_identity_sha256")
        != manifest["primary_attribution_identity_sha256"]
        or secondary.get("attribution_identity_sha256")
        != manifest["secondary_attribution_identity_sha256"]
    ):
        raise ValueError("Task 9 development target identity mismatch")
    required_raw = {
        "schema_version",
        "status",
        "run_manifest_sha256",
        "reference_sequence_count",
        "raw_mean_sequence_loglikelihoods",
        "development_plan",
        "forced_interventions",
        "generated_response_token_ids",
        "generated_response_token_ids_sha256",
        "terminal_eos_token_id",
        "unpruned_generation_trace",
        "original_visual_tokens",
        "post_btp_visual_tokens",
        "post_qtp_visual_tokens",
        "checkpoint_cache_lengths",
        "encoder_seconds",
        "prefix_decoder_seconds",
        "branch_decoder_seconds",
        "unpruned_generation_encoder_seconds",
        "unpruned_generation_decoder_seconds",
        "peak_allocated_gpu_bytes",
        "cuda_device_name",
        "raw_result_sha256",
    }
    if (
        set(raw) != required_raw
        or raw["schema_version"] != 1
        or raw["status"] != "completed-task9-regional-development"
        or raw["run_manifest_sha256"] != manifest_sha
        or not isinstance(raw["cuda_device_name"], str)
        or expected_gpu_substring not in raw["cuda_device_name"]
        or (
            raw["original_visual_tokens"],
            raw["post_btp_visual_tokens"],
            raw["post_qtp_visual_tokens"],
        )
        != expected_trace
    ):
        raise ValueError("Task 9 development raw result identity mismatch")
    generated_ids = raw["generated_response_token_ids"]
    generation_trace = raw["unpruned_generation_trace"]
    if (
        not isinstance(generated_ids, list)
        or not generated_ids
        or any(type(token) is not int or token < 0 for token in generated_ids)
        or _canonical_sha256(generated_ids) != raw["generated_response_token_ids_sha256"]
        or raw["generated_response_token_ids_sha256"]
        != manifest["generated_response_token_ids_sha256"]
        or raw["terminal_eos_token_id"] not in {151645, 151643}
        or not isinstance(generation_trace, Mapping)
        or generation_trace.get("ctp_layer") is not None
        or generation_trace.get("post_ctp_visual_tokens") != expected_trace[2]
        or (
            generation_trace.get("original_visual_tokens"),
            generation_trace.get("post_btp_visual_tokens"),
            generation_trace.get("post_qtp_visual_tokens"),
        )
        != expected_trace
    ):
        raise ValueError("Task 9 generated response or no-CTP trace is invalid")
    primary_design = primary.get("design")
    secondary_design = secondary.get("design")
    if not isinstance(primary_design, Mapping) or not isinstance(secondary_design, Mapping):
        raise ValueError("Task 9 development designs are missing")
    primary_identity = primary_design.get("attribution_identity")
    secondary_identity = secondary_design.get("attribution_identity")
    if (
        not isinstance(primary_identity, Mapping)
        or not isinstance(secondary_identity, Mapping)
        or primary_identity.get("question_id") != expected_qid
        or secondary_identity.get("question_id") != expected_qid
        or primary_identity.get("forced_boundary") != expected_boundary
        or secondary_identity.get("forced_boundary") != expected_boundary
        or primary_identity.get("mapping_artifact_sha256") != expected_mapping_sha256
        or secondary_identity.get("mapping_artifact_sha256") != expected_mapping_sha256
        or primary_identity.get("prompt_input_sha256") != manifest["prefill_input_ids_sha256"]
        or secondary_identity.get("prompt_input_sha256") != manifest["prefill_input_ids_sha256"]
        or primary_identity.get("reference_set_token_ids_sha256")
        != manifest["reference_set_token_ids_sha256"]
        or secondary_identity.get("generated_response_token_ids_sha256")
        != manifest["generated_response_token_ids_sha256"]
    ):
        raise ValueError("Task 9 development attribution identities drifted")
    expected_plan = build_regional_development_plan(
        mapping,
        primary_design,
        secondary_design,
        mapping_artifact_sha256=expected_mapping_sha256,
    )
    serialized_plan = [
        {key: value for key, value in row.items() if key != "forced_intervention"}
        for row in expected_plan
    ]
    if raw["development_plan"] != serialized_plan:
        raise ValueError("Task 9 development physical plan drifted")
    validate_regional_development_targets(
        primary,
        secondary,
        raw["development_plan"],
        raw["raw_mean_sequence_loglikelihoods"],
        reference_sequence_count=raw["reference_sequence_count"],
    )
    forced_rows = raw["forced_interventions"]
    prefix = raw["checkpoint_cache_lengths"]
    branch_times = raw["branch_decoder_seconds"]
    expected_prefix = 0 if expected_boundary == "B_input" else int(expected_boundary[2:]) + 1
    if (
        not isinstance(forced_rows, list)
        or len(forced_rows) != 96
        or type(raw["reference_sequence_count"]) is not int
        or raw["reference_sequence_count"] <= 0
        or not isinstance(prefix, list)
        or len(prefix) != expected_prefix
        or any(type(value) is not int or value <= 0 for value in prefix)
        or not isinstance(branch_times, list)
        or len(branch_times) != 96
        or any(_finite(value, "branch timing", positive=True) <= 0 for value in branch_times)
        or any(
            _finite(raw[key], "runtime timing") < 0
            for key in (
                "encoder_seconds",
                "prefix_decoder_seconds",
                "unpruned_generation_encoder_seconds",
                "unpruned_generation_decoder_seconds",
            )
        )
        or type(raw["peak_allocated_gpu_bytes"]) is not int
        or raw["peak_allocated_gpu_bytes"] < 0
    ):
        raise ValueError("Task 9 development branch or shared-prefix count is invalid")
    for plan, forced in zip(serialized_plan, forced_rows, strict=True):
        logical = (
            forced.get("logical_retained_sequence_ids") if isinstance(forced, Mapping) else None
        )
        cache = forced.get("prefill_cache_lengths") if isinstance(forced, Mapping) else None
        shape = forced.get("retained_mrope_position_shape") if isinstance(forced, Mapping) else None
        retained = plan["retained_visual_ids"]
        if (
            not isinstance(forced, Mapping)
            or forced.get("boundary") != expected_boundary
            or forced.get("mode") != "physical_delete"
            or forced.get("visual_population") != expected_geometry_count
            or forced.get("retained_visual_ids") != retained
            or forced.get("requested_budget") != len(retained)
            or forced.get("achieved_budget") != len(retained)
            or not isinstance(logical, list)
            or logical != sorted(set(logical))
            or not isinstance(cache, list)
            or len(cache) != expected_decoder_layer_count
            or cache[:expected_prefix] != prefix
            or any(value != len(logical) for value in cache[expected_prefix:])
            or not isinstance(shape, list)
            or shape != [3, 1, len(logical)]
            or not _is_sha256(forced.get("retained_mrope_position_sha256"))
        ):
            raise ValueError("Task 9 development forced cache or M-RoPE topology is invalid")
    return {
        "schema_version": 1,
        "status": "admitted-task9-regional-development",
        "qid": expected_qid,
        "boundary": expected_boundary,
        "mask_count": 96,
        "target_dataset_sha256s": [primary_sha, secondary_sha],
        "cuda_device_name": raw["cuda_device_name"],
        "completion_manifest_file_sha256": completion_file_sha,
        "completion_sha256": completion_sha,
    }


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


__all__ = [
    "admit_task9_regional_smoke",
    "publish_task9_regional_development",
    "score_task9_regional_development_once",
    "admit_task9_regional_development",
]
