#!/usr/bin/env python3
"""Run a bounded Task 9 shared-boundary regional-likelihood smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import torch

from docprune.answerers import (
    DocPruneQwenAnswerer,
    _input_ids_identity,
    _prepare_batch_with_prompt,
    prepare_task7_likelihood_target_for_question,
)
from docprune.config import load_config
from docprune.ctp_policy import btp_qtp_no_ctp_policy
from docprune.m3docrag import RetrievalOutput
from docprune.m3docvqa_factory import build_workload
from docprune.qwen2vl.preprocessing import prepare_qwen_page, prepared_raster_image
from docprune.segmentation import load_region_mapping
from docprune.task6_runtime import load_fixed_page_fixture
from docprune.task9_attribution import (
    build_region_mask_design,
    build_regional_intervention_plan,
    build_regional_target_outcome,
)


def _sha256(path: Path) -> str:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError(f"authenticated input must be an absolute regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        descriptor = -1
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor != -1:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _boundary(value: str) -> str | int:
    if value == "input":
        return value
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("boundary must be input or a block index") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("boundary block index must be nonnegative")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--index-manifest", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--fixture-sha256", required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--mapping-sha256", required=True)
    parser.add_argument("--expected-geometry-count", type=int, required=True)
    parser.add_argument("--expected-geometry-sha256", required=True)
    parser.add_argument("--qid", required=True)
    parser.add_argument("--boundary", type=_boundary, required=True)
    parser.add_argument("--split", choices=("fit", "holdout"), default="fit")
    parser.add_argument("--mask-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--runtime-commit", required=True)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def _runtime_identity(runtime_dir: Path, runtime_commit: str) -> None:
    if not runtime_dir.is_absolute() or len(runtime_commit) != 40:
        raise ValueError("Task 9 runtime identity is invalid")
    observed_commit = subprocess.run(
        ["git", "-C", str(runtime_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    observed_status = subprocess.run(
        ["git", "-C", str(runtime_dir), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if observed_commit != runtime_commit or observed_status:
        raise ValueError("Task 9 execution requires the exact clean committed runtime")


def main() -> None:
    args = _parser().parse_args()
    for path in (
        args.config,
        args.run_config,
        args.index_manifest,
        args.fixture,
        args.mapping,
        args.output,
        args.runtime_dir,
    ):
        if not path.is_absolute():
            raise ValueError("Task 9 paths must be absolute")
    if args.mask_count <= 0 or args.mask_count > (64 if args.split == "fit" else 32):
        raise ValueError("Task 9 mask count is outside the canonical split")
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"Task 9 output already exists: {args.output}")
    if not args.validate_only and not args.output.parent.is_dir():
        raise ValueError("Task 9 output parent must already exist")
    _runtime_identity(args.runtime_dir, args.runtime_commit)
    if _sha256(args.fixture) != args.fixture_sha256:
        raise ValueError("Task 9 fixed-page fixture checksum mismatch")
    if _sha256(args.mapping) != args.mapping_sha256:
        raise ValueError("Task 9 mapping checksum mismatch")
    fixture = load_fixed_page_fixture(
        args.fixture,
        expected_sha256=args.fixture_sha256,
        validate_external_bytes=False,
    )
    fixed_question = fixture.question(args.qid)
    mapping = load_region_mapping(args.mapping, validate_raw_artifacts=True)
    if (
        mapping.geometry_count != args.expected_geometry_count
        or mapping.geometry_sha256 != args.expected_geometry_sha256
        or any(
            page.fixture_question_id != args.qid
            for artifact in mapping.artifacts
            for page in artifact.pages
        )
    ):
        raise ValueError("Task 9 mapping does not match the fixed question geometry")
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "validated-without-model-or-output",
                    "qid": fixed_question.qid,
                    "boundary": "B_input" if args.boundary == "input" else f"B_{args.boundary}",
                    "split": args.split,
                    "mask_count": args.mask_count,
                    "geometry_count": mapping.geometry_count,
                    "geometry_sha256": mapping.geometry_sha256,
                    "mapping_internal_sha256": mapping.sha256,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return

    args.output.mkdir()
    workload = build_workload(
        operation="evaluate",
        config=load_config(args.config),
        page_count=4,
        output=args.output / "bootstrap",
        mode="docprune",
        run_config=args.run_config,
        index_manifest=args.index_manifest,
        sample_ids=(args.qid,),
        ctp_policy=btp_qtp_no_ctp_policy(),
        fixed_page_fixture=args.fixture,
        fixed_page_fixture_sha256=args.fixture_sha256,
        execution_runtime_commit=args.runtime_commit,
    )
    samples = tuple(workload.samples)
    if len(samples) != 1 or samples[0].question_id != args.qid:
        raise ValueError("Task 9 workload selected the wrong fixed question")
    sample = samples[0]
    runner = workload.runner
    runner.warmup(sample)
    retrieval = runner.retriever.retrieve(sample.question, 4)
    if not isinstance(retrieval, RetrievalOutput):
        raise ValueError("Task 9 fixed-page backend did not return cached retrieval features")
    images = [
        runner.page_loader.load_page(page.doc_id, page.page_index) for page in retrieval.pages
    ]
    answerer = runner.answerer
    target = prepare_task7_likelihood_target_for_question(
        answerer.processor,
        page_count=4,
        question=sample.question,
        accepted_references=sample.answers,
    )
    prepared = [prepare_qwen_page(answerer.processor, image) for image in images]
    prompt, batch = _prepare_batch_with_prompt(
        answerer.processor,
        [prepared_raster_image(page) for page in prepared],
        sample.question,
    )
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    input_shape, input_sha256 = _input_ids_identity(torch.as_tensor(batch["input_ids"]))
    if target["assistant_prompt_sha256"] != prompt_sha256:
        raise ValueError("Task 9 target and scoring prompt differ")
    target_token_ids = tuple(tuple(row) for row in target["target_token_ids"])
    reference_set_sha256 = _canonical_sha256([list(row) for row in target_token_ids])
    regions = [
        {"source_id": source.source_id, "token_cost": len(source.token_ids)}
        for source in mapping.sources
    ] + [{"source_id": source_id, "token_cost": 0} for source_id in mapping.empty_region_source_ids]
    boundary_label = "B_input" if args.boundary == "input" else f"B_{args.boundary}"
    design = build_region_mask_design(
        regions,
        question_id=args.qid,
        forced_boundary=boundary_label,
        mapping_artifact_sha256=args.mapping_sha256,
        prompt_input_sha256=input_sha256,
        target_kind="max-accepted-reference-mean-loglikelihood",
        reference_set_token_ids_sha256=reference_set_sha256,
        generated_response_token_ids_sha256=None,
    )
    full_plan = build_regional_intervention_plan(
        mapping,
        design,
        mapping_artifact_sha256=args.mapping_sha256,
        split=args.split,
    )
    selected_plan = full_plan[: args.mask_count]
    manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "configured-task9-regional-smoke",
        "runtime_commit": args.runtime_commit,
        "qid": args.qid,
        "boundary": boundary_label,
        "split": args.split,
        "mask_count": args.mask_count,
        "selected_seeds": [row["seed"] for row in selected_plan],
        "fixed_page_fixture_path": str(args.fixture),
        "fixed_page_fixture_sha256": args.fixture_sha256,
        "config_sha256": _sha256(args.config),
        "run_config_sha256": _sha256(args.run_config),
        "index_manifest_sha256": _sha256(args.index_manifest),
        "mapping_path": str(args.mapping),
        "mapping_artifact_sha256": args.mapping_sha256,
        "mapping_internal_sha256": mapping.sha256,
        "geometry_count": mapping.geometry_count,
        "geometry_sha256": mapping.geometry_sha256,
        "design_sha256": design["design_sha256"],
        "attribution_identity_sha256": design["attribution_identity_sha256"],
        "assistant_prompt_sha256": prompt_sha256,
        "prefill_input_ids_shape": list(input_shape),
        "prefill_input_ids_sha256": input_sha256,
        "reference_set_token_ids_sha256": reference_set_sha256,
        "fixed_page_provenance": True,
        "global_index_loaded": False,
        "retrieval_search_run": False,
        "cached_retrieved_pages_reused": True,
        "independent_first_mask_parity_required": True,
    }
    manifest["run_manifest_sha256"] = _canonical_sha256(manifest)
    _atomic_json(args.output / "run-manifest.json", manifest)

    started = time.perf_counter()
    shared = answerer.score_forced_intervention_likelihoods(
        images,
        sample.question,
        retrieval_output=retrieval,
        forced_interventions=tuple(row["forced_intervention"] for row in selected_plan),
        teacher_forced_target_token_ids=target_token_ids,
    )
    first = selected_plan[0]
    independent_answerer = DocPruneQwenAnswerer(
        answerer.model,
        answerer.processor,
        page_config=answerer.page_config,
        reconstruction=answerer.reconstruction,
        qa_stage="full",
        max_new_tokens=1,
        forced_intervention=first["forced_intervention"],
        teacher_forced_target_token_ids=target_token_ids,
    )
    independent = independent_answerer.answer(
        images,
        sample.question,
        retrieval_output=retrieval,
    )
    first_values = shared.result.branches[0].teacher_forced_loglikelihoods
    if (
        shared.assistant_prompt_sha256 != prompt_sha256
        or shared.prefill_input_ids_shape != input_shape
        or shared.prefill_input_ids_sha256 != input_sha256
        or independent.assistant_prompt_sha256 != prompt_sha256
        or independent.prefill_input_ids_shape != input_shape
        or independent.prefill_input_ids_sha256 != input_sha256
        or independent.teacher_forced_loglikelihoods is None
        or any(
            abs(left - right) > 1e-6
            for left, right in zip(
                first_values,
                independent.teacher_forced_loglikelihoods,
                strict=True,
            )
        )
    ):
        raise ValueError("Task 9 shared-prefix first-mask parity failed")
    outcomes = [
        build_regional_target_outcome(
            design,
            row,
            mean_sequence_loglikelihoods=branch.teacher_forced_loglikelihoods,
        )
        for row, branch in zip(selected_plan, shared.result.branches, strict=True)
    ]
    elapsed = max(time.perf_counter() - started, 1e-12)
    branch_mean = sum(shared.result.branch_decoder_seconds) / len(
        shared.result.branch_decoder_seconds
    )
    result_payload: dict[str, object] = {
        "schema_version": 1,
        "status": "completed-task9-regional-smoke",
        "run_manifest_sha256": manifest["run_manifest_sha256"],
        "design": design,
        "selected_plan": [
            {key: value for key, value in row.items() if key != "forced_intervention"}
            for row in selected_plan
        ],
        "outcomes": outcomes,
        "forced_interventions": [
            branch.forced_intervention.to_dict() for branch in shared.result.branches
        ],
        "per_reference_mean_loglikelihoods": [
            list(branch.teacher_forced_loglikelihoods) for branch in shared.result.branches
        ],
        "independent_first_mask_mean_loglikelihoods": list(
            independent.teacher_forced_loglikelihoods
        ),
        "shared_prefix_parity_max_abs_error": max(
            abs(left - right)
            for left, right in zip(
                first_values,
                independent.teacher_forced_loglikelihoods,
                strict=True,
            )
        ),
        "original_visual_tokens": shared.result.original_visual_tokens,
        "post_btp_visual_tokens": shared.result.post_btp_visual_tokens,
        "post_qtp_visual_tokens": shared.result.post_qtp_visual_tokens,
        "checkpoint_cache_lengths": list(shared.result.checkpoint_cache_lengths),
        "encoder_seconds": shared.result.encoder_seconds,
        "prefix_decoder_seconds": shared.result.prefix_decoder_seconds,
        "branch_decoder_seconds": list(shared.result.branch_decoder_seconds),
        "branch_decoder_seconds_mean": branch_mean,
        "projected_96_branch_decoder_seconds": branch_mean * 96,
        "measured_scoring_and_parity_seconds": elapsed,
        "peak_allocated_gpu_bytes": shared.peak_allocated_gpu_bytes,
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    result_payload["result_sha256"] = _canonical_sha256(result_payload)
    _atomic_json(args.output / "result.json", result_payload)
    completion = {
        "schema_version": 1,
        "status": "complete",
        "run_manifest_sha256": manifest["run_manifest_sha256"],
        "result_sha256": result_payload["result_sha256"],
        "run_manifest_file_sha256": _sha256(args.output / "run-manifest.json"),
        "result_file_sha256": _sha256(args.output / "result.json"),
    }
    completion["completion_sha256"] = _canonical_sha256(completion)
    _atomic_json(args.output / "completion-manifest.json", completion)
    print(json.dumps(completion, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
