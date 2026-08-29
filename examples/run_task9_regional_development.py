#!/usr/bin/env python3
"""Run the first one-question, 96-mask Task 9 regional development artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import torch

from docprune.answerers import (
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
    build_regional_development_plan,
    build_regional_development_targets,
    build_regional_intervention_plan,
)
from docprune.task9_live import (
    publish_task9_regional_development,
    score_task9_regional_development_once,
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
    content = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


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
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"Task 9 output already exists: {args.output}")
    if not args.output.parent.is_dir():
        raise ValueError("Task 9 output parent must already exist")
    _runtime_identity(args.runtime_dir, args.runtime_commit)
    input_hashes = {
        "config_sha256": _sha256(args.config),
        "run_config_sha256": _sha256(args.run_config),
        "index_manifest_sha256": _sha256(args.index_manifest),
        "fixed_page_fixture_sha256": _sha256(args.fixture),
        "mapping_artifact_sha256": _sha256(args.mapping),
    }
    if input_hashes["fixed_page_fixture_sha256"] != args.fixture_sha256:
        raise ValueError("Task 9 fixed-page fixture checksum mismatch")
    if input_hashes["mapping_artifact_sha256"] != args.mapping_sha256:
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
    boundary_label = "B_input" if args.boundary == "input" else f"B_{args.boundary}"
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "validated-task9-development-without-model-or-output",
                    "qid": fixed_question.qid,
                    "boundary": boundary_label,
                    "mask_count": 96,
                    "seed_order": list(range(96)),
                    "geometry_count": mapping.geometry_count,
                    "geometry_sha256": mapping.geometry_sha256,
                    "mapping_internal_sha256": mapping.sha256,
                    **input_hashes,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return

    work = args.output.parent / f".{args.output.name}.work-{os.getpid()}"
    work.mkdir()
    workload = build_workload(
        operation="evaluate",
        config=load_config(args.config),
        page_count=4,
        output=work / "bootstrap",
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
        raise ValueError("Task 9 did not receive fixed cached retrieval features")
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
    reference_ids = tuple(tuple(row) for row in target["target_token_ids"])
    reference_hash = _canonical_sha256([list(row) for row in reference_ids])
    regions = [
        {"source_id": source.source_id, "token_cost": len(source.token_ids)}
        for source in mapping.sources
    ] + [{"source_id": source_id, "token_cost": 0} for source_id in mapping.empty_region_source_ids]
    primary_design = build_region_mask_design(
        regions,
        question_id=args.qid,
        forced_boundary=boundary_label,
        mapping_artifact_sha256=args.mapping_sha256,
        prompt_input_sha256=input_sha256,
        target_kind="max-accepted-reference-mean-loglikelihood",
        reference_set_token_ids_sha256=reference_hash,
        generated_response_token_ids_sha256=None,
    )
    primary_plan = (
        *build_regional_intervention_plan(
            mapping,
            primary_design,
            mapping_artifact_sha256=args.mapping_sha256,
            split="fit",
        ),
        *build_regional_intervention_plan(
            mapping,
            primary_design,
            mapping_artifact_sha256=args.mapping_sha256,
            split="holdout",
        ),
    )
    shared = score_task9_regional_development_once(
        answerer,
        images=images,
        question=sample.question,
        retrieval_output=retrieval,
        forced_interventions=tuple(row["forced_intervention"] for row in primary_plan),
        reference_target_token_ids=reference_ids,
    )
    generated_ids = shared.unpruned_generated_response_token_ids
    terminal_eos = shared.unpruned_terminal_eos_token_id
    generation_trace = shared.unpruned_generation_trace
    if (
        not generated_ids
        or terminal_eos not in {151645, 151643}
        or generation_trace is None
        or generation_trace.ctp_layer is not None
        or generation_trace.post_ctp_visual_tokens != generation_trace.post_qtp_visual_tokens
    ):
        raise ValueError("Task 9 generated-response identity is incomplete")
    generated_hash = _canonical_sha256(list(generated_ids))
    secondary_design = build_region_mask_design(
        regions,
        question_id=args.qid,
        forced_boundary=boundary_label,
        mapping_artifact_sha256=args.mapping_sha256,
        prompt_input_sha256=input_sha256,
        target_kind="unpruned-generated-response-mean-loglikelihood",
        reference_set_token_ids_sha256=None,
        generated_response_token_ids_sha256=generated_hash,
    )
    development_plan = build_regional_development_plan(
        mapping,
        primary_design,
        secondary_design,
        mapping_artifact_sha256=args.mapping_sha256,
    )
    raw_likelihoods = [
        list(branch.teacher_forced_loglikelihoods) for branch in shared.result.branches
    ]
    targets = build_regional_development_targets(
        primary_design,
        secondary_design,
        development_plan,
        mean_sequence_loglikelihoods=raw_likelihoods,
        reference_sequence_count=len(reference_ids),
    )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "configured-task9-regional-development",
        "runtime_commit": args.runtime_commit,
        "qid": args.qid,
        "boundary": boundary_label,
        "mask_count": 96,
        "seed_order": list(range(96)),
        "config_path": str(args.config),
        "fixed_page_fixture_path": str(args.fixture),
        "mapping_path": str(args.mapping),
        "index_manifest_path": str(args.index_manifest),
        "run_config_path": str(args.run_config),
        **input_hashes,
        "mapping_internal_sha256": mapping.sha256,
        "geometry_count": mapping.geometry_count,
        "geometry_sha256": mapping.geometry_sha256,
        "assistant_prompt_sha256": prompt_sha256,
        "prefill_input_ids_shape": list(input_shape),
        "prefill_input_ids_sha256": input_sha256,
        "reference_set_token_ids_sha256": reference_hash,
        "generated_response_token_ids_sha256": generated_hash,
        "primary_attribution_identity_sha256": primary_design["attribution_identity_sha256"],
        "secondary_attribution_identity_sha256": secondary_design["attribution_identity_sha256"],
        "primary_target_dataset_sha256": targets["primary"]["target_dataset_sha256"],
        "secondary_target_dataset_sha256": targets["secondary"]["target_dataset_sha256"],
        "fixed_page_provenance": True,
        "cached_retrieved_pages_reused": True,
        "global_index_loaded": False,
        "retrieval_search_run": False,
    }
    manifest["run_manifest_sha256"] = _canonical_sha256(manifest)
    raw_result: dict[str, object] = {
        "schema_version": 1,
        "status": "completed-task9-regional-development",
        "run_manifest_sha256": manifest["run_manifest_sha256"],
        "reference_sequence_count": len(reference_ids),
        "raw_mean_sequence_loglikelihoods": raw_likelihoods,
        "development_plan": [
            {key: value for key, value in row.items() if key != "forced_intervention"}
            for row in development_plan
        ],
        "forced_interventions": [
            branch.forced_intervention.to_dict() for branch in shared.result.branches
        ],
        "generated_response_token_ids": list(generated_ids),
        "generated_response_token_ids_sha256": generated_hash,
        "terminal_eos_token_id": terminal_eos,
        "unpruned_generation_trace": {
            "original_visual_tokens": generation_trace.original_visual_tokens,
            "post_btp_visual_tokens": generation_trace.post_btp_visual_tokens,
            "post_qtp_visual_tokens": generation_trace.post_qtp_visual_tokens,
            "post_ctp_visual_tokens": generation_trace.post_ctp_visual_tokens,
            "ctp_layer": generation_trace.ctp_layer,
        },
        "original_visual_tokens": shared.result.original_visual_tokens,
        "post_btp_visual_tokens": shared.result.post_btp_visual_tokens,
        "post_qtp_visual_tokens": shared.result.post_qtp_visual_tokens,
        "checkpoint_cache_lengths": list(shared.result.checkpoint_cache_lengths),
        "encoder_seconds": shared.result.encoder_seconds,
        "prefix_decoder_seconds": shared.result.prefix_decoder_seconds,
        "branch_decoder_seconds": list(shared.result.branch_decoder_seconds),
        "unpruned_generation_encoder_seconds": shared.unpruned_generation_encoder_seconds,
        "unpruned_generation_decoder_seconds": shared.unpruned_generation_decoder_seconds,
        "peak_allocated_gpu_bytes": shared.peak_allocated_gpu_bytes,
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    raw_result["raw_result_sha256"] = _canonical_sha256(raw_result)
    completion = publish_task9_regional_development(
        args.output,
        manifest,
        raw_result,
        targets["primary"],
        targets["secondary"],
    )
    print(json.dumps(completion, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
