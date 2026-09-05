#!/usr/bin/env python3
"""Generate answers for one Task 9 preliminary question's selected regional arms."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import torch

from docprune.answerers import DocPruneQwenAnswerer, prepare_task7_likelihood_target_for_question
from docprune.config import load_config
from docprune.ctp_policy import aggregate_native_threshold_policy, btp_qtp_no_ctp_policy
from docprune.evaluation import list_em, list_f1
from docprune.m3docrag import RetrievalOutput
from docprune.m3docvqa_factory import build_workload
from docprune.segmentation import load_region_mapping
from docprune.task9_attribution import (
    _canonical_sha256,
    analyze_task9_preliminary_question,
    build_task9_mask_count_generation_plan,
    build_task9_selected_arm_plan,
)


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _publish(path: Path, value: dict[str, object]) -> None:
    content = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    try:
        os.write(descriptor, content)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _runtime_identity(runtime_dir: Path, runtime_commit: str) -> None:
    head = subprocess.run(
        ["git", "-C", str(runtime_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(runtime_dir), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if head != runtime_commit or status:
        raise ValueError("Task 9 selected-arm generation requires the exact clean runtime")


def _accepted_answers(cohort_path: Path, qid: str) -> tuple[str, ...]:
    cohort = _load(cohort_path)
    records = cohort.get("selected_records")
    matches = (
        [row for row in records if isinstance(row, dict) and row.get("question_id") == qid]
        if isinstance(records, list)
        else []
    )
    if len(matches) != 1 or not isinstance(matches[0].get("answers"), list):
        raise ValueError("Task 9 selected-arm cohort identity is invalid")
    answers = tuple(matches[0]["answers"])
    if not answers or any(not isinstance(answer, str) or not answer for answer in answers):
        raise ValueError("Task 9 selected-arm accepted answers are invalid")
    return answers


def _answer_record(
    answerer: DocPruneQwenAnswerer,
    *,
    images: list[object],
    question: str,
    retrieval: RetrievalOutput,
    accepted_answers: tuple[str, ...],
    reference_count: int,
    retained_fraction: float | None,
    arm: str,
) -> dict[str, object]:
    output = answerer.answer(images, question, retrieval_output=retrieval)
    likelihoods = output.teacher_forced_loglikelihoods
    if likelihoods is None or len(likelihoods) != reference_count + 1:
        raise ValueError("Task 9 selected-arm likelihood output is invalid")
    result: dict[str, object] = {
        "arm": arm,
        "retained_fraction": retained_fraction,
        "answer": output.answer,
        "normalized_token_f1": list_f1(output.answer, accepted_answers),
        "exact_match": bool(list_em(output.answer, accepted_answers)),
        "gold_mean_loglikelihood": max(likelihoods[:reference_count]),
        "alternative_mean_loglikelihood": likelihoods[reference_count],
        "qa_seconds": output.qa_seconds,
        "peak_allocated_gpu_bytes": output.peak_allocated_gpu_bytes,
        "encoder_seconds": output.encoder_seconds,
        "decoder_seconds": output.decoder_seconds,
        "trace": {
            "original_visual_tokens": output.trace.original_visual_tokens,
            "post_btp_visual_tokens": output.trace.post_btp_visual_tokens,
            "post_qtp_visual_tokens": output.trace.post_qtp_visual_tokens,
            "post_ctp_visual_tokens": output.trace.post_ctp_visual_tokens,
            "ctp_layer": output.trace.ctp_layer,
        },
        "forced_intervention": (
            None if output.forced_intervention is None else output.forced_intervention.to_dict()
        ),
        "policy_selection": (
            None if output.policy_selection is None else output.policy_selection.to_dict()
        ),
    }
    result["result_sha256"] = _canonical_sha256(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--runtime-commit", required=True)
    parser.add_argument("--mask-count-analysis", type=Path)
    parser.add_argument("--mask-count-analysis-sha256")
    parser.add_argument("--mask-count", type=int)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if any(
        not path.is_absolute()
        for path in (args.raw_root, args.analysis, args.output, args.runtime_dir)
    ) or (args.mask_count_analysis is not None and not args.mask_count_analysis.is_absolute()):
        raise ValueError("Task 9 selected-arm paths must be absolute")
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("Task 9 selected-arm output must be fresh")

    artifact = _load(args.analysis)
    unsigned = dict(artifact)
    observed_sha = unsigned.pop("artifact_sha256", None)
    if observed_sha != _canonical_sha256(unsigned):
        raise ValueError("Task 9 preliminary analysis digest is invalid")
    completion = _load(args.raw_root / "completion-manifest.json")
    manifest = _load(args.raw_root / "run-manifest.json")
    raw = _load(args.raw_root / "raw-result.json")
    analysis = artifact.get("analysis")
    if (
        artifact.get("completion_sha256") != completion.get("completion_sha256")
        or not isinstance(manifest.get("runtime_commit"), str)
        or len(manifest["runtime_commit"]) != 40
        or not isinstance(analysis, dict)
        or analysis.get("question_id") != manifest.get("qid")
        or analysis.get("forced_boundary") != manifest.get("boundary")
    ):
        raise ValueError("Task 9 preliminary analysis is not bound to the raw artifact")
    mapping = load_region_mapping(Path(manifest["mapping_path"]), validate_raw_artifacts=True)
    mask_mode = any(
        value is not None
        for value in (
            args.mask_count_analysis,
            args.mask_count_analysis_sha256,
            args.mask_count,
        )
    )
    if mask_mode and any(
        value is None
        for value in (
            args.mask_count_analysis,
            args.mask_count_analysis_sha256,
            args.mask_count,
        )
    ):
        raise ValueError("mask-count generation arguments must be provided together")
    generation_plan = None
    if mask_mode:
        aggregate = _load(args.mask_count_analysis)
        if aggregate.get("analysis_sha256") != args.mask_count_analysis_sha256:
            raise ValueError("mask-count aggregate identity changed")
        generation_plan = build_task9_mask_count_generation_plan(
            aggregate,
            question_id=manifest["qid"],
            mask_count=args.mask_count,
            visual_population=mapping.geometry_count,
        )
        plan = (
            ()
            if generation_plan["new_generation_count"] == 0
            else build_task9_selected_arm_plan(
                mapping,
                generation_plan["selections"],
                boundary=manifest["boundary"],
            )
        )
    else:
        plan = build_task9_selected_arm_plan(
            mapping,
            analysis["selections"],
            boundary=manifest["boundary"],
        )
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "validated-task9-selected-arms-without-model",
                    "qid": manifest["qid"],
                    "selected_arm_count": len(plan),
                    "budgets": sorted({row["retained_fraction"] for row in plan}),
                    "mask_count": args.mask_count if mask_mode else None,
                    "canonical_reuse_repeats": (
                        generation_plan["canonical_reuse_repeats"] if mask_mode else []
                    ),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return

    _runtime_identity(args.runtime_dir, args.runtime_commit)
    if mask_mode and not plan:
        result = {
            "schema_version": "docprune-task9-mask-count-selected-arms-v1",
            "status": "completed-task9-mask-count-selected-arms-by-canonical-reuse",
            "qid": manifest["qid"],
            "runtime_commit": args.runtime_commit,
            "mask_count_generation_plan": generation_plan,
            "cuda_device_name": None,
            "selected_results": [],
        }
        result["selected_arms_sha256"] = _canonical_sha256(result)
        _publish(args.output, result)
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return
    work = args.output.parent / f".{args.output.name}.work-{os.getpid()}"
    work.mkdir()
    workload = build_workload(
        operation="evaluate",
        config=load_config(Path(manifest["config_path"])),
        page_count=4,
        output=work / "bootstrap",
        mode="docprune",
        run_config=Path(manifest["run_config_path"]),
        index_manifest=Path(manifest["index_manifest_path"]),
        sample_ids=(manifest["qid"],),
        ctp_policy=btp_qtp_no_ctp_policy(),
        fixed_page_fixture=Path(manifest["fixed_page_fixture_path"]),
        fixed_page_fixture_sha256=manifest["fixed_page_fixture_sha256"],
        execution_runtime_commit=args.runtime_commit,
    )
    sample = tuple(workload.samples)[0]
    runner = workload.runner
    runner.warmup(sample)
    retrieval = runner.retriever.retrieve(sample.question, 4)
    if not isinstance(retrieval, RetrievalOutput):
        raise ValueError("Task 9 selected-arm generation requires fixed retrieval")
    images = [
        runner.page_loader.load_page(page.doc_id, page.page_index) for page in retrieval.pages
    ]
    base = runner.answerer
    accepted = _accepted_answers(Path(manifest["preliminary_cohort_path"]), manifest["qid"])
    target = prepare_task7_likelihood_target_for_question(
        base.processor,
        page_count=4,
        question=sample.question,
        accepted_references=accepted,
    )
    references = tuple(tuple(row) for row in target["target_token_ids"])
    alternative = tuple(raw["generated_response_token_ids"])
    teacher_targets = (*references, alternative)

    def configured(forced: object | None) -> DocPruneQwenAnswerer:
        kwargs: dict[str, object] = {
            "page_config": base.page_config,
            "reconstruction": base.reconstruction,
            "qa_stage": "full",
            "max_new_tokens": base.max_new_tokens,
            "teacher_forced_target_token_ids": teacher_targets,
            "frozen_post_qtp_geometry": mapping.geometry,
        }
        if forced is None:
            kwargs["ctp_policy"] = btp_qtp_no_ctp_policy()
        else:
            kwargs["forced_intervention"] = forced
        return DocPruneQwenAnswerer(base.model, base.processor, **kwargs)

    def configured_native() -> DocPruneQwenAnswerer:
        return DocPruneQwenAnswerer(
            base.model,
            base.processor,
            page_config=base.page_config,
            reconstruction=base.reconstruction,
            qa_stage="full",
            max_new_tokens=base.max_new_tokens,
            teacher_forced_target_token_ids=teacher_targets,
            frozen_post_qtp_geometry=mapping.geometry,
            ctp_policy=aggregate_native_threshold_policy(),
        )

    if mask_mode:
        selected_results = [
            _answer_record(
                configured(row["forced_intervention"]),
                images=images,
                question=sample.question,
                retrieval=retrieval,
                accepted_answers=accepted,
                reference_count=len(references),
                retained_fraction=row["retained_fraction"],
                arm=row["arm"],
            )
            for row in plan
        ]
        result = {
            "schema_version": "docprune-task9-mask-count-selected-arms-v1",
            "status": "completed-task9-mask-count-selected-arms",
            "qid": manifest["qid"],
            "raw_runtime_commit": manifest["runtime_commit"],
            "runtime_commit": args.runtime_commit,
            "mask_count_generation_plan": generation_plan,
            "cuda_device_name": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None,
            "selected_results": selected_results,
        }
        result["selected_arms_sha256"] = _canonical_sha256(result)
        _publish(args.output, result)
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return

    unpruned = _answer_record(
        configured(None),
        images=images,
        question=sample.question,
        retrieval=retrieval,
        accepted_answers=accepted,
        reference_count=len(references),
        retained_fraction=None,
        arm="unpruned",
    )
    native = _answer_record(
        configured_native(),
        images=images,
        question=sample.question,
        retrieval=retrieval,
        accepted_answers=accepted,
        reference_count=len(references),
        retained_fraction=raw["native_docprune_selection"]["achieved_budget"]
        / raw["native_docprune_selection"]["visual_population"],
        arm="native_docprune",
    )
    expected_native = raw["native_docprune_selection"]
    observed_native = native["policy_selection"]
    matched_fields = (
        "boundary",
        "native_layer",
        "visual_population",
        "achieved_budget",
        "retained_compact_visual_ids",
    )
    if not isinstance(observed_native, dict) or any(
        observed_native.get(field) != expected_native.get(field) for field in matched_fields
    ):
        raise ValueError("Task 9 native DocPrune selection did not reproduce the frozen prepass")
    selected_results = [
        _answer_record(
            configured(row["forced_intervention"]),
            images=images,
            question=sample.question,
            retrieval=retrieval,
            accepted_answers=accepted,
            reference_count=len(references),
            retained_fraction=row["retained_fraction"],
            arm=row["arm"],
        )
        for row in plan
    ]
    fraction = analysis["selections"]["budgets"][0]["retained_fraction"]
    arm_results = {row["arm"]: row for row in selected_results}
    arm_results["unpruned"] = unpruned
    arm_results["native_docprune"] = native
    question_analyses = [
        analyze_task9_preliminary_question(
            question_id=manifest["qid"],
            retained_fraction=fraction,
            arm_results=arm_results,
        )
    ]
    result: dict[str, object] = {
        "schema_version": "docprune-task9-preliminary-dynamic-selected-arms-v1",
        "status": "completed-task9-preliminary-selected-arms",
        "qid": manifest["qid"],
        "raw_runtime_commit": manifest["runtime_commit"],
        "runtime_commit": args.runtime_commit,
        "analysis_artifact_sha256": observed_sha,
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "unpruned": unpruned,
        "native_docprune": native,
        "selected_results": selected_results,
        "question_analyses": question_analyses,
    }
    result["selected_arms_sha256"] = _canonical_sha256(result)
    _publish(args.output, result)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
