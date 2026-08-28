#!/usr/bin/env python3
"""Run one sealed Task 6 QID shard while loading each model only once."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

from docprune.answerers import DocPruneQwenAnswerer
from docprune.cli import _bind_task6_result_evidence, _manifest_digest
from docprune.config import load_config
from docprune.m3docrag import DocPruneM3DocRAG, SampleInput
from docprune.m3docvqa_factory import (
    _load_index_manifest,
    _resolve_run_config,
    _validate_result_record,
    _validate_run_identity,
    build_workload,
)
from docprune.metrics import append_result_jsonl
from docprune.task6_runtime import (
    FixedPageQuestion,
    Task6ResultIdentity,
    load_fixed_page_fixture,
    task6_policy_matrix,
    validate_task6_result_record,
)


def _sha256(path: Path) -> str:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError(f"authenticated input must be an absolute regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, expected_sha256: str, label: str) -> dict[str, object]:
    if _sha256(path) != expected_sha256:
        raise ValueError(f"{label} checksum mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
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


def _bind_task7_likelihood_capture(
    record: dict[str, object], sample_result: object, target: dict[str, object]
) -> None:
    """Bind only an internally derived target to its exact processor batch."""

    prompt_sha256 = getattr(sample_result, "assistant_prompt_sha256", None)
    input_shape = getattr(sample_result, "prefill_input_ids_shape", None)
    input_sha256 = getattr(sample_result, "prefill_input_ids_sha256", None)
    if prompt_sha256 is None or input_shape is None or input_sha256 is None:
        raise ValueError("Task 7 answerer did not return exact prefill identity")
    if target.get("assistant_prompt_sha256") != prompt_sha256:
        raise ValueError("Task 7 target prompt differs from the exact processor prompt")
    from docprune.task7_runtime import bind_task7_prefill_identity

    bind_task7_prefill_identity(
        record,
        assistant_prompt_sha256=prompt_sha256,
        prefill_input_ids_shape=input_shape,
        prefill_input_ids_sha256=input_sha256,
    )


def _expected_cells(
    gate: dict[str, object],
    kind: str,
    *,
    native_boundary_cell: object | None = None,
) -> list[dict[str, object]]:
    if kind == "visual-state-native-boundary":
        if native_boundary_cell is None or not hasattr(native_boundary_cell, "to_dict"):
            raise ValueError("native-boundary diagnostic requires one authenticated cell")
        return [{"cell": 0, **native_boundary_cell.to_dict()}]
    if kind == "visual-state-fixed-grid":
        from docprune.task7_runtime import task7_intervention_matrix

        return [
            {"cell": index, **cell.to_dict()}
            for index, cell in enumerate(task7_intervention_matrix())
        ]
    key = f"{kind.replace('-', '_')}_cells"
    cells = gate.get(key)
    if not isinstance(cells, list):
        raise ValueError("gate manifest policy matrix is invalid")
    frozen = task6_policy_matrix("native" if kind == "smoke" else kind)
    if kind == "smoke":
        frozen = tuple(frozen[index] for index in (0, 1, 2, 3, 4, 5, 15, 25, 35))
    if len(cells) != len(frozen):
        raise ValueError("gate manifest policy matrix count drifted")
    for index, (persisted, expected) in enumerate(zip(cells, frozen, strict=True)):
        if not isinstance(persisted, dict):
            raise ValueError("gate manifest policy cell is invalid")
        expected_version = (
            f"task6-{'native' if kind == 'smoke' else kind}-v1"
            if expected.policy.family in {"random-top-m", "coverage-top-m"}
            else None
        )
        if persisted != {
            "cell": index,
            "policy": expected.policy.to_dict(),
            "experiment_version": expected_version,
            "repetition": expected.repetition,
        }:
            raise ValueError("gate manifest policy cell identity drifted")
    return cells


def _existing_prefix(
    path: Path,
    *,
    cells: list[dict[str, object]],
    qid: str,
    fixture_sha256: str,
    kind: str,
    sample: SampleInput | None = None,
    fixture_question: FixedPageQuestion | None = None,
    native_boundary_cell: object | None = None,
    native_boundary_manifest_sha256: str | None = None,
) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or count >= len(cells):
                raise ValueError("Task 6 resume results are not a valid cell prefix")
            cell = cells[count]
            if kind == "visual-state-native-boundary":
                from docprune.task7_native_boundary import (
                    validate_task7_native_boundary_result_record,
                )
                from docprune.task7_runtime import validate_task7_source_identity

                if native_boundary_cell is None or count != 0:
                    raise ValueError("Task 7 native resume has an invalid cell")
                expected_policy = (
                    None
                    if native_boundary_cell.ctp_policy is None
                    else native_boundary_cell.ctp_policy.to_dict()
                )
                _validate_result_record(
                    record,
                    line_number=line_number,
                    expected_page_count=4,
                    production=True,
                    expected_policy=expected_policy,
                    forbid_policy=expected_policy is None,
                    allowed_extra_fields={
                        "matrix_cell",
                        "matrix_kind",
                        "intervention_name",
                        "analysis_family",
                        "boundary_source",
                        "native_boundary_manifest_sha256",
                        "native_crossing",
                        "native_layer",
                    },
                    allow_zero_post_ctp=True,
                )
                if sample is None or fixture_question is None:
                    raise ValueError("Task 7 native resume requires sealed source identity")
                validate_task7_source_identity(record, sample, fixture_question)
                validate_task7_native_boundary_result_record(
                    record,
                    native_boundary_cell,
                    fixture_sha256=fixture_sha256,
                    native_boundary_manifest_sha256=native_boundary_manifest_sha256,
                )
                count += 1
                continue
            if kind == "visual-state-fixed-grid":
                from docprune.task7_runtime import (
                    task7_intervention_matrix,
                    validate_task7_result_record,
                    validate_task7_source_identity,
                )

                intervention = task7_intervention_matrix()[count]
                expected_policy = (
                    None if intervention.ctp_policy is None else intervention.ctp_policy.to_dict()
                )
                if (
                    record.get("question_id") != qid
                    or record.get("matrix_cell") != count
                    or record.get("matrix_kind") != "visual-state-fixed-grid"
                    or record.get("intervention_name") != intervention.name
                ):
                    raise ValueError("Task 7 resume results are not an exact cell prefix")
                _validate_result_record(
                    record,
                    line_number=line_number,
                    expected_page_count=4,
                    production=True,
                    expected_policy=expected_policy,
                    forbid_policy=expected_policy is None,
                    allowed_extra_fields={"matrix_cell", "matrix_kind", "intervention_name"}
                    | (
                        {
                            "assistant_prompt_sha256",
                            "prefill_input_ids_shape",
                            "prefill_input_ids_sha256",
                        }
                        if count < 2
                        else set()
                    ),
                    allow_zero_post_ctp=True,
                )
                if sample is None or fixture_question is None:
                    raise ValueError("Task 7 resume requires sealed source identity")
                validate_task7_source_identity(record, sample, fixture_question)
                validate_task7_result_record(
                    record,
                    intervention,
                    fixture_sha256=fixture_sha256,
                )
                count += 1
                continue
            policy = cell["policy"]
            if (
                record.get("question_id") != qid
                or record.get("matrix_cell") != count
                or record.get("matrix_kind") != kind
                or record.get("experiment_version") != cell["experiment_version"]
                or record.get("repetition") != cell["repetition"]
                or not isinstance(policy, dict)
            ):
                raise ValueError("Task 6 resume results are not an exact cell prefix")
            _validate_result_record(
                record,
                line_number=line_number,
                expected_page_count=4,
                production=True,
                expected_policy=policy,
                allowed_extra_fields={
                    "matrix_cell",
                    "matrix_kind",
                    "experiment_version",
                    "repetition",
                },
            )
            validate_task6_result_record(
                record,
                Task6ResultIdentity(
                    fixture_sha256,
                    str(policy["name"]),
                    cell["experiment_version"],
                    cell["repetition"],
                ),
            )
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--index-manifest", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--fixture-sha256", required=True)
    parser.add_argument("--gate-manifest", type=Path, required=True)
    parser.add_argument("--gate-manifest-sha256", required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--runtime-commit", required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument(
        "--kind",
        choices=(
            "smoke",
            "native",
            "native-extension",
            "holdout-primary",
            "fixed",
            "visual-state-fixed-grid",
            "visual-state-native-boundary",
        ),
        required=True,
    )
    parser.add_argument("--native-boundary-manifest", type=Path)
    parser.add_argument("--native-boundary-manifest-sha256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    for name in (
        "config",
        "run_config",
        "index_manifest",
        "fixture",
        "gate_manifest",
        "runtime_dir",
        "output",
    ):
        value = getattr(args, name)
        if not value.is_absolute():
            raise ValueError(f"--{name.replace('_', '-')} must be absolute")
    if args.kind == "visual-state-native-boundary":
        if (
            args.native_boundary_manifest is None
            or not args.native_boundary_manifest.is_absolute()
            or args.native_boundary_manifest_sha256 is None
        ):
            raise ValueError("native-boundary diagnostic requires an absolute sealed manifest")
    elif (
        args.native_boundary_manifest is not None
        or args.native_boundary_manifest_sha256 is not None
    ):
        raise ValueError("native-boundary manifest is valid only for its separate diagnostic")
    if len(args.runtime_commit) != 40 or any(
        character not in "0123456789abcdef" for character in args.runtime_commit
    ):
        raise ValueError("--runtime-commit must be a lowercase 40-character commit hash")
    observed_commit = subprocess.run(
        ["git", "-C", str(args.runtime_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    observed_status = subprocess.run(
        ["git", "-C", str(args.runtime_dir), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if observed_commit != args.runtime_commit or observed_status:
        raise ValueError("Task 6 execution runtime must be the exact clean committed checkout")

    gate = _load_json(args.gate_manifest, args.gate_manifest_sha256, "gate manifest")
    expected_gate_status = (
        "sealed-holdout-gate" if args.kind == "holdout-primary" else "sealed-development-gate"
    )
    if (
        gate.get("status") != expected_gate_status
        or gate.get("fixture_path") != str(args.fixture)
        or gate.get("fixture_sha256") != args.fixture_sha256
        or gate.get("fixed_page_provenance") is not True
        or gate.get("global_index_loaded") is not False
    ):
        raise ValueError("gate manifest fixed-page identity is invalid")
    fixture = load_fixed_page_fixture(
        args.fixture,
        expected_sha256=args.fixture_sha256,
        validate_external_bytes=False,
    )
    shards = gate.get("qid_shards")
    if not isinstance(shards, list) or not 0 <= args.shard < len(shards):
        raise ValueError("Task 6 shard is outside the sealed gate")
    shard = shards[args.shard]
    if not isinstance(shard, dict) or shard.get("shard") != args.shard:
        raise ValueError("Task 6 shard map is invalid")
    qid = shard.get("qid")
    if not isinstance(qid, str):
        raise ValueError("Task 6 shard QID is invalid")
    fixture.question(qid)
    native_boundary_manifest = None
    native_boundary_cell = None
    if args.kind == "visual-state-native-boundary":
        from docprune.task7_native_boundary import (
            load_task7_native_boundary_manifest,
            task7_native_boundary_cell,
        )

        native_boundary_manifest = load_task7_native_boundary_manifest(
            args.native_boundary_manifest,
            expected_sha256=args.native_boundary_manifest_sha256,
        )
        if native_boundary_manifest.fixture_sha256 != args.fixture_sha256:
            raise ValueError("native-boundary manifest fixture identity drifted")
        native_boundary_cell = task7_native_boundary_cell(native_boundary_manifest.question(qid))
    cells = _expected_cells(
        gate,
        args.kind,
        native_boundary_cell=native_boundary_cell,
    )

    config = load_config(args.config)
    resolved_run = _resolve_run_config(args.run_config, mode="docprune", page_count=4)
    identity = _validate_run_identity(resolved_run, mode="docprune", page_count=4)
    feature_manifest = _load_index_manifest(args.index_manifest, validate_files=False)
    if (
        feature_manifest.mode != "docprune"
        or feature_manifest.page_count != 4
        or str(fixture.feature_manifest_path) != str(args.index_manifest)
        or fixture.feature_manifest_sha256 != _sha256(args.index_manifest)
        or fixture.completion_ledger_path != feature_manifest.completion_ledger_path
        or fixture.completion_ledger_sha256 != feature_manifest.completion_ledger_sha256
    ):
        raise ValueError("Task 6 selected feature-manifest provenance is invalid")

    run_manifest = {
        "schema_version": 1,
        "status": (
            "configured-task7-fixed-grid"
            if args.kind == "visual-state-fixed-grid"
            else (
                "configured-task7-native-boundary"
                if args.kind == "visual-state-native-boundary"
                else "configured-task6-matrix"
            )
        ),
        "output": str(args.output),
        "matrix_kind": args.kind,
        "shard": args.shard,
        "qid": qid,
        "cell_count": len(cells),
        "fixture_path": str(args.fixture),
        "fixture_sha256": args.fixture_sha256,
        "gate_manifest_path": str(args.gate_manifest),
        "gate_manifest_sha256": args.gate_manifest_sha256,
        "feature_manifest_path": str(args.index_manifest),
        "feature_manifest_sha256": fixture.feature_manifest_sha256,
        "feature_build_source_order_sha256": feature_manifest.source_order_sha256,
        "fixed_page_provenance": True,
        "global_index_loaded": False,
        "feature_build_runtime_commit": identity["runtime_commit"],
        "runtime_commit": args.runtime_commit,
        "m3docrag_commit": identity["m3docrag_commit"],
        "resources": identity["resources"],
        "generation": identity["generation"],
        "cells": cells,
    }
    if args.kind == "visual-state-fixed-grid":
        run_manifest["task7_likelihood_output"] = str(args.output / "task7-likelihood.jsonl")
        run_manifest["task7_likelihood_arms"] = [
            "btp-qtp-no-ctp",
            "all-visual-drop-B_input",
        ]
    if args.kind == "visual-state-native-boundary":
        run_manifest.update(
            {
                "native_boundary_manifest_path": str(args.native_boundary_manifest),
                "native_boundary_manifest_sha256": args.native_boundary_manifest_sha256,
            }
        )
    run_manifest["run_manifest_sha256"] = _manifest_digest(run_manifest)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "validated-without-model-or-output",
                    "qid": qid,
                    "matrix_kind": args.kind,
                    "cell_count": len(cells),
                    "run_manifest_sha256": run_manifest["run_manifest_sha256"],
                },
                sort_keys=True,
            )
        )
        return 0
    manifest_path = args.output / "run_manifest.json"
    if args.resume:
        if not args.output.is_dir() or not manifest_path.is_file():
            raise ValueError("Task 6 resume requires an existing complete output")
        existing = _load_json(manifest_path, _sha256(manifest_path), "Task 6 run manifest")
        if existing != run_manifest:
            raise ValueError("Task 6 resume manifest does not exactly match")
    else:
        if args.output.exists() or args.output.is_symlink():
            raise FileExistsError(f"Task 6 output already exists: {args.output}")
        args.output.mkdir(parents=True)
        _atomic_json(manifest_path, run_manifest)

    results_path = args.output / "results.jsonl"
    completed = 0
    if args.kind not in {"visual-state-fixed-grid", "visual-state-native-boundary"}:
        completed = _existing_prefix(
            results_path,
            cells=cells,
            qid=qid,
            fixture_sha256=args.fixture_sha256,
            kind=args.kind,
        )
    bootstrap_output = args.output / "bootstrap"
    workload = build_workload(
        operation="evaluate",
        config=config,
        page_count=4,
        output=bootstrap_output,
        mode="docprune",
        run_config=args.run_config,
        index_manifest=args.index_manifest,
        sample_ids=(qid,),
        resume=args.resume,
        ctp_policy=task6_policy_matrix("native")[0].policy,
        fixed_page_fixture=args.fixture,
        fixed_page_fixture_sha256=args.fixture_sha256,
        execution_runtime_commit=args.runtime_commit,
    )
    samples = tuple(workload.samples)
    if len(samples) != 1 or samples[0].question_id != qid:
        raise ValueError("Task 6 bootstrap workload selected the wrong QID")
    sample = samples[0]
    fixture_question = fixture.question(qid)
    base_runner = workload.runner
    task7_target = None
    task7_likelihood_records: list[dict[str, object]] = []
    task7_likelihood_path = args.output / "task7-likelihood.jsonl"
    if args.kind == "visual-state-fixed-grid":
        completed = _existing_prefix(
            results_path,
            cells=cells,
            qid=qid,
            fixture_sha256=args.fixture_sha256,
            kind=args.kind,
            sample=sample,
            fixture_question=fixture_question,
        )
        if completed == 1:
            raise ValueError(
                "Task 7 cannot resume a one-cell prefix without its atomic likelihood pair"
            )
        if completed == 0 and (
            task7_likelihood_path.exists() or task7_likelihood_path.is_symlink()
        ):
            raise FileExistsError("Task 7 likelihood destination already exists")
        if completed >= 2 and (
            task7_likelihood_path.is_symlink() or not task7_likelihood_path.is_file()
        ):
            raise ValueError("Task 7 resume requires the existing likelihood pair")
        from docprune.answerers import prepare_task7_likelihood_target_for_question
        from docprune.task7_runtime import admit_task7_likelihood_pair_from_files

        task7_target = prepare_task7_likelihood_target_for_question(
            base_runner.answerer.processor,
            page_count=4,
            question=sample.question,
            accepted_references=sample.answers,
        )
        if completed >= 2:
            admit_task7_likelihood_pair_from_files(
                run_manifest_path=manifest_path,
                results_path=results_path,
                likelihood_path=task7_likelihood_path,
            )
    elif args.kind == "visual-state-native-boundary":
        completed = _existing_prefix(
            results_path,
            cells=cells,
            qid=qid,
            fixture_sha256=args.fixture_sha256,
            kind=args.kind,
            sample=sample,
            fixture_question=fixture_question,
            native_boundary_cell=native_boundary_cell,
            native_boundary_manifest_sha256=args.native_boundary_manifest_sha256,
        )
    base_runner.warmup(sample)
    base_answerer = base_runner.answerer
    append_mode = results_path.exists()
    for cell_index in range(completed, len(cells)):
        cell = cells[cell_index]
        if args.kind == "visual-state-native-boundary":
            from docprune.task7_native_boundary import (
                bind_task7_native_boundary_result_evidence,
                validate_task7_native_boundary_result_record,
            )
            from docprune.task7_runtime import validate_task7_source_identity

            answerer = DocPruneQwenAnswerer(
                base_answerer.model,
                base_answerer.processor,
                page_config=config.for_pages(4),
                qa_stage="full",
                **native_boundary_cell.answerer_kwargs(),
            )
            runner = DocPruneM3DocRAG(
                base_runner.retriever,
                base_runner.page_loader,
                answerer,
                top_k=4,
            )
            runner.inherit_warmup_state(base_runner)
            record = runner.run_sample(sample).to_dict()
            bind_task7_native_boundary_result_evidence(
                record,
                native_boundary_cell,
                fixture_sha256=args.fixture_sha256,
                native_boundary_manifest_sha256=args.native_boundary_manifest_sha256,
            )
            expected_policy = (
                None
                if native_boundary_cell.ctp_policy is None
                else native_boundary_cell.ctp_policy.to_dict()
            )
            _validate_result_record(
                record,
                line_number=1,
                expected_page_count=4,
                production=True,
                expected_policy=expected_policy,
                forbid_policy=expected_policy is None,
                allowed_extra_fields={
                    "matrix_cell",
                    "matrix_kind",
                    "intervention_name",
                    "analysis_family",
                    "boundary_source",
                    "native_boundary_manifest_sha256",
                    "native_crossing",
                    "native_layer",
                },
                allow_zero_post_ctp=True,
            )
            validate_task7_source_identity(record, sample, fixture_question)
            validate_task7_native_boundary_result_record(
                record,
                native_boundary_cell,
                fixture_sha256=args.fixture_sha256,
                native_boundary_manifest_sha256=args.native_boundary_manifest_sha256,
            )
            append_result_jsonl(results_path, record, resume=append_mode)
            append_mode = True
            continue
        if args.kind == "visual-state-fixed-grid":
            from docprune.task7_runtime import (
                bind_task7_result_evidence,
                build_task7_likelihood_record,
                task7_intervention_matrix,
                validate_task7_result_record,
                validate_task7_source_identity,
                write_task7_likelihood_artifact,
            )

            intervention = task7_intervention_matrix()[cell_index]
            answerer = DocPruneQwenAnswerer(
                base_answerer.model,
                base_answerer.processor,
                page_config=config.for_pages(4),
                qa_stage="full",
                teacher_forced_target_token_ids=(
                    tuple(tuple(value) for value in task7_target["target_token_ids"])
                    if cell_index < 2
                    else None
                ),
                **intervention.answerer_kwargs(),
            )
            runner = DocPruneM3DocRAG(
                base_runner.retriever,
                base_runner.page_loader,
                answerer,
                top_k=4,
            )
            runner.inherit_warmup_state(base_runner)
            sample_result = runner.run_sample(sample)
            record = sample_result.to_dict()
            bind_task7_result_evidence(
                record,
                intervention,
                cell_index=cell_index,
                fixture_sha256=args.fixture_sha256,
            )
            if cell_index < 2:
                _bind_task7_likelihood_capture(record, sample_result, task7_target)
            expected_policy = (
                None if intervention.ctp_policy is None else intervention.ctp_policy.to_dict()
            )
            _validate_result_record(
                record,
                line_number=cell_index + 1,
                expected_page_count=4,
                production=True,
                expected_policy=expected_policy,
                forbid_policy=expected_policy is None,
                allowed_extra_fields={"matrix_cell", "matrix_kind", "intervention_name"}
                | (
                    {
                        "assistant_prompt_sha256",
                        "prefill_input_ids_shape",
                        "prefill_input_ids_sha256",
                    }
                    if cell_index < 2
                    else set()
                ),
                allow_zero_post_ctp=True,
            )
            validate_task7_source_identity(record, sample, fixture_question)
            validate_task7_result_record(
                record,
                intervention,
                fixture_sha256=args.fixture_sha256,
            )
            if cell_index < 2:
                if sample_result.teacher_forced_loglikelihoods is None:
                    raise ValueError("Task 7 likelihood arm did not return teacher-forced values")
                task7_likelihood_records.append(
                    build_task7_likelihood_record(
                        qid=qid,
                        intervention_name=intervention.name,
                        target=task7_target,
                        per_reference_mean_loglikelihood=(
                            sample_result.teacher_forced_loglikelihoods
                        ),
                        fixture_sha256=args.fixture_sha256,
                        run_manifest_sha256=run_manifest["run_manifest_sha256"],
                        source_result_sha256=_manifest_digest(record),
                        prefill_input_ids_shape=sample_result.prefill_input_ids_shape,
                        prefill_input_ids_sha256=sample_result.prefill_input_ids_sha256,
                    )
                )
            append_result_jsonl(results_path, record, resume=append_mode)
            append_mode = True
            if cell_index == 1:
                write_task7_likelihood_artifact(
                    task7_likelihood_path,
                    tuple(task7_likelihood_records),
                )
            continue
        source_matrix = task6_policy_matrix("native" if args.kind == "smoke" else args.kind)
        if args.kind == "smoke":
            source_matrix = tuple(source_matrix[index] for index in (0, 1, 2, 3, 4, 5, 15, 25, 35))
        policy = source_matrix[cell_index].policy
        answerer = DocPruneQwenAnswerer(
            base_answerer.model,
            base_answerer.processor,
            page_config=config.for_pages(4),
            qa_stage="full",
            ctp_policy=policy,
            policy_experiment_version=cell["experiment_version"],
            policy_repetition=cell["repetition"],
        )
        runner = DocPruneM3DocRAG(
            base_runner.retriever,
            base_runner.page_loader,
            answerer,
            top_k=4,
        )
        runner.inherit_warmup_state(base_runner)
        record = runner.run_sample(sample).to_dict()
        cell_manifest = {
            "fixed_page_fixture_sha256": args.fixture_sha256,
            "fixed_page_provenance": True,
            "global_index_loaded": False,
            "ctp_policy": policy.to_dict(),
        }
        if cell["experiment_version"] is not None:
            cell_manifest["ctp_policy_context"] = {
                "experiment_version": cell["experiment_version"],
                "repetition": cell["repetition"],
                "geometry": None,
            }
        _bind_task6_result_evidence(record, cell_manifest)
        _validate_result_record(
            record,
            line_number=cell_index + 1,
            expected_page_count=4,
            production=True,
            expected_policy=policy.to_dict(),
        )
        validate_task6_result_record(
            record,
            Task6ResultIdentity(
                args.fixture_sha256,
                policy.name,
                cell["experiment_version"],
                cell["repetition"],
            ),
        )
        record.update(
            {
                "matrix_cell": cell_index,
                "matrix_kind": args.kind,
                "experiment_version": cell["experiment_version"],
                "repetition": cell["repetition"],
            }
        )
        append_result_jsonl(results_path, record, resume=append_mode)
        append_mode = True
    if args.kind == "visual-state-fixed-grid":
        from docprune.task7_runtime import admit_task7_likelihood_pair_from_files

        admit_task7_likelihood_pair_from_files(
            run_manifest_path=manifest_path,
            results_path=results_path,
            likelihood_path=task7_likelihood_path,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
