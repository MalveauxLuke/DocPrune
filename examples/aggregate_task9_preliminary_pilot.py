#!/usr/bin/env python3
"""Verify and package the completed Task 9 preliminary 48-question pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from docprune.task9_attribution import (
    _canonical_sha256,
    aggregate_task9_preliminary_pilot,
)


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_internal(value: dict[str, object], field: str, label: str) -> None:
    claimed = value.get(field)
    unsigned = dict(value)
    unsigned.pop(field, None)
    if not isinstance(claimed, str) or claimed != _canonical_sha256(unsigned):
        raise ValueError(f"{label} internal checksum mismatch")


def _arm(result: dict[str, object]) -> dict[str, object]:
    gold = float(result["gold_mean_loglikelihood"])
    alternative = float(result["alternative_mean_loglikelihood"])
    return {
        "answer": result["answer"],
        "f1": float(result["normalized_token_f1"]),
        "em": int(bool(result["exact_match"])),
        "gold": gold,
        "alternative": alternative,
        "margin": gold - alternative,
        "peak_allocated_gpu_bytes": result["peak_allocated_gpu_bytes"],
        "result_sha256": result["result_sha256"],
    }


def _metric_arm(arm: dict[str, object]) -> dict[str, object]:
    return {key: arm[key] for key in ("f1", "em", "gold", "margin")}


def _fidelity(value: dict[str, object]) -> dict[str, object]:
    return {
        "lds_spearman": value["lds_spearman"],
        "heldout_rmse": value["heldout_rmse"],
        "constant_rmse": value["constant_rmse"],
        "rmse_beats_constant": value["heldout_rmse"] < value["constant_rmse"],
    }


def _publish(path: Path, value: dict[str, object]) -> None:
    content = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    try:
        os.write(descriptor, content)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--cohort-file-sha256", required=True)
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_260_901)
    args = parser.parse_args()
    if (
        not args.cohort.is_absolute()
        or any(not root.is_absolute() for root in args.root)
        or not args.output.is_absolute()
        or args.output.exists()
    ):
        raise ValueError("pilot aggregate paths are invalid")
    if _file_sha256(args.cohort) != args.cohort_file_sha256:
        raise ValueError("sealed cohort file checksum mismatch")
    cohort = _load(args.cohort)
    _verify_internal(cohort, "cohort_sha256", "sealed cohort")
    selected_records = cohort.get("selected_records")
    if not isinstance(selected_records, list) or len(selected_records) != 48:
        raise ValueError("sealed cohort must contain 48 selected records")
    expected_qids = tuple(record["question_id"] for record in selected_records)
    cohort_by_qid = {record["question_id"]: record for record in selected_records}

    selected_paths = sorted(path for root in args.root for path in root.rglob("selected-arms.json"))
    if len(selected_paths) != 48:
        raise ValueError(f"expected 48 selected-arm artifacts, found {len(selected_paths)}")
    rows: list[dict[str, object]] = []
    aggregate_inputs: list[dict[str, object]] = []
    for selected_path in selected_paths:
        selected = _load(selected_path)
        _verify_internal(selected, "selected_arms_sha256", str(selected_path))
        qid = selected.get("qid")
        if qid not in cohort_by_qid:
            raise ValueError(f"selected artifact has unexpected QID: {qid}")
        analysis_path = selected_path.with_name("analysis.json")
        analysis_artifact = _load(analysis_path)
        _verify_internal(analysis_artifact, "artifact_sha256", str(analysis_path))
        if selected["analysis_artifact_sha256"] != analysis_artifact["artifact_sha256"]:
            raise ValueError(f"selected/analysis binding mismatch: {qid}")
        question_analyses = selected.get("question_analyses")
        if not isinstance(question_analyses, list) or len(question_analyses) != 1:
            raise ValueError(f"selected artifact question analysis is invalid: {qid}")
        question_analysis = dict(question_analyses[0])
        _verify_internal(question_analysis, "analysis_sha256", f"question analysis {qid}")
        if question_analysis["question_id"] != qid:
            raise ValueError(f"question analysis QID mismatch: {qid}")

        results = [selected["unpruned"], selected["native_docprune"], *selected["selected_results"]]
        arms: dict[str, dict[str, object]] = {}
        for result in results:
            result = dict(result)
            _verify_internal(result, "result_sha256", f"arm result {qid}/{result.get('arm')}")
            arm_name = result["arm"]
            if arm_name in arms:
                raise ValueError(f"duplicate arm result: {qid}/{arm_name}")
            arms[arm_name] = _arm(result)
        required = {
            "unpruned",
            "native_docprune",
            "contextcite_gold_support",
            "random_region_size_aware",
        }
        if not required.issubset(arms):
            raise ValueError(f"missing required arm: {qid}")
        stratum = cohort_by_qid[qid]["baseline_stratum"]
        if (stratum == "baseline_wrong") != ("contextcite_gold_margin" in arms):
            raise ValueError(f"gold-margin availability disagrees with stratum: {qid}")

        analysis = analysis_artifact["analysis"]
        support = analysis["gold_support"]
        selection = analysis["selections"]["budgets"][0]
        native_policy = selected["native_docprune"]["policy_selection"]
        compact = {
            "question_id": qid,
            "ordinal": expected_qids.index(qid),
            "baseline_stratum": stratum,
            "baseline_correct": question_analysis["baseline_correct"],
            "forced_boundary": analysis_artifact["raw_admission"]["boundary"],
            "cuda_device_name": selected["cuda_device_name"],
            "retention": {
                "visual_population": native_policy["visual_population"],
                "native_achieved_token_count": native_policy["achieved_budget"],
                "regional_requested_token_count": selection["requested_token_count"],
                "regional_achieved_token_count": selection["achieved_token_count"],
                "regional_retained_fraction": selection["retained_fraction"],
            },
            "arms": arms,
            "contextcite_fidelity": {
                "global": _fidelity(support["global_fidelity"]),
                "budget_local": _fidelity(support["budget_local_fidelity"]),
                "five_refit_top_selection": support["stability"]["top_selection_summary"],
            },
            "source": {
                "selected_arms_path": str(selected_path),
                "selected_arms_file_sha256": _file_sha256(selected_path),
                "selected_arms_internal_sha256": selected["selected_arms_sha256"],
                "analysis_path": str(analysis_path),
                "analysis_file_sha256": _file_sha256(analysis_path),
                "analysis_internal_sha256": analysis_artifact["artifact_sha256"],
                "completion_sha256": analysis_artifact["completion_sha256"],
            },
        }
        aggregate_inputs.append(
            {
                "question_id": qid,
                "baseline_stratum": stratum,
                "arms": {name: _metric_arm(value) for name, value in arms.items()},
            }
        )
        rows.append(compact)

    aggregate = aggregate_task9_preliminary_pilot(
        aggregate_inputs,
        expected_qids=expected_qids,
        natural_pool_weights=cohort["natural_pool_weights"],
        draws=args.draws,
        seed=args.seed,
    )
    rows.sort(key=lambda row: row["ordinal"])
    global_lds = [float(row["contextcite_fidelity"]["global"]["lds_spearman"]) for row in rows]
    local_lds = [float(row["contextcite_fidelity"]["budget_local"]["lds_spearman"]) for row in rows]
    result: dict[str, object] = {
        "schema_version": "docprune-task9-preliminary-dynamic48-unified-analysis-v1",
        "status": "completed-task9-preliminary-dynamic48-unified-analysis",
        "interpretation": {
            "design": (
                "48-question stratified random development pilot: 24 unpruned-correct and "
                "24 unpruned-wrong; all pruning arms use each question's native dynamic "
                "DocPrune retained-token count"
            ),
            "primary_comparison": "gold-support regional ContextCite minus native dynamic DocPrune",
            "random_control": "region-size-aware random selection at the same whole-region budget",
            "gold_margin_scope": "available only for the 24 baseline-wrong questions",
            "claim_limit": (
                "development-pilot evidence only; not a held-out estimate and not vanilla "
                "token-level ContextCite"
            ),
        },
        "provenance": {
            "runtime_commit": "90f27d7ed8b99ad10f1a5fe405c131127456ae5d",
            "cohort_path": str(args.cohort),
            "cohort_file_sha256": args.cohort_file_sha256,
            "cohort_internal_sha256": cohort["cohort_sha256"],
            "input_roots": [str(root) for root in args.root],
            "question_count": len(rows),
        },
        "outcomes": aggregate,
        "surrogate_fidelity": {
            "global_lds_mean": sum(global_lds) / len(global_lds),
            "global_lds_minimum": min(global_lds),
            "global_lds_at_least_0_5_count": sum(value >= 0.5 for value in global_lds),
            "global_rmse_beats_constant_count": sum(
                row["contextcite_fidelity"]["global"]["rmse_beats_constant"] for row in rows
            ),
            "budget_local_lds_mean": sum(local_lds) / len(local_lds),
            "budget_local_lds_minimum": min(local_lds),
            "budget_local_lds_at_least_0_5_count": sum(value >= 0.5 for value in local_lds),
            "budget_local_rmse_beats_constant_count": sum(
                row["contextcite_fidelity"]["budget_local"]["rmse_beats_constant"] for row in rows
            ),
        },
        "questions": rows,
    }
    result["analysis_sha256"] = _canonical_sha256(result)
    _publish(args.output, result)
    print(json.dumps({"output": str(args.output), "analysis_sha256": result["analysis_sha256"]}))


if __name__ == "__main__":
    main()
