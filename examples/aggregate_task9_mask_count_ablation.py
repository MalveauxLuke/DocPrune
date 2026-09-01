#!/usr/bin/env python3
"""Verify and aggregate all 48 Task 9 mask-count ablation artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path

from docprune.task9_attribution import _canonical_sha256


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


def _verify(value: dict[str, object], field: str, label: str) -> None:
    claimed = value.get(field)
    unsigned = dict(value)
    unsigned.pop(field, None)
    if not isinstance(claimed, str) or claimed != _canonical_sha256(unsigned):
        raise ValueError(f"{label} internal checksum mismatch")


def _summary(fits: list[dict[str, object]]) -> dict[str, object]:
    def values(path: tuple[str, ...]) -> list[float]:
        result: list[float] = []
        for fit in fits:
            value: object = fit
            for key in path:
                value = value[key]
            if value is not None:
                result.append(float(value))
        return result

    def distribution(items: list[float]) -> dict[str, object]:
        return {
            "defined_count": len(items),
            "mean": math.fsum(items) / len(items) if items else None,
            "median": statistics.median(items) if items else None,
            "minimum": min(items) if items else None,
            "maximum": max(items) if items else None,
        }

    global_lds = values(("global_fidelity", "lds_spearman"))
    local_lds = values(("budget_local_fidelity", "lds_spearman"))
    return {
        "fit_count": len(fits),
        "question_count": len({fit["question_id"] for fit in fits}),
        "global_lds": distribution(global_lds),
        "budget_local_lds": distribution(local_lds),
        "coefficient_spearman_vs_256": distribution(values(("coefficient_spearman_vs_256",))),
        "region_jaccard_vs_256": distribution(values(("selection_vs_256", "region_jaccard"))),
        "token_jaccard_vs_256": distribution(values(("selection_vs_256", "token_jaccard"))),
        "exact_region_set_count": sum(fit["selection_vs_256"]["exact_region_set"] for fit in fits),
        "global_lds_at_least_0_5_count": sum(value >= 0.5 for value in global_lds),
        "budget_local_lds_at_least_0_5_count": sum(value >= 0.5 for value in local_lds),
        "global_rmse_beats_constant_count": sum(
            fit["global_fidelity"]["surrogate_beats_constant"] for fit in fits
        ),
        "budget_local_rmse_beats_constant_count": sum(
            fit["budget_local_fidelity"]["surrogate_beats_constant"] for fit in fits
        ),
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
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pilot-analysis", type=Path, required=True)
    parser.add_argument("--pilot-analysis-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        not args.root.is_absolute()
        or not args.pilot_analysis.is_absolute()
        or not args.output.is_absolute()
        or args.output.exists()
    ):
        raise ValueError("mask-count aggregate paths are invalid")
    pilot = _load(args.pilot_analysis)
    _verify(pilot, "analysis_sha256", "preliminary pilot analysis")
    if pilot["analysis_sha256"] != args.pilot_analysis_sha256:
        raise ValueError("preliminary pilot analysis identity changed")
    expected = {row["question_id"] for row in pilot["questions"]}
    paths = sorted(args.root.glob("batch-*/*.json"))
    paths = [path for path in paths if path.name != "completion-manifest.json"]
    if len(paths) != 48:
        raise ValueError(f"expected 48 question artifacts, found {len(paths)}")

    artifacts: list[dict[str, object]] = []
    observed: set[str] = set()
    for path in paths:
        artifact = _load(path)
        _verify(artifact, "artifact_sha256", str(path))
        ablation = artifact["ablation"]
        _verify(ablation, "ablation_sha256", f"ablation {artifact['qid']}")
        if artifact["qid"] in observed:
            raise ValueError(f"duplicate QID: {artifact['qid']}")
        observed.add(artifact["qid"])
        for fit in ablation["fits"]:
            _verify(fit, "fit_sha256", f"fit {artifact['qid']}")
        artifacts.append(artifact)
    if observed != expected:
        raise ValueError("mask-count artifacts do not match the preliminary cohort")

    fits_by_count: dict[int, list[dict[str, object]]] = defaultdict(list)
    fits_by_count_stratum: dict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
    question_rows: list[dict[str, object]] = []
    generation_inventory: list[dict[str, object]] = []
    pilot_by_qid = {row["question_id"]: row for row in pilot["questions"]}
    for artifact in artifacts:
        qid = artifact["qid"]
        stratum = artifact["baseline_stratum"]
        ablation = artifact["ablation"]
        canonical = ablation["fits"][-1]
        canonical_ids = canonical["selection"]["top_source_ids"]
        canonical_sha = _canonical_sha256(canonical_ids)
        per_question: dict[str, object] = {
            "question_id": qid,
            "ordinal": pilot_by_qid[qid]["ordinal"],
            "baseline_stratum": stratum,
            "artifact_sha256": artifact["artifact_sha256"],
            "canonical_256_selection_sha256": canonical_sha,
            "by_mask_count": {},
        }
        unique: dict[str, dict[str, object]] = {}
        for fit in ablation["fits"]:
            compact = {**fit, "question_id": qid, "baseline_stratum": stratum}
            count = int(fit["mask_count"])
            fits_by_count[count].append(compact)
            fits_by_count_stratum[(count, stratum)].append(compact)
            if count == 256:
                continue
            source_ids = fit["selection"]["top_source_ids"]
            selection_sha = _canonical_sha256(source_ids)
            entry = unique.setdefault(
                selection_sha,
                {
                    "selection_sha256": selection_sha,
                    "retained_source_ids": source_ids,
                    "achieved_token_count": fit["selection"]["achieved_budget"],
                    "same_as_canonical_256": selection_sha == canonical_sha,
                    "candidate_fits": [],
                },
            )
            entry["candidate_fits"].append(
                {"mask_count": count, "repeat": fit["repeat"], "fit_sha256": fit["fit_sha256"]}
            )
        for count in ablation["mask_counts"]:
            selected = [fit for fit in fits_by_count[count] if fit["question_id"] == qid]
            per_question["by_mask_count"][str(count)] = _summary(selected)
        inventory = sorted(unique.values(), key=lambda row: row["selection_sha256"])
        generation_inventory.append(
            {
                "question_id": qid,
                "ordinal": pilot_by_qid[qid]["ordinal"],
                "baseline_stratum": stratum,
                "canonical_256_selection_sha256": canonical_sha,
                "unique_reduced_selection_count": len(inventory),
                "selections": inventory,
            }
        )
        question_rows.append(per_question)

    counts = sorted(fits_by_count)
    result: dict[str, object] = {
        "schema_version": "docprune-task9-mask-count-ablation-aggregate-v1",
        "status": "completed-task9-mask-count-ablation-aggregate",
        "interpretation": {
            "cpu_scope": (
                "reuse-only refits, held-out surrogate fidelity, and regional/token-set agreement; "
                "no reduced-mask answers were generated"
            ),
            "reference": "one canonical 256-mask fit per question",
            "reduced_counts": [count for count in counts if count != 256],
            "repeats": "five independent deterministic without-replacement subsets per reduced count",
            "decision_limit": (
                "selected-set or coefficient agreement cannot establish answer equivalence; use the "
                "sealed generation inventory for the bounded GPU comparison"
            ),
        },
        "provenance": {
            "runtime_commit": "d6de9d8f2e2357ec13d6e674eddc7dc0e9a22cb5",
            "job_id": "62471599",
            "root": str(args.root),
            "aggregator_path": str(Path(__file__).resolve()),
            "aggregator_file_sha256": _file_sha256(Path(__file__).resolve()),
            "pilot_analysis_path": str(args.pilot_analysis),
            "pilot_analysis_sha256": args.pilot_analysis_sha256,
            "question_count": len(artifacts),
        },
        "by_mask_count": {str(count): _summary(fits_by_count[count]) for count in counts},
        "by_mask_count_and_stratum": {
            str(count): {
                stratum: _summary(fits_by_count_stratum[(count, stratum)])
                for stratum in ("baseline_correct", "baseline_wrong")
            }
            for count in counts
        },
        "canonical_rescue_qids": [
            row["question_id"]
            for row in pilot["questions"]
            if row["arms"]["contextcite_gold_support"]["em"] == 1
            and row["arms"]["native_docprune"]["em"] == 0
        ],
        "questions": sorted(question_rows, key=lambda row: row["ordinal"]),
        "generation_inventory": sorted(generation_inventory, key=lambda row: row["ordinal"]),
    }
    result["analysis_sha256"] = _canonical_sha256(result)
    _publish(args.output, result)
    print(json.dumps({"output": str(args.output), "analysis_sha256": result["analysis_sha256"]}))


if __name__ == "__main__":
    main()
