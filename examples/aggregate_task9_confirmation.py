#!/usr/bin/env python3
"""Package the completed 100-question baseline-wrong confirmation as one analysis JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from statistics import mean

from docprune.task9_attribution import _canonical_sha256


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(value: dict[str, object], field: str, label: str) -> None:
    unsigned = dict(value)
    observed = unsigned.pop(field, None)
    if observed != _canonical_sha256(unsigned):
        raise ValueError(f"{label} internal checksum mismatch")


def _arm(value: dict[str, object]) -> dict[str, object]:
    gold = float(value["gold_mean_loglikelihood"])
    alternative = float(value["alternative_mean_loglikelihood"])
    return {
        "answer": value["answer"],
        "f1": float(value["normalized_token_f1"]),
        "em": int(bool(value["exact_match"])),
        "gold": gold,
        "alternative": alternative,
        "margin": gold - alternative,
        "peak_allocated_gpu_bytes": value["peak_allocated_gpu_bytes"],
        "result_sha256": value["result_sha256"],
    }


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = probability * (len(ordered) - 1)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _comparison(
    rows: list[dict[str, object]],
    *,
    arm: str,
    components: list[list[str]],
    draws: int,
    seed: int,
) -> dict[str, object]:
    by_qid = {row["question_id"]: row for row in rows}
    deltas = {
        qid: float(row["arms"][arm]["f1"]) - float(row["arms"]["native_docprune"]["f1"])
        for qid, row in by_qid.items()
    }
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(draws):
        selected = [components[rng.randrange(len(components))] for _ in components]
        sampled_qids = [qid for component in selected for qid in component]
        samples.append(mean(deltas[qid] for qid in sampled_qids))
    wins = sum(deltas[qid] > 0 for qid in deltas)
    ties = sum(deltas[qid] == 0 for qid in deltas)
    rescues = sum(
        bool(row["arms"][arm]["em"]) and not bool(row["arms"]["native_docprune"]["em"])
        for row in rows
    )
    return {
        "arm": arm,
        "reference_arm": "native_docprune",
        "n": len(rows),
        "mean_normalized_token_f1_difference": mean(deltas.values()),
        "support_component_bootstrap_interval_95": [
            _percentile(samples, 0.025),
            _percentile(samples, 0.975),
        ],
        "mean_exact_match_difference": mean(
            float(row["arms"][arm]["em"]) - float(row["arms"]["native_docprune"]["em"])
            for row in rows
        ),
        "win_tie_loss": {"win": wins, "tie": ties, "loss": len(rows) - wins - ties},
        "rescue_count": rescues,
        "rescue_rate": rescues / len(rows),
        "mean_gold_likelihood_difference": mean(
            float(row["arms"][arm]["gold"])
            - float(row["arms"]["native_docprune"]["gold"])
            for row in rows
        ),
        "mean_gold_vs_alternative_margin_difference": mean(
            float(row["arms"][arm]["margin"])
            - float(row["arms"]["native_docprune"]["margin"])
            for row in rows
        ),
        "bootstrap": {
            "unit": "support-document connected component",
            "draw_count": draws,
            "seed": seed,
        },
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
    parser.add_argument("--seed", type=int, default=20_260_902)
    args = parser.parse_args()
    if (
        not args.cohort.is_absolute()
        or any(not root.is_absolute() for root in args.root)
        or not args.output.is_absolute()
        or args.output.exists()
        or args.draws <= 0
        or _file_sha256(args.cohort) != args.cohort_file_sha256
    ):
        raise ValueError("confirmation aggregate arguments are invalid")
    cohort = _load(args.cohort)
    _verify(cohort, "cohort_sha256", "confirmation cohort")
    records = cohort.get("selected_records")
    if not isinstance(records, list) or len(records) != 100:
        raise ValueError("confirmation cohort must contain 100 records")
    expected_qids = [row["question_id"] for row in records]
    ordinal = {qid: index for index, qid in enumerate(expected_qids)}
    selected_paths = sorted(path for root in args.root for path in root.rglob("selected-arms.json"))
    if len(selected_paths) != 100:
        raise ValueError(f"expected 100 selected-arm artifacts, found {len(selected_paths)}")
    rows: list[dict[str, object]] = []
    for path in selected_paths:
        selected = _load(path)
        _verify(selected, "selected_arms_sha256", str(path))
        qid = selected.get("qid")
        if qid not in ordinal:
            raise ValueError(f"unexpected selected-arm QID: {qid}")
        analysis_path = path.with_name("analysis.json")
        analysis_artifact = _load(analysis_path)
        _verify(analysis_artifact, "artifact_sha256", str(analysis_path))
        if selected.get("analysis_artifact_sha256") != analysis_artifact["artifact_sha256"]:
            raise ValueError(f"selected/analysis binding mismatch: {qid}")
        values = [selected["unpruned"], selected["native_docprune"], *selected["selected_results"]]
        arms: dict[str, dict[str, object]] = {}
        for value in values:
            value = dict(value)
            _verify(value, "result_sha256", f"{qid}/{value.get('arm')}")
            arms[value["arm"]] = _arm(value)
        required = {
            "unpruned",
            "native_docprune",
            "contextcite_gold_support",
            "contextcite_gold_margin",
            "random_region_size_aware",
        }
        if set(arms) != required:
            raise ValueError(f"confirmation arms are incomplete: {qid}")
        analysis = analysis_artifact["analysis"]
        selection = analysis["selections"]["budgets"][0]
        rows.append({
            "question_id": qid,
            "ordinal": ordinal[qid],
            "supporting_document_ids": records[ordinal[qid]]["supporting_document_ids"],
            "forced_boundary": analysis["forced_boundary"],
            "retention": {
                "visual_population": selected["native_docprune"]["policy_selection"]["visual_population"],
                "native_achieved_token_count": selected["native_docprune"]["policy_selection"]["achieved_budget"],
                "regional_requested_token_count": selection["requested_token_count"],
                "regional_achieved_token_count": selection["achieved_token_count"],
            },
            "arms": arms,
            "fit_only_diagnostics": {
                "gold_support_five_refit_top_selection": analysis["gold_support"]["stability"]["top_selection_summary"],
                "gold_margin_five_refit_top_selection": analysis["gold_margin"]["stability"]["top_selection_summary"],
            },
            "source": {
                "selected_arms_path": str(path),
                "selected_arms_file_sha256": _file_sha256(path),
                "analysis_path": str(analysis_path),
                "analysis_file_sha256": _file_sha256(analysis_path),
            },
        })
    if {row["question_id"] for row in rows} != set(expected_qids):
        raise ValueError("confirmation result QIDs are incomplete")
    rows.sort(key=lambda row: row["ordinal"])
    components = [component["qids"] for component in cohort["support_components"]["components"]]
    result: dict[str, object] = {
        "schema_version": "docprune-task9-baseline-wrong100-confirmation-analysis-v1",
        "status": "completed-task9-baseline-wrong100-confirmation",
        "interpretation": {
            "cohort": "100 new randomly selected ordinary baseline-wrong questions",
            "primary_endpoint": "actual downstream answer after budget-matched pruning",
            "fit_masks_per_question": 256,
            "surrogate_holdout_masks_per_question": 0,
            "retrieval": "exact cached top-4 pages; no retrieval run",
            "claim_limit": "regional ContextCite diagnostic, not vanilla token-level ContextCite",
        },
        "cohort": {
            "path": str(args.cohort),
            "file_sha256": args.cohort_file_sha256,
            "cohort_sha256": cohort["cohort_sha256"],
            "question_count": 100,
            "support_component_count": len(components),
            "selection_used_document_fallback": cohort["selection_used_document_fallback"],
        },
        "comparisons": {
            arm: _comparison(
                rows,
                arm=arm,
                components=components,
                draws=args.draws,
                seed=args.seed + offset,
            )
            for offset, arm in enumerate((
                "contextcite_gold_support",
                "contextcite_gold_margin",
                "random_region_size_aware",
            ))
        },
        "per_question": rows,
    }
    result["analysis_sha256"] = _canonical_sha256(result)
    _publish(args.output, result)
    print(json.dumps({
        "output": str(args.output),
        "analysis_sha256": result["analysis_sha256"],
        "question_count": len(rows),
        "gold_support_vs_docprune": result["comparisons"]["contextcite_gold_support"],
    }, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
