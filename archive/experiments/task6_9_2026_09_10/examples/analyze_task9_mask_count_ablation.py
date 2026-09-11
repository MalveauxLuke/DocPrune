#!/usr/bin/env python3
"""Run one reuse-only Task 9 mask-count ablation on authenticated artifacts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from docprune.segmentation import load_region_mapping
from docprune.task9_attribution import (
    _canonical_sha256,
    analyze_contextcite_mask_count_ablation,
)


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _verify(value: dict[str, object], field: str, label: str) -> None:
    claimed = value.get(field)
    unsigned = dict(value)
    unsigned.pop(field, None)
    if not isinstance(claimed, str) or claimed != _canonical_sha256(unsigned):
        raise ValueError(f"{label} internal checksum mismatch")


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
    parser.add_argument("--question-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        not args.question_root.is_absolute()
        or not args.output.is_absolute()
        or args.output.exists()
    ):
        raise ValueError("mask-count ablation paths are invalid")

    selected = _load(args.question_root / "selected-arms.json")
    analysis_artifact = _load(args.question_root / "analysis.json")
    raw_root = args.question_root / "raw"
    manifest = _load(raw_root / "run-manifest.json")
    raw = _load(raw_root / "raw-result.json")
    primary = _load(raw_root / "primary-target.json")
    _verify(selected, "selected_arms_sha256", "selected arms")
    _verify(analysis_artifact, "artifact_sha256", "analysis")
    _verify(manifest, "run_manifest_sha256", "run manifest")
    _verify(raw, "raw_result_sha256", "raw result")
    _verify(primary, "target_dataset_sha256", "primary target")
    qid = selected["qid"]
    if (
        manifest["qid"] != qid
        or analysis_artifact["analysis"]["question_id"] != qid
        or selected["analysis_artifact_sha256"] != analysis_artifact["artifact_sha256"]
        or primary["target_dataset_sha256"] != manifest["primary_target_dataset_sha256"]
    ):
        raise ValueError("mask-count ablation input bindings disagree")
    mapping = load_region_mapping(Path(manifest["mapping_path"]), validate_raw_artifacts=True)
    regions = [
        {"source_id": source.source_id, "token_cost": len(source.token_ids)}
        for source in mapping.sources
        if source.token_ids
    ]
    reference_count = raw["reference_sequence_count"]
    local_rows = raw["budget_local_mean_sequence_loglikelihoods"]
    local_masks = [row["vector"] for row in raw["budget_local_plan"]]
    if (
        type(reference_count) is not int
        or reference_count < 1
        or len(local_rows) != 32
        or len(local_masks) != 32
    ):
        raise ValueError("mask-count ablation local holdouts are invalid")
    result = analyze_contextcite_mask_count_ablation(
        primary["design"],
        primary["outcomes"],
        regions,
        requested_budget=raw["native_docprune_selection"]["achieved_budget"],
        budget_local_masks=local_masks,
        budget_local_targets=[max(row[:reference_count]) for row in local_rows],
    )
    canonical = result["fits"][-1]
    canonical_arm = next(
        arm
        for arm in analysis_artifact["analysis"]["selections"]["budgets"][0]["arms"]
        if arm["arm"] == "contextcite_gold_support"
    )
    if (
        canonical["selection"]["top_source_ids"] != canonical_arm["retained_source_ids"]
        or canonical["selection"]["achieved_budget"] != canonical_arm["achieved_token_count"]
    ):
        raise ValueError("recomputed canonical 256 selection does not match the pilot artifact")
    artifact: dict[str, object] = {
        "schema_version": "docprune-task9-mask-count-ablation-question-v1",
        "status": "completed-task9-mask-count-ablation-question",
        "qid": qid,
        "baseline_stratum": manifest["baseline_stratum"],
        "runtime_commit": manifest["runtime_commit"],
        "input_bindings": {
            "selected_arms_sha256": selected["selected_arms_sha256"],
            "analysis_artifact_sha256": analysis_artifact["artifact_sha256"],
            "run_manifest_sha256": manifest["run_manifest_sha256"],
            "raw_result_sha256": raw["raw_result_sha256"],
            "primary_target_dataset_sha256": primary["target_dataset_sha256"],
            "mapping_internal_sha256": manifest["mapping_internal_sha256"],
        },
        "ablation": result,
    }
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    _publish(args.output, artifact)
    print(json.dumps({"qid": qid, "artifact_sha256": artifact["artifact_sha256"]}))


if __name__ == "__main__":
    main()
