#!/usr/bin/env python3
"""Analyze one admitted fit-only Task 9 confirmation artifact."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from docprune.segmentation import load_region_mapping
from docprune.task9_attribution import (
    _canonical_sha256,
    analyze_task9_confirmation_attribution,
)
from docprune.task9_live import admit_task9_regional_development


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decoder-layer-count", type=int, required=True)
    args = parser.parse_args()
    if not args.root.is_absolute() or not args.output.is_absolute() or args.output.exists():
        raise ValueError("Task 9 confirmation analysis paths are invalid")
    manifest = _load(args.root / "run-manifest.json")
    raw = _load(args.root / "raw-result.json")
    primary = _load(args.root / "primary-target.json")
    secondary = _load(args.root / "secondary-target.json")
    completion = _load(args.root / "completion-manifest.json")
    mapping = load_region_mapping(Path(manifest["mapping_path"]), validate_raw_artifacts=True)
    trace = raw["unpruned_generation_trace"]
    admission = admit_task9_regional_development(
        args.root,
        expected_runtime_commit=manifest["runtime_commit"],
        expected_qid=manifest["qid"],
        expected_boundary=manifest["boundary"],
        expected_fixture_sha256=manifest["fixed_page_fixture_sha256"],
        expected_mapping_sha256=manifest["mapping_artifact_sha256"],
        expected_mapping_internal_sha256=manifest["mapping_internal_sha256"],
        expected_geometry_count=manifest["geometry_count"],
        expected_geometry_sha256=manifest["geometry_sha256"],
        expected_trace=(
            trace["original_visual_tokens"],
            trace["post_btp_visual_tokens"],
            trace["post_qtp_visual_tokens"],
        ),
        expected_decoder_layer_count=args.decoder_layer_count,
        expected_gpu_substring="NVIDIA",
        expected_fit_mask_count=256,
        expected_holdout_mask_count=0,
        expected_budget_local_holdout_mask_count=0,
        expected_preliminary=True,
    )
    analysis = analyze_task9_confirmation_attribution(
        primary_design=primary["design"],
        primary_outcomes=primary["outcomes"],
        secondary_design=secondary["design"],
        secondary_outcomes=secondary["outcomes"],
        regions=[
            {"source_id": source.source_id, "token_cost": len(source.token_ids)}
            for source in mapping.sources
        ],
        requested_token_count=raw["native_docprune_selection"]["achieved_budget"],
        gold_margin_available=manifest["baseline_stratum"] == "baseline_wrong",
    )
    result: dict[str, object] = {
        "schema_version": "docprune-task9-confirmation-attribution-artifact-v1",
        "status": "analyzed-task9-confirmation-attribution",
        "raw_admission": admission,
        "completion_sha256": completion["completion_sha256"],
        "analysis": analysis,
    }
    result["artifact_sha256"] = _canonical_sha256(result)
    _publish(args.output, result)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
