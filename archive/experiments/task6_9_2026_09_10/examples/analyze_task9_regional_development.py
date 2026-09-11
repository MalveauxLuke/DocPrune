#!/usr/bin/env python3
"""Validate then analyze one Task 9 regional-attribution development question."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from docprune.segmentation import load_region_mapping
from docprune.task9_attribution import (
    _canonical_sha256,
    analyze_contextcite_development_question,
)
from docprune.task9_live import admit_task9_regional_development


def _publish(path: Path, payload: dict[str, object]) -> None:
    content = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
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
    parser.add_argument("--runtime-commit", required=True)
    parser.add_argument("--qid", required=True)
    parser.add_argument("--boundary", required=True)
    parser.add_argument("--fixture-sha256", required=True)
    parser.add_argument("--mapping-sha256", required=True)
    parser.add_argument("--mapping-internal-sha256", required=True)
    parser.add_argument("--geometry-count", type=int, required=True)
    parser.add_argument("--geometry-sha256", required=True)
    parser.add_argument("--original-visual-tokens", type=int, required=True)
    parser.add_argument("--post-btp-visual-tokens", type=int, required=True)
    parser.add_argument("--post-qtp-visual-tokens", type=int, required=True)
    parser.add_argument("--decoder-layer-count", type=int, required=True)
    parser.add_argument("--gpu-substring", required=True)
    parser.add_argument("--requested-budget", type=int, required=True)
    args = parser.parse_args()
    if not args.output.is_absolute() or args.output.exists() or args.output.is_symlink():
        raise FileExistsError("Task 9 analysis output must be a fresh absolute path")
    admitted = admit_task9_regional_development(
        args.root,
        expected_runtime_commit=args.runtime_commit,
        expected_qid=args.qid,
        expected_boundary=args.boundary,
        expected_fixture_sha256=args.fixture_sha256,
        expected_mapping_sha256=args.mapping_sha256,
        expected_mapping_internal_sha256=args.mapping_internal_sha256,
        expected_geometry_count=args.geometry_count,
        expected_geometry_sha256=args.geometry_sha256,
        expected_trace=(
            args.original_visual_tokens,
            args.post_btp_visual_tokens,
            args.post_qtp_visual_tokens,
        ),
        expected_decoder_layer_count=args.decoder_layer_count,
        expected_gpu_substring=args.gpu_substring,
    )
    manifest = json.loads((args.root / "run-manifest.json").read_text())
    mapping = load_region_mapping(Path(manifest["mapping_path"]), validate_raw_artifacts=True)
    regions = [
        {"source_id": source.source_id, "token_cost": len(source.token_ids)}
        for source in mapping.sources
    ]
    analyses: dict[str, object] = {}
    for name in ("primary", "secondary"):
        target = json.loads((args.root / f"{name}-target.json").read_text())
        analyses[name] = analyze_contextcite_development_question(
            target["design"],
            target["outcomes"],
            regions,
            requested_budget=args.requested_budget,
        )
    result: dict[str, object] = {
        "schema_version": 1,
        "status": "analyzed-task9-one-question-dual-target-development",
        "scope": "one-question-development-feasibility",
        "cross_question_inference_status": "deferred-to-multi-question-development",
        "raw_admission": admitted,
        "requested_budget": args.requested_budget,
        "analyses": analyses,
    }
    result["analysis_sha256"] = _canonical_sha256(result)
    _publish(args.output, result)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
