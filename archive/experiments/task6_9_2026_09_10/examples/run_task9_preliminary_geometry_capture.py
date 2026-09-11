#!/usr/bin/env python3
"""Capture one Task 9 preliminary-cohort post-BTP+QTP geometry artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from docprune.m3docvqa_factory import (
    load_pinned_colpali_query_encoder,
    load_pinned_qwen_processor,
)
from docprune.task8_geometry import capture_task8_btp_qtp_geometry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--fixture-sha256", required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--input-manifest-sha256", required=True)
    parser.add_argument("--source-results", type=Path, required=True)
    parser.add_argument("--source-results-sha256", required=True)
    parser.add_argument("--qid", required=True)
    parser.add_argument("--expected-geometry-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-commit", required=True)
    args = parser.parse_args()
    payload = capture_task8_btp_qtp_geometry(
        fixture_path=args.fixture,
        fixture_sha256=args.fixture_sha256,
        smoke_input_manifest_path=args.input_manifest,
        smoke_input_manifest_sha256=args.input_manifest_sha256,
        reference_results_path=args.source_results,
        reference_results_sha256=args.source_results_sha256,
        qid=args.qid,
        expected_geometry_count=args.expected_geometry_count,
        expected_geometry_sha256=None,
        output_path=args.output,
        runtime_commit=args.runtime_commit,
        query_encoder=load_pinned_colpali_query_encoder(),
        qwen_processor=load_pinned_qwen_processor(),
    )
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
