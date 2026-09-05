#!/usr/bin/env python3
"""Capture the authenticated Task 8 post-BTP+QTP token geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from docprune.m3docvqa_factory import (
    load_pinned_colpali_query_encoder,
    load_pinned_qwen_processor,
)
from docprune.task8_geometry import capture_task8_btp_qtp_geometry


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--fixture-sha256", required=True)
    parser.add_argument("--smoke-input-manifest", type=Path, required=True)
    parser.add_argument("--smoke-input-manifest-sha256", required=True)
    parser.add_argument("--reference-results", type=Path, required=True)
    parser.add_argument("--reference-results-sha256", required=True)
    parser.add_argument("--qid", required=True)
    parser.add_argument("--expected-geometry-count", type=int, required=True)
    parser.add_argument("--expected-geometry-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-commit", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    query_encoder = load_pinned_colpali_query_encoder()
    qwen_processor = load_pinned_qwen_processor()
    payload = capture_task8_btp_qtp_geometry(
        fixture_path=args.fixture,
        fixture_sha256=args.fixture_sha256,
        smoke_input_manifest_path=args.smoke_input_manifest,
        smoke_input_manifest_sha256=args.smoke_input_manifest_sha256,
        reference_results_path=args.reference_results,
        reference_results_sha256=args.reference_results_sha256,
        qid=args.qid,
        expected_geometry_count=args.expected_geometry_count,
        expected_geometry_sha256=args.expected_geometry_sha256,
        output_path=args.output,
        runtime_commit=args.runtime_commit,
        query_encoder=query_encoder,
        qwen_processor=qwen_processor,
    )
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
