#!/usr/bin/env python3
"""Authenticate one completed Task 9 regional-likelihood smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from docprune.task9_live import admit_task9_regional_smoke


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runtime-commit", required=True)
    parser.add_argument("--qid", required=True)
    parser.add_argument("--boundary", required=True)
    parser.add_argument("--split", choices=("fit", "holdout"), required=True)
    parser.add_argument("--mask-count", type=int, required=True)
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
    args = parser.parse_args()
    result = admit_task9_regional_smoke(
        args.root,
        expected_runtime_commit=args.runtime_commit,
        expected_qid=args.qid,
        expected_boundary=args.boundary,
        expected_split=args.split,
        expected_mask_count=args.mask_count,
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
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
