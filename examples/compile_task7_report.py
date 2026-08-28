#!/usr/bin/env python3
"""Validate and publish the canonical artifact-only Task 7 report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from docprune.task7_report_driver import compile_task7_report_from_shards


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--fixture-sha256", required=True)
    parser.add_argument("--gate-manifest", type=Path, required=True)
    parser.add_argument("--gate-manifest-sha256", required=True)
    parser.add_argument("--expected-members", type=Path, required=True)
    parser.add_argument("--expected-members-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20_260_827)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="build, validate, and discard the same private snapshot without publishing it",
    )
    args = parser.parse_args()
    report = compile_task7_report_from_shards(
        shard_root=args.shard_root,
        fixture_path=args.fixture,
        fixture_sha256=args.fixture_sha256,
        gate_path=args.gate_manifest,
        gate_sha256=args.gate_manifest_sha256,
        expected_members_path=args.expected_members,
        expected_members_sha256=args.expected_members_sha256,
        output_path=args.output,
        draws=args.draws,
        seed=args.seed,
        validate_only=args.validate_only,
    )
    print(
        json.dumps(
            {
                "status": "validated-only" if args.validate_only else "published",
                "bundle": str(args.output),
                "report": None if args.validate_only else str(args.output / "report.json"),
                "qid_count": report["member_count"],
                "canonical_report_sha256": report["canonical_report_sha256"],
                "analysis_report_sha256": report["analysis"]["report_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
