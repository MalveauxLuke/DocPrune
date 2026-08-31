#!/usr/bin/env python3
"""Authenticate and publish the Task 7 native-boundary diagnostic summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from docprune.task7_native_report import admit_task7_native_boundary_results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--fixture-sha256", required=True)
    parser.add_argument("--gate-manifest", type=Path, required=True)
    parser.add_argument("--gate-manifest-sha256", required=True)
    parser.add_argument("--native-boundary-manifest", type=Path, required=True)
    parser.add_argument("--native-boundary-manifest-sha256", required=True)
    parser.add_argument("--runtime-commit", required=True)
    parser.add_argument("--scheduler-job-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload, file_sha256 = admit_task7_native_boundary_results(
        shard_root=args.shard_root,
        fixture_path=args.fixture,
        fixture_sha256=args.fixture_sha256,
        gate_path=args.gate_manifest,
        gate_sha256=args.gate_manifest_sha256,
        native_boundary_path=args.native_boundary_manifest,
        native_boundary_sha256=args.native_boundary_manifest_sha256,
        runtime_commit=args.runtime_commit,
        scheduler_job_id=args.scheduler_job_id,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": str(args.output),
                "file_sha256": file_sha256,
                "analysis_sha256": payload["analysis_sha256"],
                "member_file_count": payload["member_file_count"],
                "summary": payload["summary"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
