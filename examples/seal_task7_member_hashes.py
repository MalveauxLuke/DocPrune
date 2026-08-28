#!/usr/bin/env python3
"""Outcome-blindly seal the exact Task 7 shard-member byte identities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from docprune.task7_report_driver import seal_task7_member_hash_authority


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--fixture-sha256", required=True)
    parser.add_argument("--gate-manifest", type=Path, required=True)
    parser.add_argument("--gate-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scheduler-job-id")
    args = parser.parse_args()
    authority, file_sha256 = seal_task7_member_hash_authority(
        shard_root=args.shard_root,
        fixture_path=args.fixture,
        fixture_sha256=args.fixture_sha256,
        gate_path=args.gate_manifest,
        gate_sha256=args.gate_manifest_sha256,
        output_path=args.output,
        scheduler_job_id=args.scheduler_job_id,
    )
    print(
        json.dumps(
            {
                "status": authority["status"],
                "output": str(args.output),
                "member_count": len(authority["members"]),
                "member_manifest_sha256": authority["member_manifest_sha256"],
                "file_sha256": file_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
