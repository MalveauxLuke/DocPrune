#!/usr/bin/env python3
"""Authenticate and publish the descriptive Task 6 fixed-retention curve."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from docprune.task6_analysis import (
    analyze_task6_fixed_sensitivity,
    publish_task6_fixed_analysis,
)


def _load_json(path: Path, expected_sha256: str, label: str) -> dict[str, object]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be an absolute regular file")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch")
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not an object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-root", type=Path, required=True)
    parser.add_argument("--gate-manifest", type=Path, required=True)
    parser.add_argument("--gate-manifest-sha256", required=True)
    parser.add_argument("--fixture-sha256", required=True)
    parser.add_argument("--trigger-analysis", type=Path, required=True)
    parser.add_argument("--trigger-analysis-sha256", required=True)
    parser.add_argument("--fixed-job-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gate = _load_json(args.gate_manifest, args.gate_manifest_sha256, "Task 6 gate manifest")
    trigger = _load_json(
        args.trigger_analysis,
        args.trigger_analysis_sha256,
        "Task 6 r20 trigger analysis",
    )
    report = analyze_task6_fixed_sensitivity(
        fixed_root=args.fixed_root,
        gate=gate,
        gate_sha256=args.gate_manifest_sha256,
        fixture_sha256=args.fixture_sha256,
        fixed_job_id=args.fixed_job_id,
        trigger_analysis=trigger,
        trigger_file_sha256=args.trigger_analysis_sha256,
    )
    file_sha256 = publish_task6_fixed_analysis(report, args.output)
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(args.output),
                "analysis_sha256": report["analysis_sha256"],
                "file_sha256": file_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
