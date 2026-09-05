#!/usr/bin/env python3
"""Authenticate and publish the canonical Task 6 20-mask analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from docprune.task6_analysis import analyze_task6_native_extension, publish_task6_analysis


def _load_gate(path: Path, expected_sha256: str) -> dict[str, object]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("Task 6 gate manifest must be an absolute regular file")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("Task 6 gate manifest SHA-256 mismatch")
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError("Task 6 gate manifest is not an object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--extension-root", type=Path, required=True)
    parser.add_argument("--gate-manifest", type=Path, required=True)
    parser.add_argument("--gate-manifest-sha256", required=True)
    parser.add_argument("--fixture-sha256", required=True)
    parser.add_argument("--native-job-id", required=True)
    parser.add_argument("--extension-job-id", required=True)
    parser.add_argument("--draws", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20_260_827)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gate = _load_gate(args.gate_manifest, args.gate_manifest_sha256)
    report = analyze_task6_native_extension(
        native_root=args.native_root,
        extension_root=args.extension_root,
        gate=gate,
        gate_sha256=args.gate_manifest_sha256,
        fixture_sha256=args.fixture_sha256,
        native_job_id=args.native_job_id,
        extension_job_id=args.extension_job_id,
        draws=args.draws,
        seed=args.seed,
    )
    file_sha256 = publish_task6_analysis(report, args.output)
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
