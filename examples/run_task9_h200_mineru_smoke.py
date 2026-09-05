#!/usr/bin/env python3
"""Prepare or run the authenticated one-page Task 9 H200 MinerU smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from docprune.task9_h200_mineru import (
    prepare_task9_one_page_mineru_smoke,
    run_task9_one_page_mineru_smoke,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="phase", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--source-fixture", type=Path, required=True)
    prepare.add_argument("--source-fixture-sha256", required=True)
    prepare.add_argument("--source-smoke-manifest", type=Path, required=True)
    prepare.add_argument("--source-model-snapshot", type=Path, required=True)
    prepare.add_argument("--model-dir", type=Path, required=True)
    prepare.add_argument("--job-root", type=Path, required=True)
    prepare.add_argument("--runtime-commit", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--job-root", type=Path, required=True)
    run.add_argument("--mineru-executable", type=Path, required=True)
    run.add_argument("--gpu-id", type=int, required=True)
    run.add_argument("--gpu-uuid", required=True)
    args = parser.parse_args()

    if args.phase == "prepare":
        result = prepare_task9_one_page_mineru_smoke(
            source_fixture_path=args.source_fixture,
            source_fixture_sha256=args.source_fixture_sha256,
            source_smoke_manifest_path=args.source_smoke_manifest,
            source_model_snapshot=args.source_model_snapshot,
            model_dir=args.model_dir,
            job_root=args.job_root,
            runtime_commit=args.runtime_commit,
        )
    else:
        result = run_task9_one_page_mineru_smoke(
            job_root=args.job_root,
            mineru_executable=args.mineru_executable,
            gpu_id=args.gpu_id,
            gpu_uuid=args.gpu_uuid,
        )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
