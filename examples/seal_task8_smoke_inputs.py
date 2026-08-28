#!/usr/bin/env python3
"""Seal or validate exact fixed-page image inputs for the Task 8 MinerU smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from docprune.task8_runtime import load_task8_smoke_inputs, seal_task8_smoke_inputs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--fixture-sha256")
    parser.add_argument("--qid")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--runtime-commit")
    parser.add_argument("--validate-manifest", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.validate_manifest is not None:
        if any(
            value is not None
            for value in (
                args.fixture,
                args.fixture_sha256,
                args.qid,
                args.output_root,
                args.runtime_commit,
            )
        ):
            raise SystemExit("--validate-manifest cannot be combined with seal arguments")
        manifest = load_task8_smoke_inputs(args.validate_manifest)
    else:
        if any(
            value is None
            for value in (
                args.fixture,
                args.fixture_sha256,
                args.qid,
                args.output_root,
                args.runtime_commit,
            )
        ):
            raise SystemExit("all seal arguments are required")
        manifest = seal_task8_smoke_inputs(
            fixture_path=args.fixture,
            fixture_sha256=args.fixture_sha256,
            qid=args.qid,
            output_root=args.output_root,
            runtime_commit=args.runtime_commit,
        )
    print(json.dumps(manifest, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
