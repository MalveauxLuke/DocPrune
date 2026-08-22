#!/usr/bin/env python3
"""Publish the paired top-4 checkpoint report without rerunning any samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from docprune.evaluation_shards import compare_paired_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-kept-run", type=Path, required=True)
    parser.add_argument("--docprune-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-questions", type=int, default=256)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"checkpoint report already exists: {args.output}")
    report = compare_paired_checkpoint(
        args.all_kept_run,
        args.docprune_run,
        expected_questions=args.expected_questions,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
