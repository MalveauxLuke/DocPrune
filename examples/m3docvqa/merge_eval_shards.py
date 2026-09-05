#!/usr/bin/env python3
"""Merge already validated shard outputs into one canonical evaluation run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from docprune.evaluation_shards import merge_evaluation_shards


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-questions", type=int, required=True)
    parser.add_argument("--allow-fixture", action="store_true")
    args = parser.parse_args()
    manifest = merge_evaluation_shards(
        plan_path=args.plan,
        shard_root=args.shard_root,
        output_dir=args.output,
        expected_questions=args.expected_questions,
        allow_fixture=args.allow_fixture,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
