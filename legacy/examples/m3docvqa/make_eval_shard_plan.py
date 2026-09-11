#!/usr/bin/env python3
"""Persist the one reusable full-corpus evaluation shard plan."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from docprune.evaluation_shards import write_shard_plan


def _source_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"source questions contain a blank line at {line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"source question {line_number} is not an object")
            rows.append(value)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=64)
    parser.add_argument("--checkpoint-size", type=int, default=256)
    parser.add_argument("--expected-source-questions", type=int, default=2441)
    parser.add_argument("--seed-label", default="docprune-paired256-v1")
    args = parser.parse_args()
    plan = write_shard_plan(
        _source_rows(args.questions),
        args.output,
        source_questions_sha256=hashlib.sha256(args.questions.read_bytes()).hexdigest(),
        expected_source_questions=args.expected_source_questions,
        shard_size=args.shard_size,
        checkpoint_size=args.checkpoint_size,
        seed_label=args.seed_label,
    )
    print(json.dumps(plan, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
