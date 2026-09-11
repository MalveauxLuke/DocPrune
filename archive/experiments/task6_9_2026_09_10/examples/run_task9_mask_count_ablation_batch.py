#!/usr/bin/env python3
"""Run one four-question CPU batch of the Task 9 mask-count ablation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from docprune.task9_attribution import _canonical_sha256

_UNIFIED = Path(
    "/scratch/lmalveau/docprune/task9-preliminary-dynamic48-90f27d7-unified-v1/analysis.json"
)
_UNIFIED_SHA256 = "985a837b2094b5a925730eeb4b9bf07c594344f9021c4a718c49c11aa655a8c5"


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _publish(path: Path, value: dict[str, object]) -> None:
    content = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    try:
        os.write(descriptor, content)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-index", type=int, required=True)
    parser.add_argument("--job-root", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--runtime-commit", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if (
        args.batch_index not in range(12)
        or not args.job_root.is_absolute()
        or not args.runtime_dir.is_absolute()
        or len(args.runtime_commit) != 40
    ):
        raise ValueError("mask-count batch arguments are invalid")
    unified = _load(_UNIFIED)
    if unified.get("analysis_sha256") != _UNIFIED_SHA256:
        raise ValueError("unified preliminary analysis identity changed")
    unsigned = dict(unified)
    unsigned.pop("analysis_sha256")
    if _canonical_sha256(unsigned) != _UNIFIED_SHA256 or len(unified["questions"]) != 48:
        raise ValueError("unified preliminary analysis is invalid")
    start = args.batch_index * 4
    questions = unified["questions"][start : start + 4]
    prepared = [
        {
            "ordinal": row["ordinal"],
            "qid": row["question_id"],
            "question_root": str(Path(row["source"]["selected_arms_path"]).parent),
        }
        for row in questions
    ]
    if args.validate_only:
        if any(not Path(row["question_root"]).is_dir() for row in prepared):
            raise ValueError("mask-count batch input root is missing")
        print(json.dumps({"status": "validated", "questions": prepared}, sort_keys=True))
        return
    if not args.job_root.is_dir() or any(args.job_root.iterdir()):
        raise ValueError("mask-count batch job root must be fresh and empty")

    def run(row: dict[str, object]) -> dict[str, object]:
        output = args.job_root / f"{row['ordinal']:02d}-{row['qid']}.json"
        subprocess.run(
            [
                sys.executable,
                str(args.runtime_dir / "examples/analyze_task9_mask_count_ablation.py"),
                "--question-root",
                str(row["question_root"]),
                "--output",
                str(output),
            ],
            check=True,
            env={
                **os.environ,
                "PYTHONPATH": os.pathsep.join(
                    (str(args.runtime_dir / "src"), os.environ.get("PYTHONPATH", ""))
                ),
            },
        )
        artifact = _load(output)
        return {
            "ordinal": row["ordinal"],
            "qid": row["qid"],
            "output": str(output),
            "artifact_sha256": artifact["artifact_sha256"],
        }

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = sorted(executor.map(run, prepared), key=lambda row: row["ordinal"])
    completion: dict[str, object] = {
        "schema_version": "docprune-task9-mask-count-ablation-batch-v1",
        "status": "complete",
        "batch_index": args.batch_index,
        "runtime_commit": args.runtime_commit,
        "results": results,
    }
    completion["completion_sha256"] = _canonical_sha256(completion)
    _publish(args.job_root / "completion-manifest.json", completion)
    print(json.dumps(completion, sort_keys=True))


if __name__ == "__main__":
    main()
