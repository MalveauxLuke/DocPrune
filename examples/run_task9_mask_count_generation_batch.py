#!/usr/bin/env python3
"""Run one four-question GPU batch for the Task 9 192-mask response check."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from docprune.task9_attribution import _canonical_sha256

_PILOT = Path(
    "/scratch/lmalveau/docprune/task9-preliminary-dynamic48-90f27d7-unified-v1/analysis.json"
)
_PILOT_SHA256 = "985a837b2094b5a925730eeb4b9bf07c594344f9021c4a718c49c11aa655a8c5"
_ABLATION = Path("/scratch/lmalveau/docprune/task9-mask-count-ablation-d6de9d8-v1/analysis.json")
_ABLATION_SHA256 = "9d6a9ac6ece6712ca9cdad33bba0bc61caf122121435c3d99a6fb40e460edc51"


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
        raise ValueError("mask-count generation batch arguments are invalid")
    pilot = _load(_PILOT)
    ablation = _load(_ABLATION)
    if (
        pilot.get("analysis_sha256") != _PILOT_SHA256
        or ablation.get("analysis_sha256") != _ABLATION_SHA256
    ):
        raise ValueError("mask-count generation inputs changed")
    start = args.batch_index * 4
    questions = pilot["questions"][start : start + 4]
    prepared = [
        {
            "ordinal": row["ordinal"],
            "qid": row["question_id"],
            "question_root": Path(row["source"]["selected_arms_path"]).parent,
        }
        for row in questions
    ]
    if len(prepared) != 4:
        raise ValueError("mask-count generation batch must contain four questions")
    if not args.validate_only and (not args.job_root.is_dir() or any(args.job_root.iterdir())):
        raise ValueError("mask-count generation job root must be fresh and empty")

    results: list[dict[str, object]] = []
    for row in prepared:
        output = args.job_root / f"{row['ordinal']:02d}-{row['qid']}.json"
        command = [
            "/home/lmalveau/mamba-envs/docprune-sol/bin/python",
            str(args.runtime_dir / "examples/run_task9_preliminary_selected_arms.py"),
            "--raw-root",
            str(row["question_root"] / "raw"),
            "--analysis",
            str(row["question_root"] / "analysis.json"),
            "--output",
            str(output),
            "--runtime-dir",
            str(args.runtime_dir),
            "--runtime-commit",
            args.runtime_commit,
            "--mask-count-analysis",
            str(_ABLATION),
            "--mask-count-analysis-sha256",
            _ABLATION_SHA256,
            "--mask-count",
            "192",
        ]
        if args.validate_only:
            command.append("--validate-only")
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        if args.validate_only:
            results.append(json.loads(completed.stdout))
        else:
            artifact = _load(output)
            results.append(
                {
                    "ordinal": row["ordinal"],
                    "qid": row["qid"],
                    "output": str(output),
                    "selected_arms_sha256": artifact["selected_arms_sha256"],
                }
            )
    if args.validate_only:
        print(json.dumps({"status": "validated", "results": results}, sort_keys=True))
        return
    completion: dict[str, object] = {
        "schema_version": "docprune-task9-mask-count-generation-batch-v1",
        "status": "complete",
        "batch_index": args.batch_index,
        "mask_count": 192,
        "runtime_commit": args.runtime_commit,
        "results": results,
    }
    completion["completion_sha256"] = _canonical_sha256(completion)
    _publish(args.job_root / "completion-manifest.json", completion)
    print(json.dumps(completion, sort_keys=True))


if __name__ == "__main__":
    main()
