#!/usr/bin/env python3
"""Run one configurable batch of the 100-question Task 9 confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from docprune.task9_attribution import _canonical_sha256


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _publish(path: Path, value: dict[str, object]) -> None:
    content = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    try:
        os.write(descriptor, content)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _run(command: list[str], env: dict[str, str]) -> None:
    subprocess.run(command, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-index", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--job-root", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--runtime-commit", required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--index-manifest", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--fixture-sha256", required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--cohort-file-sha256", required=True)
    parser.add_argument("--mappings-dir", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    paths = (
        args.job_root,
        args.runtime_dir,
        args.python,
        args.config,
        args.run_config,
        args.index_manifest,
        args.fixture,
        args.cohort,
        args.mappings_dir,
    )
    if (
        any(not path.is_absolute() for path in paths)
        or args.batch_index < 0
        or args.batch_size <= 0
        or len(args.runtime_commit) != 40
        or _sha256(args.fixture) != args.fixture_sha256
        or _sha256(args.cohort) != args.cohort_file_sha256
    ):
        raise ValueError("Task 9 confirmation batch arguments are invalid")
    cohort = _load(args.cohort)
    records = cohort.get("selected_records")
    if not isinstance(records, list) or len(records) != 100:
        raise ValueError("Task 9 confirmation cohort must contain 100 records")
    start = args.batch_index * args.batch_size
    selected = records[start : start + args.batch_size]
    if not selected:
        raise ValueError("Task 9 confirmation batch index is outside the cohort")

    prepared: list[dict[str, object]] = []
    for ordinal, record in enumerate(selected, start=start):
        qid = record.get("question_id") if isinstance(record, dict) else None
        matches = sorted(args.mappings_dir.glob(f"{ordinal:03d}-*{qid}*.json"))
        if not isinstance(qid, str) or len(matches) != 1:
            raise ValueError(f"Task 9 confirmation mapping identity is invalid: {ordinal}/{qid}")
        mapping_path = matches[0]
        mapping = _load(mapping_path)
        if (
            mapping.get("geometry_count") != len(mapping.get("geometry", []))
            or not isinstance(mapping.get("geometry_sha256"), str)
            or not isinstance(mapping.get("sha256"), str)
        ):
            raise ValueError(f"Task 9 confirmation mapping geometry is invalid: {qid}")
        prepared.append({
            "ordinal": ordinal,
            "qid": qid,
            "mapping_path": mapping_path,
            "mapping_sha256": _sha256(mapping_path),
            "mapping_internal_sha256": mapping["sha256"],
            "geometry_count": mapping["geometry_count"],
            "geometry_sha256": mapping["geometry_sha256"],
        })
    if args.validate_only:
        print(json.dumps({
            "status": "validated-task9-confirmation-batch-without-model",
            "batch_index": args.batch_index,
            "ordinals": [row["ordinal"] for row in prepared],
            "qids": [row["qid"] for row in prepared],
            "fit_mask_count": 256,
            "holdout_mask_count": 0,
        }, sort_keys=True, separators=(",", ":")))
        return
    if not args.job_root.is_dir() or any(args.job_root.iterdir()):
        raise ValueError("Task 9 confirmation job root must be fresh and empty")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(args.runtime_dir / "src")
    results: list[dict[str, object]] = []
    for item in prepared:
        ordinal, qid = item["ordinal"], item["qid"]
        question_root = args.job_root / f"{ordinal:03d}-{qid}"
        question_root.mkdir()
        raw_root = question_root / "raw"
        _run([
            str(args.python),
            str(args.runtime_dir / "examples/run_task9_regional_development.py"),
            "--config", str(args.config),
            "--run-config", str(args.run_config),
            "--index-manifest", str(args.index_manifest),
            "--fixture", str(args.fixture),
            "--fixture-sha256", args.fixture_sha256,
            "--preliminary-cohort", str(args.cohort),
            "--preliminary-cohort-sha256", args.cohort_file_sha256,
            "--mapping", str(item["mapping_path"]),
            "--mapping-sha256", str(item["mapping_sha256"]),
            "--expected-geometry-count", str(item["geometry_count"]),
            "--expected-geometry-sha256", str(item["geometry_sha256"]),
            "--qid", str(qid),
            "--boundary", "dynamic",
            "--fit-mask-count", "256",
            "--holdout-mask-count", "0",
            "--budget-local-holdout-mask-count", "0",
            "--output", str(raw_root),
            "--runtime-dir", str(args.runtime_dir),
            "--runtime-commit", args.runtime_commit,
        ], env)
        analysis_path = question_root / "analysis.json"
        _run([
            str(args.python),
            str(args.runtime_dir / "examples/analyze_task9_confirmation_attribution.py"),
            "--root", str(raw_root),
            "--output", str(analysis_path),
            "--decoder-layer-count", "28",
        ], env)
        selected_path = question_root / "selected-arms.json"
        _run([
            str(args.python),
            str(args.runtime_dir / "examples/run_task9_preliminary_selected_arms.py"),
            "--raw-root", str(raw_root),
            "--analysis", str(analysis_path),
            "--output", str(selected_path),
            "--runtime-dir", str(args.runtime_dir),
            "--runtime-commit", args.runtime_commit,
        ], env)
        results.append({
            "ordinal": ordinal,
            "qid": qid,
            "raw_root": str(raw_root),
            "analysis_path": str(analysis_path),
            "selected_arms_path": str(selected_path),
            "selected_arms_file_sha256": _sha256(selected_path),
        })
    completion: dict[str, object] = {
        "schema_version": "docprune-task9-confirmation-batch-completion-v1",
        "status": "complete",
        "batch_index": args.batch_index,
        "batch_size": args.batch_size,
        "runtime_commit": args.runtime_commit,
        "results": results,
    }
    completion["completion_sha256"] = _canonical_sha256(completion)
    _publish(args.job_root / "completion-manifest.json", completion)
    print(json.dumps(completion, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
