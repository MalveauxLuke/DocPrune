#!/usr/bin/env python3
"""Run one four-question batch of the Task 9 preliminary random pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from docprune.task9_attribution import _canonical_sha256

_SOL_PYTHON = Path("/home/lmalveau/mamba-envs/docprune-sol/bin/python")
_CONTEXTCITE_PYTHON = Path(
    "/scratch/lmalveau/docprune/tool-envs/random-coverage-attribution-v2/contextcite/bin/python"
)
_SOL_SITE_PACKAGES = Path("/home/lmalveau/mamba-envs/docprune-sol/lib/python3.10/site-packages")
_FIXTURE = Path(
    "/scratch/lmalveau/docprune/task9-preliminary-mapping-inputs-5f7ff93-v1/fixture.json"
)
_FIXTURE_SHA256 = "704ca552c79263a6398a596d82dccecc0c64b07b03dec5e1acbcb13ac686b930"
_COHORT = Path("/scratch/lmalveau/docprune/task9-preliminary-random48-v1/cohort.json")
_COHORT_SHA256 = "123607a6a1226b4e3436f43cb82d45e64e8a3008e9ab6deefd7526efeabd0273"
_MAPPINGS = Path("/scratch/lmalveau/docprune/task9-preliminary-mappings-ecd1a87-v1")
_ATTEMPT = Path("/scratch/lmalveau/docprune/benchmark-4e2473b/attempt-2")
_RUN_CONFIG = _ATTEMPT / "run-configs/docprune-top4-task6-clean-m3-v1.json"
_INDEX_MANIFEST = _ATTEMPT / "indexes/docprune/top4/docprune/manifest.json"
_REUSE_Q0_ROOT = Path("/scratch/lmalveau/docprune/task9-preliminary-attribution-smoke-d66a523-v1")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
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


def _run(command: list[str], *, env: dict[str, str]) -> None:
    subprocess.run(command, check=True, env=env)


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
        raise ValueError("Task 9 preliminary batch arguments are invalid")
    if _sha256(_FIXTURE) != _FIXTURE_SHA256 or _sha256(_COHORT) != _COHORT_SHA256:
        raise ValueError("Task 9 preliminary batch fixture or cohort changed")
    cohort = _load(_COHORT)
    records = cohort.get("selected_records")
    if not isinstance(records, list) or len(records) != 48:
        raise ValueError("Task 9 preliminary cohort must contain 48 records")
    start = args.batch_index * 4
    selected = records[start : start + 4]
    if len(selected) != 4:
        raise ValueError("Task 9 preliminary batch must contain four questions")

    prepared: list[dict[str, object]] = []
    for ordinal, record in enumerate(selected, start=start):
        qid = record.get("question_id") if isinstance(record, dict) else None
        matches = sorted(_MAPPINGS.glob(f"{ordinal:02d}-{qid}.json"))
        if not isinstance(qid, str) or len(matches) != 1:
            raise ValueError("Task 9 preliminary batch mapping identity is invalid")
        mapping_path = matches[0]
        mapping = _load(mapping_path)
        if (
            mapping.get("geometry_count") != len(mapping.get("geometry", []))
            or not isinstance(mapping.get("geometry_sha256"), str)
            or not isinstance(mapping.get("sha256"), str)
        ):
            raise ValueError("Task 9 preliminary batch mapping geometry is invalid")
        prepared.append(
            {
                "ordinal": ordinal,
                "qid": qid,
                "mapping_path": mapping_path,
                "mapping_sha256": _sha256(mapping_path),
                "mapping_internal_sha256": mapping["sha256"],
                "geometry_count": mapping["geometry_count"],
                "geometry_sha256": mapping["geometry_sha256"],
            }
        )
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "validated-task9-preliminary-batch-without-model",
                    "batch_index": args.batch_index,
                    "ordinals": [row["ordinal"] for row in prepared],
                    "qids": [row["qid"] for row in prepared],
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return
    if not args.job_root.is_dir() or any(args.job_root.iterdir()):
        raise ValueError("Task 9 preliminary batch job root must be fresh and empty")

    base_env = dict(os.environ)
    base_env["PYTHONPATH"] = str(args.runtime_dir / "src")
    analysis_env = dict(base_env)
    analysis_env["PYTHONPATH"] = os.pathsep.join(
        (str(args.runtime_dir / "src"), str(_SOL_SITE_PACKAGES))
    )
    config = args.runtime_dir / "configs/docprune-m3docvqa.toml"
    results: list[dict[str, object]] = []
    for item in prepared:
        ordinal = item["ordinal"]
        qid = item["qid"]
        question_root = args.job_root / f"{ordinal:02d}-{qid}"
        question_root.mkdir()
        if ordinal == 0:
            raw_root = _REUSE_Q0_ROOT / "output"
            analysis_path = _REUSE_Q0_ROOT / "analysis.json"
        else:
            raw_root = question_root / "raw"
            _run(
                [
                    str(_SOL_PYTHON),
                    str(args.runtime_dir / "examples/run_task9_regional_development.py"),
                    "--config",
                    str(config),
                    "--run-config",
                    str(_RUN_CONFIG),
                    "--index-manifest",
                    str(_INDEX_MANIFEST),
                    "--fixture",
                    str(_FIXTURE),
                    "--fixture-sha256",
                    _FIXTURE_SHA256,
                    "--preliminary-cohort",
                    str(_COHORT),
                    "--preliminary-cohort-sha256",
                    _COHORT_SHA256,
                    "--mapping",
                    str(item["mapping_path"]),
                    "--mapping-sha256",
                    str(item["mapping_sha256"]),
                    "--expected-geometry-count",
                    str(item["geometry_count"]),
                    "--expected-geometry-sha256",
                    str(item["geometry_sha256"]),
                    "--qid",
                    str(qid),
                    "--boundary",
                    "13",
                    "--fit-mask-count",
                    "256",
                    "--holdout-mask-count",
                    "32",
                    "--budget-local-holdout-mask-count",
                    "32",
                    "--output",
                    str(raw_root),
                    "--runtime-dir",
                    str(args.runtime_dir),
                    "--runtime-commit",
                    args.runtime_commit,
                ],
                env=base_env,
            )
            analysis_path = question_root / "analysis.json"
            _run(
                [
                    str(_CONTEXTCITE_PYTHON),
                    str(args.runtime_dir / "examples/analyze_task9_preliminary_attribution.py"),
                    "--root",
                    str(raw_root),
                    "--output",
                    str(analysis_path),
                    "--decoder-layer-count",
                    "28",
                ],
                env=analysis_env,
            )
        selected_path = question_root / "selected-arms.json"
        _run(
            [
                str(_SOL_PYTHON),
                str(args.runtime_dir / "examples/run_task9_preliminary_selected_arms.py"),
                "--raw-root",
                str(raw_root),
                "--analysis",
                str(analysis_path),
                "--output",
                str(selected_path),
                "--runtime-dir",
                str(args.runtime_dir),
                "--runtime-commit",
                args.runtime_commit,
            ],
            env=base_env,
        )
        results.append(
            {
                "ordinal": ordinal,
                "qid": qid,
                "raw_root": str(raw_root),
                "analysis_path": str(analysis_path),
                "selected_arms_path": str(selected_path),
                "selected_arms_file_sha256": _sha256(selected_path),
            }
        )
    completion: dict[str, object] = {
        "schema_version": "docprune-task9-preliminary-batch-completion-v1",
        "status": "complete",
        "batch_index": args.batch_index,
        "runtime_commit": args.runtime_commit,
        "results": results,
    }
    completion["completion_sha256"] = _canonical_sha256(completion)
    _publish(args.job_root / "completion-manifest.json", completion)
    print(json.dumps(completion, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
