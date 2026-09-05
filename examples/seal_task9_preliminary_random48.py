#!/usr/bin/env python3
"""Seal the Task 9 preliminary 24-correct/24-wrong development cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from docprune.experiment_design import build_task9_preliminary_random_cohort


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSON object required at {path}:{line_number}")
            rows.append(value)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", required=True)
    args = parser.parse_args()

    analysis_path = args.analysis.resolve()
    output_path = args.output.resolve()
    if output_path.exists() or output_path.is_symlink():
        raise SystemExit(f"output already exists: {output_path}")

    analysis = _load_json(analysis_path)
    question_ids = analysis.get("question_ids")
    source_roots = analysis.get("source_roots")
    if (
        analysis.get("question_count") != 245
        or not isinstance(question_ids, list)
        or len(question_ids) != 245
        or len(set(question_ids)) != 245
        or analysis.get("retrieval_is_identical_across_stages") is not True
        or not isinstance(source_roots, dict)
    ):
        raise ValueError("stage245 analysis is not the authenticated 245-question pool")

    first64 = source_roots.get("btp_qtp_first64")
    incremental_root = source_roots.get("incremental")
    if (
        not isinstance(first64, list)
        or not first64
        or any(not isinstance(path, str) for path in first64)
        or not isinstance(incremental_root, str)
    ):
        raise ValueError("stage245 BTP+QTP result roots are invalid")
    result_paths = [Path(path).resolve() for path in first64]
    result_paths.extend(
        sorted(
            Path(incremental_root).resolve().glob(
                "btp-qtp/shard-*/run/results.jsonl"
            )
        )
    )
    if len(set(result_paths)) != len(result_paths):
        raise ValueError("duplicate BTP+QTP result path")

    rows: list[dict[str, object]] = []
    result_files: list[dict[str, str]] = []
    for path in result_paths:
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"regular BTP+QTP result file required: {path}")
        rows.extend(_load_jsonl(path))
        result_files.append({"path": str(path), "sha256": _sha256(path)})
    observed_qids = [row.get("question_id") for row in rows]
    if len(observed_qids) != 245 or set(observed_qids) != set(question_ids):
        raise ValueError("BTP+QTP result QIDs do not match the stage245 analysis")

    cohort = build_task9_preliminary_random_cohort(rows, seed=args.seed, per_stratum=24)
    if cohort["eligible_counts"] != {"baseline_correct": 90, "baseline_wrong": 155}:
        raise ValueError("canonical BTP+QTP baseline strata are not the frozen 90/155 pools")
    cohort.pop("cohort_sha256", None)
    cohort["source_authority"] = {
        "stage_analysis_path": str(analysis_path),
        "stage_analysis_file_sha256": _sha256(analysis_path),
        "stage_analysis_sha256": analysis.get("analysis_sha256"),
        "question_ids_sha256": analysis.get("question_ids_sha256"),
        "result_files": result_files,
    }
    cohort["cohort_sha256"] = _canonical_sha256(cohort)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("x", encoding="utf-8") as handle:
            json.dump(cohort, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as error:
        raise SystemExit(f"output already exists: {output_path}") from error

    print(
        json.dumps(
            {
                "output": str(output_path),
                "cohort_sha256": cohort["cohort_sha256"],
                "eligible_counts": cohort["eligible_counts"],
                "selected_counts": {
                    name: len(qids) for name, qids in cohort["selected_qids"].items()
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
