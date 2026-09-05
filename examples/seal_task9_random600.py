#!/usr/bin/env python3
"""Seal a 300-correct/300-wrong Task 9 cohort from cached top-4 results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from docprune.experiment_design import (
    _canonical_json_sha256,
    build_task9_preliminary_random_cohort,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--results-sha256", required=True)
    args = parser.parse_args()

    results = args.results.resolve()
    output = args.output.resolve()
    if not results.is_file() or results.is_symlink():
        raise ValueError(f"regular results file required: {results}")
    observed_sha256 = _sha256(results)
    if observed_sha256 != args.results_sha256:
        raise ValueError("results SHA-256 mismatch")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output already exists: {output}")

    rows = []
    with results.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSON object required at {results}:{line_number}")
            rows.append(value)

    cohort = build_task9_preliminary_random_cohort(
        rows,
        seed=args.seed,
        per_stratum=300,
    )
    cohort.pop("cohort_sha256")
    cohort["purpose"] = "Task 9 shared-probe stratified-random 600-question cohort"
    cohort["source_authority"] = {
        "results_path": str(results),
        "results_sha256": observed_sha256,
    }
    cohort["cohort_sha256"] = _canonical_json_sha256(cohort)

    output.parent.mkdir(parents=True, exist_ok=False)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(cohort, stream, indent=2, sort_keys=True)
        stream.write("\n")

    print(json.dumps({
        "output": str(output),
        "cohort_sha256": cohort["cohort_sha256"],
        "eligible_counts": cohort["eligible_counts"],
        "selected_counts": {
            name: len(qids) for name, qids in cohort["selected_qids"].items()
        },
    }, sort_keys=True))


if __name__ == "__main__":
    main()
