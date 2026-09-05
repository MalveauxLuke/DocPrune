#!/usr/bin/env python3
"""Publish the authenticated Task 9 shared-probe cohort without replacement."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from docprune.task9_shared_probe import (
    build_task9_shared_probe_cohort,
    publish_task9_shared_probe_cohort,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSON object required at {path}:{line_number}")
            rows.append(value)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cached-results", type=Path, required=True)
    parser.add_argument("--preliminary-cohort", type=Path, required=True)
    parser.add_argument("--confirmation-cohort", type=Path, required=True)
    parser.add_argument("--eligibility-records", type=Path)
    parser.add_argument("--split-counts", type=Path, help="Optional JSON split contract for CPU fixtures")
    parser.add_argument("--selection-seed", "--seed", dest="selection_seed", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = tuple(path.resolve() for path in (args.cached_results, args.preliminary_cohort, args.confirmation_cohort))
    if args.eligibility_records is not None:
        paths += (args.eligibility_records.resolve(),)
    if args.split_counts is not None:
        paths += (args.split_counts.resolve(),)
    if any(not path.is_file() or path.is_symlink() for path in paths):
        raise ValueError("cohort inputs must be regular files")
    rows = _jsonl(paths[0])
    preliminary = _json_object(paths[1])
    confirmation = _json_object(paths[2])
    eligibility = None
    if args.eligibility_records is not None:
        eligibility_value = json.loads(args.eligibility_records.resolve().read_text(encoding="utf-8"))
        if isinstance(eligibility_value, dict) and isinstance(eligibility_value.get("eligible_records"), list):
            eligibility = eligibility_value["eligible_records"]
        elif isinstance(eligibility_value, list):
            eligibility = eligibility_value
        else:
            eligibility = _jsonl(args.eligibility_records.resolve())
    split_counts = _json_object(args.split_counts.resolve()) if args.split_counts is not None else None
    input_hashes = {str(path): _sha256(path) for path in paths}
    cohort = build_task9_shared_probe_cohort(
        rows,
        preliminary_cohort=preliminary,
        confirmation_cohort=confirmation,
        selection_seed=args.selection_seed,
        split_counts=split_counts,
        eligibility_records=eligibility,
        input_file_hashes=input_hashes,
    )
    digest = publish_task9_shared_probe_cohort(cohort, args.output.resolve())
    print(json.dumps({
        "output": str(args.output.resolve()),
        "cohort_sha256": digest,
        "question_count": cohort["question_count"],
        "eligible_component_count": cohort["eligible_component_count"],
        "selected_component_count": cohort["selected_component_count"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
