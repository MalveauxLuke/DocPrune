#!/usr/bin/env python3
"""Seal 100 new baseline-wrong Task 9 questions from authenticated cached artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from docprune.experiment_design import (
    _canonical_json_sha256,
    build_task9_baseline_wrong_confirmation_cohort,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSON object required at {path}:{number}")
            rows.append(value)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cached-results", type=Path, required=True)
    parser.add_argument("--holdout-manifest", type=Path, required=True)
    parser.add_argument("--previous-cohort", type=Path, required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = tuple(path.resolve() for path in (
        args.cached_results,
        args.holdout_manifest,
        args.previous_cohort,
    ))
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output already exists: {output}")
    if any(not path.is_file() or path.is_symlink() for path in inputs):
        raise ValueError("confirmation cohort inputs must be regular files")

    cached_path, manifest_path, previous_path = inputs
    rows = _jsonl(cached_path)
    manifest = _json(manifest_path)
    previous = _json(previous_path)
    records = manifest.get("selected_records")
    previous_records = previous.get("selected_records")
    if (
        manifest.get("status") != "sealed"
        or manifest.get("required_n") != 1213
        or not isinstance(records, list)
        or len(records) != 1213
        or not isinstance(previous_records, list)
        or len(previous_records) != 48
    ):
        raise ValueError("Task 6 holdout or prior Task 9 cohort identity is invalid")
    row_by_qid = {row.get("question_id"): row for row in rows}
    record_by_qid = {record.get("qid"): record for record in records if isinstance(record, dict)}
    if (
        len(row_by_qid) != len(rows)
        or len(record_by_qid) != 1213
        or not set(record_by_qid).issubset(row_by_qid)
    ):
        raise ValueError("sealed holdout QIDs are not uniquely covered by cached results")
    row_by_qid = {qid: row_by_qid[qid] for qid in record_by_qid}
    rows = list(row_by_qid.values())
    for qid, row in row_by_qid.items():
        cached_pages = record_by_qid[qid].get("cached_pages")
        result_pages = row.get("retrieved_pages")
        if (
            not isinstance(cached_pages, list)
            or not isinstance(result_pages, list)
            or len(cached_pages) != 4
            or len(result_pages) != 4
            or [
                (page.get("doc_id"), page.get("page_index")) for page in cached_pages
            ]
            != [(page.get("doc_id"), page.get("page_index")) for page in result_pages]
        ):
            raise ValueError(f"cached top-4 page identity mismatch: {qid}")
    excluded = [
        record.get("question_id") for record in previous_records if isinstance(record, dict)
    ]
    cohort = build_task9_baseline_wrong_confirmation_cohort(
        rows,
        eligibility_records=records,
        excluded_qids=excluded,
        seed=args.seed,
        sample_size=100,
    )
    cohort.pop("cohort_sha256")
    cohort["source_authority"] = {
        "cached_results_path": str(cached_path),
        "cached_results_file_sha256": _sha256(cached_path),
        "holdout_manifest_path": str(manifest_path),
        "holdout_manifest_file_sha256": _sha256(manifest_path),
        "holdout_manifest_sha256": manifest.get("manifest_sha256"),
        "previous_cohort_path": str(previous_path),
        "previous_cohort_file_sha256": _sha256(previous_path),
        "previous_cohort_sha256": previous.get("cohort_sha256"),
    }
    cohort["cohort_sha256"] = _canonical_json_sha256(cohort)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(cohort, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({
        "output": str(output),
        "cohort_sha256": cohort["cohort_sha256"],
        "eligible_baseline_wrong_count": cohort["eligible_baseline_wrong_count"],
        "eligible_support_component_count": cohort["eligible_support_component_count"],
        "selection_used_document_fallback": cohort["selection_used_document_fallback"],
        "selected_count": len(cohort["selected_qids"]),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
