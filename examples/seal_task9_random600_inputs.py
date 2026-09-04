#!/usr/bin/env python3
"""Materialize the Task 9 random-600 cohort as retrieval-free fixed-page inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from docprune.experiment_design import _canonical_json_sha256
from docprune.task6_runtime import build_fixed_page_fixture, publish_fixed_page_fixture
from docprune.task8_runtime import seal_task8_smoke_inputs


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--cohort-file-sha256", required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--splits-file-sha256", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--results-sha256", required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--pdf-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--runtime-commit", required=True)
    args = parser.parse_args()

    cohort_path = args.cohort.resolve()
    splits_path = args.splits.resolve()
    source_path = args.source.resolve()
    results_path = args.results.resolve()
    root = args.output_root.absolute()
    for path, expected, label in (
        (cohort_path, args.cohort_file_sha256, "cohort"),
        (splits_path, args.splits_file_sha256, "splits"),
        (source_path, args.source_sha256, "source"),
        (results_path, args.results_sha256, "results"),
    ):
        if not path.is_file() or path.is_symlink() or _sha256(path) != expected:
            raise ValueError(f"Task 9 random-600 {label} identity mismatch")
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"Task 9 random-600 input root already exists: {root}")

    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    unsigned = dict(cohort)
    cohort_sha = unsigned.pop("cohort_sha256", None)
    if cohort_sha != _canonical_json_sha256(unsigned):
        raise ValueError("Task 9 random-600 cohort checksum is invalid")
    selected_qids = [
        *cohort["selected_qids"]["baseline_correct"],
        *cohort["selected_qids"]["baseline_wrong"],
    ]
    records = cohort.get("selected_records")
    if len(selected_qids) != 600 or not isinstance(records, list) or len(records) != 600:
        raise ValueError("Task 9 random-600 cohort must contain 600 records")
    by_qid = {record.get("question_id"): record for record in records if isinstance(record, dict)}
    if set(by_qid) != set(selected_qids):
        raise ValueError("Task 9 random-600 selected record identity mismatch")

    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    unsigned_splits = dict(splits)
    splits_sha = unsigned_splits.pop("manifest_sha256", None)
    if splits_sha != _canonical_json_sha256(unsigned_splits):
        raise ValueError("Task 9 random-600 split checksum is invalid")
    split_by_qid: dict[str, str] = {}
    for split, payload in splits["splits"].items():
        for qid in payload["qids"]:
            if qid in split_by_qid:
                raise ValueError(f"duplicate split QID: {qid}")
            split_by_qid[qid] = split
    if set(split_by_qid) != set(selected_qids):
        raise ValueError("split manifest is not an exact cohort partition")

    supports: dict[str, list[str]] = {}
    with source_path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            qid = row.get("qid")
            if qid not in by_qid:
                continue
            context = row.get("supporting_context")
            if not isinstance(context, list) or not context:
                raise ValueError(f"missing source support context: {qid}")
            supports[qid] = list(dict.fromkeys(entry["doc_id"] for entry in context))
    if set(supports) != set(selected_qids):
        raise ValueError("source does not resolve every selected QID")

    selected_results: dict[str, dict[str, object]] = {}
    with results_path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            qid = row.get("question_id")
            if qid in by_qid:
                selected_results[qid] = row
    if set(selected_results) != set(selected_qids):
        raise ValueError("cached results do not resolve every selected QID")

    reference_rows: dict[str, object] = {}
    eligible: list[dict[str, str]] = []
    for qid in selected_qids:
        record = by_qid[qid]
        pages = record.get("retrieved_pages")
        if not isinstance(pages, list) or len(pages) != 4:
            raise ValueError(f"selected record does not contain four pages: {qid}")
        reference_rows[qid] = {"retrieved_pages": pages}
        eligible.append({"qid": qid, "question": record["question"]})
    reference = {
        "schema_version": 1,
        "selection_is_outcome_blind": False,
        "fixed_page_selection_is_outcome_blind": True,
        "question_selection": "300-baseline-correct/300-baseline-wrong",
        "cohort_sha256": cohort_sha,
        "split_manifest_sha256": splits_sha,
        "question_ids": selected_qids,
        "rows": reference_rows,
    }

    root.mkdir(parents=True)
    try:
        reference_path = root / "reference.json"
        eligible_path = root / "eligible.jsonl"
        fixture_path = root / "fixture.json"
        selected_results_path = root / "selected-source-results.jsonl"
        _write_json(reference_path, reference)
        eligible_path.write_text("".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in eligible
        ), encoding="utf-8")
        selected_results_path.write_text("".join(
            json.dumps(selected_results[qid], sort_keys=True, separators=(",", ":")) + "\n"
            for qid in selected_qids
        ), encoding="utf-8")
        fixture = build_fixed_page_fixture(
            reference_path=reference_path,
            eligible_questions_path=eligible_path,
            feature_manifest_path=args.feature_manifest.resolve(),
            pdf_dir=args.pdf_dir.resolve(),
            fixture_version="task9-shared-probe-random600-v1",
            page_count=4,
            allow_outcome_stratified_questions=True,
        )
        fixture_sha = publish_fixed_page_fixture(fixture, fixture_path)

        qid_inputs: list[dict[str, object]] = []
        qid_root = root / "qid-inputs"
        qid_root.mkdir()
        for index, qid in enumerate(selected_qids):
            input_root = qid_root / f"{index:03d}-{qid}"
            smoke = seal_task8_smoke_inputs(
                fixture_path=fixture_path,
                fixture_sha256=fixture_sha,
                qid=qid,
                output_root=input_root,
                runtime_commit=args.runtime_commit,
            )
            trace = selected_results[qid].get("trace")
            if not isinstance(trace, dict) or type(trace.get("post_qtp_visual_tokens")) is not int:
                raise ValueError(f"cached result lacks post-QTP count: {qid}")
            manifest_path = input_root / "smoke-input-manifest.json"
            qid_inputs.append({
                "array_index": index,
                "qid": qid,
                "split": split_by_qid[qid],
                "baseline_stratum": by_qid[qid]["baseline_stratum"],
                "supporting_document_ids": supports[qid],
                "input_root": str(input_root),
                "input_manifest": str(manifest_path),
                "input_manifest_sha256": _sha256(manifest_path),
                "expected_post_qtp_visual_tokens": trace["post_qtp_visual_tokens"],
                "page_count": len(smoke["pages"]),
            })
        preprocessing = {
            "schema_version": "docprune-task9-shared-probe-inputs-v1",
            "status": "sealed",
            "runtime_commit": args.runtime_commit,
            "cohort_path": str(cohort_path),
            "cohort_file_sha256": args.cohort_file_sha256,
            "cohort_sha256": cohort_sha,
            "splits_path": str(splits_path),
            "splits_file_sha256": args.splits_file_sha256,
            "split_manifest_sha256": splits_sha,
            "fixture_path": str(fixture_path),
            "fixture_sha256": fixture_sha,
            "selected_source_results_path": str(selected_results_path),
            "selected_source_results_sha256": _sha256(selected_results_path),
            "question_count": 600,
            "page_count": 2400,
            "qid_inputs": qid_inputs,
            "global_index_loaded": False,
            "retrieval_run": False,
        }
        _write_json(root / "preprocessing-manifest.json", preprocessing)
    except BaseException:
        shutil.rmtree(root)
        raise
    print(json.dumps({
        "output_root": str(root),
        "fixture_sha256": fixture_sha,
        "question_count": 600,
        "page_count": 2400,
        "preprocessing_manifest_sha256": _sha256(root / "preprocessing-manifest.json"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
