#!/usr/bin/env python3
"""Materialize a sealed Task 9 confirmation cohort as retrieval-free fixed-page inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from docprune.experiment_design import build_task9_confirmation_fixture_inputs
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--cohort-file-sha256", required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--pdf-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--runtime-commit", required=True)
    args = parser.parse_args()
    cohort_path = args.cohort.resolve()
    root = args.output_root.absolute()
    if _sha256(cohort_path) != args.cohort_file_sha256:
        raise ValueError("Task 9 confirmation cohort file checksum mismatch")
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    if not isinstance(cohort, dict):
        raise ValueError("Task 9 confirmation cohort must be a JSON object")
    reference, eligible = build_task9_confirmation_fixture_inputs(cohort)
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"Task 9 confirmation input root already exists: {root}")
    source = cohort.get("source_authority")
    results_path = Path(str(source.get("cached_results_path"))) if isinstance(source, dict) else None
    if results_path is None or _sha256(results_path) != source.get("cached_results_file_sha256"):
        raise ValueError("Task 9 confirmation cached result authority changed")

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
        fixture = build_fixed_page_fixture(
            reference_path=reference_path,
            eligible_questions_path=eligible_path,
            feature_manifest_path=args.feature_manifest.resolve(),
            pdf_dir=args.pdf_dir.resolve(),
            fixture_version="task9-baseline-wrong100-confirmation-v1",
            page_count=4,
        )
        fixture_sha = publish_fixed_page_fixture(fixture, fixture_path)
        selected_qids = [row["qid"] for row in eligible]
        selected_set = set(selected_qids)
        selected_rows: dict[str, dict[str, object]] = {}
        with results_path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                qid = row.get("question_id") if isinstance(row, dict) else None
                if qid in selected_set:
                    selected_rows[qid] = row
        if set(selected_rows) != selected_set:
            raise ValueError("Task 9 confirmation selected source results are incomplete")
        selected_results_path.write_text("".join(
            json.dumps(selected_rows[qid], sort_keys=True, separators=(",", ":")) + "\n"
            for qid in selected_qids
        ), encoding="utf-8")

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
            source_trace = selected_rows[qid].get("trace")
            if not isinstance(source_trace, dict) or type(source_trace.get("post_qtp_visual_tokens")) is not int:
                raise ValueError(f"selected source result lacks post-QTP count: {qid}")
            manifest_path = input_root / "smoke-input-manifest.json"
            qid_inputs.append({
                "array_index": index,
                "qid": qid,
                "baseline_stratum": "baseline_wrong",
                "supporting_document_ids": cohort["selected_records"][index]["supporting_document_ids"],
                "input_root": str(input_root),
                "input_manifest": str(manifest_path),
                "input_manifest_sha256": _sha256(manifest_path),
                "expected_post_qtp_visual_tokens": source_trace["post_qtp_visual_tokens"],
                "page_count": len(smoke["pages"]),
            })
        manifest = {
            "schema_version": "docprune-task9-confirmation-inputs-v1",
            "status": "sealed",
            "runtime_commit": args.runtime_commit,
            "cohort_path": str(cohort_path),
            "cohort_file_sha256": args.cohort_file_sha256,
            "cohort_sha256": cohort["cohort_sha256"],
            "fixture_path": str(fixture_path),
            "fixture_sha256": fixture_sha,
            "selected_source_results_path": str(selected_results_path),
            "selected_source_results_sha256": _sha256(selected_results_path),
            "question_count": len(selected_qids),
            "page_count": 4 * len(selected_qids),
            "qid_inputs": qid_inputs,
            "global_index_loaded": False,
            "retrieval_run": False,
        }
        _write_json(root / "preprocessing-manifest.json", manifest)
    except BaseException:
        shutil.rmtree(root)
        raise
    print(json.dumps(manifest, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
