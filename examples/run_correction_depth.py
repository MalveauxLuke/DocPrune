#!/usr/bin/env python3
"""Run sealed correction depth-comparison stages. No model import for validate."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from docprune.correction_depth import (read, resource, validate_case, run_baseline,
    run_comparison, summarize)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["validate", "baseline", "compare", "summarize"])
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-commit", required=True)
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--boundaries", nargs="+", default=["input", "dynamic"])
    parser.add_argument("--holdout-masks", type=int, default=0)
    parser.add_argument("--adjudications", type=Path)
    args = parser.parse_args()
    if len(args.runtime_commit) != 40 or any(c not in "0123456789abcdef" for c in args.runtime_commit):
        parser.error("runtime-commit must identify the packaged runtime's 40-character Git revision")
    if args.holdout_masks < 0:
        parser.error("holdout masks must be nonnegative")
    cases = [validate_case(c) for c in read(args.corpus)["cases"]]
    if args.case_id:
        requested = set(args.case_id)
        cases = [c for c in cases if c["case_id"] in requested]
        if {c["case_id"] for c in cases} != requested:
            parser.error("unknown case-id")
    resources = read(args.resources)
    root = args.resources.resolve().parent
    if args.phase == "validate":
        for key in ("config", "run_config", "index_manifest", "fixture"):
            resource(resources[key], root)
        from docprune.task6_runtime import load_fixed_page_fixture
        fixture = load_fixed_page_fixture(resource(resources["fixture"], root),
            expected_sha256=resources["fixture"]["sha256"], validate_external_bytes=True)
        for case in cases:
            sample = fixture.selected_samples((case["qid"],))[0]
            if sample.question != case["question"]:
                raise ValueError("fixture/corpus question mismatch")
        print(json.dumps({"status": "validated_without_model", "case_count": len(cases)}))
        return
    if args.phase == "summarize":
        print(json.dumps(summarize(cases, args.output), indent=2, ensure_ascii=False))
        return
    reviews = read(args.adjudications) if args.adjudications else None
    for case in cases:
        if args.phase == "baseline":
            result = run_baseline(case, resources, root, args.output, args.runtime_commit)
            print(json.dumps({"case_id": case["case_id"], "status": "baseline_completed", "score": result["unpruned"]["contract_score"]}), flush=True)
        else:
            result = run_comparison(case, resources, root, args.output, args.runtime_commit,
                depth_spec=args.boundaries, holdout=args.holdout_masks, adjudications=reviews)
            print(json.dumps({"case_id": case["case_id"], "result": "comparison_completed" if isinstance(result, list) else result}), flush=True)

if __name__ == "__main__":
    main()
