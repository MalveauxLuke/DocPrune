#!/usr/bin/env python3
"""Compare the four sealed Task 6 portability-smoke artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(root: Path) -> list[dict[str, object]]:
    path = root / "run" / "results.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != 9 or [row.get("matrix_cell") for row in rows] != list(range(9)):
        raise ValueError(f"smoke root is not an exact nine-cell result: {root}")
    return rows


def _selection_identity(row: dict[str, object]) -> dict[str, object]:
    selection = row.get("policy_selection")
    if not isinstance(selection, dict):
        raise ValueError("smoke row lacks policy selection evidence")
    return {
        key: selection.get(key)
        for key in (
            "policy",
            "boundary",
            "native_layer",
            "visual_population",
            "requested_budget",
            "achieved_budget",
            "retained_compact_visual_ids",
            "geometry_count",
            "geometry_sha256",
            "prefill_cache_lengths",
            "retained_mrope_position_shape",
            "retained_mrope_position_sha256",
        )
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("a30", "a100_40", "h100", "l40s"):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    roots = {
        "A30": args.a30,
        "A100-40GB": args.a100_40,
        "H100": args.h100,
        "L40S": args.l40s,
    }
    all_rows = {name: _rows(root) for name, root in roots.items()}
    canonical = all_rows["L40S"]
    comparisons: dict[str, object] = {}
    admitted = True
    for family, rows in all_rows.items():
        cell_reports = []
        for index, (expected, observed) in enumerate(zip(canonical, rows, strict=True)):
            input_identity_equal = all(
                observed.get(key) == expected.get(key)
                for key in (
                    "question_id",
                    "question",
                    "answers",
                    "retrieved_pages",
                    "fixed_page_fixture_sha256",
                    "fixed_page_provenance",
                    "global_index_loaded",
                    "policy_context",
                )
            )
            selection_equal = _selection_identity(observed) == _selection_identity(expected)
            trace_equal = observed.get("trace") == expected.get("trace")
            answer_equal = observed.get("predicted_answer") == expected.get("predicted_answer")
            cell_admitted = input_identity_equal and selection_equal and trace_equal
            admitted = admitted and cell_admitted
            cell_reports.append(
                {
                    "cell": index,
                    "input_identity_equal": input_identity_equal,
                    "selection_cache_mrope_equal": selection_equal,
                    "trace_equal": trace_equal,
                    "answer_equal": answer_equal,
                    "admitted": cell_admitted,
                }
            )
        comparisons[family] = cell_reports
    report = {
        "schema_version": 1,
        "status": "admitted" if admitted else "rejected",
        "canonical_gpu_family": "L40S",
        "quality_pooling_permitted": False,
        "timing_pooling_permitted": False,
        "comparisons": comparisons,
        "artifacts": {
            family: {
                "root": str(root.resolve()),
                "results_sha256": _sha256(root / "run" / "results.jsonl"),
                "run_manifest_sha256": _sha256(root / "run" / "run_manifest.json"),
            }
            for family, root in roots.items()
        },
    }
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"smoke analysis output exists: {args.output}")
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if admitted else 2


if __name__ == "__main__":
    raise SystemExit(main())
