#!/usr/bin/env python3
"""Seal the approved maximum-available Task 6 holdout without retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from docprune.experiment_design import (
    approve_maximum_available_power,
    plan_equivalence_power,
    seal_holdout,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _eligibility_input(record: dict[str, object]) -> dict[str, object]:
    source = record["source"]
    if not isinstance(source, dict):
        raise ValueError("projection source is invalid")
    return {
        "qid": record["qid"],
        "metadata": {
            "type": source["type"],
            "supporting_document_ids": record["supporting_document_ids"],
            "source_path": source["path"],
            "source_sha256": source["sha256"],
        },
        "cached_pages": record["cached_pages"],
        "persisted_features": record["persisted_features"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--projection", type=Path, required=True)
    parser.add_argument("--development-analysis", type=Path, required=True)
    parser.add_argument("--development-registry", type=Path, required=True)
    parser.add_argument("--runtime-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    projection = json.loads(args.projection.read_text(encoding="utf-8"))
    analysis = json.loads(args.development_analysis.read_text(encoding="utf-8"))
    registry = json.loads(args.development_registry.read_text(encoding="utf-8"))
    records = projection.get("eligible_records")
    primary = analysis.get("primary_r20")
    calibration = analysis.get("random_calibrations", {}).get("global-uniform-random")
    labels = registry.get("required_labels")
    if (
        projection.get("status") != "outcome-blind-cached-page-eligibility"
        or not isinstance(records, list)
        or len(records) != projection.get("eligible_count")
        or not isinstance(primary, dict)
        or not isinstance(calibration, dict)
        or not isinstance(labels, list)
    ):
        raise ValueError("holdout source artifacts are incomplete")
    original_power = plan_equivalence_power(
        developmental_mean_f1=float(primary["point_estimate"]),
        developmental_sd_f1=float(primary["developmental_sd_f1"]),
        cluster_design_effect=1.0,
        eligible_pool_size=len(records),
        equivalence_margin_f1=float(primary["classification"]["equivalence_margin_f1"]),
        target_power=0.8,
    )
    power = approve_maximum_available_power(original_power)

    input_hashes = {
        str(args.projection.resolve()): _sha256(args.projection.resolve()),
        str(args.development_analysis.resolve()): _sha256(args.development_analysis.resolve()),
        str(args.development_registry.resolve()): _sha256(args.development_registry.resolve()),
    }
    for source in projection.get("source_files", {}).values():
        if not isinstance(source, dict) or not isinstance(source.get("path"), str):
            raise ValueError("projection source-file inventory is invalid")
        source_path = Path(source["path"])
        expected_digest = source.get("sha256", source.get("file_sha256"))
        if not isinstance(expected_digest, str) or _sha256(source_path) != expected_digest:
            raise ValueError(f"projection source checksum mismatch: {source_path}")
        input_hashes[str(source_path)] = expected_digest

    manifest = seal_holdout(
        [_eligibility_input(record) for record in records],
        development_registry_path=args.development_registry.resolve(),
        required_registry_labels=tuple(labels),
        power=power,
        calibration=calibration,
        runtime_pins={
            "runtime_commit": args.runtime_commit,
            "m3docrag_commit": "29e6ac2294d6b87075a1d45b8a8df175b214248a",
            "model": "Qwen/Qwen2-VL-7B-Instruct",
            "matrix": "aggregate-score-top-m vs global-uniform-random-r20",
        },
        input_file_hashes=input_hashes,
        destination=args.output,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "manifest_sha256": manifest["manifest_sha256"],
                "selected_n": len(manifest["selected_qids"]),
                "planned_required_n": power["planned_required_n"],
                "achieved_power": power["achieved_power"],
                "fixed_page_provenance": True,
                "global_index_loaded": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
