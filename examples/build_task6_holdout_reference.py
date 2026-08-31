#!/usr/bin/env python3
"""Project cached ordered pages into an outcome-blind fixed-page reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from docprune.experiment_design import validate_holdout_for_launch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout-root", type=Path, required=True)
    parser.add_argument("--development-registry", type=Path, required=True)
    parser.add_argument("--required-label", action="append", required=True)
    parser.add_argument("--cached-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    holdout = validate_holdout_for_launch(
        args.holdout_root,
        required_registry_path=args.development_registry,
        required_registry_labels=tuple(args.required_label),
    )
    selected = holdout["selected_records"]
    selected_qids = holdout["selected_qids"]
    wanted = set(selected_qids)
    retrieved_by_qid: dict[str, object] = {}
    with args.cached_results.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = row.get("question_id")
            if qid in wanted:
                if qid in retrieved_by_qid:
                    raise ValueError(f"duplicate cached result QID: {qid}")
                retrieved_by_qid[qid] = row.get("retrieved_pages")
    if set(retrieved_by_qid) != wanted:
        raise ValueError("cached results do not cover the exact selected holdout")
    rows: dict[str, object] = {}
    for record in selected:
        qid = record["qid"]
        pages = retrieved_by_qid[qid]
        expected = [
            (page["doc_id"], page["page_index"]) for page in record["cached_pages"]
        ]
        observed = [(page["doc_id"], page["page_index"]) for page in pages]
        if observed != expected:
            raise ValueError(f"cached ordered page identity mismatch for QID {qid}")
        rows[qid] = {"retrieved_pages": pages}
    payload = {
        "schema_version": 1,
        "purpose": "maximum-available Task 6 primary holdout fixed-page reference",
        "selection_is_outcome_blind": True,
        "question_ids": selected_qids,
        "rows": rows,
        "holdout_manifest_path": str((args.holdout_root / "manifest.json").resolve()),
        "holdout_manifest_file_sha256": _sha256((args.holdout_root / "manifest.json").resolve()),
        "holdout_manifest_sha256": holdout["manifest_sha256"],
        "cached_results_path": str(args.cached_results.resolve()),
        "cached_results_sha256": _sha256(args.cached_results.resolve()),
    }
    payload["reference_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{args.output.name}.", dir=args.output.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        descriptor = -1
        os.link(temporary, args.output)
        directory = os.open(args.output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor != -1:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "sha256": _sha256(args.output.resolve()),
                "question_count": len(selected_qids),
                "selection_is_outcome_blind": True,
                "retrieval_run": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
