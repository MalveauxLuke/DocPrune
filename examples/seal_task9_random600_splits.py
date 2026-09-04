#!/usr/bin/env python3
"""Seal document-disjoint train/validation/test splits for the Task 9 random 600."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from docprune.experiment_design import _canonical_json_sha256, support_document_components


TARGETS = {
    "train": {"baseline_correct": 180, "baseline_wrong": 180},
    "validation": {"baseline_correct": 60, "baseline_wrong": 60},
    "test": {"baseline_correct": 60, "baseline_wrong": 60},
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _key(seed: str, *parts: str) -> str:
    return hashlib.sha256((seed + "\0" + "\0".join(parts)).encode()).hexdigest()


def _load_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--cohort-file-sha256", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cohort_path = args.cohort.resolve()
    source_path = args.source.resolve()
    output = args.output.resolve()
    for path, expected in (
        (cohort_path, args.cohort_file_sha256),
        (source_path, args.source_sha256),
    ):
        if not path.is_file() or path.is_symlink() or _sha256(path) != expected:
            raise ValueError(f"input identity mismatch: {path}")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output already exists: {output}")

    cohort = _load_object(cohort_path)
    unsigned_cohort = dict(cohort)
    cohort_digest = unsigned_cohort.pop("cohort_sha256", None)
    if cohort_digest != _canonical_json_sha256(unsigned_cohort):
        raise ValueError("cohort canonical digest mismatch")
    selected = {
        qid: stratum
        for stratum, qids in cohort["selected_qids"].items()
        for qid in qids
    }
    if len(selected) != 600 or set(selected.values()) != {
        "baseline_correct", "baseline_wrong"
    }:
        raise ValueError("cohort is not the balanced random 600")

    supports: dict[str, list[str]] = {}
    with source_path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            qid = row.get("qid")
            if qid not in selected:
                continue
            context = row.get("supporting_context")
            if not isinstance(context, list) or not context:
                raise ValueError(f"missing supporting context: {qid}")
            doc_ids = [entry.get("doc_id") for entry in context if isinstance(entry, dict)]
            if len(doc_ids) != len(context) or any(not isinstance(doc, str) or not doc for doc in doc_ids):
                raise ValueError(f"invalid supporting context: {qid}")
            supports[qid] = list(dict.fromkeys(doc_ids))
    if set(supports) != set(selected):
        raise ValueError("source does not resolve every selected QID")

    component_payload = support_document_components([
        {"qid": qid, "supporting_document_ids": supports[qid]}
        for qid in sorted(selected)
    ])
    components = list(component_payload["components"])
    assigned: dict[str, list[dict[str, object]]] = {name: [] for name in TARGETS}
    counts = {
        name: {"baseline_correct": 0, "baseline_wrong": 0}
        for name in TARGETS
    }

    def profile(component: dict[str, object]) -> dict[str, int]:
        qids = component["qids"]
        return {
            stratum: sum(selected[qid] == stratum for qid in qids)
            for stratum in ("baseline_correct", "baseline_wrong")
        }

    multi = [component for component in components if len(component["qids"]) > 1]
    single = [component for component in components if len(component["qids"]) == 1]
    multi.sort(key=lambda component: (
        -len(component["qids"]),
        _key(args.seed, "component", str(component["component_id"])),
        str(component["component_id"]),
    ))
    for component in multi:
        addition = profile(component)
        candidates = [
            name for name, target in TARGETS.items()
            if all(counts[name][stratum] + addition[stratum] <= target[stratum]
                   for stratum in addition)
        ]
        if not candidates:
            raise ValueError("document component cannot fit an exact split")
        destination = min(candidates, key=lambda name: (
            sum(
                ((counts[name][stratum] + addition[stratum]) / TARGETS[name][stratum]) ** 2
                for stratum in addition
            ),
            _key(args.seed, "destination", str(component["component_id"]), name),
            name,
        ))
        assigned[destination].append(component)
        for stratum in addition:
            counts[destination][stratum] += addition[stratum]

    for stratum in ("baseline_correct", "baseline_wrong"):
        pool = [component for component in single if selected[component["qids"][0]] == stratum]
        pool.sort(key=lambda component: (
            _key(args.seed, "singleton", stratum, str(component["component_id"])),
            str(component["component_id"]),
        ))
        cursor = 0
        for name in TARGETS:
            required = TARGETS[name][stratum] - counts[name][stratum]
            assigned[name].extend(pool[cursor:cursor + required])
            counts[name][stratum] += required
            cursor += required
        if cursor != len(pool):
            raise ValueError(f"singleton accounting failed: {stratum}")

    splits: dict[str, object] = {}
    all_qids: list[str] = []
    doc_sets: dict[str, set[str]] = {}
    for name in TARGETS:
        qids = sorted(
            (qid for component in assigned[name] for qid in component["qids"]),
            key=lambda qid: (_key(args.seed, "qid", name, qid), qid),
        )
        docs = {doc for qid in qids for doc in supports[qid]}
        doc_sets[name] = docs
        all_qids.extend(qids)
        splits[name] = {
            "qids": qids,
            "question_count": len(qids),
            "baseline_counts": {
                stratum: sum(selected[qid] == stratum for qid in qids)
                for stratum in ("baseline_correct", "baseline_wrong")
            },
            "component_ids": sorted(str(component["component_id"]) for component in assigned[name]),
            "supporting_document_ids": sorted(docs),
        }

    if len(all_qids) != len(set(all_qids)) or set(all_qids) != set(selected):
        raise ValueError("split coverage is not an exact partition of the cohort")
    names = list(TARGETS)
    if any(doc_sets[names[i]] & doc_sets[names[j]] for i in range(len(names)) for j in range(i + 1, len(names))):
        raise ValueError("supporting documents cross splits")
    if any(splits[name]["baseline_counts"] != TARGETS[name] for name in TARGETS):
        raise ValueError("split stratum counts do not match targets")

    manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "sealed",
        "purpose": "Task 9 random-600 document-disjoint probe splits",
        "selection_method": "group selected QIDs by connected gold supporting-document components; deterministically place multi-question components, then fill exact stratum targets with seeded singleton order",
        "seed": args.seed,
        "cohort_path": str(cohort_path),
        "cohort_file_sha256": args.cohort_file_sha256,
        "cohort_sha256": cohort_digest,
        "source_path": str(source_path),
        "source_sha256": args.source_sha256,
        "question_count": 600,
        "component_count": len(components),
        "targets": TARGETS,
        "splits": splits,
        "document_disjoint": True,
        "teacher_boundary": "B13",
        "teacher_mask_count_per_question": 32,
    }
    manifest["manifest_sha256"] = _canonical_json_sha256(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({
        "output": str(output),
        "manifest_sha256": manifest["manifest_sha256"],
        "component_count": len(components),
        "split_counts": {name: splits[name]["question_count"] for name in TARGETS},
        "document_disjoint": True,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
