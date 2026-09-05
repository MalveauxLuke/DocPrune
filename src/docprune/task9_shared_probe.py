"""Outcome-blind, document-disjoint cohort contract for Task 9 shared probes."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from docprune.experiment_design import (
    _authenticate_eligibility_records,
    _canonical_json_sha256,
    _require_sha256,
    support_document_components,
)

DEFAULT_SHARED_PROBE_SPLITS: dict[str, dict[str, int]] = {
    "train": {"total": 360, "baseline_correct": 180, "baseline_wrong": 180},
    "validation": {"total": 120, "baseline_correct": 60, "baseline_wrong": 60},
    "primary_test": {"total": 60, "baseline_correct": 30, "baseline_wrong": 30},
    "secondary_test": {"total": 60, "natural_prevalence": 60},
}
_SPLIT_NAMES = ("train", "validation", "primary_test", "secondary_test")
_BALANCED_SPLITS = ("train", "validation", "primary_test")
_STRATA = ("baseline_correct", "baseline_wrong")
_FORBIDDEN_OUTCOME_KEYS = frozenset({
    "em", "f1", "exact_match", "token_f1", "quality", "metrics", "trace", "timing",
    "intervention", "interventions", "mask_outcomes", "generated_answer", "gold_answer",
    "loss", "outcome", "result_metrics", "evaluation",
})
_FORBIDDEN_ACCESS_KEYS = frozenset({
    "retrieval", "retrieval_index", "global_index", "index", "retrieval_run", "global_index_loaded",
})
_METADATA_KEYS = ("template_id", "template", "vendor", "vendor_id", "layout_family", "document_template")


def _sha256_file(path: Path) -> str:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError(f"input file must be an absolute regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_key(seed: str, *parts: str) -> str:
    return hashlib.sha256((seed + "\0" + "\0".join(parts)).encode("utf-8")).hexdigest()


def _normalise_split_counts(
    split_counts: Mapping[str, Mapping[str, int]] | None,
) -> dict[str, dict[str, int]]:
    supplied = DEFAULT_SHARED_PROBE_SPLITS if split_counts is None else split_counts
    if not isinstance(supplied, Mapping) or set(supplied) != set(_SPLIT_NAMES):
        raise ValueError("shared-probe split counts must name all four splits")
    checked: dict[str, dict[str, int]] = {}
    for split in _BALANCED_SPLITS:
        value = supplied[split]
        if not isinstance(value, Mapping) or set(value) not in ({"total", *_STRATA}, set(_STRATA)):
            raise ValueError(f"{split} split counts must include both baseline strata")
        counts = {key: value[key] for key in _STRATA}
        supplied_total = value.get("total", sum(counts.values()))
        counts["total"] = supplied_total
        if any(type(number) is not int or number < 0 for number in counts.values()):
            raise ValueError(f"{split} split counts must be nonnegative integers")
        if counts["total"] != counts["baseline_correct"] + counts["baseline_wrong"]:
            raise ValueError(f"{split} split total does not equal its baseline strata")
        checked[split] = counts
    secondary = supplied["secondary_test"]
    if not isinstance(secondary, Mapping) or set(secondary) not in ({"total", "natural_prevalence"}, {"total"}):
        raise ValueError("secondary_test split counts must include total")
    natural_total = secondary.get("total")
    natural_prevalence = secondary.get("natural_prevalence", natural_total)
    if (
        type(natural_total) is not int or type(natural_prevalence) is not int
        or natural_total < 0 or natural_prevalence != natural_total
    ):
        raise ValueError("secondary_test split counts are invalid")
    checked["secondary_test"] = {"total": natural_total, "natural_prevalence": natural_prevalence}
    return checked


def _prior_qids(value: Mapping[str, object], *, label: str) -> tuple[str, ...]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} cohort must be a mapping")
    supplied_digest = value.get("cohort_sha256")
    unsigned = dict(value)
    unsigned.pop("cohort_sha256", None)
    if (
        value.get("status") not in {"selected", "sealed"}
        or not isinstance(supplied_digest, str)
        or supplied_digest != _canonical_json_sha256(unsigned)
    ):
        raise ValueError(f"{label} cohort is not authenticated")
    selected = value.get("selected_qids")
    if label == "preliminary-48":
        if (
            not isinstance(selected, Mapping) or set(selected) != set(_STRATA)
            or any(not isinstance(selected[name], list) or len(selected[name]) != 24 for name in _STRATA)
        ):
            raise ValueError("preliminary-48 cohort must contain 24 questions per baseline stratum")
        qids = tuple(selected[name][index] for name in _STRATA for index in range(24))
    else:
        if not isinstance(selected, list) or len(selected) != 100:
            raise ValueError("confirmation-100 cohort must contain exactly 100 questions")
        qids = tuple(selected)
    if any(not isinstance(qid, str) or not qid for qid in qids) or len(set(qids)) != len(qids):
        raise ValueError(f"{label} cohort QIDs must be unique nonempty strings")
    return qids


def _identity(value: object, *, label: str) -> tuple[str, int]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    doc_id, page_index = value.get("doc_id"), value.get("page_index")
    if not isinstance(doc_id, str) or not doc_id or type(page_index) is not int or page_index < 0:
        raise ValueError(f"{label} has invalid page identity")
    return doc_id, page_index


def _source_from_row(row: Mapping[str, object], metadata: Mapping[str, object]) -> dict[str, object]:
    raw = row.get("source")
    if raw is None:
        raw = {
            "path": row.get("source_path", metadata.get("source_path")),
            "sha256": row.get("source_sha256", metadata.get("source_sha256")),
            "type": row.get("source_type", metadata.get("type", "single_hop")),
        }
    if not isinstance(raw, Mapping):
        raise ValueError("question source identity must be a mapping")
    path, digest = raw.get("path", raw.get("source_path")), raw.get("sha256", raw.get("source_sha256"))
    source_type = raw.get("type", "single_hop")
    if not isinstance(path, str) or not path.startswith("/"):
        raise ValueError("question source path must be absolute")
    _require_sha256(digest, label="question source SHA-256")
    if source_type not in {"single_hop", "TextQ", "TableQ", "ImageQ", "ImageListQ"}:
        raise ValueError("shared-probe cohort requires single-hop source rows")
    return {"path": path, "sha256": digest, "type": "single_hop"}


def _support_ids(row: Mapping[str, object], metadata: Mapping[str, object]) -> list[str]:
    supports = row.get("supporting_document_ids", row.get("support_document_ids"))
    if supports is None:
        supports = metadata.get("supporting_document_ids")
    if supports is None and isinstance(row.get("supporting_context"), list):
        supports = [entry.get("doc_id") for entry in row["supporting_context"] if isinstance(entry, Mapping)]
    if supports is None and isinstance(row.get("supporting_documents"), list):
        supports = [entry.get("doc_id") for entry in row["supporting_documents"] if isinstance(entry, Mapping)]
    if (
        not isinstance(supports, Sequence) or isinstance(supports, str | bytes) or not supports
        or any(not isinstance(doc_id, str) or not doc_id for doc_id in supports)
        or len(set(supports)) != len(supports)
    ):
        raise ValueError("supporting document identities must be unique nonempty strings")
    return list(supports)


def _pages_from_row(row: Mapping[str, object]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    pages_value = row.get("cached_pages", row.get("fixed_pages", row.get("fixed_page_identities")))
    features_value = row.get("persisted_features", row.get("fixed_features", row.get("feature_identities")))
    if not isinstance(pages_value, Sequence) or isinstance(pages_value, str | bytes) or len(pages_value) != 4:
        raise ValueError("shared-probe row must contain four cached fixed pages")
    if not isinstance(features_value, Sequence) or isinstance(features_value, str | bytes) or len(features_value) != 4:
        raise ValueError("shared-probe row must contain four persisted page features")
    pages: list[dict[str, object]] = []
    features: list[dict[str, object]] = []
    seen: set[tuple[str, int]] = set()
    for position, (raw_page, raw_feature) in enumerate(zip(pages_value, features_value, strict=True)):
        if not isinstance(raw_page, Mapping) or not isinstance(raw_feature, Mapping):
            raise ValueError(f"cached page/feature identity is invalid at position {position}")
        identity = _identity(raw_page, label="cached page")
        if identity in seen:
            raise ValueError("cached page identities must be unique")
        seen.add(identity)
        source_path, source_sha = raw_page.get("source_path"), raw_page.get("source_sha256")
        if not isinstance(source_path, str) or not source_path.startswith("/"):
            raise ValueError("cached page source path must be absolute")
        _require_sha256(source_sha, label="cached page source SHA-256")
        feature_identity = _identity(raw_feature, label="persisted feature")
        if feature_identity != identity:
            raise ValueError("cached page and persisted feature identities differ")
        feature_path, feature_sha = raw_feature.get("feature_path"), raw_feature.get("feature_sha256")
        index_path, index_sha = raw_feature.get("index_manifest_path"), raw_feature.get("index_manifest_sha256")
        if not isinstance(feature_path, str) or not feature_path.startswith("/"):
            raise ValueError("persisted feature path must be absolute")
        if not isinstance(index_path, str) or not index_path.startswith("/"):
            raise ValueError("feature index manifest path must be absolute")
        _require_sha256(feature_sha, label="persisted feature SHA-256")
        _require_sha256(index_sha, label="feature index manifest SHA-256")
        pages.append({"doc_id": identity[0], "page_index": identity[1], "source_path": source_path, "source_sha256": source_sha})
        features.append({
            "doc_id": identity[0], "page_index": identity[1], "feature_path": feature_path,
            "feature_sha256": feature_sha, "index_manifest_path": index_path, "index_manifest_sha256": index_sha,
        })
    retrieved = row.get("retrieved_pages")
    if retrieved is not None:
        if not isinstance(retrieved, Sequence) or isinstance(retrieved, str | bytes) or len(retrieved) != 4:
            raise ValueError("cached retrieved pages must contain exactly four pages")
        observed = [_identity(page, label="retrieved page") for page in retrieved]
        if observed != [_identity(page, label="cached page") for page in pages]:
            raise ValueError("cached retrieved page order differs from fixed page identity")
    return pages, features


def _normalise_row(row: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(row, Mapping):
        raise ValueError("Task 6 result row must be a mapping")
    forbidden = sorted(_FORBIDDEN_OUTCOME_KEYS & set(row))
    if forbidden:
        raise ValueError(f"outcome field is not permitted in cohort input: {forbidden[0]}")
    access_flags = sorted(_FORBIDDEN_ACCESS_KEYS & set(row))
    if access_flags:
        flag = access_flags[0]
        if flag in {"retrieval_run", "global_index_loaded"} and row[flag] is False:
            pass
        else:
            raise ValueError(f"retrieval/global index declaration is forbidden: {flag}")
    qid = row.get("qid", row.get("question_id"))
    if not isinstance(qid, str) or not qid:
        raise ValueError("cohort question ID must be a nonempty string")
    question, answers = row.get("question"), row.get("answers")
    prediction = row.get("predicted_answer", row.get("unpruned_predicted_answer"))
    if not isinstance(question, str) or not question:
        raise ValueError(f"question text is missing: {qid}")
    if not isinstance(answers, list) or not answers or any(not isinstance(answer, str) or not answer for answer in answers):
        raise ValueError(f"question answers are invalid: {qid}")
    if not isinstance(prediction, str):
        raise ValueError(f"baseline prediction is missing: {qid}")
    from docprune.evaluation import list_em

    baseline_correct = list_em(prediction, answers) == 1.0
    baseline_stratum = "baseline_correct" if baseline_correct else "baseline_wrong"
    if row.get("baseline_stratum") is not None and row["baseline_stratum"] != baseline_stratum:
        raise ValueError(f"baseline stratum does not match baseline prediction: {qid}")
    if row.get("baseline_em") is not None and row["baseline_em"] != (1.0 if baseline_correct else 0.0):
        raise ValueError(f"baseline EM does not match baseline prediction: {qid}")
    raw_metadata = row.get("metadata", {})
    if raw_metadata is not None and not isinstance(raw_metadata, Mapping):
        raise ValueError(f"row metadata is invalid: {qid}")
    metadata = raw_metadata if isinstance(raw_metadata, Mapping) else {}
    source = _source_from_row(row, metadata)
    supports = _support_ids(row, metadata)
    pages, features = _pages_from_row(row)
    preserved_metadata: dict[str, object] = {}
    for key in _METADATA_KEYS:
        if key in row:
            preserved_metadata[key] = row[key]
        elif key in metadata:
            preserved_metadata[key] = metadata[key]
        if key in preserved_metadata and preserved_metadata[key] is not None and not isinstance(
            preserved_metadata[key], (str, int, float, bool)
        ):
            raise ValueError(f"row metadata field is not scalar: {key}")
    record: dict[str, object] = {
        "qid": qid, "question": question, "answers": list(answers),
        "unpruned_predicted_answer": prediction, "baseline_em": 1.0 if baseline_correct else 0.0,
        "baseline_stratum": baseline_stratum, "source": source,
        "supporting_document_ids": supports, "cached_pages": pages, "persisted_features": features,
        "metadata": preserved_metadata,
    }
    record.update(preserved_metadata)
    return record


def _merge_eligibility(
    rows: Sequence[Mapping[str, object]],
    eligibility_records: Sequence[Mapping[str, object]] | Mapping[str, Mapping[str, object]] | None,
) -> list[dict[str, object]]:
    if eligibility_records is None:
        return [dict(row) for row in rows]
    if isinstance(eligibility_records, Mapping) and isinstance(eligibility_records.get("eligible_records"), list):
        values = list(eligibility_records["eligible_records"])
    else:
        values = list(eligibility_records.values()) if isinstance(eligibility_records, Mapping) else list(eligibility_records)
    by_qid: dict[str, Mapping[str, object]] = {}
    for record in values:
        if not isinstance(record, Mapping):
            raise ValueError("eligibility record must be a mapping")
        qid = record.get("qid", record.get("question_id"))
        if not isinstance(qid, str) or not qid or qid in by_qid:
            raise ValueError("eligibility QIDs must be unique nonempty strings")
        by_qid[qid] = record
    merged: list[dict[str, object]] = []
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ValueError("Task 6 result row must be a mapping")
        qid = raw.get("qid", raw.get("question_id"))
        if not isinstance(qid, str) or qid not in by_qid:
            raise ValueError(f"result row is absent from authenticated eligibility records: {qid}")
        combined = dict(raw)
        eligibility = by_qid[qid]
        for key in ("source", "supporting_document_ids", "cached_pages", "persisted_features"):
            if key not in combined and key in eligibility:
                combined[key] = eligibility[key]
        merged.append(combined)
    return merged


def _record_eligibility(record: Mapping[str, object]) -> dict[str, object]:
    return {
        "qid": record["qid"], "source": record["source"],
        "supporting_document_ids": record["supporting_document_ids"],
        "cached_pages": record["cached_pages"], "persisted_features": record["persisted_features"],
    }


def _source_hash_inventory(records: Sequence[Mapping[str, object]]) -> dict[str, dict[str, str]]:
    source: dict[str, str] = {}
    pages: dict[str, str] = {}
    features: dict[str, str] = {}
    manifests: dict[str, str] = {}
    for record in records:
        row_source = record["source"]
        source[str(row_source["path"])] = str(row_source["sha256"])
        for page in record["cached_pages"]:
            pages[str(page["source_path"])] = str(page["source_sha256"])
        for feature in record["persisted_features"]:
            features[str(feature["feature_path"])] = str(feature["feature_sha256"])
            manifests[str(feature["index_manifest_path"])] = str(feature["index_manifest_sha256"])
    return {
        "question_sources": dict(sorted(source.items())),
        "cached_page_sources": dict(sorted(pages.items())),
        "persisted_features": dict(sorted(features.items())),
        "feature_index_manifests": dict(sorted(manifests.items())),
    }


def _components_for_records(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    return support_document_components([
        {"qid": record["qid"], "supporting_document_ids": record["supporting_document_ids"]}
        for record in records
    ])


def build_task9_shared_probe_cohort(
    result_rows: Sequence[Mapping[str, object]],
    *,
    preliminary_cohort: Mapping[str, object],
    confirmation_cohort: Mapping[str, object],
    selection_seed: str = "task9-shared-probe-cohort-v1",
    seed: str | None = None,
    split_counts: Mapping[str, Mapping[str, int]] | None = None,
    eligibility_records: Sequence[Mapping[str, object]] | Mapping[str, Mapping[str, object]] | None = None,
    input_file_hashes: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Build a deterministic document-disjoint shared-probe cohort.

    Result rows may carry authenticated eligibility fields directly, or
    ``eligibility_records`` may supply those fields by QID.  Only the baseline
    prediction is used for stratification; later outcomes are rejected.
    """
    if seed is not None:
        if selection_seed != "task9-shared-probe-cohort-v1":
            raise ValueError("provide only one of selection_seed and seed")
        selection_seed = seed
    if not isinstance(selection_seed, str) or not selection_seed:
        raise ValueError("selection_seed must be a nonempty string")
    checked_splits = _normalise_split_counts(split_counts)
    target_count = sum(value["total"] for value in checked_splits.values())
    if target_count <= 0:
        raise ValueError("shared-probe split counts must select at least one question")
    preliminary_qids = _prior_qids(preliminary_cohort, label="preliminary-48")
    confirmation_qids = _prior_qids(confirmation_cohort, label="confirmation-100")
    excluded_qids = tuple(dict.fromkeys((*preliminary_qids, *confirmation_qids)))
    excluded_set = set(excluded_qids)
    by_qid: dict[str, dict[str, object]] = {}
    for raw_row in _merge_eligibility(result_rows, eligibility_records):
        record = _normalise_row(raw_row)
        qid = str(record["qid"])
        if qid in by_qid:
            raise ValueError(f"duplicate question ID: {qid}")
        by_qid[qid] = record
    eligible = [record for qid, record in by_qid.items() if qid not in excluded_set]
    if not eligible:
        raise ValueError("no eligible shared-probe questions remain after prior-cohort exclusion")
    _authenticate_eligibility_records([_record_eligibility(record) for record in eligible])
    components = _components_for_records(eligible)
    component_values = components["components"]
    if not isinstance(component_values, list):
        raise ValueError("support component inventory is invalid")
    required_balanced = sum(checked_splits[split]["total"] for split in _BALANCED_SPLITS)
    if len(component_values) < target_count:
        raise ValueError(
            f"shared-probe cohort requires at least {target_count} independent support-document components; "
            f"only {len(component_values)} are available"
        )
    balanced_correct = sum(checked_splits[split]["baseline_correct"] for split in _BALANCED_SPLITS)
    balanced_wrong = sum(checked_splits[split]["baseline_wrong"] for split in _BALANCED_SPLITS)
    categories: dict[str, list[Mapping[str, object]]] = {"correct_only": [], "wrong_only": [], "mixed": []}
    for component in component_values:
        members = component["qids"]
        strata = {str(by_qid[qid]["baseline_stratum"]) for qid in members}
        category = "mixed" if len(strata) == 2 else "correct_only" if "baseline_correct" in strata else "wrong_only"
        categories[category].append(component)
    mixed_correct = max(0, balanced_correct - len(categories["correct_only"]))
    mixed_wrong = max(0, balanced_wrong - len(categories["wrong_only"]))
    if mixed_correct + mixed_wrong > len(categories["mixed"]):
        raise ValueError(
            "insufficient independent components for balanced baseline strata: "
            f"correct={len(categories['correct_only'])}, wrong={len(categories['wrong_only'])}, mixed={len(categories['mixed'])}"
        )

    def ordered(values: Sequence[Mapping[str, object]], kind: str) -> list[Mapping[str, object]]:
        return sorted(
            values,
            key=lambda component: (
                _hash_key(selection_seed, kind, str(component["component_id"])),
                str(component["component_id"]),
            ),
        )

    selected_components: list[tuple[Mapping[str, object], str]] = []
    ordered_correct = ordered(categories["correct_only"], "balanced-correct")
    ordered_wrong = ordered(categories["wrong_only"], "balanced-wrong")
    ordered_mixed = ordered(categories["mixed"], "balanced-mixed")
    selected_components.extend((component, "baseline_correct") for component in ordered_correct[: balanced_correct - mixed_correct])
    selected_components.extend((component, "baseline_wrong") for component in ordered_wrong[: balanced_wrong - mixed_wrong])
    selected_components.extend((component, "baseline_correct") for component in ordered_mixed[:mixed_correct])
    selected_components.extend((component, "baseline_wrong") for component in ordered_mixed[mixed_correct : mixed_correct + mixed_wrong])
    if len(selected_components) != required_balanced:
        raise ValueError("balanced split assignment could not satisfy the requested strata")
    selected_component_ids = {str(component["component_id"]) for component, _ in selected_components}
    remaining_components = [component for component in component_values if str(component["component_id"]) not in selected_component_ids]
    secondary_count = checked_splits["secondary_test"]["total"]
    if len(remaining_components) < secondary_count:
        raise ValueError("not enough independent components remain for secondary_test")
    secondary_components = sorted(
        remaining_components,
        key=lambda component: (_hash_key(selection_seed, "secondary", str(component["component_id"])), str(component["component_id"])),
    )[:secondary_count]

    def choose_qid(component: Mapping[str, object], desired: str | None, kind: str) -> str:
        members = [str(qid) for qid in component["qids"]]
        if desired is not None:
            members = [qid for qid in members if by_qid[qid]["baseline_stratum"] == desired]
        if not members:
            raise ValueError("support component has no question in its assigned baseline stratum")
        return min(
            members,
            key=lambda qid: (_hash_key(selection_seed, kind, str(component["component_id"]), qid), qid),
        )

    assigned: dict[str, list[tuple[str, str]]] = {split: [] for split in _SPLIT_NAMES}
    balanced_by_stratum: dict[str, list[tuple[str, str]]] = {stratum: [] for stratum in _STRATA}
    for component, desired in selected_components:
        qid = choose_qid(component, desired, "representative")
        balanced_by_stratum[desired].append((qid, str(component["component_id"])))
    for stratum in _STRATA:
        ordered_members = sorted(
            balanced_by_stratum[stratum],
            key=lambda value: (_hash_key(selection_seed, "split", stratum, value[1]), value[1], value[0]),
        )
        cursor = 0
        for split in _BALANCED_SPLITS:
            count = checked_splits[split][stratum]
            assigned[split].extend(ordered_members[cursor : cursor + count])
            cursor += count
    for component in secondary_components:
        qid = choose_qid(component, None, "secondary-representative")
        assigned["secondary_test"].append((qid, str(component["component_id"])))
    selected_qids = {split: [qid for qid, _ in assigned[split]] for split in _SPLIT_NAMES}
    selected_records = {split: [by_qid[qid] for qid in selected_qids[split]] for split in _SPLIT_NAMES}
    selected_flat = [record for split in _SPLIT_NAMES for record in selected_records[split]]
    split_payload = {
        split: {
            "qids": selected_qids[split],
            "records": selected_records[split],
            "baseline_counts": {
                stratum: sum(record["baseline_stratum"] == stratum for record in selected_records[split])
                for stratum in _STRATA
            },
        }
        for split in _SPLIT_NAMES
    }
    checked_input_hashes: dict[str, str] = {}
    for raw_path, digest in (input_file_hashes or {}).items():
        if not isinstance(raw_path, str) or not raw_path.startswith("/"):
            raise ValueError("input file hash paths must be absolute")
        checked = _require_sha256(digest, label=f"input hash for {raw_path}")
        if checked != _sha256_file(Path(raw_path)):
            raise ValueError(f"input file SHA-256 mismatch: {raw_path}")
        checked_input_hashes[raw_path] = checked
    cohort: dict[str, object] = {
        "schema_version": 1,
        "status": "selected",
        "purpose": "Task 9 shared-probe document-disjoint cohort",
        "selection_method": "authenticated cached top-4 pages/features; prior-cohort exclusion; support-document components; one seeded representative QID per component; balanced component assignment and seeded natural-prevalence remainder",
        "selection_seed": selection_seed,
        "question_count": len(selected_flat),
        "split_counts": checked_splits,
        "selected_qids": selected_qids,
        "selected_records": selected_records,
        "splits": split_payload,
        "excluded_qids": list(excluded_qids),
        "exclusion_sources": {"preliminary_48": list(preliminary_qids), "confirmation_100": list(confirmation_qids)},
        "prior_cohort_hashes": {
            "preliminary_48": str(preliminary_cohort["cohort_sha256"]),
            "confirmation_100": str(confirmation_cohort["cohort_sha256"]),
        },
        "eligible_question_count": len(eligible),
        "eligible_component_count": len(component_values),
        "selected_component_count": len(selected_flat),
        "eligible_support_components": components,
        "support_components": _components_for_records(selected_flat),
        "selection_audit": {
            "excluded_qid_count": len(excluded_qids),
            "balanced_component_count": required_balanced,
            "secondary_component_count": secondary_count,
            "one_question_per_component": True,
            "component_cross_split": False,
        },
        "source_hashes": _source_hash_inventory(selected_flat),
        "input_file_hashes": dict(sorted(checked_input_hashes.items())),
        "retrieval_run": False,
        "global_index_loaded": False,
        "teacher_boundary": "B13",
        "selection_is_outcome_blind": True,
    }
    cohort["cohort_sha256"] = _canonical_json_sha256(cohort)
    return cohort


def validate_task9_shared_probe_cohort(
    cohort: Mapping[str, object], *, authenticate_files: bool = True
) -> dict[str, object]:
    """Validate the canonical cohort, including split and document isolation."""
    if not isinstance(cohort, Mapping):
        raise ValueError("shared-probe cohort must be a mapping")
    unsigned = dict(cohort)
    supplied = unsigned.pop("cohort_sha256", None)
    if not isinstance(supplied, str) or supplied != _canonical_json_sha256(unsigned):
        raise ValueError("shared-probe cohort canonical digest mismatch")
    if (
        cohort.get("schema_version") != 1
        or cohort.get("status") not in {"selected", "sealed"}
        or cohort.get("teacher_boundary") != "B13"
        or cohort.get("retrieval_run") is not False
        or cohort.get("global_index_loaded") is not False
        or cohort.get("selection_is_outcome_blind") is not True
    ):
        raise ValueError("shared-probe cohort contract flags are invalid")
    splits = cohort.get("splits")
    selected_qids = cohort.get("selected_qids")
    selected_records = cohort.get("selected_records")
    split_counts = cohort.get("split_counts")
    checked_counts = _normalise_split_counts(split_counts if isinstance(split_counts, Mapping) else None)
    if (
        not isinstance(splits, Mapping) or set(splits) != set(_SPLIT_NAMES)
        or not isinstance(selected_qids, Mapping) or set(selected_qids) != set(_SPLIT_NAMES)
        or not isinstance(selected_records, Mapping) or set(selected_records) != set(_SPLIT_NAMES)
    ):
        raise ValueError("shared-probe split inventory is invalid")
    flat: list[Mapping[str, object]] = []
    flat_qids: list[str] = []
    for split in _SPLIT_NAMES:
        qids, records, payload = selected_qids[split], selected_records[split], splits[split]
        if not isinstance(qids, list) or not isinstance(records, list) or not isinstance(payload, Mapping):
            raise ValueError(f"shared-probe {split} inventory is invalid")
        if qids != [record.get("qid") for record in records if isinstance(record, Mapping)]:
            raise ValueError(f"shared-probe {split} QID order does not match records")
        if len(qids) != checked_counts[split]["total"] or payload.get("qids") != qids or payload.get("records") != records:
            raise ValueError(f"shared-probe {split} count or payload mismatch")
        expected_counts = {
            stratum: sum(record.get("baseline_stratum") == stratum for record in records if isinstance(record, Mapping))
            for stratum in _STRATA
        }
        if payload.get("baseline_counts") != expected_counts:
            raise ValueError(f"shared-probe {split} baseline counts mismatch")
        if split in _BALANCED_SPLITS and expected_counts != {stratum: checked_counts[split][stratum] for stratum in _STRATA}:
            raise ValueError(f"shared-probe {split} is not balanced")
        for record in records:
            if not isinstance(record, Mapping):
                raise ValueError(f"shared-probe {split} record is invalid")
            if _FORBIDDEN_OUTCOME_KEYS & set(record):
                raise ValueError("outcome field is not permitted in sealed cohort")
            flat.append(record)
            qid = record.get("qid")
            if not isinstance(qid, str) or not qid:
                raise ValueError("sealed cohort QID is invalid")
            flat_qids.append(qid)
    if len(flat_qids) != cohort.get("question_count") or len(set(flat_qids)) != len(flat_qids):
        raise ValueError("shared-probe selected QIDs are not unique")
    if len(flat_qids) != cohort.get("selected_component_count"):
        raise ValueError("shared-probe selected component count mismatch")
    exclusions = cohort.get("excluded_qids")
    if not isinstance(exclusions, list) or len(exclusions) != 148 or len(set(exclusions)) != 148:
        raise ValueError("shared-probe prior-cohort exclusion inventory is invalid")
    if set(flat_qids) & set(exclusions):
        raise ValueError("shared-probe selected QID overlaps an excluded prior cohort")
    if cohort.get("support_components") != _components_for_records(flat):
        raise ValueError("shared-probe component identity mismatch")
    support_sets = [set(record["supporting_document_ids"]) for record in flat]
    if any(not left.isdisjoint(right) for index, left in enumerate(support_sets) for right in support_sets[index + 1:]):
        raise ValueError("support-document component crosses selected questions or splits")
    if authenticate_files:
        _authenticate_eligibility_records([_record_eligibility(record) for record in flat])
    if cohort.get("source_hashes") != _source_hash_inventory(flat):
        raise ValueError("shared-probe source hash inventory mismatch")
    input_hashes = cohort.get("input_file_hashes")
    if not isinstance(input_hashes, Mapping):
        raise ValueError("shared-probe input hash inventory is invalid")
    for path, digest in input_hashes.items():
        checked = _require_sha256(digest, label=f"input hash for {path}")
        if not isinstance(path, str) or not path.startswith("/"):
            raise ValueError("input file hash paths must be absolute")
        if authenticate_files and checked != _sha256_file(Path(path)):
            raise ValueError(f"shared-probe input file SHA-256 mismatch: {path}")
    return dict(cohort)


def publish_task9_shared_probe_cohort(
    cohort: Mapping[str, object], destination: Path, *, authenticate_files: bool = True
) -> str:
    """Publish a canonical cohort JSON atomically without replacement."""
    validated = validate_task9_shared_probe_cohort(cohort, authenticate_files=authenticate_files)
    output = Path(destination)
    if not output.is_absolute():
        raise ValueError("shared-probe cohort output must be an absolute path")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"shared-probe cohort output already exists: {output}")
    payload = (json.dumps(validated, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, staging_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    staging = Path(staging_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(staging, output, follow_symlinks=False)
        except FileExistsError as error:
            raise FileExistsError(f"shared-probe cohort output already exists: {output}") from error
        directory = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        staging.unlink(missing_ok=True)
    return str(validated["cohort_sha256"])


build_shared_probe_cohort = build_task9_shared_probe_cohort
validate_shared_probe_cohort = validate_task9_shared_probe_cohort
publish_shared_probe_cohort = publish_task9_shared_probe_cohort
seal_task9_shared_probe_cohort = publish_task9_shared_probe_cohort
seal_shared_probe_cohort = publish_task9_shared_probe_cohort

__all__ = [
    "DEFAULT_SHARED_PROBE_SPLITS",
    "build_task9_shared_probe_cohort",
    "build_shared_probe_cohort",
    "validate_task9_shared_probe_cohort",
    "validate_shared_probe_cohort",
    "publish_task9_shared_probe_cohort",
    "publish_shared_probe_cohort",
    "seal_task9_shared_probe_cohort",
    "seal_shared_probe_cohort",
]
