"""Concrete, fail-closed integration factory for the pinned M3DocVQA run."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import subprocess
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from docprune.answerers import (
    AllKeptQwenAnswerer,
    DocPruneQwenAnswerer,
    _resolved_eos_token_ids,
)
from docprune.artifacts import IndexManifest
from docprune.benchmark_config import (
    COLPALI_BACKBONE_MODEL,
    COLPALI_BACKBONE_REVISION,
    COLPALI_MODEL,
    COLPALI_REVISION,
    M3DOCRAG_COMMIT,
    MAX_NEW_TOKENS,
    MODES,
    PAGE_COUNTS,
    QWEN_EOS_TOKEN_IDS,
    QWEN_MODEL,
    QWEN_REVISION,
    SHORT_ANSWER_TEMPLATE,
    BenchmarkRunConfig,
    CorpusIdentity,
    sha256_file,
)
from docprune.cli import EvaluationWorkload
from docprune.colpali.compat import assert_supported_colpali
from docprune.config import DocPruneConfig
from docprune.ctp_controls import VisualTokenGeometry
from docprune.ctp_policy import (
    CTPPolicy,
    PolicySelectionContext,
    aggregate_native_threshold_policy,
    aggregate_score_top_m_policy,
    btp_qtp_no_ctp_policy,
    literal_native_threshold_policy,
    literal_score_top_m_policy,
)
from docprune.indexing import IndexBuildConfig, IndexBuildResult, build_index
from docprune.m3docrag import DocPruneM3DocRAG, OfficialM3DocRAGBoundary, SampleInput
from docprune.metrics import measurement_identity
from docprune.processor_probe import validate_processor_contract
from docprune.task6_runtime import (
    AuthenticatedFixedPageLoader,
    AuthenticatedFixedPageRetriever,
    FixedPageFixture,
    load_fixed_page_fixture,
    render_task6_pdf_page,
    task6_policy,
)

DEFAULT_FACTORY = "docprune.m3docvqa_factory:build_workload"

_EXPECTED_CONTRACT_RESOURCES = {
    "qwen": {"model": QWEN_MODEL, "revision": QWEN_REVISION},
    "colpali": {"model": COLPALI_MODEL, "revision": COLPALI_REVISION},
    "colpali_backbone": {
        "model": COLPALI_BACKBONE_MODEL,
        "revision": COLPALI_BACKBONE_REVISION,
    },
}
_CLI_PLACEHOLDER_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "command",
        "config",
        "factory",
        "output",
        "mode",
        "run_config",
        "run_config_sha256",
        "index_manifest",
        "index_manifest_sha256",
        "limit",
        "sample_ids",
        "page_count",
        "upstream",
        "paper_values",
        "reconstruction_defaults",
        "measurement",
    }
)


def _value(container: object, name: str, default: object = None) -> Any:
    if isinstance(container, Mapping):
        return container.get(name, default)
    return getattr(container, name, default)


def _required(container: object, name: str) -> Any:
    value = _value(container, name)
    if value is None:
        raise ValueError(f"run configuration is missing {name}")
    return value


def _sha256_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _is_cli_placeholder(
    existing: Mapping[str, object],
    payload: Mapping[str, object],
    *,
    invocation: Mapping[str, object] | None = None,
) -> bool:
    if invocation is None:
        return False
    if any(existing.get(key) != invocation.get(key) for key in _CLI_PLACEHOLDER_KEYS):
        return False
    return (
        set(existing) == _CLI_PLACEHOLDER_KEYS
        and existing.get("schema_version") == 2
        and existing.get("status") == "configured"
        and existing.get("command") == payload.get("operation")
        and existing.get("mode") == payload.get("mode")
        and existing.get("page_count") == payload.get("page_count")
        and existing.get("output") == payload.get("output")
        and "runtime_commit" not in existing
    )


def _source_file_identity(value: str | Path | None, *, label: str) -> tuple[str | None, str | None]:
    if value is None:
        return None, None
    path = Path(value)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file: {path}")
    resolved = path.resolve()
    return str(resolved), sha256_file(resolved)


def validate_processor_contract_file(path: Path) -> dict[str, object]:
    """Read and validate the immutable processor-probe contract."""

    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"processor contract is missing or not regular: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"processor contract is not JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError("processor contract must be a JSON object")
    if payload.get("schema_version") != 2:
        raise ValueError("processor contract schema_version must equal 2")
    validate_processor_contract(payload)
    resources = payload.get("resources")
    if not isinstance(resources, Mapping):
        raise ValueError("processor contract is missing resources")
    if dict(resources) != _EXPECTED_CONTRACT_RESOURCES:
        raise ValueError("processor contract resources do not match the pinned resources")
    return payload


def filter_samples(
    samples: Iterable[SampleInput],
    *,
    limit: int | None = None,
    sample_ids: Sequence[str] | None = None,
) -> tuple[SampleInput, ...]:
    """Select source-order samples without changing the immutable corpus order."""

    values = tuple(samples)
    source_ids = tuple(sample.question_id for sample in values)
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("source samples contain duplicate question IDs")
    requested = None if sample_ids is None else tuple(str(value) for value in sample_ids)
    if requested is not None and len(requested) != len(set(requested)):
        raise ValueError("sample IDs contain a duplicate")
    requested_set = None if requested is None else set(requested)
    if requested_set is not None:
        unknown = requested_set - set(source_ids)
        if unknown:
            raise ValueError(
                f"sample IDs are not present in the source corpus: {sorted(unknown)!r}"
            )
    selected = tuple(
        sample for sample in values if requested_set is None or sample.question_id in requested_set
    )
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        selected = selected[:limit]
    return selected


def load_completed_qids(
    path: Path,
    *,
    expected_qids: Sequence[str] | None = None,
    expected_samples: Sequence[SampleInput] | None = None,
    expected_page_count: int | None = None,
    production: bool = False,
    expected_policy: Mapping[str, object] | None = None,
    forbid_policy: bool = False,
) -> set[str]:
    """Validate a result JSONL's qid set before permitting resume."""

    if expected_page_count is not None and expected_page_count not in PAGE_COUNTS:
        raise ValueError("expected_page_count must be 1, 2, or 4")
    path = Path(path)
    if path.is_symlink():
        raise ValueError("results JSONL must be a regular file")
    if expected_qids is None and expected_samples is None:
        raise ValueError("resume validation requires expected question IDs or samples")
    if not path.exists():
        return set()
    if not path.is_file():
        raise ValueError("results JSONL must be a regular file")
    expected_by_qid = (
        {sample.question_id: sample for sample in expected_samples}
        if expected_samples is not None
        else {}
    )
    expected = set(expected_qids or expected_by_qid)
    completed: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"results JSONL contains a blank line at {line_number}")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"results JSONL is invalid at line {line_number}") from error
            if not isinstance(record, Mapping):
                raise ValueError(f"results JSONL record {line_number} is not an object")
            record = _canonicalize_result_record(record, line_number=line_number)
            _validate_result_record(
                record,
                line_number=line_number,
                expected_page_count=expected_page_count,
                production=production,
                expected_policy=expected_policy,
                forbid_policy=forbid_policy,
            )
            qid = record["question_id"]
            if not isinstance(qid, str) or not qid:
                raise ValueError(f"results JSONL record {line_number} is missing question_id")
            if qid not in expected:
                raise ValueError(f"results JSONL contains unexpected question ID: {qid}")
            if qid in completed:
                raise ValueError(f"results JSONL contains duplicate question ID: {qid}")
            if qid in expected_by_qid:
                sample = expected_by_qid[qid]
                if record.get("question") != sample.question:
                    raise ValueError(f"results JSONL question drift for question ID: {qid}")
                answers = record.get("answers")
                if (
                    not isinstance(answers, list)
                    or tuple(str(value) for value in answers) != sample.answers
                ):
                    raise ValueError(f"results JSONL answers drift for question ID: {qid}")
            completed.add(qid)
    return completed


def _canonicalize_result_record(
    record: Mapping[str, object], *, line_number: int
) -> dict[str, object]:
    normalized = dict(record)
    question_id = normalized.get("question_id")
    legacy_qid = normalized.get("qid")
    if question_id is not None and legacy_qid is not None and question_id != legacy_qid:
        raise ValueError(f"results JSONL record {line_number} has conflicting question IDs")
    if question_id is None and legacy_qid is not None:
        normalized["question_id"] = legacy_qid
    normalized.pop("qid", None)
    return normalized


def _validate_policy_selection_record(value: object, *, line_number: int) -> None:
    """Validate durable native/ranking evidence independently of Task 3 forced records."""

    if not isinstance(value, Mapping):
        raise ValueError(f"results JSONL record {line_number} has invalid policy selection")
    required = {
        "policy",
        "boundary",
        "native_layer",
        "visual_population",
        "requested_budget",
        "achieved_budget",
        "retained_compact_visual_ids",
        "aggregate_native_reference_ids",
        "aggregate_threshold_tied_ids",
        "symmetric_difference_ids",
        "native_threshold",
        "seed_sha256",
        "geometry_count",
        "geometry_sha256",
        "prefill_cache_lengths",
        "retained_mrope_position_shape",
        "retained_mrope_position_sha256",
    }
    if set(value) != required:
        raise ValueError(f"results JSONL record {line_number} has invalid policy selection")
    policy_data = value["policy"]
    if not isinstance(policy_data, Mapping):
        raise ValueError(f"results JSONL record {line_number} has invalid policy selection")
    policy_keys = {
        "family",
        "name",
        "selection_kind",
        "score_semantics",
        "budget_source",
        "fixed_retention",
    }
    if set(policy_data) != policy_keys or not all(
        isinstance(policy_data[key], str) for key in policy_keys - {"fixed_retention"}
    ):
        raise ValueError(f"results JSONL record {line_number} has invalid policy selection")
    fixed = policy_data["fixed_retention"]
    try:
        fixed_retention = None if fixed is None else Fraction(str(fixed))
        policy = CTPPolicy(
            str(policy_data["name"]),
            str(policy_data["family"]),
            str(policy_data["selection_kind"]),
            str(policy_data["score_semantics"]),
            str(policy_data["budget_source"]),
            fixed_retention,
        )
    except (TypeError, ValueError, ZeroDivisionError) as error:
        raise ValueError(
            f"results JSONL record {line_number} has invalid policy selection"
        ) from error
    if policy.to_dict() != dict(policy_data):
        raise ValueError(f"results JSONL record {line_number} has invalid policy selection")
    population = value["visual_population"]
    if not isinstance(population, int) or isinstance(population, bool) or population < 0:
        raise ValueError(f"results JSONL record {line_number} has invalid policy selection")
    geometry_count = value["geometry_count"]
    geometry_sha256 = value["geometry_sha256"]
    if (geometry_count is None) != (geometry_sha256 is None):
        raise ValueError(f"results JSONL record {line_number} has invalid policy selection")
    if geometry_count is not None and (
        not isinstance(geometry_count, int)
        or isinstance(geometry_count, bool)
        or geometry_count != population
        or not isinstance(geometry_sha256, str)
        or len(geometry_sha256) != 64
        or any(character not in "0123456789abcdef" for character in geometry_sha256)
    ):
        raise ValueError(f"results JSONL record {line_number} has invalid policy selection")
    cache_lengths = value["prefill_cache_lengths"]
    position_shape = value["retained_mrope_position_shape"]
    position_sha256 = value["retained_mrope_position_sha256"]
    empty_runtime_evidence = (
        cache_lengths == [] and position_shape is None and position_sha256 is None
    )
    valid_runtime_evidence = (
        isinstance(cache_lengths, list)
        and bool(cache_lengths)
        and all(type(length) is int and length >= 0 for length in cache_lengths)
        and isinstance(position_shape, list)
        and len(position_shape) == 3
        and all(type(size) is int and size >= 0 for size in position_shape)
        and isinstance(position_sha256, str)
        and len(position_sha256) == 64
        and all(character in "0123456789abcdef" for character in position_sha256)
    )
    if not empty_runtime_evidence and not valid_runtime_evidence:
        raise ValueError(f"results JSONL record {line_number} has invalid policy selection")
    for name in ("requested_budget", "achieved_budget"):
        budget = value[name]
        if not isinstance(budget, int) or isinstance(budget, bool) or not 0 <= budget <= population:
            raise ValueError(f"results JSONL record {line_number} has invalid policy selection")
    retained = value["retained_compact_visual_ids"]
    tied = value["aggregate_threshold_tied_ids"]
    difference = value["symmetric_difference_ids"]
    if not all(isinstance(items, list) for items in (retained, tied, difference)):
        raise ValueError(f"results JSONL record {line_number} has invalid policy selection")
    for items in (retained, tied, difference):
        if items != sorted(set(items)) or any(
            not isinstance(identifier, int)
            or isinstance(identifier, bool)
            or not 0 <= identifier < population
            for identifier in items
        ):
            raise ValueError(f"results JSONL record {line_number} has invalid policy selection")
    if value["requested_budget"] != len(retained) or value["achieved_budget"] != len(retained):
        raise ValueError(f"results JSONL record {line_number} has invalid policy selection")
    reference = value["aggregate_native_reference_ids"]
    if reference is not None and (
        not isinstance(reference, list)
        or reference != sorted(set(reference))
        or any(
            not isinstance(identifier, int)
            or isinstance(identifier, bool)
            or not 0 <= identifier < population
            for identifier in reference
        )
    ):
        raise ValueError(f"results JSONL record {line_number} has invalid policy selection")
    if reference is not None and difference != sorted(
        set(retained).symmetric_difference(reference)
    ):
        raise ValueError(f"results JSONL record {line_number} has invalid policy selection")
    boundary, native_layer = value["boundary"], value["native_layer"]
    if boundary is None or native_layer is None:
        if boundary is not None or native_layer is not None:
            raise ValueError(f"results JSONL record {line_number} has invalid policy selection")
        if (
            retained != list(range(population))
            or value["requested_budget"] != population
            or value["achieved_budget"] != population
            or reference is not None
            or tied
            or difference
            or value["native_threshold"] is not None
            or value["seed_sha256"] is not None
        ):
            raise ValueError(f"results JSONL record {line_number} has invalid policy selection")
        return
    if (
        not isinstance(boundary, str)
        or not boundary.startswith("B_")
        or not isinstance(native_layer, int)
        or isinstance(native_layer, bool)
        or native_layer < 0
        or boundary != f"B_{native_layer}"
    ):
        raise ValueError(f"results JSONL record {line_number} has invalid policy selection")
    seed = value["seed_sha256"]
    if seed is not None and (
        not isinstance(seed, str)
        or len(seed) != 64
        or any(character not in "0123456789abcdef" for character in seed)
    ):
        raise ValueError(f"results JSONL record {line_number} has invalid policy selection")
    if policy.family in {"random-top-m", "coverage-top-m"} and seed is None:
        raise ValueError(f"results JSONL record {line_number} has invalid policy selection")
    if policy.family in {"random-top-m", "coverage-top-m"} and geometry_count is None:
        raise ValueError(f"results JSONL record {line_number} has invalid policy selection")
    if policy.family not in {"random-top-m", "coverage-top-m"} and seed is not None:
        raise ValueError(f"results JSONL record {line_number} has invalid policy selection")
    if policy.family == "no-ctp":
        raise ValueError(f"results JSONL record {line_number} has invalid policy selection")
    threshold = value["native_threshold"]
    if (
        not isinstance(threshold, int | float)
        or isinstance(threshold, bool)
        or not math.isfinite(float(threshold))
    ):
        raise ValueError(f"results JSONL record {line_number} has invalid policy selection")
    if policy.family == "native-threshold":
        if policy.score_semantics.startswith("literal-"):
            if reference is not None or tied or difference:
                raise ValueError(f"results JSONL record {line_number} has invalid policy selection")
        elif reference is None or retained != reference or difference:
            raise ValueError(f"results JSONL record {line_number} has invalid policy selection")
    elif reference is None:
        raise ValueError(f"results JSONL record {line_number} has invalid policy selection")
    if reference is not None and not set(tied).issubset(reference):
        raise ValueError(f"results JSONL record {line_number} has invalid policy selection")
    if policy.fixed_retention is not None:
        if value["requested_budget"] != policy.resolve_budget(
            population, aggregate_native_budget=None
        ):
            raise ValueError(f"results JSONL record {line_number} has invalid policy selection")
    elif policy.family == "score-top-m" and value["requested_budget"] != len(reference):
        raise ValueError(f"results JSONL record {line_number} has invalid policy selection")
    if policy.name == "aggregate-score-top-m" and not tied:
        if retained != reference:
            raise ValueError(f"results JSONL record {line_number} has invalid policy selection")


def _validate_forced_intervention_record(value: object, *, line_number: int) -> None:
    """Keep the admitted Task 3 forced record strict while adding policy evidence."""

    if not isinstance(value, Mapping):
        raise ValueError(f"results JSONL record {line_number} has invalid forced intervention")
    required = {
        "boundary",
        "mode",
        "selection_kind",
        "visual_population",
        "requested_budget",
        "achieved_budget",
        "retained_visual_ids",
        "logical_retained_sequence_ids",
        "prefill_cache_lengths",
        "retained_mrope_position_shape",
        "retained_mrope_position_sha256",
    }
    if set(value) != required or value.get("selection_kind") != "forced":
        raise ValueError(f"results JSONL record {line_number} has invalid forced intervention")
    population = value.get("visual_population")
    requested, achieved = value.get("requested_budget"), value.get("achieved_budget")
    retained = value.get("retained_visual_ids")
    if (
        not isinstance(value.get("boundary"), str)
        or not str(value["boundary"]).startswith("B_")
        or value.get("mode") not in {"physical_delete", "zero_mask"}
        or not isinstance(population, int)
        or isinstance(population, bool)
        or population < 0
        or not isinstance(requested, int)
        or not isinstance(achieved, int)
        or isinstance(requested, bool)
        or isinstance(achieved, bool)
        or requested != achieved
        or not 0 <= requested <= population
        or not isinstance(retained, list)
        or retained != sorted(set(retained))
        or len(retained) != achieved
        or any(
            not isinstance(item, int) or isinstance(item, bool) or not 0 <= item < population
            for item in retained
        )
    ):
        raise ValueError(f"results JSONL record {line_number} has invalid forced intervention")
    logical_ids = value.get("logical_retained_sequence_ids")
    cache_lengths = value.get("prefill_cache_lengths")
    removed_visual_count = population - achieved if value.get("mode") == "physical_delete" else 0
    if (
        not isinstance(logical_ids, list)
        or not logical_ids
        or logical_ids != sorted(set(logical_ids))
        or any(type(item) is not int or item < 0 for item in logical_ids)
        or max(logical_ids) >= len(logical_ids) + removed_visual_count
        or not isinstance(cache_lengths, list)
        or not cache_lengths
        or any(type(item) is not int or item < 0 for item in cache_lengths)
    ):
        raise ValueError(f"results JSONL record {line_number} has invalid forced intervention")
    shape = value.get("retained_mrope_position_shape")
    digest = value.get("retained_mrope_position_sha256")
    if (
        shape != [3, 1, len(logical_ids)]
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"results JSONL record {line_number} has invalid forced intervention")


def _validate_result_record(
    record: Mapping[str, object],
    *,
    line_number: int,
    expected_page_count: int | None = None,
    mode: str | None = None,
    production: bool = False,
    expected_policy: Mapping[str, object] | None = None,
    forbid_policy: bool = False,
    allowed_extra_fields: set[str] | frozenset[str] = frozenset(),
    allow_zero_post_ctp: bool = False,
) -> None:
    required = {
        "question_id",
        "question",
        "answers",
        "predicted_answer",
        "retrieved_pages",
        "trace",
        "timing",
    }
    fields = set(record)
    missing = required - fields
    if missing:
        raise ValueError(f"results JSONL record {line_number} is missing {sorted(missing)!r}")
    if any(not isinstance(field, str) for field in allowed_extra_fields):
        raise TypeError("allowed result fields must be strings")
    if not isinstance(allow_zero_post_ctp, bool):
        raise TypeError("allow_zero_post_ctp must be a boolean")
    extra = (
        fields
        - required
        - {
            "policy_selection",
            "forced_intervention",
            "fixed_page_fixture_sha256",
            "fixed_page_provenance",
            "global_index_loaded",
            "policy_context",
        }
        - set(allowed_extra_fields)
    )
    if extra:
        raise ValueError(f"results JSONL record {line_number} has unknown fields {sorted(extra)!r}")
    if "policy_selection" in record and "forced_intervention" in record:
        raise ValueError(
            f"results JSONL record {line_number} cannot contain both policy and forced evidence"
        )
    if expected_policy is not None:
        if record.get("policy_selection") is None:
            raise ValueError(
                f"results JSONL record {line_number} is missing manifest policy evidence"
            )
        selection_policy = (
            record["policy_selection"].get("policy")
            if isinstance(record["policy_selection"], Mapping)
            else None
        )
        if selection_policy != expected_policy:
            raise ValueError(
                f"results JSONL record {line_number} policy evidence does not match manifest policy"
            )
    elif forbid_policy and "policy_selection" in record:
        raise ValueError(f"results JSONL record {line_number} manifest forbids policy evidence")
    if "policy_selection" in record:
        _validate_policy_selection_record(record["policy_selection"], line_number=line_number)
    if "forced_intervention" in record:
        _validate_forced_intervention_record(record["forced_intervention"], line_number=line_number)
    if not isinstance(record["question_id"], str) or not record["question_id"]:
        raise ValueError(f"results JSONL record {line_number} has an invalid question ID")
    if not isinstance(record["question"], str) or not isinstance(record["predicted_answer"], str):
        raise ValueError(f"results JSONL record {line_number} has invalid answer fields")
    answers = record["answers"]
    if not isinstance(answers, list) or not all(isinstance(answer, str) for answer in answers):
        raise ValueError(f"results JSONL record {line_number} has invalid answers")
    pages = record["retrieved_pages"]
    if not isinstance(pages, list) or not pages:
        raise ValueError(f"results JSONL record {line_number} has invalid retrieved_pages")
    if expected_page_count is not None and len(pages) != expected_page_count:
        raise ValueError(
            f"results JSONL record {line_number} has the wrong page count: "
            f"expected {expected_page_count}, got {len(pages)}"
        )
    page_ids: set[tuple[str, int]] = set()
    for page in pages:
        if not isinstance(page, Mapping):
            raise ValueError(f"results JSONL record {line_number} has an invalid page")
        doc_id = page.get("doc_id")
        page_index = page.get("page_index")
        score = page.get("score")
        if (
            not isinstance(doc_id, str)
            or not doc_id
            or not isinstance(page_index, int)
            or isinstance(page_index, bool)
            or page_index < 0
            or not isinstance(score, int | float)
            or isinstance(score, bool)
            or not math.isfinite(float(score))
        ):
            raise ValueError(f"results JSONL record {line_number} has an invalid page")
        identity = (doc_id, page_index)
        if identity in page_ids:
            raise ValueError(f"results JSONL record {line_number} has duplicate retrieved pages")
        page_ids.add(identity)
    trace = record["trace"]
    if not isinstance(trace, Mapping):
        raise ValueError(f"results JSONL record {line_number} has an invalid trace")
    trace_field_order = (
        "original_visual_tokens",
        "post_btp_visual_tokens",
        "post_qtp_visual_tokens",
        "post_ctp_visual_tokens",
        "ctp_layer",
    )
    trace_fields = set(trace_field_order)
    if set(trace) != trace_fields:
        raise ValueError(f"results JSONL record {line_number} has an invalid trace schema")
    counts = [trace[name] for name in trace_field_order[:4]]
    early_counts, post_ctp_count = counts[:3], counts[3]
    if (
        any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in early_counts
        )
        or not isinstance(post_ctp_count, int)
        or isinstance(post_ctp_count, bool)
        or post_ctp_count < 0
        or (post_ctp_count == 0 and not allow_zero_post_ctp)
    ):
        raise ValueError(f"results JSONL record {line_number} has non-positive trace counts")
    if not all(left >= right for left, right in zip(counts, counts[1:])):
        raise ValueError(f"results JSONL record {line_number} trace is not monotonic")
    ctp_layer = trace["ctp_layer"]
    if ctp_layer is not None and (
        not isinstance(ctp_layer, int) or isinstance(ctp_layer, bool) or not 0 <= ctp_layer < 28
    ):
        raise ValueError(f"results JSONL record {line_number} has invalid ctp_layer")
    if mode == "all-kept" and (len(set(counts)) != 1 or ctp_layer is not None):
        raise ValueError(f"results JSONL record {line_number} violates the all-kept trace contract")
    if mode == "docprune" and ctp_layer is None and counts[3] != counts[2]:
        raise ValueError(
            f"results JSONL record {line_number} has a docprune trace without a CTP decision"
        )
    selection = record.get("policy_selection")
    if isinstance(selection, Mapping):
        native_layer = selection["native_layer"]
        if native_layer != ctp_layer:
            raise ValueError(
                f"results JSONL record {line_number} policy selection does not match the CTP trace"
            )
        if selection["achieved_budget"] != counts[3]:
            raise ValueError(
                f"results JSONL record {line_number} policy selection budget disagrees with trace"
            )
        if selection["visual_population"] != counts[2]:
            raise ValueError(
                f"results JSONL record {line_number} policy selection population disagrees with trace"
            )
    forced = record.get("forced_intervention")
    if isinstance(forced, Mapping):
        if ctp_layer is not None:
            raise ValueError(
                f"results JSONL record {line_number} forced intervention disagrees with trace layer"
            )
        if forced["visual_population"] != counts[2]:
            raise ValueError(
                f"results JSONL record {line_number} forced intervention population disagrees with trace"
            )
        if forced["mode"] == "physical_delete" and forced["achieved_budget"] != counts[3]:
            raise ValueError(
                f"results JSONL record {line_number} forced intervention budget disagrees with trace"
            )
        if forced["mode"] == "zero_mask" and counts[3] != counts[2]:
            raise ValueError(
                f"results JSONL record {line_number} zero-mask forced intervention disagrees with trace"
            )
    timing = record["timing"]
    timing_fields = {"retrieval_seconds", "qa_seconds"}
    optional_timing_fields = {
        "encoder_seconds",
        "decoder_seconds",
        "page_load_seconds",
        "total_sample_seconds",
        "peak_allocated_gpu_bytes",
        "warmup_excluded",
        "profiler_enabled",
        "profiler_definition",
        "flops",
    }
    if not isinstance(timing, Mapping) or not timing_fields <= set(timing):
        raise ValueError(f"results JSONL record {line_number} has invalid timing schema")
    production_timing_fields = {
        "retrieval_seconds",
        "page_load_seconds",
        "qa_seconds",
        "total_sample_seconds",
        "encoder_seconds",
        "decoder_seconds",
    }
    if production and not production_timing_fields <= set(timing):
        raise ValueError(
            f"results JSONL record {line_number} production timing is missing stage boundaries: "
            f"{sorted(production_timing_fields - set(timing))!r}"
        )
    if set(timing) - timing_fields - optional_timing_fields:
        raise ValueError(f"results JSONL record {line_number} has invalid timing schema")
    if any(
        not isinstance(timing[name], int | float)
        or isinstance(timing[name], bool)
        or not math.isfinite(float(timing[name]))
        or (float(timing[name]) <= 0 if production else float(timing[name]) < 0)
        for name in timing
        if name
        in {
            "retrieval_seconds",
            "qa_seconds",
            "encoder_seconds",
            "decoder_seconds",
            "page_load_seconds",
            "total_sample_seconds",
        }
    ):
        raise ValueError(f"results JSONL record {line_number} has invalid timings")
    if "peak_allocated_gpu_bytes" in timing and (
        not isinstance(timing["peak_allocated_gpu_bytes"], int)
        or isinstance(timing["peak_allocated_gpu_bytes"], bool)
        or timing["peak_allocated_gpu_bytes"] < 0
    ):
        raise ValueError(f"results JSONL record {line_number} has invalid peak GPU bytes")
    if "warmup_excluded" in timing and not isinstance(timing["warmup_excluded"], bool):
        raise ValueError(f"results JSONL record {line_number} has invalid warmup flag")
    if "profiler_enabled" not in timing:
        raise ValueError(f"results JSONL record {line_number} is missing profiler_enabled")
    profiler_enabled = timing["profiler_enabled"]
    if not isinstance(profiler_enabled, bool):
        raise ValueError(f"results JSONL record {line_number} has invalid profiler flag")
    if profiler_enabled:
        definition = timing.get("profiler_definition")
        flops = timing.get("flops")
        if not isinstance(definition, str) or not definition:
            raise ValueError(f"results JSONL record {line_number} has invalid profiler definition")
        if (
            not isinstance(flops, int | float)
            or isinstance(flops, bool)
            or not math.isfinite(float(flops))
            or float(flops) < 0
        ):
            raise ValueError(f"results JSONL record {line_number} has invalid profiler FLOPs")
    elif "profiler_definition" in timing or "flops" in timing:
        raise ValueError(
            f"results JSONL record {line_number} records profiler data while profiling is disabled"
        )


def _resolve_mode(mode: str | None, run_config: object | None) -> str:
    resolved = mode or _value(run_config, "mode") or os.environ.get("DOCPRUNE_MODE", "all-kept")
    if resolved not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}")
    return str(resolved)


def _resolve_run_config(
    run_config: BenchmarkRunConfig | Path | Mapping[str, object] | None,
    *,
    mode: str,
    page_count: int,
) -> object:
    if isinstance(run_config, BenchmarkRunConfig):
        resolved: object = run_config
    elif run_config is None:
        resolved = BenchmarkRunConfig.from_env(mode, page_count)
    elif isinstance(run_config, str | Path):
        path = Path(run_config)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"run configuration is missing: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError("run configuration file must be JSON") from error
        if not isinstance(payload, Mapping):
            raise ValueError("run configuration file must contain a JSON object")
        resolved = _normalise_run_config_mapping(payload, base=path.parent)
    elif isinstance(run_config, Mapping):
        resolved = _normalise_run_config_mapping(run_config)
    else:
        resolved = run_config
    if str(_required(resolved, "mode")) != mode:
        raise ValueError("run configuration mode does not match requested mode")
    if int(_required(resolved, "page_count")) != page_count:
        raise ValueError("run configuration page_count does not match requested page_count")
    return resolved


def _normalise_run_config_mapping(
    payload: Mapping[str, object], *, base: Path | None = None
) -> object:
    """Build a complete run-config object without consulting the environment."""

    if isinstance(payload.get("run_config"), Mapping):
        payload = payload["run_config"]  # type: ignore[assignment]
    values = dict(payload)
    resources = values.get("resources")
    if isinstance(resources, Mapping):
        for field, group in (
            ("qwen_model", "qwen"),
            ("qwen_revision", "qwen"),
            ("colpali_model", "colpali"),
            ("colpali_revision", "colpali"),
            ("colpali_backbone_model", "colpali_backbone"),
            ("colpali_backbone_revision", "colpali_backbone"),
        ):
            if field not in values and isinstance(resources.get(group), Mapping):
                suffix = "model" if field.endswith("model") else "revision"
                values[field] = resources[group].get(suffix)
            elif field in values and isinstance(resources.get(group), Mapping):
                suffix = "model" if field.endswith("model") else "revision"
                nested_value = resources[group].get(suffix)
                if nested_value is not None and values[field] != nested_value:
                    raise ValueError(f"run configuration {field} conflicts with resources")
    generation = values.get("generation")
    if isinstance(generation, Mapping):
        for field in ("max_new_tokens", "do_sample", "num_beams"):
            if field in generation:
                if field in values and values[field] != generation[field]:
                    raise ValueError(f"run configuration {field} conflicts with generation")
                values[field] = generation[field]
        generation_prompt = generation.get("prompt", generation.get("short_answer_template"))
        if generation_prompt is not None:
            if "prompt" in values and values["prompt"] != generation_prompt:
                raise ValueError("run configuration prompt conflicts with generation")
            values["prompt"] = generation_prompt
    required = (
        "mode",
        "page_count",
        "runtime_commit",
        "m3docrag_commit",
        "qwen_model",
        "qwen_revision",
        "colpali_model",
        "colpali_revision",
        "colpali_backbone_model",
        "colpali_backbone_revision",
        "processor_contract_path",
        "corpus",
        "max_new_tokens",
        "do_sample",
        "num_beams",
        "prompt",
    )
    missing = [name for name in required if values.get(name) is None]
    if missing:
        raise ValueError(f"run configuration is missing required fields: {', '.join(missing)}")
    contract = Path(str(values["processor_contract_path"]))
    if base is not None and not contract.is_absolute():
        contract = (base / contract).resolve()
    values["processor_contract_path"] = contract
    corpus = values["corpus"]
    if isinstance(corpus, Mapping):
        corpus_values = dict(corpus)
        path_fields = (
            "root",
            "questions_path",
            "document_ids_path",
            "pdf_dir",
            "integrity_report_path",
            "archive_checksum_manifest_path",
        )
        if base is not None:
            for field in path_fields:
                if field in corpus_values and not Path(str(corpus_values[field])).is_absolute():
                    corpus_values[field] = str((base / str(corpus_values[field])).resolve())
        corpus_required = (
            *path_fields,
            "integrity_sha256",
            "archive_checksum_manifest_sha256",
            "questions_sha256",
            "document_ids_sha256",
            "expected_question_count",
            "expected_pdf_count",
            "expected_page_count",
        )
        missing_corpus = [name for name in corpus_required if corpus_values.get(name) is None]
        if missing_corpus:
            raise ValueError(
                "run configuration corpus is missing required fields: " + ", ".join(missing_corpus)
            )
        archive_hashes = corpus_values.get("archive_hashes", {})
        if not isinstance(archive_hashes, Mapping):
            raise ValueError("run configuration corpus archive_hashes must be a mapping")
        corpus_values["archive_hashes"] = dict(archive_hashes)
        if "is_fixture" not in corpus_values:
            corpus_values["is_fixture"] = False
        elif type(corpus_values["is_fixture"]) is not bool:
            raise ValueError("run configuration corpus is_fixture must be a boolean")
        try:
            corpus = CorpusIdentity(**corpus_values)
        except (TypeError, ValueError) as error:
            raise ValueError("run configuration corpus identity is invalid") from error
        values["corpus"] = corpus
    elif not callable(getattr(corpus, "validate", None)):
        raise ValueError("run configuration corpus must be a complete identity")
    return SimpleNamespace(**values)


def _corpus_identity_payload(corpus: object) -> dict[str, object]:
    """Serialize every immutable corpus input into the run identity."""

    path_fields = (
        "root",
        "questions_path",
        "document_ids_path",
        "pdf_dir",
        "integrity_report_path",
        "archive_checksum_manifest_path",
    )
    payload: dict[str, object] = {
        name: str(Path(_required(corpus, name)).resolve()) for name in path_fields
    }
    is_fixture = _value(corpus, "is_fixture", False)
    if type(is_fixture) is not bool:
        raise TypeError("corpus is_fixture must be a boolean")
    payload.update(
        {
            "integrity_sha256": str(_required(corpus, "integrity_sha256")),
            "archive_checksum_manifest_sha256": str(
                _required(corpus, "archive_checksum_manifest_sha256")
            ),
            "questions_sha256": str(_required(corpus, "questions_sha256")),
            "document_ids_sha256": str(_required(corpus, "document_ids_sha256")),
            "expected_question_count": int(_required(corpus, "expected_question_count")),
            "expected_pdf_count": int(_required(corpus, "expected_pdf_count")),
            "expected_page_count": int(_required(corpus, "expected_page_count")),
            "is_fixture": is_fixture,
            "archive_hashes": dict(_value(corpus, "archive_hashes", {})),
        }
    )
    return payload


def _validate_run_identity(run_config: object, *, mode: str, page_count: int) -> dict[str, object]:
    if str(_required(run_config, "m3docrag_commit")) != M3DOCRAG_COMMIT:
        raise ValueError(f"m3docrag checkout must use pinned commit {M3DOCRAG_COMMIT}")
    runtime_commit = str(_required(run_config, "runtime_commit"))
    if len(runtime_commit) != 40 or any(
        char not in "0123456789abcdefABCDEF" for char in runtime_commit
    ):
        raise ValueError("runtime_commit must be a 40-character hexadecimal commit hash")
    expected_resources = {
        "qwen_model": QWEN_MODEL,
        "qwen_revision": QWEN_REVISION,
        "colpali_model": COLPALI_MODEL,
        "colpali_revision": COLPALI_REVISION,
        "colpali_backbone_model": COLPALI_BACKBONE_MODEL,
        "colpali_backbone_revision": COLPALI_BACKBONE_REVISION,
    }
    for name, expected in expected_resources.items():
        if str(_required(run_config, name)) != expected:
            raise ValueError(f"{name} does not match its pinned revision/resource")
    contract = Path(_required(run_config, "processor_contract_path"))
    contract_payload = validate_processor_contract_file(contract)
    declared_contract_sha = _value(run_config, "processor_contract_sha256")
    actual_contract_sha = sha256_file(contract)
    if (
        declared_contract_sha is not None
        and str(declared_contract_sha).lower() != actual_contract_sha
    ):
        raise ValueError("processor contract SHA-256 does not match the run configuration")
    max_new_tokens = _required(run_config, "max_new_tokens")
    do_sample = _required(run_config, "do_sample")
    num_beams = _required(run_config, "num_beams")
    prompt = _required(run_config, "prompt")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
        raise ValueError("max_new_tokens must be an integer")
    if not isinstance(do_sample, bool):
        raise ValueError("do_sample must be a boolean")
    if isinstance(num_beams, bool) or not isinstance(num_beams, int):
        raise ValueError("num_beams must be an integer")
    if not isinstance(prompt, str):
        raise ValueError("prompt must be a string")
    if max_new_tokens != MAX_NEW_TOKENS:
        raise ValueError(f"max_new_tokens must equal {MAX_NEW_TOKENS}")
    if do_sample or num_beams != 1:
        raise ValueError("generation must use greedy decoding (do_sample=False, num_beams=1)")
    if prompt != SHORT_ANSWER_TEMPLATE:
        raise ValueError("prompt must equal the official short-answer prompt")
    corpus = _value(run_config, "corpus")
    if corpus is None:
        raise ValueError("run configuration is missing corpus identity")
    validate = getattr(corpus, "validate", None)
    if callable(validate):
        validate()
    return {
        "mode": mode,
        "page_count": page_count,
        "runtime_commit": runtime_commit.lower(),
        "m3docrag_commit": M3DOCRAG_COMMIT,
        "resources": {
            "qwen": {"model": QWEN_MODEL, "revision": QWEN_REVISION},
            "colpali": {"model": COLPALI_MODEL, "revision": COLPALI_REVISION},
            "colpali_backbone": {
                "model": COLPALI_BACKBONE_MODEL,
                "revision": COLPALI_BACKBONE_REVISION,
            },
        },
        "processor_contract_path": str(contract.resolve()),
        "processor_contract_sha256": actual_contract_sha,
        "processor_contract": contract_payload,
        "corpus": _corpus_identity_payload(corpus),
        "generation": {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "num_beams": num_beams,
            "eos_token_ids": list(QWEN_EOS_TOKEN_IDS),
            "prompt": prompt,
        },
    }


def _task6_execution_identity(
    feature_identity: Mapping[str, object], execution_runtime_commit: str
) -> dict[str, object]:
    """Separate historical feature construction from the code executing Task 6."""

    feature_commit = feature_identity.get("runtime_commit")
    if (
        not isinstance(feature_commit, str)
        or len(feature_commit) != 40
        or any(character not in "0123456789abcdefABCDEF" for character in feature_commit)
    ):
        raise ValueError("feature-build runtime commit must be a 40-character hexadecimal hash")
    if (
        not isinstance(execution_runtime_commit, str)
        or len(execution_runtime_commit) != 40
        or any(character not in "0123456789abcdefABCDEF" for character in execution_runtime_commit)
    ):
        raise ValueError("execution runtime commit must be a 40-character hexadecimal hash")
    return {
        **dict(feature_identity),
        "feature_build_runtime_commit": feature_commit.lower(),
        "runtime_commit": execution_runtime_commit.lower(),
    }


def _validate_m3docrag_checkout(run_config: object) -> Path:
    root_value = (
        _value(run_config, "m3docrag_root")
        or _value(run_config, "upstream_root")
        or os.environ.get("DOCPRUNE_M3DOCRAG_ROOT")
        or os.environ.get("M3DOCRAG_ROOT")
        or "/home/lmalveau/src/m3docrag-benchmark-29e6ac2"
    )
    root = Path(root_value)
    if not root.is_dir():
        raise FileNotFoundError(f"pinned M3DocRAG checkout is missing: {root}")
    try:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"unable to inspect pinned M3DocRAG checkout: {root}") from error
    if revision.lower() != M3DOCRAG_COMMIT:
        raise ValueError(
            f"M3DocRAG checkout revision mismatch: expected {M3DOCRAG_COMMIT}, got {revision}"
        )
    if dirty:
        raise ValueError(f"M3DocRAG checkout is dirty: {root}")
    return root


def _cached_snapshot(repo_id: str, revision: str) -> Path:
    """Resolve only an already-cached immutable HF snapshot."""

    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(repo_id=repo_id, revision=revision, local_files_only=True)
        return Path(path)
    except Exception as error:
        raise FileNotFoundError(
            f"cached model snapshot is missing for {repo_id}@{revision}"
        ) from error


def _load_colpali(run_config: object) -> tuple[object, object]:
    device = _require_benchmark_cuda()
    from colpali_engine.models import ColPali, ColPaliProcessor

    backbone = _cached_snapshot(COLPALI_BACKBONE_MODEL, COLPALI_BACKBONE_REVISION)
    adapter = _cached_snapshot(COLPALI_MODEL, COLPALI_REVISION)
    model = ColPali.from_pretrained(
        str(backbone), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.load_adapter(str(adapter))
    model = model.to(device).eval()
    processor = ColPaliProcessor.from_pretrained(str(adapter))
    assert_supported_colpali(model, processor)
    return model, processor


def _validate_qwen_generation_identity(model: object) -> None:
    observed = _resolved_eos_token_ids(model)
    if observed != QWEN_EOS_TOKEN_IDS:
        raise ValueError(f"Qwen generation EOS IDs must be {QWEN_EOS_TOKEN_IDS}, got {observed}")


def _load_qwen(run_config: object) -> tuple[object, object]:
    device = _require_benchmark_cuda()
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    snapshot = _cached_snapshot(QWEN_MODEL, QWEN_REVISION)
    model = (
        Qwen2VLForConditionalGeneration.from_pretrained(
            str(snapshot),
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
        )
        .to(device)
        .eval()
    )
    vision_config = getattr(getattr(model, "config", None), "vision_config", None)
    if vision_config is not None:
        vision_config.torch_dtype = torch.bfloat16
    _validate_qwen_generation_identity(model)
    processor = AutoProcessor.from_pretrained(str(snapshot))
    return model, processor


def load_pinned_qwen_processor() -> object:
    """Load only the pinned cached Qwen processor, never the generation model."""

    from transformers import AutoConfig, AutoProcessor

    snapshot = _cached_snapshot(QWEN_MODEL, QWEN_REVISION)
    config = AutoConfig.from_pretrained(str(snapshot))
    processor = AutoProcessor.from_pretrained(str(snapshot))
    image_token_id = getattr(config, "image_token_id", None)
    tokenizer = getattr(processor, "tokenizer", None)
    convert_token = getattr(tokenizer, "convert_tokens_to_ids", None)
    convert_id = getattr(tokenizer, "convert_ids_to_tokens", None)
    if (
        not isinstance(image_token_id, int)
        or isinstance(image_token_id, bool)
        or not callable(convert_token)
        or not callable(convert_id)
        or convert_token("<|image_pad|>") != image_token_id
        or convert_id(image_token_id) != "<|image_pad|>"
    ):
        raise ValueError("pinned Qwen config and tokenizer image token identity mismatch")
    setattr(processor, "image_token_id", image_token_id)
    return processor


def _require_benchmark_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "the production M3DocVQA benchmark requires a CUDA GPU; CUDA is unavailable"
        )
    return torch.device("cuda")


def _make_dataset(run_config: object) -> object:
    make_dataset = getattr(run_config, "make_dataset", None)
    if callable(make_dataset):
        return make_dataset()
    from docprune.m3docvqa_dataset import M3DocVQADevDataset

    return M3DocVQADevDataset(_required(run_config, "corpus"), expected_question_count=2441)


def _load_index_manifest(
    value: IndexManifest | Path | str, *, validate_files: bool = True
) -> IndexManifest:
    """Load an exact schema-5 identity, optionally without opening derived artifacts.

    ``validate_files=False`` is reserved for the authenticated Task 6 fixed-page
    path.  That path separately authenticates its selected document shards and
    must not touch the global embedding tensor, token map, or FAISS index.
    """
    if type(value) is IndexManifest:
        try:
            payload = IndexManifest.to_dict(value)
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("index manifest is missing required immutable fields") from error
    elif not isinstance(value, str | Path):
        raise TypeError("index manifest must be an IndexManifest or a manifest JSON path")
    else:
        path = Path(value)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"index manifest is missing: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError("index manifest must be valid JSON") from error
        if not isinstance(payload, Mapping):
            raise ValueError("index manifest must be a JSON object")
        if payload.get("schema_version") != 5:
            raise ValueError("index manifest schema_version must equal 5")
        supplied_digest = payload.get("manifest_sha256")
        unsigned_payload = dict(payload)
        unsigned_payload.pop("manifest_sha256", None)
        if supplied_digest != _sha256_json(unsigned_payload):
            raise ValueError("index manifest canonical SHA-256 is invalid")
    resources = payload.get("resources")
    if not isinstance(resources, Mapping):
        raise ValueError("index manifest is missing resources")
    qwen = resources.get("qwen", {})
    colpali = resources.get("colpali", {})
    backbone = resources.get("colpali_backbone", {})
    if not all(isinstance(item, Mapping) for item in (qwen, colpali, backbone)):
        raise ValueError("index manifest resources are invalid")
    try:
        manifest = IndexManifest(
            mode=str(payload["mode"]),
            page_count=int(payload["page_count"]),
            corpus_integrity_sha256=str(payload["corpus_integrity_sha256"]),
            source_order_sha256=str(payload["source_order_sha256"]),
            runtime_commit=str(payload["runtime_commit"]),
            m3docrag_commit=str(payload["m3docrag_commit"]),
            qwen_model=str(qwen["model"]),
            qwen_revision=str(qwen["revision"]),
            colpali_model=str(colpali["model"]),
            colpali_revision=str(colpali["revision"]),
            colpali_backbone_model=str(backbone["model"]),
            colpali_backbone_revision=str(backbone["revision"]),
            processor_contract_path=Path(payload["processor_contract_path"]),
            processor_contract_sha256=str(payload["processor_contract_sha256"]),
            pruning_config=payload["pruning_config"],
            artifact_root=Path(payload["artifact_root"]),
            embeddings_path=Path(payload["embeddings_path"]),
            embedding_metadata_path=Path(payload["embedding_metadata_path"]),
            embedding_shape=tuple(payload["embeddings"]["shape"]),
            embedding_dtype=str(payload["embeddings"]["dtype"]),
            embeddings_sha256=str(payload["embeddings_sha256"]),
            token2pageuid_path=Path(payload["token2pageuid_path"]),
            token2pageuid_sha256=str(payload["token2pageuid_sha256"]),
            completion_ledger_path=Path(payload["completion_ledger_path"]),
            completion_ledger_sha256=str(payload["completion_ledger_sha256"]),
            index_path=Path(payload["index_path"]),
            index_sha256=str(payload["index_sha256"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("index manifest is missing required immutable fields") from error
    manifest_payload = IndexManifest.to_dict(manifest)
    if payload != manifest_payload:
        raise ValueError("index manifest payload does not exactly match IndexManifest")
    if manifest_payload.get("manifest_sha256") != _sha256_json(
        {key: value for key, value in manifest_payload.items() if key != "manifest_sha256"}
    ):
        raise ValueError("index manifest canonical SHA-256 is invalid")
    if validate_files:
        IndexManifest.validate_files(manifest)
    return manifest


def _require_source_order_sha256(dataset: object) -> str:
    """Require a nonempty authoritative source-order hash from the dataset."""

    source_order = getattr(dataset, "source_order_sha256", None)
    if not isinstance(source_order, str) or len(source_order) != 64:
        raise ValueError("dataset must provide an authoritative source_order_sha256")
    if any(character not in "0123456789abcdefABCDEF" for character in source_order):
        raise ValueError("dataset source_order_sha256 must be a SHA-256")
    return source_order.lower()


class _ColPaliQueryAdapter:
    def __init__(self, model: object, processor: object) -> None:
        self.model = model
        self.processor = processor

    def encode_queries(self, queries: Sequence[str]) -> list[torch.Tensor]:
        process = getattr(self.processor, "process_queries", None)
        if not callable(process):
            raise ValueError("ColPali processor must expose process_queries")
        batch = process(list(queries))
        device = next(self.model.parameters()).device if hasattr(self.model, "parameters") else None
        moved = {
            key: value.to(device)
            if isinstance(value, torch.Tensor) and device is not None
            else value
            for key, value in batch.items()
        }
        with torch.no_grad():
            output = self.model(**moved)
        output = torch.as_tensor(output)
        if output.ndim != 3 or output.shape[0] != len(queries):
            raise ValueError("ColPali query embeddings must have shape [batch, tokens, width]")
        if output.shape[2] != 128:
            raise ValueError("ColPali query embeddings must have width 128")
        attention_mask = torch.as_tensor(batch.get("attention_mask", torch.ones(output.shape[:2])))
        if attention_mask.shape != output.shape[:2]:
            raise ValueError("ColPali query attention mask must match query embeddings")
        return [rows[mask.bool()] for rows, mask in zip(output, attention_mask, strict=True)]


def load_pinned_colpali_query_encoder() -> object:
    """Load the pinned ColPali model as the query-only retrieval adapter."""

    model, processor = _load_colpali(None)
    return _ColPaliQueryAdapter(model, processor)


def _load_task6_fixed_page_samples(
    fixture_path: Path,
    fixture_sha256: str,
    *,
    sample_ids: Sequence[str] | None,
    limit: int | None,
    page_count: int,
) -> tuple[FixedPageFixture, tuple[SampleInput, ...]]:
    """Select Task 6 samples without constructing or scanning the global dataset."""

    if sample_ids is None:
        raise ValueError("Task 6 fixed-page runs require explicit sealed sample IDs")
    fixture = load_fixed_page_fixture(
        fixture_path,
        expected_sha256=fixture_sha256,
        validate_external_bytes=False,
    )
    samples = filter_samples(
        fixture.selected_samples(sample_ids),
        limit=limit,
        sample_ids=sample_ids,
    )
    selected_qids = tuple(sample.question_id for sample in samples)
    fixture.validate_external_bytes(selected_qids=selected_qids)
    if any(len(fixture.question(qid).pages) != page_count for qid in selected_qids):
        raise ValueError("Task 6 fixed page count does not match the requested run")
    return fixture, samples


def _build_task6_fixed_page_components(
    fixture_path: Path,
    fixture_sha256: str,
    query_encoder: object,
    *,
    samples: Sequence[SampleInput],
    page_count: int,
) -> tuple[FixedPageFixture, AuthenticatedFixedPageRetriever, AuthenticatedFixedPageLoader]:
    """Build the fixed-page runtime without importing or opening search artifacts."""

    fixture = load_fixed_page_fixture(
        fixture_path,
        expected_sha256=fixture_sha256,
        validate_external_bytes=False,
    )
    selected_qids = tuple(sample.question_id for sample in samples)
    if len(selected_qids) != len(set(selected_qids)):
        raise ValueError("Task 6 selected samples contain duplicate QIDs")
    for sample in samples:
        try:
            fixed = fixture.question(sample.question_id)
        except KeyError as error:
            raise ValueError("Task 6 selected QID is absent from the fixed-page fixture") from error
        question_sha256 = hashlib.sha256(sample.question.encode("utf-8")).hexdigest()
        if fixed.question_sha256 != question_sha256:
            raise ValueError("Task 6 question text does not match the fixed-page fixture")
        if len(fixed.pages) != page_count:
            raise ValueError("Task 6 fixed page count does not match the requested run")
    fixture.validate_external_bytes(selected_qids=selected_qids)
    retriever = AuthenticatedFixedPageRetriever(
        fixture, query_encoder, validate_external_bytes=False
    )
    loader = AuthenticatedFixedPageLoader(
        fixture, render_task6_pdf_page, validate_external_bytes=False
    )
    return fixture, retriever, loader


def _run_manifest(
    identity: Mapping[str, object],
    *,
    operation: str,
    output: Path,
    index_manifest: IndexManifest | None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 2,
        "status": "configured",
        "operation": operation,
        "output": str(Path(output).resolve()),
        **dict(identity),
    }
    if index_manifest is not None:
        payload["index_manifest"] = index_manifest.to_dict()
    payload["run_manifest_sha256"] = _sha256_json(payload)
    return payload


def _expected_pruning_identity(
    config: DocPruneConfig, *, mode: str, page_count: int
) -> dict[str, object]:
    return {
        "enabled": mode == "docprune",
        "page_settings": asdict(config.for_pages(page_count)),
        "reconstruction_defaults": asdict(config.reconstruction_defaults),
        "siglip_patch_size": 14,
    }


def _selection_identity(
    samples: Sequence[SampleInput], *, limit: int | None, sample_ids: Sequence[str] | None
) -> dict[str, object]:
    return {
        "requested_sample_ids": None
        if sample_ids is None
        else [str(value) for value in sample_ids],
        "limit": limit,
        "resolved_question_ids": [sample.question_id for sample in samples],
        "count": len(samples),
    }


def _canonical_json_value(value: object, *, label: str) -> object:
    """Copy a value into the narrow JSON domain used by immutable manifest identity."""

    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool | float):
        raise ValueError(f"{label} must be JSON-safe without ambiguous numeric types")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, list):
        return [_canonical_json_value(item, label=label) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _canonical_json_value(value[key], label=label) for key in sorted(value)}
    raise ValueError(f"{label} must be JSON-safe and canonical")


def _ctp_policy_context_identity(
    experiment_version: object,
    repetition: object,
    geometry: Sequence[VisualTokenGeometry] | None,
) -> dict[str, object]:
    """Serialize only the immutable Task 4 seed inputs, never runtime tensors."""

    if experiment_version is None or repetition is None:
        raise ValueError("random policy context requires experiment version and repetition")
    canonical_geometry: list[list[int]] | None = None
    if geometry is not None:
        if not isinstance(geometry, Sequence) or isinstance(geometry, str | bytes):
            raise ValueError("policy geometry must be validated VisualTokenGeometry values")
        canonical_geometry = []
        for token in geometry:
            if not isinstance(token, VisualTokenGeometry):
                raise ValueError("policy geometry must be validated VisualTokenGeometry values")
            fields = [token.page_index, token.row, token.column, token.height, token.width]
            if any(not isinstance(value, int) or isinstance(value, bool) for value in fields):
                raise ValueError("policy geometry must be validated VisualTokenGeometry values")
            page, row, column, height, width = fields
            if (
                page < 0
                or height <= 0
                or width <= 0
                or not 0 <= row < height
                or not 0 <= column < width
            ):
                raise ValueError("policy geometry must be validated VisualTokenGeometry values")
            canonical_geometry.append(fields)
    geometry_identity: dict[str, object] | None = None
    if canonical_geometry is not None:
        geometry_bytes = json.dumps(
            canonical_geometry, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        geometry_identity = {
            "count": len(canonical_geometry),
            "sha256": hashlib.sha256(geometry_bytes).hexdigest(),
        }
    return {
        "experiment_version": _canonical_json_value(
            experiment_version, label="policy experiment version"
        ),
        "repetition": _canonical_json_value(repetition, label="policy repetition"),
        "geometry": geometry_identity,
    }


def _validate_ctp_policy_context_identity(value: object) -> None:
    """Reject a malformed persisted seed context before any workload is resumed."""

    if not isinstance(value, Mapping) or set(value) != {
        "experiment_version",
        "repetition",
        "geometry",
    }:
        raise ValueError("ctp_policy_context must be a canonical JSON-safe identity")
    try:
        if value["experiment_version"] is None or value["repetition"] is None:
            raise ValueError("missing seed inputs")
        if (
            _canonical_json_value(value["experiment_version"], label="policy experiment version")
            != value["experiment_version"]
        ):
            raise ValueError("noncanonical experiment version")
        if (
            _canonical_json_value(value["repetition"], label="policy repetition")
            != value["repetition"]
        ):
            raise ValueError("noncanonical repetition")
    except (KeyError, ValueError) as error:
        raise ValueError("ctp_policy_context must be a canonical JSON-safe identity") from error
    geometry = value["geometry"]
    if geometry is None:
        return
    if (
        not isinstance(geometry, Mapping)
        or set(geometry) != {"count", "sha256"}
        or not isinstance(geometry["count"], int)
        or isinstance(geometry["count"], bool)
        or geometry["count"] < 0
        or not isinstance(geometry["sha256"], str)
        or len(geometry["sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in geometry["sha256"])
    ):
        raise ValueError("ctp_policy_context must be a canonical JSON-safe identity")


def _write_run_manifest(
    output: Path,
    payload: Mapping[str, object],
    *,
    resume: bool,
    invocation_manifest: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Publish the complete identity before loading any model weights."""

    policy_data = payload.get("ctp_policy")
    policy_context = payload.get("ctp_policy_context")
    if policy_context is not None:
        _validate_ctp_policy_context_identity(policy_context)
        if not isinstance(policy_data, Mapping) or policy_data.get("family") not in {
            "random-top-m",
            "coverage-top-m",
        }:
            raise ValueError("ctp_policy_context requires a random or coverage CTP policy")
    elif isinstance(policy_data, Mapping) and policy_data.get("family") in {
        "random-top-m",
        "coverage-top-m",
    }:
        raise ValueError("random or coverage CTP policy requires ctp_policy_context")
    if output.is_symlink():
        raise ValueError("output must not be a symbolic link")
    if resume and (not output.exists() or not output.is_dir()):
        raise FileNotFoundError("resume requires an existing regular output directory")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "run_manifest.json"
    if resume and not manifest_path.is_file():
        raise ValueError("resume requires a complete run manifest")
    existing: dict[str, object] = {}
    if manifest_path.exists():
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ValueError("run manifest must be a regular file")
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError("run manifest is not valid JSON") from error
        if not isinstance(raw, dict):
            raise ValueError("run manifest must be a JSON object")
        existing = raw
        identity_keys = (
            "output",
            "operation",
            "mode",
            "page_count",
            "runtime_commit",
            "m3docrag_commit",
            "resources",
            "processor_contract_path",
            "processor_contract_sha256",
            "processor_contract",
            "run_config_source_path",
            "run_config_source_sha256",
            "index_manifest_source_path",
            "index_manifest_source_sha256",
            "corpus",
            "generation",
            "pruning_config",
            "ctp_policy",
            "ctp_policy_context",
            "fixed_page_fixture_path",
            "fixed_page_fixture_sha256",
            "fixed_page_provenance",
            "global_index_loaded",
            "geometry_derivation",
            "selection",
            "index_manifest",
            "measurement",
        )
        if resume:
            if existing.get("schema_version") != 2 or existing.get("status") != "configured":
                raise ValueError("resume manifest is incomplete")
            for path_key, digest_key, label in (
                ("run_config_source_path", "run_config_source_sha256", "run configuration"),
                ("index_manifest_source_path", "index_manifest_source_sha256", "index manifest"),
            ):
                source_path = existing.get(path_key)
                source_digest = existing.get(digest_key)
                if source_path is not None:
                    current_path, current_digest = _source_file_identity(
                        str(source_path), label=label
                    )
                    if current_path != source_path or current_digest != source_digest:
                        raise ValueError(f"{label} source bytes changed since the recorded run")
            if any(key not in existing for key in identity_keys if key in payload):
                raise ValueError("resume manifest is missing an immutable identity field")
            supplied_digest = existing.get("run_manifest_sha256")
            unsigned_existing = dict(existing)
            unsigned_existing.pop("run_manifest_sha256", None)
            if supplied_digest != _sha256_json(unsigned_existing):
                raise ValueError("resume manifest digest is invalid")
        placeholder = _is_cli_placeholder(
            existing,
            payload,
            invocation=invocation_manifest,
        )
        if not resume:
            if not placeholder:
                label = "complete run" if "runtime_commit" in existing else "unrelated run"
                raise FileExistsError(f"output directory already contains a {label}: {output}")
        else:
            requested_identity_keys = tuple(key for key in identity_keys if key in payload)
            if any(key not in existing for key in requested_identity_keys):
                raise ValueError("resume manifest is missing an immutable identity field")
            if any(existing.get(key) != payload.get(key) for key in requested_identity_keys):
                raise ValueError("resume manifest does not exactly match the requested run")
    elif any(output.iterdir()) and not resume:
        raise FileExistsError(f"output directory already contains artifacts: {output}")
    merged = {**existing, **dict(payload)}
    merged.pop("run_manifest_sha256", None)
    merged["run_manifest_sha256"] = _sha256_json(merged)
    descriptor, raw_temp = tempfile.mkstemp(prefix=".run_manifest.", dir=output)
    temporary = Path(raw_temp)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(merged, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        descriptor = -1
        os.replace(temporary, manifest_path)
        _fsync_directory(output)
    finally:
        if descriptor != -1:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
    return merged


def build_workload(
    *,
    operation: str,
    config: DocPruneConfig,
    page_count: int,
    output: Path,
    mode: str | None = None,
    run_config: BenchmarkRunConfig | Path | Mapping[str, object] | None = None,
    index_manifest: IndexManifest | Path | str | None = None,
    limit: int | None = None,
    sample_ids: Sequence[str] | None = None,
    resume: bool = False,
    invocation_manifest: Mapping[str, object] | None = None,
    qa_stage: str = "full",
    ctp_policy: CTPPolicy | None = None,
    policy_experiment_version: object | None = None,
    policy_repetition: object | None = None,
    policy_geometry: Sequence[VisualTokenGeometry] | None = None,
    fixed_page_fixture: Path | None = None,
    fixed_page_fixture_sha256: str | None = None,
    execution_runtime_commit: str | None = None,
) -> EvaluationWorkload | IndexBuildResult:
    """Build the concrete index operation or lazy end-to-end evaluation workload."""

    if operation not in {"embed", "evaluate"}:
        raise ValueError("operation must be 'embed' or 'evaluate'")
    if qa_stage not in {"full", "btp-only", "btp-qtp"}:
        raise ValueError("qa_stage must be full, btp-only, or btp-qtp")
    if ctp_policy is not None and not isinstance(ctp_policy, CTPPolicy):
        raise TypeError("ctp_policy must be a CTPPolicy")
    fixed_page_run = fixed_page_fixture is not None or fixed_page_fixture_sha256 is not None
    if fixed_page_run and (fixed_page_fixture is None or fixed_page_fixture_sha256 is None):
        raise ValueError("fixed-page fixture path and checksum must be supplied together")
    if fixed_page_run and execution_runtime_commit is None:
        raise ValueError("Task 6 fixed-page runs require the execution runtime commit")
    if not fixed_page_run and execution_runtime_commit is not None:
        raise ValueError("execution runtime commit is reserved for Task 6 fixed-page runs")
    if not isinstance(config, DocPruneConfig):
        raise TypeError("config must be a DocPruneConfig")
    if page_count not in PAGE_COUNTS:
        raise ValueError("page_count must be 1, 2, or 4")
    if config.m3docrag_commit != M3DOCRAG_COMMIT:
        raise ValueError("config must use the pinned M3DocRAG commit")
    resolved_mode = _resolve_mode(mode, run_config)
    if qa_stage != "full" and (operation != "evaluate" or resolved_mode != "docprune"):
        raise ValueError("diagnostic QA stages require a docprune evaluation")
    if ctp_policy is not None and (operation != "evaluate" or resolved_mode != "docprune"):
        raise ValueError("corrected CTP policies require a docprune evaluation")
    if fixed_page_run and (
        operation != "evaluate"
        or resolved_mode != "docprune"
        or page_count != 4
        or ctp_policy is None
    ):
        raise ValueError("Task 6 fixed-page runs require top-4 DocPrune evaluation and a policy")
    if fixed_page_run and policy_geometry is not None:
        raise ValueError("Task 6 geometry must be derived dynamically after BTP and QTP")
    if (
        ctp_policy is not None
        and ctp_policy.family in {"random-top-m", "coverage-top-m"}
        and (policy_experiment_version is None or policy_repetition is None)
    ):
        raise ValueError("random corrected CTP policy requires experiment version and repetition")
    policy_context: dict[str, object] | None = None
    if ctp_policy is not None:
        if ctp_policy.family in {"random-top-m", "coverage-top-m"}:
            policy_context = _ctp_policy_context_identity(
                policy_experiment_version,
                policy_repetition,
                policy_geometry,
            )
        elif any(
            value is not None
            for value in (policy_experiment_version, policy_repetition, policy_geometry)
        ):
            raise ValueError("policy context is valid only for random or coverage CTP policies")
    resolved_run = _resolve_run_config(run_config, mode=resolved_mode, page_count=page_count)
    identity = _validate_run_identity(resolved_run, mode=resolved_mode, page_count=page_count)
    if fixed_page_run:
        identity = _task6_execution_identity(identity, str(execution_runtime_commit))
    _validate_m3docrag_checkout(resolved_run)
    run_config_source_path, run_config_source_sha256 = _source_file_identity(
        run_config if isinstance(run_config, str | Path) else None,
        label="run configuration",
    )
    index_manifest_source_path, index_manifest_source_sha256 = _source_file_identity(
        index_manifest if isinstance(index_manifest, str | Path) else None,
        label="index manifest",
    )
    identity = {
        **identity,
        "run_config_source_path": run_config_source_path,
        "run_config_source_sha256": run_config_source_sha256,
        "index_manifest_source_path": index_manifest_source_path,
        "index_manifest_source_sha256": index_manifest_source_sha256,
    }
    output = Path(output)
    if output.is_symlink():
        raise ValueError("output must not be a symbolic link")
    if resume and (not output.exists() or not output.is_dir()):
        raise FileNotFoundError("resume requires an existing regular output directory")
    if output.exists() and output.is_file():
        raise FileExistsError(f"output must be a directory: {output}")
    if operation == "evaluate" and output.exists() and not output.is_dir():
        raise ValueError(f"output must be a directory: {output}")
    if operation == "evaluate" and output.exists() and not resume:
        entries = tuple(output.iterdir())
        if any(entry.name != "run_manifest.json" for entry in entries):
            raise FileExistsError(f"evaluation output already exists: {output}")
        manifest_path = output / "run_manifest.json"
        if manifest_path.is_file():
            try:
                existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                raise ValueError("existing run manifest is not valid JSON") from error
            if isinstance(existing_manifest, Mapping) and "runtime_commit" in existing_manifest:
                raise FileExistsError(f"evaluation output already exists: {output}")
    task6_fixture: FixedPageFixture | None = None
    if fixed_page_run:
        task6_fixture, samples = _load_task6_fixed_page_samples(
            Path(fixed_page_fixture),
            str(fixed_page_fixture_sha256),
            sample_ids=sample_ids,
            limit=limit,
            page_count=page_count,
        )
        dataset = None
        dataset_source_order = None
    else:
        dataset = _make_dataset(resolved_run)
        dataset_source_order = _require_source_order_sha256(dataset)
        samples = (
            filter_samples(dataset, limit=limit, sample_ids=sample_ids)
            if operation == "evaluate"
            else ()
        )
    identity = {
        **identity,
        "pruning_config": _expected_pruning_identity(
            config, mode=resolved_mode, page_count=page_count
        ),
        "selection": _selection_identity(samples, limit=limit, sample_ids=sample_ids),
        "measurement": measurement_identity(
            sample_ids=tuple(sample.question_id for sample in samples),
            warmup_count=1,
        ),
    }
    if ctp_policy is not None:
        identity["ctp_policy"] = ctp_policy.to_dict()
        if policy_context is not None:
            identity["ctp_policy_context"] = policy_context
    if fixed_page_run:
        if task6_fixture is None:
            raise AssertionError("Task 6 fixture must be loaded before identity construction")
        identity.update(
            {
                "fixed_page_fixture_path": str(Path(fixed_page_fixture).resolve()),
                "fixed_page_fixture_sha256": str(fixed_page_fixture_sha256),
                "fixed_page_provenance": True,
                "global_index_loaded": False,
                "geometry_derivation": {
                    "version": "task6-post-btp-qtp-grid-v1",
                    "order": "sealed-page-major-row-major-filtered-by-combined-mask",
                },
            }
        )

    if operation == "embed":
        if sample_ids is not None or limit is not None:
            raise ValueError("sample selection applies only to evaluation")
        _write_run_manifest(
            output,
            _run_manifest(identity, operation=operation, output=output, index_manifest=None),
            resume=resume,
            invocation_manifest=invocation_manifest,
        )
        model, processor = _load_colpali(resolved_run)
        result = build_index(
            IndexBuildConfig(
                run_config=resolved_run,
                dataset=dataset,
                model=model,
                processor=processor,
                pruning_config=config,
            ),
            resolved_mode,
            output,
        )
        if not isinstance(result, IndexBuildResult):
            raise TypeError("embed factory must return IndexBuildResult")
        return result

    if index_manifest is None:
        raise ValueError("evaluate requires --index-manifest for the selected mode")
    manifest = _load_index_manifest(index_manifest, validate_files=not fixed_page_run)
    if manifest.mode != resolved_mode or manifest.page_count != page_count:
        raise ValueError("index manifest mode/page_count does not match the requested run")
    if manifest.m3docrag_commit != M3DOCRAG_COMMIT:
        raise ValueError("index manifest uses the wrong M3DocRAG commit")
    if manifest.runtime_commit != str(_required(resolved_run, "runtime_commit")).lower():
        raise ValueError("index manifest runtime commit does not match the run configuration")
    if manifest.processor_contract_sha256 != str(identity["processor_contract_sha256"]):
        raise ValueError("index manifest processor contract does not match the run configuration")
    feature_runtime_commit = str(
        identity.get("feature_build_runtime_commit", identity["runtime_commit"])
    )
    if manifest.runtime_commit != feature_runtime_commit:
        raise ValueError("index manifest runtime commit does not match the run configuration")
    if manifest.to_dict()["resources"] != identity["resources"]:
        raise ValueError("index manifest model resources do not match the run configuration")
    if manifest.pruning_config != identity["pruning_config"]:
        raise ValueError(
            "index manifest pruning configuration does not match the run configuration"
        )
    corpus = identity["corpus"]
    if manifest.corpus_integrity_sha256 != corpus["integrity_sha256"]:
        raise ValueError("index manifest corpus identity does not match the run configuration")
    if task6_fixture is None and manifest.source_order_sha256 != dataset_source_order:
        raise ValueError("index manifest source order does not match the corpus")
    if task6_fixture is not None:
        identity["feature_build_source_order_sha256"] = manifest.source_order_sha256
        manifest_path, manifest_sha256 = _source_file_identity(
            index_manifest if isinstance(index_manifest, str | Path) else None,
            label="index manifest",
        )
        if (
            manifest_path != str(task6_fixture.feature_manifest_path)
            or manifest_sha256 != task6_fixture.feature_manifest_sha256
            or manifest.completion_ledger_path != task6_fixture.completion_ledger_path
            or manifest.completion_ledger_sha256 != task6_fixture.completion_ledger_sha256
        ):
            raise ValueError("Task 6 fixture feature-manifest provenance does not match")

    complete_manifest = _write_run_manifest(
        output,
        _run_manifest(identity, operation=operation, output=output, index_manifest=manifest),
        resume=resume,
        invocation_manifest=invocation_manifest,
    )

    colpali_model, colpali_processor = _load_colpali(resolved_run)
    query_adapter = _ColPaliQueryAdapter(colpali_model, colpali_processor)
    fixed_loader: AuthenticatedFixedPageLoader | None = None
    if task6_fixture is not None:
        retriever = AuthenticatedFixedPageRetriever(
            task6_fixture, query_adapter, validate_external_bytes=False
        )
        fixed_loader = AuthenticatedFixedPageLoader(
            task6_fixture, render_task6_pdf_page, validate_external_bytes=False
        )
        for sample in samples:
            retriever.retrieve(sample.question, page_count)
        retriever.release_query_encoder()
        del query_adapter, colpali_model, colpali_processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        import faiss
        from safetensors.torch import load_file

        embedding_tensors = load_file(manifest.embeddings_path, device="cpu")
        embeddings = embedding_tensors["embeddings"]
        raster_indices = embedding_tensors["raster_indices"]
        metadata = json.loads(manifest.embedding_metadata_path.read_text(encoding="utf-8"))
        source_hw = tuple(metadata["source_hw"])
        token_map = json.loads(manifest.token2pageuid_path.read_text(encoding="utf-8"))
        if not isinstance(token_map, list) or len(token_map) != len(embeddings):
            raise ValueError("index token2pageuid rows do not match embeddings")
        index = faiss.read_index(str(manifest.index_path))
        rag_model = SimpleNamespace(retrieval_model=query_adapter)
        retriever = OfficialM3DocRAGBoundary(
            rag_model=rag_model,
            dataset=dataset,
            docid2embs={},
            index=index,
            token2pageuid=token_map,
            all_token_embeddings=embeddings,
            raster_indices=raster_indices,
            source_hw=source_hw,
        )
    qwen_model, qwen_processor = _load_qwen(resolved_run)
    if resolved_mode == "all-kept":
        answerer = AllKeptQwenAnswerer(qwen_model, qwen_processor)
    else:
        answerer = DocPruneQwenAnswerer(
            qwen_model,
            qwen_processor,
            page_config=config.for_pages(page_count),
            qa_stage=qa_stage,
            ctp_policy=ctp_policy,
            selection_context=(
                None
                if policy_context is None or policy_geometry is None
                else PolicySelectionContext(
                    policy_context["experiment_version"],
                    None,
                    None,
                    policy_context["repetition"],
                    tuple(policy_geometry),
                )
            ),
            policy_experiment_version=(
                None if policy_context is None else policy_context["experiment_version"]
            ),
            policy_repetition=None if policy_context is None else policy_context["repetition"],
        )
    runner = (
        DocPruneM3DocRAG(retriever, fixed_loader, answerer, top_k=page_count)
        if fixed_loader is not None
        else DocPruneM3DocRAG(retriever, dataset, answerer, top_k=page_count)
    )
    return EvaluationWorkload(
        runner=runner,
        samples=samples,
        manifest=complete_manifest,
    )


def _build_diagnostic_workload(qa_stage: str, **kwargs: object) -> object:
    if kwargs.get("operation") != "evaluate" or kwargs.get("mode") != "docprune":
        raise ValueError("diagnostic QA stages require a docprune evaluation")
    return build_workload(qa_stage=qa_stage, **kwargs)


def build_btp_only_workload(**kwargs: object) -> object:
    """Build the fixed DocPrune evaluation with only QA-stage BTP enabled."""

    return _build_diagnostic_workload("btp-only", **kwargs)


def build_btp_qtp_workload(**kwargs: object) -> object:
    """Build the fixed DocPrune evaluation with QA-stage BTP and QTP enabled."""

    return _build_diagnostic_workload("btp-qtp", **kwargs)


def _build_corrected_policy_workload(policy: CTPPolicy, **kwargs: object) -> object:
    if kwargs.get("operation") != "evaluate" or kwargs.get("mode") != "docprune":
        raise ValueError("corrected CTP policies require a docprune evaluation")
    return build_workload(qa_stage="full", ctp_policy=policy, **kwargs)


def build_btp_qtp_no_ctp_workload(**kwargs: object) -> object:
    return _build_corrected_policy_workload(btp_qtp_no_ctp_policy(), **kwargs)


def build_literal_native_threshold_workload(**kwargs: object) -> object:
    return _build_corrected_policy_workload(literal_native_threshold_policy(), **kwargs)


def build_aggregate_native_threshold_workload(**kwargs: object) -> object:
    return _build_corrected_policy_workload(aggregate_native_threshold_policy(), **kwargs)


def build_literal_score_top_m_workload(**kwargs: object) -> object:
    return _build_corrected_policy_workload(literal_score_top_m_policy(), **kwargs)


def build_aggregate_score_top_m_workload(**kwargs: object) -> object:
    return _build_corrected_policy_workload(aggregate_score_top_m_policy(), **kwargs)


def _task6_environment_identity() -> tuple[CTPPolicy, Path, str, object | None, int | None, str]:
    """Resolve the closed Task 6 launcher identity without loading any model."""

    raw_fixture = os.environ.get("DOCPRUNE_TASK6_FIXED_PAGE_FIXTURE")
    fixture_sha256 = os.environ.get("DOCPRUNE_TASK6_FIXED_PAGE_FIXTURE_SHA256")
    policy_name = os.environ.get("DOCPRUNE_TASK6_POLICY")
    runtime_commit = os.environ.get("DOCPRUNE_TASK6_RUNTIME_COMMIT")
    if not raw_fixture or not fixture_sha256 or not policy_name or not runtime_commit:
        raise ValueError(
            "Task 6 fixed-page fixture, checksum, policy, and runtime commit are required"
        )
    fixture = Path(raw_fixture)
    if not fixture.is_absolute() or fixture.is_symlink() or not fixture.is_file():
        raise ValueError("Task 6 fixed-page fixture must be an absolute regular file")
    if len(fixture_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in fixture_sha256
    ):
        raise ValueError("Task 6 fixed-page fixture checksum must be a lowercase SHA-256")
    policy = task6_policy(policy_name)
    _task6_execution_identity({"runtime_commit": runtime_commit}, runtime_commit)
    raw_version = os.environ.get("DOCPRUNE_TASK6_EXPERIMENT_VERSION")
    raw_repetition = os.environ.get("DOCPRUNE_TASK6_REPETITION")
    if policy.family in {"random-top-m", "coverage-top-m"}:
        if not raw_version or raw_repetition is None:
            raise ValueError("Task 6 random policy requires experiment version and repetition")
        try:
            repetition = int(raw_repetition)
        except ValueError as error:
            raise ValueError("Task 6 repetition must be a non-negative integer") from error
        if repetition < 0 or str(repetition) != raw_repetition:
            raise ValueError("Task 6 repetition must be a canonical non-negative integer")
        return policy, fixture, fixture_sha256, raw_version, repetition, runtime_commit.lower()
    if raw_version is not None or raw_repetition is not None:
        raise ValueError("Task 6 deterministic policy must not declare random seed context")
    return policy, fixture, fixture_sha256, None, None, runtime_commit.lower()


def build_task6_fixed_page_workload(**kwargs: object) -> object:
    """Build one closed fixed-page Task 6 cell with no search/index path."""

    if kwargs.get("operation") != "evaluate" or kwargs.get("mode") != "docprune":
        raise ValueError("Task 6 fixed-page policy requires a DocPrune evaluation")
    policy, fixture, fixture_sha256, version, repetition, runtime_commit = (
        _task6_environment_identity()
    )
    return build_workload(
        qa_stage="full",
        ctp_policy=policy,
        fixed_page_fixture=fixture,
        fixed_page_fixture_sha256=fixture_sha256,
        policy_experiment_version=version,
        policy_repetition=repetition,
        execution_runtime_commit=runtime_commit,
        **kwargs,
    )


def controlled_policy_identity(factory_spec: str) -> dict[str, object] | None:
    """Resolve only this module's static corrected-policy factories before model load."""

    policies = {
        "docprune.m3docvqa_factory:build_btp_qtp_no_ctp_workload": btp_qtp_no_ctp_policy,
        "docprune.m3docvqa_factory:build_literal_native_threshold_workload": literal_native_threshold_policy,
        "docprune.m3docvqa_factory:build_aggregate_native_threshold_workload": aggregate_native_threshold_policy,
        "docprune.m3docvqa_factory:build_literal_score_top_m_workload": literal_score_top_m_policy,
        "docprune.m3docvqa_factory:build_aggregate_score_top_m_workload": aggregate_score_top_m_policy,
    }
    if factory_spec == "docprune.m3docvqa_factory:build_task6_fixed_page_workload":
        return _task6_environment_identity()[0].to_dict()
    factory = policies.get(factory_spec)
    return None if factory is None else factory().to_dict()


__all__ = [
    "DEFAULT_FACTORY",
    "build_btp_only_workload",
    "build_btp_qtp_workload",
    "build_btp_qtp_no_ctp_workload",
    "build_literal_native_threshold_workload",
    "build_aggregate_native_threshold_workload",
    "build_literal_score_top_m_workload",
    "build_aggregate_score_top_m_workload",
    "build_task6_fixed_page_workload",
    "controlled_policy_identity",
    "build_workload",
    "filter_samples",
    "load_pinned_colpali_query_encoder",
    "load_pinned_qwen_processor",
    "load_completed_qids",
    "validate_processor_contract_file",
]
