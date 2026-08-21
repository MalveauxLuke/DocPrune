"""Narrow executable validators shared by the corrected benchmark launchers."""

from __future__ import annotations

import math
import os
import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path


def validate_runtime_identity(runtime_commit: object, runtime_dir: Path | str) -> str:
    if not isinstance(runtime_commit, str) or re.fullmatch(r"[0-9a-f]{40}", runtime_commit) is None:
        raise ValueError("runtime commit must be an exact 40-hex SHA")
    path = _absolute_path(runtime_dir, label="runtime checkout")
    if not path.is_dir():
        raise ValueError(f"runtime checkout is missing: {path}")
    try:
        head = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"unable to inspect runtime checkout: {path}") from error
    if head != runtime_commit:
        raise ValueError(f"runtime checkout revision mismatch: {head}")
    if dirty:
        raise ValueError(f"runtime checkout is not clean: {dirty.splitlines()[0]}")
    return runtime_commit


def validate_attempt_root(attempt_root: Path | str, expected_attempt_root: Path | str) -> str:
    actual = os.path.abspath(os.fspath(attempt_root))
    expected = os.path.abspath(os.fspath(expected_attempt_root))
    if actual != expected or not re.fullmatch(
        r"/scratch/lmalveau/docprune/benchmark-[0-9a-f]{7,40}/attempt-1", expected
    ):
        raise ValueError(f"attempt root is not the exact fresh attempt root: {actual}")
    return actual


def validate_mini_index_manifest(payload: object) -> None:
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 5:
        raise ValueError("mini-index schema_version must equal 5")
    if payload.get("fixture") is not True:
        raise ValueError("mini-index must be explicitly marked fixture")


def align_retrieved_page_context(
    pages: Sequence[tuple[str, int]],
    features: Sequence[object],
    load_page: Callable[[str, int], object],
) -> tuple[object, ...]:
    if len(pages) != len(features):
        raise ValueError("retrieved page/features alignment has different lengths")
    loaded: list[object] = []
    for (doc_id, page_index), feature in zip(pages, features, strict=True):
        if not isinstance(feature, Mapping):
            raise ValueError("retrieved page/features alignment has an invalid feature")
        if feature.get("doc_id") != doc_id or feature.get("page_index") != page_index:
            raise ValueError("retrieved page/features alignment does not match page identity")
        loaded.append(load_page(doc_id, page_index))
    return tuple(loaded)


def validate_gate_evidence_payload(
    semantic: object,
    *,
    expected_qids: Sequence[str],
) -> None:
    if not isinstance(semantic, Mapping):
        raise ValueError("semantic evidence must be an object")
    qids = tuple(str(qid) for qid in expected_qids)
    if semantic.get("status") != "passed":
        raise ValueError("semantic evidence did not pass")
    if semantic.get("baseline_equivalence") is not True:
        raise ValueError("semantic baseline equivalence is missing")
    if semantic.get("complete_colpali_equivalence") is not True:
        raise ValueError("complete ColPali equivalence is missing")
    if semantic.get("equivalence_qids") != list(qids):
        raise ValueError("semantic evidence qids are not the fixed tuple")
    supporting = semantic.get("supporting_documents")
    if (
        not isinstance(supporting, Mapping)
        or set(supporting) != set(qids)
        or any(not isinstance(value, str) or not value for value in supporting.values())
    ):
        raise ValueError("semantic supporting documents are incomplete")
    validate_mini_index_manifest(semantic.get("schema5_mini_index"))

    backend = semantic.get("flash_attention_2")
    if not isinstance(backend, Mapping) or backend.get("attn_implementation") != "flash_attention_2":
        raise ValueError("exact FlashAttention-2 identity is missing")
    for field in ("cuda", "torch", "transformers"):
        if not isinstance(backend.get(field), str) or not backend[field]:
            raise ValueError(f"FlashAttention-2 identity is missing {field}")
    ctp_policy = semantic.get("ctp_policy")
    expected_ctp_policy = {
        "head_aggregation": "arithmetic_mean",
        "score_scale": "current_visual_token_count",
        "raw_attention_semantics": "mean_head_attention_scores_before_visual_count_scaling",
        "transformed_attention_semantics": "raw_mean_times_current_visual_count",
        "prefill_query_token": "last_prompt_token",
    }
    if ctp_policy != expected_ctp_policy:
        raise ValueError("CTP policy evidence is missing or does not match the sealed reconstruction")

    upstream = semantic.get("upstream_retrieval_orders")
    runtime = semantic.get("runtime_retrieval_orders")
    legacy = semantic.get("retrieval_orders")
    _validate_retrieval_orders(upstream, qids, supporting, label="upstream")
    _validate_retrieval_orders(runtime, qids, supporting, label="runtime")
    if upstream != runtime:
        raise ValueError("upstream and runtime retrieval orders differ")
    if legacy is not None and legacy != runtime:
        raise ValueError("retrieval evidence differs from runtime order")

    counters = semantic.get("colpali_counters")
    if not isinstance(counters, Mapping):
        raise ValueError("ColPali counters are missing")
    try:
        before = _counter_snapshot(counters, "before_retrieval")
        after_retrieval = _counter_snapshot(counters, "after_retrieval")
        after_qa = _counter_snapshot(counters, "after_qa")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("ColPali counters are incomplete") from error
    if after_retrieval["processor_queries"] - before["processor_queries"] != len(qids):
        raise ValueError("ColPali query encode count is not exactly one per fixed QID")
    if after_qa != after_retrieval:
        raise ValueError("ColPali counters changed during QA; re-encoding is present")
    if after_retrieval["processor_queries"] <= 0:
        raise ValueError("ColPali query counter is fabricated or empty")

    traces = semantic.get("docprune_traces")
    if not isinstance(traces, Mapping):
        raise ValueError("semantic evidence lacks the exact page traces")
    if set(traces) == {"1", "2", "4"}:
        trace_values = tuple(traces.values())
    elif set(traces) == set(qids):
        trace_values = tuple(
            page_values
            for per_qid in traces.values()
            if isinstance(per_qid, Mapping)
            for page_values in (
                per_qid.get("1"),
                per_qid.get("2"),
                per_qid.get("4"),
            )
        )
        if len(trace_values) != len(qids) * 3:
            raise ValueError("semantic evidence lacks the exact per-QID page traces")
    else:
        raise ValueError("semantic evidence lacks the exact page traces")
    for values in trace_values:
        if (
            not isinstance(values, Sequence)
            or isinstance(values, str | bytes)
            or len(values) != 4
            or any(type(value) is not int or value <= 0 for value in values)
            or any(left < right for left, right in zip(values, values[1:]))
        ):
            raise ValueError("semantic evidence has invalid or non-monotonic traces")

    timings = semantic.get("timings")
    if not isinstance(timings, Mapping) or set(timings) != set(qids):
        raise ValueError("semantic timing probes are incomplete")
    fields = (
        "retrieval_seconds",
        "page_load_seconds",
        "qa_seconds",
        "total_sample_seconds",
        "encoder_seconds",
        "decoder_seconds",
    )
    for qid in qids:
        values = timings.get(qid)
        if isinstance(values, Mapping) and set(fields) <= set(values):
            probes = (values,)
        elif isinstance(values, Mapping) and set(values) == {"1", "2", "4"}:
            probes = tuple(values[str(page)] for page in (1, 2, 4))
        else:
            probes = ()
        if any(
            not isinstance(probe, Mapping)
            or any(
                not isinstance(probe.get(field), int | float)
                or isinstance(probe.get(field), bool)
                or not math.isfinite(float(probe[field]))
                or float(probe[field]) <= 0
                for field in fields
            )
            for probe in probes
        ) or not probes:
            raise ValueError(f"semantic timing probes are invalid for {qid}")


def _absolute_path(value: Path | str, *, label: str) -> Path:
    if not isinstance(value, str | os.PathLike) or not value:
        raise ValueError(f"{label} path must be non-empty")
    path = Path(os.path.abspath(os.fspath(value)))
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(f"{label} path contains a symlink: {path}")
    return path


def _counter_snapshot(counters: Mapping[object, object], name: str) -> dict[str, int]:
    value = counters[name]
    if not isinstance(value, Mapping):
        raise TypeError(name)
    result: dict[str, int] = {}
    for key in ("processor_queries", "processor_images", "model_forwards"):
        count = value[key]
        if type(count) is not int or count < 0:
            raise ValueError(name)
        result[key] = count
    return result


def _validate_retrieval_orders(
    value: object,
    qids: Sequence[str],
    supporting: Mapping[object, object],
    *,
    label: str,
) -> None:
    if not isinstance(value, Mapping) or set(value) != set(qids):
        raise ValueError(f"{label} retrieval evidence is incomplete")
    allowed_docs = set(str(doc) for doc in supporting.values())
    for qid in qids:
        per_qid = value.get(qid)
        if not isinstance(per_qid, Mapping) or set(per_qid) != {"1", "2", "4"}:
            raise ValueError(f"{label} retrieval evidence is incomplete for {qid}")
        for top_k in (1, 2, 4):
            pages = per_qid[str(top_k)]
            if not isinstance(pages, Sequence) or isinstance(pages, str | bytes) or len(pages) != top_k:
                raise ValueError(f"{label} retrieval order is invalid for {qid} top-{top_k}")
            seen: set[tuple[str, int]] = set()
            for page in pages:
                if not isinstance(page, Mapping):
                    raise ValueError(f"{label} retrieval order has an invalid page")
                doc_id = page.get("doc_id")
                page_index = page.get("page_index")
                identity = (doc_id, page_index)
                if (
                    not isinstance(doc_id, str)
                    or doc_id not in allowed_docs
                    or type(page_index) is not int
                    or page_index < 0
                    or identity in seen
                ):
                    raise ValueError(f"{label} retrieval order is invalid for {qid}")
                seen.add(identity)
