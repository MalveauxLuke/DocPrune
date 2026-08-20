"""Concrete, fail-closed integration factory for the pinned M3DocVQA run."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from docprune.answerers import AllKeptQwenAnswerer, DocPruneQwenAnswerer
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
from docprune.indexing import IndexBuildConfig, IndexBuildResult, build_index
from docprune.m3docrag import DocPruneM3DocRAG, OfficialM3DocRAGBoundary, SampleInput
from docprune.processor_probe import validate_processor_contract

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


def _validate_result_record(
    record: Mapping[str, object],
    *,
    line_number: int,
    expected_page_count: int | None = None,
    mode: str | None = None,
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
    extra = fields - required
    if extra:
        raise ValueError(f"results JSONL record {line_number} has unknown fields {sorted(extra)!r}")
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
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in counts):
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
    timing = record["timing"]
    timing_fields = {"retrieval_seconds", "qa_seconds"}
    optional_timing_fields = {
        "peak_allocated_gpu_bytes",
        "warmup_excluded",
        "profiler_enabled",
        "profiler_definition",
        "flops",
    }
    if not isinstance(timing, Mapping) or not timing_fields <= set(timing):
        raise ValueError(f"results JSONL record {line_number} has invalid timing schema")
    if set(timing) - timing_fields - optional_timing_fields:
        raise ValueError(f"results JSONL record {line_number} has invalid timing schema")
    if any(
        not isinstance(timing[name], int | float)
        or isinstance(timing[name], bool)
        or not math.isfinite(float(timing[name]))
        or float(timing[name]) < 0
        for name in ("retrieval_seconds", "qa_seconds")
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
            "prompt": prompt,
        },
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


def _load_qwen(run_config: object) -> tuple[object, object]:
    device = _require_benchmark_cuda()
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    snapshot = _cached_snapshot(QWEN_MODEL, QWEN_REVISION)
    model = (
        Qwen2VLForConditionalGeneration.from_pretrained(
            str(snapshot), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
        )
        .to(device)
        .eval()
    )
    processor = AutoProcessor.from_pretrained(str(snapshot))
    return model, processor


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


def _load_index_manifest(value: IndexManifest | Path | str) -> IndexManifest:
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


def _write_run_manifest(
    output: Path,
    payload: Mapping[str, object],
    *,
    resume: bool,
    invocation_manifest: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Publish the complete identity before loading any model weights."""

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
            "selection",
            "index_manifest",
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
) -> EvaluationWorkload | IndexBuildResult:
    """Build the concrete index operation or lazy end-to-end evaluation workload."""

    if operation not in {"embed", "evaluate"}:
        raise ValueError("operation must be 'embed' or 'evaluate'")
    if not isinstance(config, DocPruneConfig):
        raise TypeError("config must be a DocPruneConfig")
    if page_count not in PAGE_COUNTS:
        raise ValueError("page_count must be 1, 2, or 4")
    if config.m3docrag_commit != M3DOCRAG_COMMIT:
        raise ValueError("config must use the pinned M3DocRAG commit")
    resolved_mode = _resolve_mode(mode, run_config)
    resolved_run = _resolve_run_config(run_config, mode=resolved_mode, page_count=page_count)
    identity = _validate_run_identity(resolved_run, mode=resolved_mode, page_count=page_count)
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
    }

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
    manifest = _load_index_manifest(index_manifest)
    if manifest.mode != resolved_mode or manifest.page_count != page_count:
        raise ValueError("index manifest mode/page_count does not match the requested run")
    if manifest.m3docrag_commit != M3DOCRAG_COMMIT:
        raise ValueError("index manifest uses the wrong M3DocRAG commit")
    if manifest.runtime_commit != str(_required(resolved_run, "runtime_commit")).lower():
        raise ValueError("index manifest runtime commit does not match the run configuration")
    if manifest.processor_contract_sha256 != str(identity["processor_contract_sha256"]):
        raise ValueError("index manifest processor contract does not match the run configuration")
    if manifest.runtime_commit != str(identity["runtime_commit"]):
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
    if manifest.source_order_sha256 != dataset_source_order:
        raise ValueError("index manifest source order does not match the corpus")

    complete_manifest = _write_run_manifest(
        output,
        _run_manifest(identity, operation=operation, output=output, index_manifest=manifest),
        resume=resume,
        invocation_manifest=invocation_manifest,
    )

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
    colpali_model, colpali_processor = _load_colpali(resolved_run)
    query_adapter = _ColPaliQueryAdapter(colpali_model, colpali_processor)
    rag_model = SimpleNamespace(retrieval_model=query_adapter)
    boundary = OfficialM3DocRAGBoundary(
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
        )
    runner = DocPruneM3DocRAG(boundary, dataset, answerer, top_k=page_count)
    return EvaluationWorkload(
        runner=runner,
        samples=samples,
        manifest=complete_manifest,
    )


__all__ = [
    "DEFAULT_FACTORY",
    "build_workload",
    "filter_samples",
    "load_completed_qids",
    "validate_processor_contract_file",
]
