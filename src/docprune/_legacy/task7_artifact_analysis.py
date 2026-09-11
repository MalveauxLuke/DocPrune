"""Pure-CPU Task 7 artifact admission and report-row assembly.

This module deliberately depends only on the standard library and the pinned
``word2number`` evaluator dependency.  It must remain importable and usable
without model, retrieval, image, tensor, or indexing packages.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import string
from collections.abc import Mapping, Sequence
from functools import cache
from pathlib import Path

from word2number.w2n import word_to_num

_DECODER_LAYER_COUNT = 28
_RENDERER_CONTRACT = (
    "pdf2image-1.17.0|poppler-pdftoppm-26.05.0|dpi-144|rgb-contiguous|"
    "pdftoppm-sha256-1102bc3f4a12f3d3d207ac8e39f463d0fb3c403511fb4c817e776e5251b53f33"
)
_BOUNDARIES = ("B_input", "B_0", "B_6", "B_13", "B_20", "B_23", "B_26")
_INTERVENTION_NAMES = (
    "btp-qtp-no-ctp",
    "all-visual-drop-B_input",
    "all-visual-drop-B_0",
    "all-visual-drop-B_6",
    "all-visual-drop-B_13",
    "all-visual-drop-B_20",
    "all-visual-drop-B_23",
    "all-visual-drop-B_26",
)
_REFERENCE_POLICY = {
    "family": "no-ctp",
    "name": "btp-qtp-no-ctp",
    "selection_kind": "none",
    "score_semantics": "none",
    "budget_source": "none",
    "fixed_retention": None,
}
_FIXTURE_KEYS = {
    "schema_version",
    "fixture_version",
    "fixed_page_provenance",
    "global_index_loaded",
    "reference_path",
    "reference_sha256",
    "eligible_questions_path",
    "eligible_questions_sha256",
    "feature_manifest_path",
    "feature_manifest_sha256",
    "completion_ledger_path",
    "completion_ledger_sha256",
    "questions",
}
_QUESTION_KEYS = {"qid", "question_sha256", "pages"}
_PAGE_KEYS = {
    "rank",
    "doc_id",
    "page_index",
    "score",
    "source_pdf_path",
    "source_pdf_sha256",
    "rendered_rgb_width",
    "rendered_rgb_height",
    "rendered_rgb_sha256",
    "renderer_contract",
    "feature_shard_path",
    "feature_shard_sha256",
    "feature_page_index",
}
_LIKELIHOOD_TARGET_KEYS = {
    "schema_version",
    "target_name",
    "normalization",
    "aggregation",
    "eos_included",
    "answer_item_semantics",
    "tokenization",
    "assistant_prompt_sha256",
    "accepted_references",
    "target_token_ids",
    "likelihood_target_sha256",
}
_LIKELIHOOD_RECORD_KEYS = {
    "schema_version",
    "qid",
    "intervention_name",
    "target",
    "likelihood_target_sha256",
    "fixture_sha256",
    "run_manifest_sha256",
    "source_result_sha256",
    "assistant_prompt_sha256",
    "prefill_input_ids_shape",
    "prefill_input_ids_sha256",
    "per_reference_mean_loglikelihood",
    "best_reference_index",
    "best_reference_mean_loglikelihood",
    "likelihood_record_sha256",
}
_PUNCTUATION = set(string.punctuation)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _open_directory_nofollow(path: Path) -> int:
    absolute = Path(os.path.abspath(path))
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in absolute.parts[1:]:
            next_descriptor = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
    except OSError as error:
        os.close(descriptor)
        raise ValueError(
            f"artifact parent contains a symlink or invalid component: {absolute}"
        ) from error
    return descriptor


def _read_regular_file_bytes(path: Path, label: str) -> bytes:
    file_path = Path(path)
    if not file_path.is_absolute():
        raise ValueError(f"{label} must be an absolute regular file")
    parent_fd = _open_directory_nofollow(file_path.parent)
    descriptor: int | None = None
    try:
        try:
            descriptor = os.open(
                file_path.name,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=parent_fd,
            )
        except OSError as error:
            raise ValueError(f"{label} must be an absolute regular file") from error
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"{label} must be an absolute regular file")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)


def _authenticated_file(path: Path, expected_sha256: str, label: str) -> bytes:
    if not _is_sha256(expected_sha256):
        raise ValueError(f"{label} checksum must be a lowercase SHA-256")
    content = _read_regular_file_bytes(path, label)
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError(f"{label} checksum mismatch")
    return content


def _authenticated_bytes(content: bytes, expected_sha256: str, label: str) -> bytes:
    if not isinstance(content, bytes) or not _is_sha256(expected_sha256):
        raise ValueError(f"{label} bytes or checksum are invalid")
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError(f"{label} checksum mismatch")
    return content


def _json_mapping_bytes(content: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(content.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return dict(value)


def _jsonl_mappings_bytes(content: bytes, label: str) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    try:
        for line_number, line in enumerate(content.decode("utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"{label} row {line_number} is not an object")
            rows.append(dict(value))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"{label} is not valid JSONL") from error
    if not rows:
        raise ValueError(f"{label} must not be empty")
    return tuple(rows)


def task7_intervention_identities() -> tuple[dict[str, object], ...]:
    """Return pure mappings for the reference and seven all-drop cells."""

    return (
        {
            "name": _INTERVENTION_NAMES[0],
            "analysis_family": "reference",
            "ctp_policy": dict(_REFERENCE_POLICY),
            "forced_intervention": None,
        },
        *(
            {
                "name": name,
                "analysis_family": "fixed-grid",
                "ctp_policy": None,
                "forced_intervention": {
                    "boundary": boundary,
                    "mode": "physical_delete",
                    "retained_visual_ids": [],
                },
            }
            for name, boundary in zip(_INTERVENTION_NAMES[1:], _BOUNDARIES, strict=True)
        ),
    )


def _validated_fixture(content: bytes) -> dict[str, object]:
    fixture = _json_mapping_bytes(content, "fixed page fixture")
    if (
        set(fixture) != _FIXTURE_KEYS
        or fixture.get("schema_version") != 2
        or fixture.get("fixed_page_provenance") is not True
        or fixture.get("global_index_loaded") is not False
        or not isinstance(fixture.get("fixture_version"), str)
        or not fixture.get("fixture_version")
    ):
        raise ValueError("fixed page fixture schema or provenance is invalid")
    for path_key, digest_key in (
        ("reference_path", "reference_sha256"),
        ("eligible_questions_path", "eligible_questions_sha256"),
        ("feature_manifest_path", "feature_manifest_sha256"),
        ("completion_ledger_path", "completion_ledger_sha256"),
    ):
        path = fixture.get(path_key)
        if (
            not isinstance(path, str)
            or not Path(path).is_absolute()
            or not _is_sha256(fixture.get(digest_key))
        ):
            raise ValueError("fixed page fixture external identity is invalid")
    raw_questions = fixture.get("questions")
    if not isinstance(raw_questions, list) or not raw_questions:
        raise ValueError("fixed page fixture questions are invalid")
    qids: set[str] = set()
    question_hashes: set[str] = set()
    checked_questions: list[dict[str, object]] = []
    for raw_question in raw_questions:
        if not isinstance(raw_question, Mapping) or set(raw_question) != _QUESTION_KEYS:
            raise ValueError("fixed question schema is invalid")
        qid, question_sha256, raw_pages = (
            raw_question.get("qid"),
            raw_question.get("question_sha256"),
            raw_question.get("pages"),
        )
        if (
            not isinstance(qid, str)
            or not qid
            or qid in qids
            or not _is_sha256(question_sha256)
            or question_sha256 in question_hashes
            or not isinstance(raw_pages, list)
            or not raw_pages
        ):
            raise ValueError("fixed question identity is invalid")
        pages: list[dict[str, object]] = []
        page_ids: set[tuple[str, int]] = set()
        for rank, raw_page in enumerate(raw_pages):
            if not isinstance(raw_page, Mapping) or set(raw_page) != _PAGE_KEYS:
                raise ValueError("fixed page record schema is invalid")
            page = dict(raw_page)
            doc_id, page_index, score = (
                page.get("doc_id"),
                page.get("page_index"),
                page.get("score"),
            )
            if (
                page.get("rank") != rank
                or not isinstance(doc_id, str)
                or not doc_id
                or type(page_index) is not int
                or page_index < 0
                or (doc_id, page_index) in page_ids
                or type(score) not in {int, float}
                or not math.isfinite(float(score))
                or page.get("feature_page_index") != page_index
            ):
                raise ValueError("fixed page record identity is invalid")
            for key in ("source_pdf_path", "feature_shard_path"):
                if not isinstance(page.get(key), str) or not Path(page[key]).is_absolute():
                    raise ValueError("fixed page path identity is invalid")
            for key in ("source_pdf_sha256", "rendered_rgb_sha256", "feature_shard_sha256"):
                if not _is_sha256(page.get(key)):
                    raise ValueError("fixed page checksum identity is invalid")
            if (
                type(page.get("rendered_rgb_width")) is not int
                or page["rendered_rgb_width"] <= 0
                or type(page.get("rendered_rgb_height")) is not int
                or page["rendered_rgb_height"] <= 0
                or page.get("renderer_contract") != _RENDERER_CONTRACT
            ):
                raise ValueError("fixed page raster identity is invalid")
            page_ids.add((doc_id, page_index))
            page["score"] = float(score)
            pages.append(page)
        qids.add(qid)
        question_hashes.add(question_sha256)
        checked_questions.append({"qid": qid, "question_sha256": question_sha256, "pages": pages})
    fixture["questions"] = checked_questions
    return fixture


def _validated_run_manifest(
    content: bytes,
    *,
    fixture_path: Path,
    fixture_sha256: str,
    results_path: Path,
    likelihood_path: Path,
) -> dict[str, object]:
    value = _json_mapping_bytes(content, "Task 7 run manifest")
    unsigned = dict(value)
    digest = unsigned.pop("run_manifest_sha256", None)
    if not _is_sha256(digest) or digest != _canonical_sha256(unsigned):
        raise ValueError("Task 7 run manifest identity is invalid")
    if (
        value.get("schema_version") != 1
        or value.get("status") != "configured-task7-fixed-grid"
        or value.get("matrix_kind") != "visual-state-fixed-grid"
        or value.get("cell_count") != len(_INTERVENTION_NAMES)
        or value.get("fixture_path") != str(fixture_path)
        or value.get("fixture_sha256") != fixture_sha256
        or value.get("output") != str(results_path.parent)
        or value.get("fixed_page_provenance") is not True
        or value.get("global_index_loaded") is not False
        or value.get("task7_likelihood_output") != str(likelihood_path)
        or value.get("task7_likelihood_arms") != ["btp-qtp-no-ctp", "all-visual-drop-B_input"]
    ):
        raise ValueError("Task 7 run manifest does not bind the fixed-grid artifacts")
    return value


def _validate_forced(value: object, *, line_number: int) -> None:
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
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError(f"results JSONL record {line_number} has invalid forced intervention")
    population = value.get("visual_population")
    logical = value.get("logical_retained_sequence_ids")
    cache_lengths = value.get("prefill_cache_lengths")
    if (
        value.get("selection_kind") != "forced"
        or value.get("mode") != "physical_delete"
        or not isinstance(value.get("boundary"), str)
        or not value["boundary"].startswith("B_")
        or type(population) is not int
        or population < 0
        or value.get("requested_budget") != value.get("achieved_budget")
        or type(value.get("achieved_budget")) is not int
        or not 0 <= value["achieved_budget"] <= population
        or not isinstance(value.get("retained_visual_ids"), list)
        or value["retained_visual_ids"] != sorted(set(value["retained_visual_ids"]))
        or len(value["retained_visual_ids"]) != value["achieved_budget"]
        or any(
            type(identifier) is not int or not 0 <= identifier < population
            for identifier in value["retained_visual_ids"]
        )
        or not isinstance(logical, list)
        or not logical
        or logical != sorted(set(logical))
        or any(type(identifier) is not int or identifier < 0 for identifier in logical)
        or not isinstance(cache_lengths, list)
        or not cache_lengths
        or any(type(length) is not int or length < 0 for length in cache_lengths)
        or value.get("retained_mrope_position_shape") != [3, 1, len(logical)]
        or not _is_sha256(value.get("retained_mrope_position_sha256"))
    ):
        raise ValueError(f"results JSONL record {line_number} has invalid forced intervention")


def _validate_generic_result(record: Mapping[str, object], *, line_number: int) -> None:
    required = {
        "question_id",
        "question",
        "answers",
        "predicted_answer",
        "retrieved_pages",
        "trace",
        "timing",
    }
    allowed = required | {
        "policy_selection",
        "forced_intervention",
        "fixed_page_fixture_sha256",
        "fixed_page_provenance",
        "global_index_loaded",
        "policy_context",
        "matrix_cell",
        "matrix_kind",
        "intervention_name",
    }
    if line_number <= 2:
        allowed |= {
            "assistant_prompt_sha256",
            "prefill_input_ids_shape",
            "prefill_input_ids_sha256",
        }
    if not required <= set(record) or set(record) - allowed:
        raise ValueError(f"results JSONL record {line_number} has invalid fields")
    if (
        not isinstance(record.get("question_id"), str)
        or not record["question_id"]
        or not isinstance(record.get("question"), str)
        or not isinstance(record.get("predicted_answer"), str)
        or not isinstance(record.get("answers"), list)
        or any(not isinstance(answer, str) for answer in record["answers"])
    ):
        raise ValueError(f"results JSONL record {line_number} has invalid answer fields")
    pages = record.get("retrieved_pages")
    if not isinstance(pages, list) or len(pages) != 4:
        raise ValueError(f"results JSONL record {line_number} has invalid retrieved_pages")
    page_ids: set[tuple[str, int]] = set()
    for page in pages:
        if not isinstance(page, Mapping):
            raise ValueError(f"results JSONL record {line_number} has an invalid page")
        identity = (page.get("doc_id"), page.get("page_index"))
        if (
            not isinstance(identity[0], str)
            or not identity[0]
            or type(identity[1]) is not int
            or identity[1] < 0
            or identity in page_ids
            or type(page.get("score")) not in {int, float}
            or not math.isfinite(float(page["score"]))
        ):
            raise ValueError(f"results JSONL record {line_number} has an invalid page")
        page_ids.add(identity)
    trace = record.get("trace")
    trace_names = (
        "original_visual_tokens",
        "post_btp_visual_tokens",
        "post_qtp_visual_tokens",
        "post_ctp_visual_tokens",
        "ctp_layer",
    )
    if not isinstance(trace, Mapping) or set(trace) != set(trace_names):
        raise ValueError(f"results JSONL record {line_number} has an invalid trace schema")
    counts = [trace[name] for name in trace_names[:4]]
    if (
        any(type(value) is not int or value <= 0 for value in counts[:3])
        or type(counts[3]) is not int
        or counts[3] < 0
        or not all(left >= right for left, right in zip(counts, counts[1:]))
        or trace["ctp_layer"] is not None
    ):
        raise ValueError(f"results JSONL record {line_number} has invalid trace counts")
    timing = record.get("timing")
    required_timing = {
        "retrieval_seconds",
        "page_load_seconds",
        "qa_seconds",
        "total_sample_seconds",
        "encoder_seconds",
        "decoder_seconds",
        "profiler_enabled",
    }
    optional_timing = {
        "peak_allocated_gpu_bytes",
        "warmup_excluded",
        "profiler_definition",
        "flops",
    }
    if (
        not isinstance(timing, Mapping)
        or not required_timing <= set(timing)
        or set(timing) - required_timing - optional_timing
    ):
        raise ValueError(f"results JSONL record {line_number} has invalid timing schema")
    for name in required_timing - {"profiler_enabled"}:
        value = timing[name]
        if type(value) not in {int, float} or not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"results JSONL record {line_number} has invalid timings")
    if "peak_allocated_gpu_bytes" in timing and (
        type(timing["peak_allocated_gpu_bytes"]) is not int
        or timing["peak_allocated_gpu_bytes"] < 0
    ):
        raise ValueError(f"results JSONL record {line_number} has invalid peak GPU bytes")
    if "warmup_excluded" in timing and type(timing["warmup_excluded"]) is not bool:
        raise ValueError(f"results JSONL record {line_number} has invalid warmup flag")
    profiler_enabled = timing["profiler_enabled"]
    if type(profiler_enabled) is not bool:
        raise ValueError(f"results JSONL record {line_number} has invalid profiler flag")
    if profiler_enabled:
        definition, flops = timing.get("profiler_definition"), timing.get("flops")
        if (
            not isinstance(definition, str)
            or not definition
            or type(flops) not in {int, float}
            or not math.isfinite(float(flops))
            or float(flops) < 0
        ):
            raise ValueError(f"results JSONL record {line_number} has invalid profiler data")
    elif "profiler_definition" in timing or "flops" in timing:
        raise ValueError(f"results JSONL record {line_number} has disabled profiler data")
    if line_number <= 2:
        _validated_prefill_identity(record)


def _validate_task7_result(
    record: Mapping[str, object],
    *,
    index: int,
    fixture_sha256: str,
) -> None:
    name = _INTERVENTION_NAMES[index]
    if (
        record.get("fixed_page_fixture_sha256") != fixture_sha256
        or record.get("fixed_page_provenance") is not True
        or record.get("global_index_loaded") is not False
        or record.get("matrix_kind") != "visual-state-fixed-grid"
        or record.get("matrix_cell") != index
        or record.get("intervention_name") != name
    ):
        raise ValueError("Task 7 result has invalid fixed-grid identity")
    trace = record["trace"]
    population, retained = trace["post_qtp_visual_tokens"], trace["post_ctp_visual_tokens"]
    if index == 0:
        selection = record.get("policy_selection")
        selection_keys = {
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
        if (
            "forced_intervention" in record
            or not isinstance(selection, Mapping)
            or set(selection) != selection_keys
            or selection.get("policy") != _REFERENCE_POLICY
            or selection.get("native_layer") is not None
            or selection.get("boundary") is not None
            or selection.get("visual_population") != population
            or selection.get("requested_budget") != population
            or selection.get("achieved_budget") != population
            or selection.get("retained_compact_visual_ids") != list(range(population))
            or selection.get("aggregate_native_reference_ids") is not None
            or selection.get("aggregate_threshold_tied_ids") != []
            or selection.get("symmetric_difference_ids") != []
            or selection.get("native_threshold") is not None
            or selection.get("seed_sha256") is not None
            or selection.get("geometry_count") is not None
            or selection.get("geometry_sha256") is not None
            or selection.get("prefill_cache_lengths") != []
            or selection.get("retained_mrope_position_shape") is not None
            or selection.get("retained_mrope_position_sha256") is not None
            or retained != population
        ):
            raise ValueError("Task 7 result has invalid reference identity")
        return
    if "policy_selection" in record:
        raise ValueError("Task 7 all-drop result contains policy evidence")
    forced = record.get("forced_intervention")
    _validate_forced(forced, line_number=index + 1)
    if (
        forced.get("boundary") != _BOUNDARIES[index - 1]
        or forced.get("visual_population") != population
        or forced.get("requested_budget") != 0
        or forced.get("achieved_budget") != 0
        or forced.get("retained_visual_ids") != []
        or retained != 0
    ):
        raise ValueError("Task 7 result has invalid all-drop identity")
    logical = forced["logical_retained_sequence_ids"]
    if max(logical) >= len(logical) + population or len(forced["prefill_cache_lengths"]) != 28:
        raise ValueError("Task 7 result has invalid physical all-drop evidence")
    compact, full = len(logical), len(logical) + population
    raw_boundary = _BOUNDARIES[index - 1]
    if raw_boundary == "B_input":
        expected_cache = [compact] * _DECODER_LAYER_COUNT
    else:
        boundary = int(raw_boundary.removeprefix("B_"))
        expected_cache = [full] * (boundary + 1) + [compact] * (_DECODER_LAYER_COUNT - boundary - 1)
    if forced["prefill_cache_lengths"] != expected_cache:
        raise ValueError("Task 7 result has invalid physical all-drop evidence")


def _validated_prefill_identity(record: Mapping[str, object]) -> dict[str, object]:
    shape = record.get("prefill_input_ids_shape")
    identity = {
        "assistant_prompt_sha256": record.get("assistant_prompt_sha256"),
        "prefill_input_ids_shape": shape,
        "prefill_input_ids_sha256": record.get("prefill_input_ids_sha256"),
    }
    if (
        not _is_sha256(identity["assistant_prompt_sha256"])
        or not isinstance(shape, list)
        or len(shape) != 2
        or any(type(value) is not int or value <= 0 for value in shape)
        or not _is_sha256(identity["prefill_input_ids_sha256"])
    ):
        raise ValueError("Task 7 result has invalid prompt or prefill identity")
    return identity


def _likelihood_target_payload(
    accepted_references: Sequence[str],
    target_token_ids: Sequence[Sequence[int]],
    assistant_prompt_sha256: object,
) -> dict[str, object]:
    references = tuple(accepted_references)
    targets = tuple(tuple(target) for target in target_token_ids)
    if (
        not references
        or len(references) != len(targets)
        or any(not isinstance(reference, str) for reference in references)
        or not _is_sha256(assistant_prompt_sha256)
        or any(
            not target or any(type(token) is not int or token < 0 for token in target)
            for target in targets
        )
    ):
        raise ValueError("Task 7 likelihood target has invalid references or token IDs")
    payload: dict[str, object] = {
        "schema_version": 1,
        "target_name": "best-reference-full-gold-sequence",
        "normalization": "mean-log-probability-per-supplied-answer-token",
        "aggregation": "maximum-over-official-answer-items",
        "eos_included": False,
        "answer_item_semantics": "local-max-reference-adaptation-not-official-m3docvqa-multispan-scoring",
        "tokenization": "exact-assistant-prompt-prefix-suffix|tokenizer-encode|add-special-tokens-false|no-eos",
        "assistant_prompt_sha256": assistant_prompt_sha256,
        "accepted_references": list(references),
        "target_token_ids": [list(target) for target in targets],
    }
    payload["likelihood_target_sha256"] = _canonical_sha256(payload)
    return payload


def _validated_likelihood_target(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _LIKELIHOOD_TARGET_KEYS:
        raise ValueError("Task 7 likelihood target schema is invalid")
    rebuilt = _likelihood_target_payload(
        value.get("accepted_references", ()),
        value.get("target_token_ids", ()),
        value.get("assistant_prompt_sha256"),
    )
    if value != rebuilt:
        raise ValueError("Task 7 likelihood target identity is invalid")
    return rebuilt


def _validated_likelihood_record(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _LIKELIHOOD_RECORD_KEYS:
        raise ValueError("Task 7 likelihood record schema is invalid")
    target = _validated_likelihood_target(value.get("target"))
    values = value.get("per_reference_mean_loglikelihood")
    shape = value.get("prefill_input_ids_shape")
    if (
        value.get("schema_version") != 1
        or not isinstance(value.get("qid"), str)
        or not value["qid"]
        or not isinstance(value.get("intervention_name"), str)
        or not value["intervention_name"]
        or any(
            not _is_sha256(value.get(key))
            for key in (
                "fixture_sha256",
                "run_manifest_sha256",
                "source_result_sha256",
                "prefill_input_ids_sha256",
            )
        )
        or value.get("likelihood_target_sha256") != target["likelihood_target_sha256"]
        or value.get("assistant_prompt_sha256") != target["assistant_prompt_sha256"]
        or not isinstance(shape, list)
        or len(shape) != 2
        or any(type(item) is not int or item <= 0 for item in shape)
        or not isinstance(values, list)
        or len(values) != len(target["accepted_references"])
        or any(type(item) not in {int, float} or not math.isfinite(float(item)) for item in values)
    ):
        raise ValueError("Task 7 likelihood record provenance or values are invalid")
    best_index = max(range(len(values)), key=lambda index: float(values[index]))
    if value.get("best_reference_index") != best_index or value.get(
        "best_reference_mean_loglikelihood"
    ) != float(values[best_index]):
        raise ValueError("Task 7 likelihood best-reference identity is invalid")
    unsigned = dict(value)
    digest = unsigned.pop("likelihood_record_sha256")
    if not _is_sha256(digest) or digest != _canonical_sha256(unsigned):
        raise ValueError("Task 7 likelihood record identity is invalid")
    return dict(value)


def _validate_likelihood_pair(
    run_manifest: Mapping[str, object],
    results: Sequence[Mapping[str, object]],
    likelihoods: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], dict[str, object]]:
    checked = tuple(_validated_likelihood_record(value) for value in likelihoods)
    expected_names = _INTERVENTION_NAMES[:2]
    qid = run_manifest.get("qid")
    if (
        len(checked) != 2
        or tuple(row["intervention_name"] for row in checked) != expected_names
        or any(row["qid"] != qid for row in checked)
        or checked[0]["target"] != checked[1]["target"]
        or checked[0]["target"]["accepted_references"] != results[0].get("answers")
    ):
        raise ValueError("Task 7 likelihoods must match reference and B_input in order")
    for likelihood, result in zip(checked, results[:2], strict=True):
        input_identity = _validated_prefill_identity(result)
        if (
            likelihood["fixture_sha256"] != run_manifest.get("fixture_sha256")
            or likelihood["run_manifest_sha256"] != run_manifest.get("run_manifest_sha256")
            or likelihood["source_result_sha256"] != _canonical_sha256(result)
            or likelihood["assistant_prompt_sha256"] != input_identity["assistant_prompt_sha256"]
            or likelihood["prefill_input_ids_shape"] != input_identity["prefill_input_ids_shape"]
            or likelihood["prefill_input_ids_sha256"] != input_identity["prefill_input_ids_sha256"]
        ):
            raise ValueError("Task 7 likelihood source result identity does not match")
    return checked


def _official_answer_text(answer: object) -> str:
    if not isinstance(answer, Mapping) or "answer" not in answer:
        raise ValueError("answer must be an object with an answer key")
    return str(answer["answer"])


def _normalize_number(text: str) -> str:
    if _is_number(text):
        return str(float(text))
    try:
        return str(float(word_to_num(text)))
    except (TypeError, ValueError):
        return text


def _normalize_answer(value: object) -> str:
    normalized: list[str] = []
    for token in re.split(" |-", str(value).lower()):
        if not token:
            continue
        if not _is_number(token):
            token = "".join(character for character in token if character not in _PUNCTUATION)
        token = " ".join(token.split())
        token = " ".join(re.sub(r"\b(a|an|the)\b", " ", token).split())
        token = _normalize_number(token)
        if token.strip():
            normalized.append(token.strip())
    return " ".join(normalized).strip()


def _answer_bags(answer: object) -> tuple[list[str], list[set[str]]]:
    spans = answer if isinstance(answer, list | tuple) else [answer]
    normalized = [_normalize_answer(span) for span in spans]
    return normalized, [set(span.split()) for span in normalized]


def _match_numbers(gold: set[str], predicted: set[str]) -> bool:
    gold_numbers = {token for token in gold if _is_number(token)}
    predicted_numbers = {token for token in predicted if _is_number(token)}
    return not gold_numbers or bool(gold_numbers.intersection(predicted_numbers))


def _is_number(text: str) -> bool:
    try:
        float(text)
    except (TypeError, ValueError):
        return False
    return True


def _bag_f1(predicted: set[str], gold: set[str]) -> float:
    intersection = len(predicted.intersection(gold))
    precision = 1.0 if not predicted else intersection / len(predicted)
    recall = 1.0 if not gold else intersection / len(gold)
    return 0.0 if precision == recall == 0.0 else 2 * precision * recall / (precision + recall)


def _list_scores(predicted: object, gold: object) -> tuple[bool, float]:
    predicted_spans, predicted_bags = _answer_bags(predicted)
    gold_spans, gold_bags = _answer_bags(gold)
    if len(predicted_bags) > 20:
        raise ValueError("answer lists longer than 20 spans are unsupported")
    scores = [
        [_bag_f1(pred, ref) if _match_numbers(ref, pred) else 0.0 for pred in predicted_bags]
        for ref in gold_bags
    ]

    @cache
    def best(row: int, used: int) -> float:
        if row == len(gold_bags):
            return 0.0
        value = best(row + 1, used)
        for column, score in enumerate(scores[row]):
            if not used & (1 << column):
                value = max(value, score + best(row + 1, used | (1 << column)))
        return value

    em = set(predicted_spans) == set(gold_spans) and len(predicted_spans) == len(gold_spans)
    f1 = (
        0.0
        if not gold_bags and not predicted_bags
        else round(best(0, 0) / max(len(gold_bags), len(predicted_bags)), 2)
    )
    return em, f1 * 100.0


def _ordered_supporting_document_ids(source: Mapping[str, object]) -> list[str]:
    contexts = source.get("supporting_context")
    if not isinstance(contexts, Sequence) or isinstance(contexts, str | bytes):
        raise ValueError("Task 7 source has invalid supporting_context")
    values: list[str] = []
    for context in contexts:
        doc_id = context.get("doc_id") if isinstance(context, Mapping) else context
        if not isinstance(doc_id, str) or not doc_id:
            raise ValueError("Task 7 source has invalid supporting document identity")
        if doc_id not in values:
            values.append(doc_id)
    if not values:
        raise ValueError("Task 7 source has no supporting document identity")
    return values


def assemble_task7_analysis_bundle_from_artifacts(
    *,
    fixture_path: Path,
    fixture_sha256: str,
    run_manifest_path: Path,
    run_manifest_file_sha256: str,
    results_path: Path,
    results_sha256: str,
    likelihood_path: Path,
    likelihood_sha256: str,
    fixture_authority_path: Path | None = None,
    results_authority_path: Path | None = None,
    likelihood_authority_path: Path | None = None,
    eligible_questions_path: Path | None = None,
    fixture_content: bytes | None = None,
    run_manifest_content: bytes | None = None,
    results_content: bytes | None = None,
    likelihood_content: bytes | None = None,
    eligible_questions_content: bytes | None = None,
) -> dict[str, object]:
    """Authenticate one fixed-grid shard and derive its complete report bundle."""

    supplied_contents = (
        fixture_content,
        run_manifest_content,
        results_content,
        likelihood_content,
        eligible_questions_content,
    )
    if any(content is not None for content in supplied_contents) and any(
        content is None for content in supplied_contents
    ):
        raise ValueError("Task 7 artifact bytes must be supplied as one complete snapshot")
    if fixture_content is None:
        fixture_content = _authenticated_file(fixture_path, fixture_sha256, "fixture artifact")
        manifest_content = _authenticated_file(
            run_manifest_path, run_manifest_file_sha256, "Task 7 run manifest"
        )
        results_content = _authenticated_file(results_path, results_sha256, "results artifact")
        likelihood_content = _authenticated_file(
            likelihood_path, likelihood_sha256, "likelihood artifact"
        )
    else:
        fixture_content = _authenticated_bytes(fixture_content, fixture_sha256, "fixture artifact")
        manifest_content = _authenticated_bytes(
            run_manifest_content, run_manifest_file_sha256, "Task 7 run manifest"
        )
        results_content = _authenticated_bytes(results_content, results_sha256, "results artifact")
        likelihood_content = _authenticated_bytes(
            likelihood_content, likelihood_sha256, "likelihood artifact"
        )
    fixture = _validated_fixture(fixture_content)
    fixture_authority_path = (
        fixture_path if fixture_authority_path is None else Path(fixture_authority_path)
    )
    results_authority_path = (
        results_path if results_authority_path is None else Path(results_authority_path)
    )
    likelihood_authority_path = (
        likelihood_path if likelihood_authority_path is None else Path(likelihood_authority_path)
    )
    manifest = _validated_run_manifest(
        manifest_content,
        fixture_path=fixture_authority_path,
        fixture_sha256=fixture_sha256,
        results_path=results_authority_path,
        likelihood_path=likelihood_authority_path,
    )
    qid = manifest.get("qid")
    questions = [question for question in fixture["questions"] if question["qid"] == qid]
    if len(questions) != 1:
        raise ValueError("Task 7 QID must occur exactly once in the fixture")
    question = questions[0]
    eligible_path = (
        Path(fixture["eligible_questions_path"])
        if eligible_questions_path is None
        else Path(eligible_questions_path)
    )
    eligible_content = (
        _authenticated_file(
            eligible_path, fixture["eligible_questions_sha256"], "eligible questions artifact"
        )
        if eligible_questions_content is None
        else _authenticated_bytes(
            eligible_questions_content,
            fixture["eligible_questions_sha256"],
            "eligible questions artifact",
        )
    )
    eligible_rows = _jsonl_mappings_bytes(eligible_content, "eligible questions")
    sources = [row for row in eligible_rows if row.get("qid", row.get("question_id")) == qid]
    if len(sources) != 1:
        raise ValueError("Task 7 QID must occur exactly once in eligible questions")
    source = sources[0]
    source_qid = source.get("qid", source.get("question_id"))
    source_question = source.get("question")
    source_answers = source.get("answers")
    if (
        not isinstance(source_qid, str)
        or not isinstance(source_question, str)
        or hashlib.sha256(source_question.encode("utf-8")).hexdigest()
        != question["question_sha256"]
        or not isinstance(source_answers, list)
    ):
        raise ValueError("Task 7 eligible source does not match the fixture sample")
    answers = [_official_answer_text(answer) for answer in source_answers]
    results = _jsonl_mappings_bytes(results_content, "Task 7 results")
    if len(results) != len(_INTERVENTION_NAMES):
        raise ValueError("Task 7 results must contain the exact eight-cell fixed grid")
    expected_pages = [
        {"doc_id": page["doc_id"], "page_index": page["page_index"], "score": page["score"]}
        for page in question["pages"]
    ]
    for index, result in enumerate(results):
        _validate_generic_result(result, line_number=index + 1)
        _validate_task7_result(result, index=index, fixture_sha256=fixture_sha256)
        if (
            result.get("question_id") != source_qid
            or result.get("question") != source_question
            or result.get("answers") != answers
            or result.get("retrieved_pages") != expected_pages
        ):
            raise ValueError("Task 7 result does not match the sealed source identity")
    likelihoods = _validate_likelihood_pair(
        manifest,
        results,
        _jsonl_mappings_bytes(likelihood_content, "Task 7 likelihoods"),
    )
    scores = {
        name: _list_scores(result["predicted_answer"], answers)
        for name, result in zip(_INTERVENTION_NAMES, results, strict=True)
    }
    reference_em, reference_f1 = scores[_INTERVENTION_NAMES[0]]
    input_em, _ = scores[_INTERVENTION_NAMES[1]]
    curve_row = {
        "qid": qid,
        "reference_f1": reference_f1,
        "all_drop_f1": {
            boundary: scores[name][1]
            for boundary, name in zip(_BOUNDARIES, _INTERVENTION_NAMES[1:], strict=True)
        },
    }
    opportunity_row = {
        "qid": qid,
        "supporting_document_ids": _ordered_supporting_document_ids(source),
        "retrieved_document_ids": [page["doc_id"] for page in results[0]["retrieved_pages"]],
        "reference": {
            "name": _INTERVENTION_NAMES[0],
            "result_sha256": _canonical_sha256(results[0]),
            "em_correct": reference_em,
            "f1": reference_f1,
        },
        "input_all_drop": {
            "name": _INTERVENTION_NAMES[1],
            "boundary": "B_input",
            "mode": "physical_delete",
            "retained_visual_ids": [],
            "result_sha256": _canonical_sha256(results[1]),
            "em_correct": input_em,
            "best_reference_loglikelihood_drop_per_token": float(
                likelihoods[0]["best_reference_mean_loglikelihood"]
            )
            - float(likelihoods[1]["best_reference_mean_loglikelihood"]),
            "likelihood_target": likelihoods[0]["target"]["target_name"],
            "likelihood_target_sha256": likelihoods[0]["target"]["likelihood_target_sha256"],
        },
    }
    provenance = {
        "fixture_sha256": fixture_sha256,
        "run_manifest_file_sha256": run_manifest_file_sha256,
        "run_manifest_sha256": manifest["run_manifest_sha256"],
        "results_file_sha256": results_sha256,
        "likelihood_file_sha256": likelihood_sha256,
        "reference_result_sha256": opportunity_row["reference"]["result_sha256"],
        "input_all_drop_result_sha256": opportunity_row["input_all_drop"]["result_sha256"],
    }
    bundle: dict[str, object] = {
        "schema_version": 1,
        "qid": qid,
        "curve": {"provenance": dict(provenance), "row": curve_row},
        "opportunity": {"provenance": dict(provenance), "row": opportunity_row},
    }
    bundle["analysis_bundle_sha256"] = _canonical_sha256(bundle)
    return bundle


__all__ = ["assemble_task7_analysis_bundle_from_artifacts", "task7_intervention_identities"]
