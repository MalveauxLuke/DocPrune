"""Fail-closed runtime identities for the visual-state removal curve."""

from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from docprune.ctp_policy import CTPPolicy, btp_qtp_no_ctp_policy
from docprune.evaluation import evaluate_m3docvqa
from docprune.experiment_design import _open_directory_nofollow, _renameat2_noreplace
from docprune.m3docrag import SampleInput
from docprune.qwen2vl.decoder import ForcedVisualIntervention
from docprune.task6_runtime import FixedPageFixture, FixedPageQuestion

_TASK7_DECODER_LAYER_COUNT = 28
_TASK7_LIKELIHOOD_TARGET_KEYS = {
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
_TASK7_LIKELIHOOD_RECORD_KEYS = {
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


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_regular_file_bytes(path: Path, label: str) -> bytes:
    """Read one file through a no-symlink-component descriptor chain."""

    file_path = Path(path)
    if not file_path.is_absolute():
        raise ValueError(f"{label} must be an absolute regular file")
    parent_fd = _open_directory_nofollow(file_path.parent)
    descriptor: int | None = None
    try:
        try:
            descriptor = os.open(
                file_path.name,
                os.O_RDONLY | os.O_NOFOLLOW,
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
    """Return exactly the bytes authenticated through one opened descriptor."""

    if not _is_sha256(expected_sha256):
        raise ValueError(f"{label} checksum must be a lowercase SHA-256")
    content = _read_regular_file_bytes(path, label)
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError(f"{label} checksum mismatch")
    return content


def _jsonl_mappings_bytes(content: bytes, label: str) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    try:
        text = content.decode("utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"{label} row {line_number} is not an object")
            rows.append(dict(row))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"{label} is not valid JSONL") from error
    if not rows:
        raise ValueError(f"{label} must not be empty")
    return tuple(rows)


def tokenize_task7_gold_answers(
    tokenizer: object,
    accepted_references: Sequence[str],
    *,
    assistant_prompt: str,
) -> tuple[tuple[int, ...], ...]:
    """Encode exact prompt continuations without BOS, chat, or EOS additions."""

    references = tuple(accepted_references)
    encode = getattr(tokenizer, "encode", None)
    if (
        not references
        or any(not isinstance(reference, str) for reference in references)
        or not isinstance(assistant_prompt, str)
        or not assistant_prompt
        or not callable(encode)
    ):
        raise ValueError(
            "Task 7 gold tokenization requires an assistant prompt, text references, and a tokenizer"
        )
    prompt_ids = encode(assistant_prompt, add_special_tokens=False)
    if (
        not isinstance(prompt_ids, Sequence)
        or isinstance(prompt_ids, str | bytes)
        or not prompt_ids
        or any(type(token) is not int or token < 0 for token in prompt_ids)
    ):
        raise ValueError("Task 7 assistant prompt produced invalid token IDs")
    prompt_prefix = tuple(prompt_ids)
    tokenized: list[tuple[int, ...]] = []
    for reference in references:
        combined = encode(assistant_prompt + reference, add_special_tokens=False)
        if (
            not isinstance(combined, Sequence)
            or isinstance(combined, str | bytes)
            or any(type(token) is not int or token < 0 for token in combined)
        ):
            raise ValueError("Task 7 prompt-plus-answer produced invalid token IDs")
        combined_ids = tuple(combined)
        if combined_ids[: len(prompt_prefix)] != prompt_prefix:
            raise ValueError("Task 7 answer changes the exact assistant-prompt continuation prefix")
        continuation = combined_ids[len(prompt_prefix) :]
        if not continuation:
            raise ValueError("Task 7 gold answer produced an empty continuation token sequence")
        tokenized.append(continuation)
    return tuple(tokenized)


def _likelihood_target_payload(
    accepted_references: Sequence[str],
    target_token_ids: Sequence[Sequence[int]],
    assistant_prompt_sha256: str,
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
        "answer_item_semantics": (
            "local-max-reference-adaptation-not-official-m3docvqa-multispan-scoring"
        ),
        "tokenization": (
            "exact-assistant-prompt-prefix-suffix|tokenizer-encode|add-special-tokens-false|no-eos"
        ),
        "assistant_prompt_sha256": assistant_prompt_sha256,
        "accepted_references": list(references),
        "target_token_ids": [list(target) for target in targets],
    }
    payload["likelihood_target_sha256"] = _canonical_sha256(payload)
    return payload


def build_task7_likelihood_target(
    accepted_references: Sequence[str],
    target_token_ids: Sequence[Sequence[int]],
    *,
    assistant_prompt: str,
) -> dict[str, object]:
    """Bind the exact prompt and local max-reference full-answer adaptation."""

    if not isinstance(assistant_prompt, str) or not assistant_prompt:
        raise ValueError("Task 7 likelihood target requires the exact assistant prompt")
    return _likelihood_target_payload(
        accepted_references,
        target_token_ids,
        hashlib.sha256(assistant_prompt.encode("utf-8")).hexdigest(),
    )


def prepare_task7_likelihood_target(
    tokenizer: object,
    accepted_references: Sequence[str],
    *,
    assistant_prompt: str,
) -> dict[str, object]:
    """Tokenize and bind one exact-prompt likelihood target in a single step."""

    target_token_ids = tokenize_task7_gold_answers(
        tokenizer,
        accepted_references,
        assistant_prompt=assistant_prompt,
    )
    return build_task7_likelihood_target(
        accepted_references,
        target_token_ids,
        assistant_prompt=assistant_prompt,
    )


def _validated_likelihood_target(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _TASK7_LIKELIHOOD_TARGET_KEYS:
        raise ValueError("Task 7 likelihood target schema is invalid")
    target = dict(value)
    digest = target.pop("likelihood_target_sha256")
    rebuilt = _likelihood_target_payload(
        target.get("accepted_references", ()),
        target.get("target_token_ids", ()),
        target.get("assistant_prompt_sha256"),
    )
    if not _is_sha256(digest) or digest != rebuilt["likelihood_target_sha256"] or value != rebuilt:
        raise ValueError("Task 7 likelihood target identity is invalid")
    return rebuilt


def build_task7_likelihood_record(
    *,
    qid: str,
    intervention_name: str,
    target: Mapping[str, object],
    per_reference_mean_loglikelihood: Sequence[float],
    fixture_sha256: str,
    run_manifest_sha256: str,
    source_result_sha256: str,
    prefill_input_ids_shape: Sequence[int],
    prefill_input_ids_sha256: str,
) -> dict[str, object]:
    """Build one authenticated intervention likelihood record."""

    if not isinstance(qid, str) or not qid:
        raise ValueError("Task 7 likelihood record requires a nonempty QID")
    if not isinstance(intervention_name, str) or not intervention_name:
        raise ValueError("Task 7 likelihood record requires an intervention name")
    validated_target = _validated_likelihood_target(target)
    input_shape = tuple(prefill_input_ids_shape)
    if (
        any(
            not _is_sha256(value)
            for value in (
                fixture_sha256,
                run_manifest_sha256,
                source_result_sha256,
                prefill_input_ids_sha256,
            )
        )
        or len(input_shape) != 2
        or any(type(value) is not int or value <= 0 for value in input_shape)
    ):
        raise ValueError("Task 7 likelihood provenance identity is invalid")
    values = tuple(per_reference_mean_loglikelihood)
    if len(values) != len(validated_target["accepted_references"]) or any(
        type(value) not in {int, float} or not math.isfinite(float(value)) for value in values
    ):
        raise ValueError("Task 7 likelihood values must be finite and match the references")
    best_index = max(range(len(values)), key=lambda index: float(values[index]))
    payload: dict[str, object] = {
        "schema_version": 1,
        "qid": qid,
        "intervention_name": intervention_name,
        "target": validated_target,
        "likelihood_target_sha256": validated_target["likelihood_target_sha256"],
        "fixture_sha256": fixture_sha256,
        "run_manifest_sha256": run_manifest_sha256,
        "source_result_sha256": source_result_sha256,
        "assistant_prompt_sha256": validated_target["assistant_prompt_sha256"],
        "prefill_input_ids_shape": list(input_shape),
        "prefill_input_ids_sha256": prefill_input_ids_sha256,
        "per_reference_mean_loglikelihood": [float(value) for value in values],
        "best_reference_index": best_index,
        "best_reference_mean_loglikelihood": float(values[best_index]),
    }
    payload["likelihood_record_sha256"] = _canonical_sha256(payload)
    return payload


def _validated_likelihood_record(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _TASK7_LIKELIHOOD_RECORD_KEYS:
        raise ValueError("Task 7 likelihood record schema is invalid")
    record = dict(value)
    digest = record.pop("likelihood_record_sha256")
    rebuilt = build_task7_likelihood_record(
        qid=record.get("qid"),
        intervention_name=record.get("intervention_name"),
        target=record.get("target"),
        per_reference_mean_loglikelihood=record.get("per_reference_mean_loglikelihood", ()),
        fixture_sha256=record.get("fixture_sha256"),
        run_manifest_sha256=record.get("run_manifest_sha256"),
        source_result_sha256=record.get("source_result_sha256"),
        prefill_input_ids_shape=record.get("prefill_input_ids_shape", ()),
        prefill_input_ids_sha256=record.get("prefill_input_ids_sha256"),
    )
    if not _is_sha256(digest) or digest != rebuilt["likelihood_record_sha256"] or value != rebuilt:
        raise ValueError("Task 7 likelihood record identity is invalid")
    return rebuilt


def write_task7_likelihood_artifact(path: Path, rows: Sequence[Mapping[str, object]]) -> str:
    """Atomically publish the exact reference/B_input likelihood pair once."""

    destination = Path(path)
    if not destination.is_absolute():
        raise ValueError("Task 7 likelihood destination requires an absolute real parent")
    validated = tuple(_validated_likelihood_record(row) for row in rows)
    if (
        len(validated) != 2
        or tuple(row["intervention_name"] for row in validated)
        != ("btp-qtp-no-ctp", "all-visual-drop-B_input")
        or validated[0]["qid"] != validated[1]["qid"]
        or validated[0]["target"] != validated[1]["target"]
        or validated[0]["fixture_sha256"] != validated[1]["fixture_sha256"]
        or validated[0]["run_manifest_sha256"] != validated[1]["run_manifest_sha256"]
        or validated[0]["assistant_prompt_sha256"] != validated[1]["assistant_prompt_sha256"]
        or validated[0]["prefill_input_ids_shape"] != validated[1]["prefill_input_ids_shape"]
        or validated[0]["prefill_input_ids_sha256"] != validated[1]["prefill_input_ids_sha256"]
    ):
        raise ValueError("Task 7 likelihoods must match reference and B_input in order")
    payload = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode(
            "utf-8"
        )
        for row in validated
    )
    parent_fd = _open_directory_nofollow(destination.parent)
    temporary_name = f".{destination.name}.{secrets.token_hex(16)}"
    descriptor: int | None = None
    temporary_created = False
    try:
        try:
            os.stat(destination.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(f"Task 7 likelihood destination already exists: {destination}")
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=parent_fd,
        )
        temporary_created = True
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            remaining = remaining[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        _renameat2_noreplace(parent_fd, temporary_name, parent_fd, destination.name)
        temporary_created = False
        os.fsync(parent_fd)
        return hashlib.sha256(payload).hexdigest()
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_created:
            try:
                os.unlink(temporary_name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        os.close(parent_fd)


@dataclass(frozen=True, slots=True)
class Task7InterventionCell:
    """One mutually exclusive reference or fixed-boundary all-drop arm."""

    name: str
    ctp_policy: CTPPolicy | None
    forced_intervention: ForcedVisualIntervention | None

    def answerer_kwargs(self) -> dict[str, object]:
        """Return the mutually exclusive arguments consumed by the Qwen answerer."""

        return {
            "ctp_policy": self.ctp_policy,
            "forced_intervention": self.forced_intervention,
        }

    def to_dict(self) -> dict[str, object]:
        """Serialize the mutually exclusive fixed-grid arm identity."""

        forced = self.forced_intervention
        return {
            "name": self.name,
            "analysis_family": "reference" if forced is None else "fixed-grid",
            "ctp_policy": None if self.ctp_policy is None else self.ctp_policy.to_dict(),
            "forced_intervention": (
                None
                if forced is None
                else {
                    "boundary": (
                        "B_input" if forced.boundary == "input" else f"B_{forced.boundary}"
                    ),
                    "mode": forced.mode,
                    "retained_visual_ids": list(forced.retained_visual_ids),
                }
            ),
        }


def task7_intervention_matrix() -> tuple[Task7InterventionCell, ...]:
    """Return the fixed reference/all-drop matrix."""

    boundaries: tuple[str | int, ...] = ("input", 0, 6, 13, 20, 23, 26)
    return (
        Task7InterventionCell("btp-qtp-no-ctp", btp_qtp_no_ctp_policy(), None),
        *(
            Task7InterventionCell(
                f"all-visual-drop-B_{boundary}",
                None,
                ForcedVisualIntervention(boundary, "physical_delete", ()),
            )
            for boundary in boundaries
        ),
    )


def bind_task7_result_evidence(
    record: dict[str, object],
    cell: Task7InterventionCell,
    *,
    cell_index: int,
    fixture_sha256: str,
) -> None:
    """Attach only manifest-derived Task 7 evidence before validation and write."""

    matrix = task7_intervention_matrix()
    if (
        type(cell_index) is not int
        or not 0 <= cell_index < len(matrix)
        or matrix[cell_index] != cell
    ):
        raise ValueError("Task 7 result uses an invalid matrix cell")
    if len(fixture_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in fixture_sha256
    ):
        raise ValueError("Task 7 fixture checksum must be a lowercase SHA-256")
    evidence = {
        "fixed_page_fixture_sha256": fixture_sha256,
        "fixed_page_provenance": True,
        "global_index_loaded": False,
        "matrix_kind": "visual-state-fixed-grid",
        "matrix_cell": cell_index,
        "intervention_name": cell.name,
    }
    for key, value in evidence.items():
        if key in record and record[key] != value:
            raise ValueError(f"Task 7 result attempts to replace immutable evidence: {key}")
        record[key] = value


def bind_task7_prefill_identity(
    record: dict[str, object],
    *,
    assistant_prompt_sha256: str,
    prefill_input_ids_shape: Sequence[int],
    prefill_input_ids_sha256: str,
) -> None:
    """Attach the exact complete prompt and processor input identity once."""

    shape = tuple(prefill_input_ids_shape)
    if (
        not _is_sha256(assistant_prompt_sha256)
        or not _is_sha256(prefill_input_ids_sha256)
        or len(shape) != 2
        or any(type(value) is not int or value <= 0 for value in shape)
    ):
        raise ValueError("Task 7 prefill identity is invalid")
    evidence = {
        "assistant_prompt_sha256": assistant_prompt_sha256,
        "prefill_input_ids_shape": list(shape),
        "prefill_input_ids_sha256": prefill_input_ids_sha256,
    }
    for key, value in evidence.items():
        if key in record and record[key] != value:
            raise ValueError(f"Task 7 result attempts to replace immutable prefill identity: {key}")
        record[key] = value


def validate_task7_source_identity(
    record: Mapping[str, object],
    sample: SampleInput,
    fixture_question: FixedPageQuestion,
) -> None:
    """Bind a Task 7 result to the sealed sample and ordered fixed-page tuple."""

    expected_pages = [
        {
            "doc_id": page.doc_id,
            "page_index": page.page_index,
            "score": float(page.score),
        }
        for page in fixture_question.pages
    ]
    if (
        sample.question_id != fixture_question.qid
        or record.get("question_id") != sample.question_id
        or record.get("question") != sample.question
        or record.get("answers") != list(sample.answers)
        or record.get("retrieved_pages") != expected_pages
    ):
        raise ValueError("Task 7 result does not match the sealed source identity")


def validate_task7_result_record(
    record: Mapping[str, object],
    cell: Task7InterventionCell,
    *,
    fixture_sha256: str,
) -> None:
    """Cross-bind one generic validated result to its fixed-grid Task 7 arm."""

    if (
        len(fixture_sha256) != 64
        or any(character not in "0123456789abcdef" for character in fixture_sha256)
        or record.get("fixed_page_fixture_sha256") != fixture_sha256
        or record.get("fixed_page_provenance") is not True
        or record.get("global_index_loaded") is not False
        or record.get("matrix_kind") != "visual-state-fixed-grid"
    ):
        raise ValueError("Task 7 result has invalid fixed-page identity")
    matrix = task7_intervention_matrix()
    try:
        cell_index = matrix.index(cell)
    except ValueError as error:
        raise ValueError("Task 7 result uses an unknown intervention cell") from error
    if record.get("matrix_cell") != cell_index or record.get("intervention_name") != cell.name:
        raise ValueError("Task 7 result has invalid matrix-cell identity")
    trace = record.get("trace")
    if not isinstance(trace, Mapping):
        raise ValueError("Task 7 result has invalid pruning trace")
    population = trace.get("post_qtp_visual_tokens")
    retained = trace.get("post_ctp_visual_tokens")
    if (
        not isinstance(population, int)
        or isinstance(population, bool)
        or population < 0
        or not isinstance(retained, int)
        or isinstance(retained, bool)
        or retained < 0
        or trace.get("ctp_layer") is not None
    ):
        raise ValueError("Task 7 result has invalid pruning trace")

    if cell.forced_intervention is None:
        selection = record.get("policy_selection")
        policy = selection.get("policy") if isinstance(selection, Mapping) else None
        if (
            "forced_intervention" in record
            or not isinstance(policy, Mapping)
            or policy.get("name") != "btp-qtp-no-ctp"
            or policy.get("family") != "no-ctp"
            or policy.get("selection_kind") != "none"
            or retained != population
        ):
            raise ValueError("Task 7 result has invalid reference identity")
        return

    forced = record.get("forced_intervention")
    expected_boundary = (
        "B_input"
        if cell.forced_intervention.boundary == "input"
        else f"B_{cell.forced_intervention.boundary}"
    )
    if (
        "policy_selection" in record
        or not isinstance(forced, Mapping)
        or forced.get("boundary") != expected_boundary
        or forced.get("mode") != "physical_delete"
        or forced.get("selection_kind") != "forced"
        or forced.get("visual_population") != population
        or forced.get("requested_budget") != 0
        or forced.get("achieved_budget") != 0
        or forced.get("retained_visual_ids") != []
        or retained != 0
    ):
        raise ValueError("Task 7 result has invalid all-drop identity")

    logical_ids = forced.get("logical_retained_sequence_ids")
    cache_lengths = forced.get("prefill_cache_lengths")
    position_shape = forced.get("retained_mrope_position_shape")
    if (
        not isinstance(logical_ids, list)
        or not logical_ids
        or logical_ids != sorted(set(logical_ids))
        or any(type(identifier) is not int or identifier < 0 for identifier in logical_ids)
        or max(logical_ids) >= len(logical_ids) + population
        or not isinstance(cache_lengths, list)
        or len(cache_lengths) != _TASK7_DECODER_LAYER_COUNT
        or position_shape != [3, 1, len(logical_ids)]
    ):
        raise ValueError("Task 7 result has invalid physical all-drop evidence")

    compact_length = len(logical_ids)
    full_length = compact_length + population
    boundary = cell.forced_intervention.boundary
    if boundary == "input":
        expected_cache_lengths = [compact_length] * _TASK7_DECODER_LAYER_COUNT
    else:
        expected_cache_lengths = [full_length] * (boundary + 1) + [compact_length] * (
            _TASK7_DECODER_LAYER_COUNT - boundary - 1
        )
    if cache_lengths != expected_cache_lengths:
        raise ValueError("Task 7 result has invalid physical all-drop evidence")


def _eligible_source_row(rows: Sequence[Mapping[str, object]], qid: str) -> dict[str, object]:
    matches = [dict(row) for row in rows if row.get("qid", row.get("question_id")) == qid]
    if len(matches) != 1:
        raise ValueError("Task 7 QID must occur exactly once in eligible questions")
    return matches[0]


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


def _ordered_retrieved_document_ids(record: Mapping[str, object]) -> list[str]:
    pages = record.get("retrieved_pages")
    if not isinstance(pages, Sequence) or isinstance(pages, str | bytes) or len(pages) != 4:
        raise ValueError("Task 7 reference must contain four cached retrieved pages")
    values: list[str] = []
    for page in pages:
        doc_id = page.get("doc_id") if isinstance(page, Mapping) else None
        if not isinstance(doc_id, str) or not doc_id:
            raise ValueError("Task 7 reference has invalid retrieved document identity")
        values.append(doc_id)
    return values


def _task7_result_score(
    record: Mapping[str, object], source: Mapping[str, object]
) -> tuple[bool, float]:
    qid = record.get("question_id")
    metrics = evaluate_m3docvqa((record,), (source,))
    score = metrics.per_question.get(qid)
    if score is None:
        raise ValueError("Task 7 result could not be scored against its source")
    return score["list_em"] == 1.0, score["list_f1"] * 100.0


def _validated_task7_run_manifest(
    content: bytes,
    *,
    fixture_path: Path,
    fixture_sha256: str,
    results_path: Path,
    likelihood_path: Path,
) -> dict[str, object]:
    try:
        value = json.loads(content.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("Task 7 run manifest is not valid JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError("Task 7 run manifest is not an object")
    manifest = dict(value)
    digest = manifest.pop("run_manifest_sha256", None)
    if not _is_sha256(digest) or digest != _canonical_sha256(manifest):
        raise ValueError("Task 7 run manifest identity is invalid")
    if (
        value.get("schema_version") != 1
        or value.get("status") != "configured-task7-fixed-grid"
        or value.get("matrix_kind") != "visual-state-fixed-grid"
        or value.get("cell_count") != len(task7_intervention_matrix())
        or value.get("fixture_path") != str(fixture_path)
        or value.get("fixture_sha256") != fixture_sha256
        or value.get("output") != str(results_path.parent)
        or value.get("fixed_page_provenance") is not True
        or value.get("global_index_loaded") is not False
        or value.get("task7_likelihood_output") != str(likelihood_path)
        or value.get("task7_likelihood_arms") != ["btp-qtp-no-ctp", "all-visual-drop-B_input"]
    ):
        raise ValueError("Task 7 run manifest does not bind the fixed-grid artifacts")
    return dict(value)


def _validate_task7_likelihood_pair(
    run_manifest: Mapping[str, object],
    result_rows: Sequence[Mapping[str, object]],
    likelihood_rows: Sequence[Mapping[str, object]],
) -> None:
    """Bind the exact two likelihood rows to the run and first two QA rows."""

    expected_names = ("btp-qtp-no-ctp", "all-visual-drop-B_input")
    if len(result_rows) < 2:
        raise ValueError("Task 7 results must contain the reference and B_input prefix")
    first_two = tuple(dict(row) for row in result_rows[:2])
    if (
        tuple(row.get("intervention_name") for row in first_two) != expected_names
        or tuple(row.get("matrix_cell") for row in first_two) != (0, 1)
        or any(row.get("matrix_kind") != "visual-state-fixed-grid" for row in first_two)
    ):
        raise ValueError("Task 7 results must begin with reference and B_input in order")
    qid = run_manifest.get("qid")
    if (
        not isinstance(qid, str)
        or not qid
        or any(row.get("question_id") != qid for row in first_two)
    ):
        raise ValueError("Task 7 likelihood pair QID does not match the run")
    answers = first_two[0].get("answers")
    if (
        not isinstance(answers, list)
        or not answers
        or any(not isinstance(answer, str) for answer in answers)
        or first_two[1].get("answers") != answers
    ):
        raise ValueError("Task 7 likelihood pair answers do not match the first two results")
    validated = tuple(_validated_likelihood_record(row) for row in likelihood_rows)
    if (
        len(validated) != 2
        or tuple(row["intervention_name"] for row in validated) != expected_names
        or any(row["qid"] != qid for row in validated)
    ):
        raise ValueError("Task 7 likelihoods must match reference and B_input in order")
    target = validated[0]["target"]
    if validated[1]["target"] != target or target["accepted_references"] != answers:
        raise ValueError("Task 7 likelihood target does not match the sealed gold answers")
    for likelihood, result in zip(validated, first_two, strict=True):
        if (
            likelihood["fixture_sha256"] != run_manifest.get("fixture_sha256")
            or likelihood["run_manifest_sha256"] != run_manifest.get("run_manifest_sha256")
            or likelihood["source_result_sha256"] != _canonical_sha256(result)
        ):
            raise ValueError("Task 7 likelihood source result identity does not match")
        result_input = _validated_result_prefill_identity(result)
        if (
            likelihood["assistant_prompt_sha256"] != result_input["assistant_prompt_sha256"]
            or likelihood["target"]["assistant_prompt_sha256"]
            != result_input["assistant_prompt_sha256"]
            or likelihood["prefill_input_ids_shape"] != result_input["prefill_input_ids_shape"]
            or likelihood["prefill_input_ids_sha256"] != result_input["prefill_input_ids_sha256"]
        ):
            raise ValueError("Task 7 likelihood prompt or prefill identity does not match")


def admit_task7_likelihood_pair_from_files(
    *,
    run_manifest_path: Path,
    results_path: Path,
    likelihood_path: Path,
) -> str:
    """Admit an exact Task 7 likelihood pair for resume or final postflight."""

    manifest_content = _read_regular_file_bytes(run_manifest_path, "Task 7 run manifest")
    try:
        unsigned_manifest = json.loads(manifest_content.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("Task 7 run manifest is not valid JSON") from error
    if not isinstance(unsigned_manifest, Mapping):
        raise ValueError("Task 7 run manifest is not an object")
    fixture_path = unsigned_manifest.get("fixture_path")
    fixture_sha256 = unsigned_manifest.get("fixture_sha256")
    if not isinstance(fixture_path, str) or not _is_sha256(fixture_sha256):
        raise ValueError("Task 7 run manifest fixture identity is invalid")
    results_content = _read_regular_file_bytes(results_path, "Task 7 results")
    likelihood_content = _read_regular_file_bytes(likelihood_path, "Task 7 likelihoods")
    run_manifest = _validated_task7_run_manifest(
        manifest_content,
        fixture_path=Path(fixture_path),
        fixture_sha256=fixture_sha256,
        results_path=Path(results_path),
        likelihood_path=Path(likelihood_path),
    )
    result_rows = _jsonl_mappings_bytes(results_content, "Task 7 results")
    likelihood_rows = _jsonl_mappings_bytes(likelihood_content, "Task 7 likelihoods")
    _validate_task7_likelihood_pair(run_manifest, result_rows, likelihood_rows)
    return hashlib.sha256(likelihood_content).hexdigest()


def _validated_result_prefill_identity(record: Mapping[str, object]) -> dict[str, object]:
    identity = {
        "assistant_prompt_sha256": record.get("assistant_prompt_sha256"),
        "prefill_input_ids_shape": record.get("prefill_input_ids_shape"),
        "prefill_input_ids_sha256": record.get("prefill_input_ids_sha256"),
    }
    shape = identity["prefill_input_ids_shape"]
    if (
        not _is_sha256(identity["assistant_prompt_sha256"])
        or not isinstance(shape, list)
        or len(shape) != 2
        or any(type(value) is not int or value <= 0 for value in shape)
        or not _is_sha256(identity["prefill_input_ids_sha256"])
    ):
        raise ValueError("Task 7 result has invalid prompt or prefill identity")
    return identity


def assemble_task7_opportunity_row_from_artifacts(
    *,
    fixture_path: Path,
    fixture_sha256: str,
    run_manifest_path: Path,
    run_manifest_file_sha256: str,
    results_path: Path,
    results_sha256: str,
    likelihood_path: Path,
    likelihood_sha256: str,
) -> dict[str, object]:
    """Authenticate raw Task 7 artifacts and derive one opportunity-strata row.

    This function deliberately derives scores and support/retrieval identities
    from the sealed sources.  It never accepts a precomputed stratum label.
    """

    fixture_content = _authenticated_file(fixture_path, fixture_sha256, "fixture artifact")
    results_content = _authenticated_file(results_path, results_sha256, "results artifact")
    likelihood_content = _authenticated_file(
        likelihood_path, likelihood_sha256, "likelihood artifact"
    )
    manifest_content = _authenticated_file(
        run_manifest_path, run_manifest_file_sha256, "Task 7 run manifest"
    )
    run_manifest = _validated_task7_run_manifest(
        manifest_content,
        fixture_path=fixture_path,
        fixture_sha256=fixture_sha256,
        results_path=results_path,
        likelihood_path=likelihood_path,
    )
    try:
        fixture_payload = json.loads(fixture_content.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("fixed page fixture is not valid JSON") from error
    fixture = FixedPageFixture.from_dict(fixture_payload)
    eligible_content = _authenticated_file(
        fixture.eligible_questions_path,
        fixture.eligible_questions_sha256,
        "eligible questions artifact",
    )
    eligible_rows = _jsonl_mappings_bytes(eligible_content, "eligible questions")
    result_rows = _jsonl_mappings_bytes(results_content, "Task 7 results")
    matrix = task7_intervention_matrix()
    if len(result_rows) != len(matrix):
        raise ValueError("Task 7 results must contain the exact eight-cell fixed grid")
    qids = {row.get("question_id") for row in result_rows}
    if len(qids) != 1 or not isinstance(next(iter(qids)), str):
        raise ValueError("Task 7 results must contain exactly one nonempty QID")
    qid = next(iter(qids))
    if not qid:
        raise ValueError("Task 7 results must contain exactly one nonempty QID")
    if run_manifest.get("qid") != qid:
        raise ValueError("Task 7 run manifest QID does not match the results")
    fixture_question = fixture.question(qid)
    source = _eligible_source_row(eligible_rows, qid)
    sample = SampleInput.from_mapping(source)
    if hashlib.sha256(sample.question.encode("utf-8")).hexdigest() != (
        fixture_question.question_sha256
    ):
        raise ValueError("Task 7 eligible source does not match the fixture sample")

    from docprune.m3docvqa_factory import _validate_result_record

    results_by_name: dict[str, dict[str, object]] = {}
    for line_number, (record, cell) in enumerate(zip(result_rows, matrix, strict=True), start=1):
        expected_policy = None if cell.ctp_policy is None else cell.ctp_policy.to_dict()
        _validate_result_record(
            record,
            line_number=line_number,
            expected_page_count=4,
            production=True,
            expected_policy=expected_policy,
            forbid_policy=expected_policy is None,
            allowed_extra_fields={"matrix_cell", "matrix_kind", "intervention_name"}
            | (
                {
                    "assistant_prompt_sha256",
                    "prefill_input_ids_shape",
                    "prefill_input_ids_sha256",
                }
                if line_number <= 2
                else set()
            ),
            allow_zero_post_ctp=True,
        )
        validate_task7_source_identity(record, sample, fixture_question)
        validate_task7_result_record(record, cell, fixture_sha256=fixture_sha256)
        if line_number <= 2:
            _validated_result_prefill_identity(record)
        results_by_name[cell.name] = record

    likelihood_rows = tuple(
        _validated_likelihood_record(row)
        for row in _jsonl_mappings_bytes(likelihood_content, "Task 7 likelihoods")
    )
    expected_likelihood_names = ("btp-qtp-no-ctp", "all-visual-drop-B_input")
    if (
        len(likelihood_rows) != len(expected_likelihood_names)
        or tuple(row["intervention_name"] for row in likelihood_rows) != expected_likelihood_names
        or any(row["qid"] != qid for row in likelihood_rows)
    ):
        raise ValueError("Task 7 likelihoods must match reference and B_input in order")
    target = likelihood_rows[0]["target"]
    if likelihood_rows[1]["target"] != target or target["accepted_references"] != list(
        sample.answers
    ):
        raise ValueError("Task 7 likelihood target does not match the sealed gold answers")

    reference_result = results_by_name["btp-qtp-no-ctp"]
    input_result = results_by_name["all-visual-drop-B_input"]
    _validate_task7_likelihood_pair(run_manifest, result_rows, likelihood_rows)
    for likelihood, result in zip(
        likelihood_rows,
        (reference_result, input_result),
        strict=True,
    ):
        if (
            likelihood["fixture_sha256"] != fixture_sha256
            or likelihood["run_manifest_sha256"] != run_manifest["run_manifest_sha256"]
            or likelihood["source_result_sha256"] != _canonical_sha256(result)
        ):
            raise ValueError("Task 7 likelihood source result identity does not match")
        result_input = _validated_result_prefill_identity(result)
        if (
            likelihood["assistant_prompt_sha256"] != result_input["assistant_prompt_sha256"]
            or likelihood["target"]["assistant_prompt_sha256"]
            != result_input["assistant_prompt_sha256"]
            or likelihood["prefill_input_ids_shape"] != result_input["prefill_input_ids_shape"]
            or likelihood["prefill_input_ids_sha256"] != result_input["prefill_input_ids_sha256"]
        ):
            raise ValueError("Task 7 likelihood prompt or prefill identity does not match")
    reference_em, reference_f1 = _task7_result_score(reference_result, source)
    input_em, _ = _task7_result_score(input_result, source)
    reference_likelihood, input_likelihood = likelihood_rows
    likelihood_drop = float(reference_likelihood["best_reference_mean_loglikelihood"]) - float(
        input_likelihood["best_reference_mean_loglikelihood"]
    )
    return {
        "qid": qid,
        "supporting_document_ids": _ordered_supporting_document_ids(source),
        "retrieved_document_ids": _ordered_retrieved_document_ids(reference_result),
        "reference": {
            "name": "btp-qtp-no-ctp",
            "result_sha256": _canonical_sha256(reference_result),
            "em_correct": reference_em,
            "f1": reference_f1,
        },
        "input_all_drop": {
            "name": "all-visual-drop-B_input",
            "boundary": "B_input",
            "mode": "physical_delete",
            "retained_visual_ids": [],
            "result_sha256": _canonical_sha256(input_result),
            "em_correct": input_em,
            "best_reference_loglikelihood_drop_per_token": likelihood_drop,
            "likelihood_target": target["target_name"],
            "likelihood_target_sha256": target["likelihood_target_sha256"],
        },
    }


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
) -> dict[str, object]:
    """Bind one fixed-grid curve/opportunity pair to the same admitted artifacts.

    The opportunity assembler is the single admission gate for the fixture,
    manifest, eight result cells, and the paired likelihood artifact.  This
    companion then re-authenticates the result bytes, derives every score with
    the official evaluator, and emits duplicated exact provenance so report
    assembly can reject same-QID rows sourced from different artifacts.
    """

    admitted = assemble_task7_opportunity_row_from_artifacts(
        fixture_path=fixture_path,
        fixture_sha256=fixture_sha256,
        run_manifest_path=run_manifest_path,
        run_manifest_file_sha256=run_manifest_file_sha256,
        results_path=results_path,
        results_sha256=results_sha256,
        likelihood_path=likelihood_path,
        likelihood_sha256=likelihood_sha256,
    )
    results_content = _authenticated_file(results_path, results_sha256, "results artifact")
    fixture_content = _authenticated_file(fixture_path, fixture_sha256, "fixture artifact")
    _authenticated_file(likelihood_path, likelihood_sha256, "likelihood artifact")
    manifest_content = _authenticated_file(
        run_manifest_path, run_manifest_file_sha256, "Task 7 run manifest"
    )
    try:
        fixture_payload = json.loads(fixture_content.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("fixed page fixture is not valid JSON") from error
    fixture = FixedPageFixture.from_dict(fixture_payload)
    qid = admitted["qid"]
    eligible_content = _authenticated_file(
        fixture.eligible_questions_path,
        fixture.eligible_questions_sha256,
        "eligible questions artifact",
    )
    source = _eligible_source_row(
        _jsonl_mappings_bytes(eligible_content, "eligible questions"), qid
    )
    rows = _jsonl_mappings_bytes(results_content, "Task 7 results")
    matrix = task7_intervention_matrix()
    if len(rows) != len(matrix) or any(
        row.get("question_id") != qid or row.get("intervention_name") != cell.name
        for row, cell in zip(rows, matrix, strict=True)
    ):
        raise ValueError("Task 7 curve results changed after artifact admission")
    scores = {
        cell.name: _task7_result_score(row, source)[1]
        for row, cell in zip(rows, matrix, strict=True)
    }
    boundary_names = (
        ("B_input", "all-visual-drop-B_input"),
        ("B_0", "all-visual-drop-B_0"),
        ("B_6", "all-visual-drop-B_6"),
        ("B_13", "all-visual-drop-B_13"),
        ("B_20", "all-visual-drop-B_20"),
        ("B_23", "all-visual-drop-B_23"),
        ("B_26", "all-visual-drop-B_26"),
    )
    curve_row = {
        "qid": qid,
        "reference_f1": scores["btp-qtp-no-ctp"],
        "all_drop_f1": {
            boundary: scores[intervention] for boundary, intervention in boundary_names
        },
    }
    run_manifest = _validated_task7_run_manifest(
        manifest_content,
        fixture_path=fixture_path,
        fixture_sha256=fixture_sha256,
        results_path=results_path,
        likelihood_path=likelihood_path,
    )
    provenance = {
        "fixture_sha256": fixture_sha256,
        "run_manifest_file_sha256": run_manifest_file_sha256,
        "run_manifest_sha256": run_manifest["run_manifest_sha256"],
        "results_file_sha256": results_sha256,
        "likelihood_file_sha256": likelihood_sha256,
        "reference_result_sha256": admitted["reference"]["result_sha256"],
        "input_all_drop_result_sha256": admitted["input_all_drop"]["result_sha256"],
    }
    bundle: dict[str, object] = {
        "schema_version": 1,
        "qid": qid,
        "curve": {"provenance": dict(provenance), "row": curve_row},
        "opportunity": {"provenance": dict(provenance), "row": admitted},
    }
    bundle["analysis_bundle_sha256"] = _canonical_sha256(bundle)
    return bundle
