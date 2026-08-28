"""Authenticated CPU-only admission for Task 7 native-boundary replays."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from docprune.cli import _manifest_digest
from docprune.experiment_design import _open_directory_nofollow
from docprune.m3docvqa_factory import _validate_result_record
from docprune.task6_runtime import load_fixed_page_fixture
from docprune.task7_artifact_analysis import _list_scores
from docprune.task7_native_boundary import (
    Task7NativeBoundaryEntry,
    load_task7_native_boundary_manifest,
    task7_native_boundary_cell,
    validate_task7_native_boundary_result_record,
)
from docprune.task7_report_driver import (
    _authenticated_file,
    _canonical_sha256,
    _json_mapping,
    _publish_regular_file_noreplace,
    _read_regular_file_bytes,
    _sealed_qids,
)
from docprune.task7_runtime import validate_task7_source_identity

_SHARD_COUNT = 64
_SHARD_MEMBERS = ("run_manifest.json", "results.jsonl")
_RUN_MANIFEST_KEYS = {
    "schema_version",
    "status",
    "output",
    "matrix_kind",
    "shard",
    "qid",
    "cell_count",
    "fixture_path",
    "fixture_sha256",
    "gate_manifest_path",
    "gate_manifest_sha256",
    "feature_manifest_path",
    "feature_manifest_sha256",
    "feature_build_source_order_sha256",
    "fixed_page_provenance",
    "global_index_loaded",
    "feature_build_runtime_commit",
    "runtime_commit",
    "m3docrag_commit",
    "resources",
    "generation",
    "cells",
    "native_boundary_manifest_path",
    "native_boundary_manifest_sha256",
    "run_manifest_sha256",
}


def _require_exact_native_shard_tree(root: Path) -> None:
    """Require exactly 64 real shard directories containing two regular files."""

    root = Path(root)
    if not root.is_absolute():
        raise ValueError("Task 7 native shard root must be absolute")
    root_fd = _open_directory_nofollow(root)
    try:
        expected = {f"shard-{index:04d}" for index in range(_SHARD_COUNT)}
        if set(os.listdir(root_fd)) != expected:
            raise ValueError("Task 7 native shard root has missing or extra entries")
        for shard_name in sorted(expected):
            shard_fd: int | None = None
            try:
                shard_fd = os.open(
                    shard_name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=root_fd,
                )
                if not stat.S_ISDIR(os.fstat(shard_fd).st_mode):
                    raise ValueError("Task 7 native shard is not a real directory")
                if set(os.listdir(shard_fd)) != set(_SHARD_MEMBERS):
                    raise ValueError("Task 7 native shard has missing or extra files")
                for member in _SHARD_MEMBERS:
                    member_fd = os.open(
                        member,
                        os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW,
                        dir_fd=shard_fd,
                    )
                    try:
                        if not stat.S_ISREG(os.fstat(member_fd).st_mode):
                            raise ValueError("Task 7 native shard member is not a regular file")
                    finally:
                        os.close(member_fd)
            except OSError as error:
                raise ValueError("Task 7 native shard tree contains a substituted path") from error
            finally:
                if shard_fd is not None:
                    os.close(shard_fd)
    finally:
        os.close(root_fd)


def _validate_run_manifest(
    manifest: Mapping[str, object],
    *,
    shard_root: Path,
    shard_index: int,
    qid: str,
    fixture_path: Path,
    fixture_sha256: str,
    gate_path: Path,
    gate_sha256: str,
    native_boundary_path: Path,
    native_boundary_sha256: str,
    runtime_commit: str,
    expected_cell: Mapping[str, object],
    feature_manifest_path: Path | None = None,
    feature_manifest_sha256: str | None = None,
) -> None:
    supplied = dict(manifest)
    digest = supplied.pop("run_manifest_sha256", None)
    expected_shard = shard_root / f"shard-{shard_index:04d}"
    if (
        set(manifest) != _RUN_MANIFEST_KEYS
        or digest != _manifest_digest(supplied)
        or manifest.get("schema_version") != 1
        or manifest.get("status") != "configured-task7-native-boundary"
        or manifest.get("output") != str(expected_shard)
        or manifest.get("matrix_kind") != "visual-state-native-boundary"
        or manifest.get("shard") != shard_index
        or manifest.get("qid") != qid
        or manifest.get("cell_count") != 1
        or manifest.get("fixture_path") != str(fixture_path)
        or manifest.get("fixture_sha256") != fixture_sha256
        or manifest.get("gate_manifest_path") != str(gate_path)
        or manifest.get("gate_manifest_sha256") != gate_sha256
        or manifest.get("fixed_page_provenance") is not True
        or manifest.get("global_index_loaded") is not False
        or manifest.get("runtime_commit") != runtime_commit
        or (
            feature_manifest_path is not None
            and manifest.get("feature_manifest_path") != str(feature_manifest_path)
        )
        or (
            feature_manifest_sha256 is not None
            and manifest.get("feature_manifest_sha256") != feature_manifest_sha256
        )
        or manifest.get("native_boundary_manifest_path") != str(native_boundary_path)
        or manifest.get("native_boundary_manifest_sha256") != native_boundary_sha256
        or manifest.get("cells") != [dict(expected_cell)]
    ):
        raise ValueError("Task 7 run manifest differs from the sealed native replay")


def _one_jsonl_mapping(content: bytes) -> dict[str, object]:
    try:
        rows = [json.loads(line) for line in content.decode("utf-8").splitlines() if line.strip()]
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("Task 7 native results are not valid JSONL") from error
    if len(rows) != 1 or not isinstance(rows[0], Mapping):
        raise ValueError("Task 7 native shard must contain exactly one result")
    return dict(rows[0])


def _summarize_native_records(
    records: Sequence[Mapping[str, object]],
    entries: Sequence[Task7NativeBoundaryEntry],
) -> dict[str, object]:
    if len(records) != len(entries) or not records:
        raise ValueError("Task 7 native summary requires one record per boundary entry")
    scored: list[tuple[bool, float]] = []
    crossing_scores: list[tuple[bool, float]] = []
    layer_counts: Counter[str] = Counter()
    for record, entry in zip(records, entries, strict=True):
        if record.get("question_id") != entry.qid:
            raise ValueError("Task 7 native summary QID order drifted")
        answers = record.get("answers")
        if not isinstance(answers, list):
            raise ValueError("Task 7 native result answers are invalid")
        score = _list_scores(record.get("predicted_answer"), answers)
        scored.append(score)
        if entry.native_crossing:
            crossing_scores.append(score)
            assert entry.native_layer is not None
            layer_counts[str(entry.native_layer)] += 1

    def mean(values: Sequence[tuple[bool, float]], index: int) -> float | None:
        if not values:
            return None
        scale = 100.0 if index == 0 else 1.0
        return scale * sum(float(value[index]) for value in values) / len(values)

    return {
        "interpretation": "descriptive-native-boundary-diagnostic-not-information-horizon",
        "qid_count": len(records),
        "native_crossing_count": len(crossing_scores),
        "native_no_crossing_noop_count": len(records) - len(crossing_scores),
        "native_layer_counts": dict(sorted(layer_counts.items(), key=lambda pair: int(pair[0]))),
        "overall_em": mean(scored, 0),
        "overall_f1": mean(scored, 1),
        "crossing_only_em": mean(crossing_scores, 0),
        "crossing_only_f1": mean(crossing_scores, 1),
    }


def admit_task7_native_boundary_results(
    *,
    shard_root: Path,
    fixture_path: Path,
    fixture_sha256: str,
    gate_path: Path,
    gate_sha256: str,
    native_boundary_path: Path,
    native_boundary_sha256: str,
    runtime_commit: str,
    scheduler_job_id: str,
    output_path: Path,
) -> tuple[dict[str, object], str]:
    """Authenticate all 128 members and publish a no-replace descriptive summary."""

    paths = tuple(
        Path(path)
        for path in (
            shard_root,
            fixture_path,
            gate_path,
            native_boundary_path,
            output_path,
        )
    )
    if any(not path.is_absolute() for path in paths):
        raise ValueError("Task 7 native admission paths must all be absolute")
    shard_root, fixture_path, gate_path, native_boundary_path, output_path = tuple(
        Path(os.path.abspath(path)) for path in paths
    )
    if output_path == shard_root or output_path.is_relative_to(shard_root):
        raise ValueError("Task 7 native admission must remain outside the shard root")
    if (
        len(runtime_commit) != 40
        or any(character not in "0123456789abcdef" for character in runtime_commit)
        or not scheduler_job_id
    ):
        raise ValueError("Task 7 native execution identity is invalid")

    _authenticated_file(fixture_path, fixture_sha256, "Task 7 fixture")
    gate_content = _authenticated_file(gate_path, gate_sha256, "Task 7 gate manifest")
    qids = _sealed_qids(
        gate_content,
        fixture_path=fixture_path,
        fixture_sha256=fixture_sha256,
    )
    fixture = load_fixed_page_fixture(
        fixture_path,
        expected_sha256=fixture_sha256,
        validate_external_bytes=False,
    )
    samples = fixture.selected_samples(qids)
    sample_by_qid = {sample.question_id: sample for sample in samples}
    boundary_manifest = load_task7_native_boundary_manifest(
        native_boundary_path,
        expected_sha256=native_boundary_sha256,
    )
    if boundary_manifest.qids != qids or boundary_manifest.fixture_sha256 != fixture_sha256:
        raise ValueError("Task 7 native-boundary manifest differs from the sealed gate")

    _require_exact_native_shard_tree(shard_root)
    members: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    shared_run_identity: dict[str, object] | None = None
    for index, (qid, entry) in enumerate(zip(qids, boundary_manifest.entries, strict=True)):
        shard = shard_root / f"shard-{index:04d}"
        contents = {
            name: _read_regular_file_bytes(shard / name, f"Task 7 native {name}")
            for name in _SHARD_MEMBERS
        }
        manifest = _json_mapping(contents["run_manifest.json"], "Task 7 native run manifest")
        cell = task7_native_boundary_cell(entry)
        expected_cell = {"cell": 0, **cell.to_dict()}
        _validate_run_manifest(
            manifest,
            shard_root=shard_root,
            shard_index=index,
            qid=qid,
            fixture_path=fixture_path,
            fixture_sha256=fixture_sha256,
            gate_path=gate_path,
            gate_sha256=gate_sha256,
            native_boundary_path=native_boundary_path,
            native_boundary_sha256=native_boundary_sha256,
            runtime_commit=runtime_commit,
            expected_cell=expected_cell,
            feature_manifest_path=fixture.feature_manifest_path,
            feature_manifest_sha256=fixture.feature_manifest_sha256,
        )
        current_shared_identity = {
            key: manifest[key]
            for key in (
                "feature_manifest_path",
                "feature_manifest_sha256",
                "feature_build_source_order_sha256",
                "feature_build_runtime_commit",
                "m3docrag_commit",
                "resources",
                "generation",
            )
        }
        if shared_run_identity is None:
            shared_run_identity = current_shared_identity
        elif current_shared_identity != shared_run_identity:
            raise ValueError("Task 7 native run identity differs across shards")
        record = _one_jsonl_mapping(contents["results.jsonl"])
        expected_policy = None if cell.ctp_policy is None else cell.ctp_policy.to_dict()
        _validate_result_record(
            record,
            line_number=1,
            expected_page_count=4,
            production=True,
            expected_policy=expected_policy,
            forbid_policy=expected_policy is None,
            allowed_extra_fields={
                "matrix_cell",
                "matrix_kind",
                "intervention_name",
                "analysis_family",
                "boundary_source",
                "native_boundary_manifest_sha256",
                "native_crossing",
                "native_layer",
            },
            allow_zero_post_ctp=True,
        )
        validate_task7_source_identity(record, sample_by_qid[qid], fixture.question(qid))
        validate_task7_native_boundary_result_record(
            record,
            cell,
            fixture_sha256=fixture_sha256,
            native_boundary_manifest_sha256=native_boundary_sha256,
        )
        records.append(record)
        members.append(
            {
                "shard": index,
                "qid": qid,
                "files": {
                    name: hashlib.sha256(content).hexdigest()
                    for name, content in contents.items()
                },
            }
        )
    _require_exact_native_shard_tree(shard_root)

    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "admitted-task7-native-boundary-diagnostic",
        "interpretation": "descriptive-native-boundary-diagnostic-not-information-horizon",
        "shard_root": str(shard_root),
        "scheduler_job_id": scheduler_job_id,
        "runtime_commit": runtime_commit,
        "fixture_path": str(fixture_path),
        "fixture_sha256": fixture_sha256,
        "gate_manifest_path": str(gate_path),
        "gate_manifest_sha256": gate_sha256,
        "native_boundary_manifest_path": str(native_boundary_path),
        "native_boundary_manifest_sha256": native_boundary_sha256,
        "fixed_page_provenance": True,
        "global_index_loaded": False,
        "shared_run_identity": shared_run_identity,
        "member_file_count": _SHARD_COUNT * len(_SHARD_MEMBERS),
        "members": members,
        "member_digest_sha256": _canonical_sha256(members),
        "summary": _summarize_native_records(records, boundary_manifest.entries),
    }
    payload["analysis_sha256"] = _canonical_sha256(payload)
    content = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    file_sha256 = hashlib.sha256(content).hexdigest()
    _publish_regular_file_noreplace(output_path, content, "Task 7 native admission")
    return payload, file_sha256


__all__ = ["admit_task7_native_boundary_results"]
