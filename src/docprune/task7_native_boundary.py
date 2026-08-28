"""Sealed native-boundary evidence and the separate Task 7 replay arm.

The publisher reads only fixed-page Task 6 provenance and policy-selection
evidence.  It deliberately omits questions, answers, predictions, timings, and
quality metrics from the boundary manifest.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
import stat
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from docprune.cli import _manifest_digest
from docprune.ctp_policy import CTPPolicy, btp_qtp_no_ctp_policy
from docprune.experiment_design import _open_directory_nofollow, _renameat2_noreplace
from docprune.qwen2vl.decoder import ForcedVisualIntervention
from docprune.task6_runtime import task6_policy_matrix

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on the pinned Python 3.10 runtime
    import tomli as tomllib

_SHA256_LENGTH = 64
_COMMIT_LENGTH = 40
_DECODER_LAYER_COUNT = 28
_SOURCE_CONFIG_RELATIVE_PATH = Path("configs/docprune-m3docvqa.toml")
_SOURCE_LAUNCHER_RELATIVE_PATH = Path("examples/sbatch/35_docprune_task6_l40s_matrix.sbatch")
_MANIFEST_KEYS = {
    "schema_version",
    "status",
    "matrix_root",
    "source_runtime_commit",
    "source_runtime_dir",
    "qids",
    "fixture_path",
    "fixture_sha256",
    "gate_manifest_path",
    "gate_manifest_sha256",
    "feature_manifest_path",
    "feature_manifest_sha256",
    "config_path",
    "config_sha256",
    "source_config_relative_path",
    "source_launcher_path",
    "source_launcher_sha256",
    "source_launcher_relative_path",
    "source_admission_path",
    "source_admission_sha256",
    "source_admission_internal_sha256",
    "source_member_digest_sha256",
    "comprehension_threshold",
    "fixed_page_provenance",
    "global_index_loaded",
    "questions",
}
_QUESTION_KEYS = {
    "qid",
    "source_shard",
    "source_run_manifest_path",
    "source_run_manifest_sha256",
    "source_results_path",
    "source_results_sha256",
    "native_crossing",
    "native_layer",
    "boundary",
    "post_qtp_visual_tokens",
    "source_literal_policy_name",
    "source_aggregate_policy_name",
}


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: object, label: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return str(value)


def _require_commit(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _COMMIT_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase 40-character commit")
    return value


def _read_regular_file_bytes(path: Path, label: str) -> bytes:
    """Capture one regular file through a no-symlink descriptor chain."""

    target = Path(path)
    if not target.is_absolute():
        raise ValueError(f"{label} must be an absolute regular file")
    parent_fd = _open_directory_nofollow(target.parent)
    descriptor: int | None = None
    try:
        try:
            descriptor = os.open(target.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd)
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


def _authenticated_file_bytes(path: Path, expected_sha256: str, label: str) -> tuple[bytes, str]:
    expected = _require_sha256(expected_sha256, f"{label} checksum")
    content = _read_regular_file_bytes(path, label)
    actual = hashlib.sha256(content).hexdigest()
    if actual != expected:
        raise ValueError(f"{label} checksum mismatch")
    return content, actual


def _authenticate_file(path: Path, expected_sha256: str, label: str) -> str:
    return _authenticated_file_bytes(path, expected_sha256, label)[1]


def _require_destination_parent_identity(path: Path, descriptor: int) -> None:
    """Reject replacing the named destination parent after its secure open."""

    opened = os.fstat(descriptor)
    try:
        named = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise ValueError("native-boundary destination parent was replaced") from error
    if (
        not stat.S_ISDIR(named.st_mode)
        or named.st_dev != opened.st_dev
        or named.st_ino != opened.st_ino
    ):
        raise ValueError("native-boundary destination parent was replaced")


def _publish_new_file(content: bytes, destination: Path) -> None:
    """Durably publish one new file relative to one retained parent descriptor."""

    target = Path(destination)
    parent_fd = _open_directory_nofollow(target.parent)
    temporary_name: str | None = None
    temporary_fd: int | None = None
    published = False
    try:
        _require_destination_parent_identity(target.parent, parent_fd)
        try:
            os.stat(target.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise ValueError("native-boundary destination must be a new absolute path")
        for _ in range(128):
            candidate = f".{target.name}.{secrets.token_hex(16)}"
            try:
                temporary_fd = os.open(
                    candidate,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=parent_fd,
                )
            except FileExistsError:
                continue
            temporary_name = candidate
            break
        if temporary_fd is None or temporary_name is None:
            raise FileExistsError("could not create a private native-boundary temporary")
        remaining = memoryview(content)
        while remaining:
            written = os.write(temporary_fd, remaining)
            remaining = remaining[written:]
        os.fsync(temporary_fd)
        os.close(temporary_fd)
        temporary_fd = None
        _require_destination_parent_identity(target.parent, parent_fd)
        _renameat2_noreplace(parent_fd, temporary_name, parent_fd, target.name)
        temporary_name = None
        published = True
        os.fsync(parent_fd)
        _require_destination_parent_identity(target.parent, parent_fd)
    except BaseException:
        if published:
            try:
                os.unlink(target.name, dir_fd=parent_fd)
                os.fsync(parent_fd)
            except FileNotFoundError:
                pass
        if temporary_name is not None:
            try:
                os.unlink(temporary_name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        raise
    finally:
        if temporary_fd is not None:
            os.close(temporary_fd)
        os.close(parent_fd)


def _load_json_bytes(content: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(content.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _native_cells() -> list[dict[str, object]]:
    return [
        {
            "cell": index,
            "policy": cell.policy.to_dict(),
            "experiment_version": (
                "task6-native-v1"
                if cell.policy.family in {"random-top-m", "coverage-top-m"}
                else None
            ),
            "repetition": cell.repetition,
        }
        for index, cell in enumerate(task6_policy_matrix("native"))
    ]


def _ordered_rows(content: bytes, *, qid: str) -> tuple[list[dict[str, object]], str]:
    digest = hashlib.sha256(content).hexdigest()
    rows: list[dict[str, object]] = []
    try:
        for line in content.decode("utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("Task 6 source result row must be a JSON object")
            rows.append(value)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("Task 6 source results are not valid JSONL") from error
    expected_count = len(_native_cells())
    if len(rows) != expected_count or any(
        row.get("question_id") != qid
        or row.get("matrix_kind") != "native"
        or row.get("matrix_cell") != index
        for index, row in enumerate(rows)
    ):
        raise ValueError("Task 6 source results are not the complete ordered native matrix")
    return rows, digest


def _validate_policy_row(
    row: Mapping[str, object],
    *,
    policy_index: int,
    fixture_sha256: str,
) -> tuple[int | None, int]:
    expected_policy = _native_cells()[policy_index]["policy"]
    selection = row.get("policy_selection")
    trace = row.get("trace")
    if (
        row.get("fixed_page_fixture_sha256") != fixture_sha256
        or row.get("fixed_page_provenance") is not True
    ):
        raise ValueError("Task 6 source row fixture identity drifted")
    if row.get("global_index_loaded") is not False:
        raise ValueError("Task 6 source row permits a global index")
    if row.get("experiment_version") is not None or row.get("repetition") is not None:
        raise ValueError("Task 6 native policy row has random-policy context")
    if (
        not isinstance(selection, Mapping)
        or selection.get("policy") != expected_policy
        or not isinstance(trace, Mapping)
    ):
        raise ValueError("Task 6 source row has invalid native policy evidence")
    layer = selection.get("native_layer")
    boundary = selection.get("boundary")
    if layer is None:
        if boundary is not None or trace.get("ctp_layer") is not None:
            raise ValueError("Task 6 source row has an invalid native boundary")
    elif (
        type(layer) is not int
        or not 0 <= layer < _DECODER_LAYER_COUNT
        or boundary != f"B_{layer}"
        or trace.get("ctp_layer") != layer
    ):
        raise ValueError("Task 6 source row has an invalid native boundary")
    population = trace.get("post_qtp_visual_tokens")
    if (
        type(population) is not int
        or population <= 0
        or selection.get("visual_population") != population
    ):
        raise ValueError("Task 6 source row has an invalid visual population")
    return layer, population


def _comprehension_threshold(config_content: bytes) -> float:
    try:
        payload = tomllib.loads(config_content.decode("utf-8"))
        value = payload["paper"]["top4"]["comprehension_threshold"]
    except (UnicodeDecodeError, tomllib.TOMLDecodeError, KeyError, TypeError) as error:
        raise ValueError("config does not define paper.top4.comprehension_threshold") from error
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError("config comprehension threshold must be finite")
    return float(value)


def _authenticate_task6_admission(
    path: Path,
    *,
    expected_sha256: str,
    expected_internal_sha256: str,
    expected_member_digest_sha256: str,
    matrix_root: Path,
    source_runtime_commit: str,
    fixture_sha256: str,
    gate_sha256: str,
    qid_count: int,
) -> tuple[str, str, str, dict[Path, bytes]]:
    """Authenticate the admitted Task 6 analysis and every ordered source member."""

    content, file_digest = _authenticated_file_bytes(
        path, expected_sha256, "Task 6 admission analysis"
    )
    internal_digest = _require_sha256(
        expected_internal_sha256, "Task 6 admission internal checksum"
    )
    member_digest = _require_sha256(expected_member_digest_sha256, "Task 6 admitted member digest")
    payload = _load_json_bytes(content, "Task 6 admission analysis")
    unsigned = dict(payload)
    supplied_internal = unsigned.pop("analysis_sha256", None)
    if supplied_internal != internal_digest or _manifest_digest(unsigned) != internal_digest:
        raise ValueError("Task 6 admission analysis internal checksum mismatch")
    admission = payload.get("admission")
    members = payload.get("members")
    expected_member_count = qid_count * 2
    if (
        payload.get("schema_version") != 1
        or payload.get("status") != "admitted-development-r10-extension-required"
        or payload.get("root") != str(matrix_root)
        or payload.get("runtime_commit") != source_runtime_commit
        or payload.get("fixture_sha256") != fixture_sha256
        or payload.get("gate_sha256") != gate_sha256
        or not isinstance(admission, Mapping)
        or admission.get("fixed_page_provenance") is not True
        or admission.get("global_index_loaded") is not False
        or admission.get("shards") != qid_count
        or admission.get("rows") != qid_count * len(_native_cells())
        or admission.get("member_file_count") != expected_member_count
        or admission.get("member_digest_sha256") != member_digest
        or not isinstance(members, list)
        or len(members) != expected_member_count
    ):
        raise ValueError("Task 6 admission analysis identity is invalid")
    member_contents: dict[Path, bytes] = {}
    for member_index, member in enumerate(members):
        shard_index, file_index = divmod(member_index, 2)
        name = ("run_manifest.json", "results.jsonl")[file_index]
        expected_path = matrix_root / f"shard-{shard_index:04d}" / name
        if (
            not isinstance(member, Mapping)
            or set(member) != {"path", "sha256"}
            or member.get("path") != str(expected_path)
        ):
            raise ValueError("Task 6 admission has invalid ordered member identity")
        expected_member_sha256 = _require_sha256(
            member.get("sha256"), "Task 6 admitted member checksum"
        )
        content = _read_regular_file_bytes(expected_path, "Task 6 admitted member")
        if hashlib.sha256(content).hexdigest() != expected_member_sha256:
            raise ValueError("Task 6 admitted member checksum mismatch")
        member_contents[expected_path] = content
    canonical_members = [(member["path"], member["sha256"]) for member in members]
    computed_member_digest = hashlib.sha256(
        json.dumps(canonical_members, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if computed_member_digest != member_digest:
        raise ValueError("Task 6 admitted member digest mismatch")
    return file_digest, internal_digest, member_digest, member_contents


def _authenticate_source_runtime(
    runtime_dir: Path,
    *,
    source_runtime_commit: str,
    config_path: Path,
    config_sha256: str,
    launcher_path: Path,
    launcher_sha256: str,
) -> tuple[str, str, bytes]:
    """Bind source pins to canonical tracked files in the exact clean Task 6 checkout."""

    root = Path(runtime_dir)
    config = Path(config_path)
    launcher = Path(launcher_path)
    if (
        not root.is_absolute()
        or root.is_symlink()
        or not root.is_dir()
        or config != root / _SOURCE_CONFIG_RELATIVE_PATH
        or launcher != root / _SOURCE_LAUNCHER_RELATIVE_PATH
        or config.resolve() != config
        or launcher.resolve() != launcher
    ):
        raise ValueError("source pins are not canonical files in the admitted runtime")
    try:
        observed_commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        observed_status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        tracked = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "ls-files",
                "--error-unmatch",
                str(_SOURCE_CONFIG_RELATIVE_PATH),
                str(_SOURCE_LAUNCHER_RELATIVE_PATH),
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("source runtime is not the exact clean admitted checkout") from error
    if (
        observed_commit != source_runtime_commit
        or observed_status
        or tracked != [str(_SOURCE_CONFIG_RELATIVE_PATH), str(_SOURCE_LAUNCHER_RELATIVE_PATH)]
    ):
        raise ValueError("source runtime is not the exact clean admitted checkout")
    config_content, config_digest = _authenticated_file_bytes(
        config, config_sha256, "source config"
    )
    launcher_digest = _authenticate_file(launcher, launcher_sha256, "source launcher")
    return config_digest, launcher_digest, config_content


@dataclass(frozen=True, slots=True)
class Task7NativeBoundaryEntry:
    """One question's authenticated native comprehension boundary."""

    qid: str
    native_crossing: bool
    native_layer: int | None
    boundary: str | None
    post_qtp_visual_tokens: int
    source_literal_policy_name: str
    source_aggregate_policy_name: str


@dataclass(frozen=True, slots=True)
class Task7NativeBoundaryManifest:
    """Strictly loaded native-boundary manifest."""

    qids: tuple[str, ...]
    comprehension_threshold: float
    fixture_sha256: str
    source_runtime_commit: str
    source_runtime_dir: str
    source_admission_path: str
    source_admission_sha256: str
    source_admission_internal_sha256: str
    source_member_digest_sha256: str
    source_config_sha256: str
    source_config_relative_path: str
    source_launcher_sha256: str
    source_launcher_relative_path: str
    entries: tuple[Task7NativeBoundaryEntry, ...]

    def question(self, qid: str) -> Task7NativeBoundaryEntry:
        matches = tuple(entry for entry in self.entries if entry.qid == qid)
        if len(matches) != 1:
            raise ValueError(
                "native-boundary manifest does not contain the requested QID exactly once"
            )
        return matches[0]


@dataclass(frozen=True, slots=True)
class Task7NativeBoundaryCell:
    """The separate per-question native-boundary all-drop diagnostic."""

    name: str
    entry: Task7NativeBoundaryEntry
    ctp_policy: CTPPolicy | None
    forced_intervention: ForcedVisualIntervention | None

    def answerer_kwargs(self) -> dict[str, object]:
        return {
            "ctp_policy": self.ctp_policy,
            "forced_intervention": self.forced_intervention,
        }

    def to_dict(self) -> dict[str, object]:
        forced = self.forced_intervention
        return {
            "name": self.name,
            "analysis_family": "native-boundary-diagnostic",
            "boundary_source": "admitted-task6-native-comprehension",
            "native_crossing": self.entry.native_crossing,
            "native_layer": self.entry.native_layer,
            "boundary": self.entry.boundary,
            "post_qtp_visual_tokens": self.entry.post_qtp_visual_tokens,
            "ctp_policy": (None if self.ctp_policy is None else self.ctp_policy.to_dict()),
            "forced_intervention": (
                None
                if forced is None
                else {
                    "boundary": f"B_{forced.boundary}",
                    "mode": forced.mode,
                    "retained_visual_ids": list(forced.retained_visual_ids),
                }
            ),
        }


def task7_native_boundary_cell(entry: Task7NativeBoundaryEntry) -> Task7NativeBoundaryCell:
    """Build the per-question replay, preserving no-crossing as a no-op."""

    if entry.native_crossing:
        if entry.native_layer is None or entry.boundary != f"B_{entry.native_layer}":
            raise ValueError("native-boundary entry has inconsistent crossing evidence")
        return Task7NativeBoundaryCell(
            f"all-visual-drop-native-{entry.boundary}",
            entry,
            None,
            ForcedVisualIntervention(entry.native_layer, "physical_delete", ()),
        )
    if entry.native_layer is not None or entry.boundary is not None:
        raise ValueError("native-boundary no-crossing entry fabricates a layer")
    return Task7NativeBoundaryCell(
        "native-no-crossing-noop",
        entry,
        btp_qtp_no_ctp_policy(),
        None,
    )


def bind_task7_native_boundary_result_evidence(
    record: dict[str, object],
    cell: Task7NativeBoundaryCell,
    *,
    fixture_sha256: str,
    native_boundary_manifest_sha256: str,
) -> None:
    """Attach immutable manifest-derived evidence for the separate replay."""

    fixture_digest = _require_sha256(fixture_sha256, "fixture checksum")
    boundary_digest = _require_sha256(
        native_boundary_manifest_sha256, "native-boundary manifest checksum"
    )
    evidence = {
        "fixed_page_fixture_sha256": fixture_digest,
        "fixed_page_provenance": True,
        "global_index_loaded": False,
        "matrix_kind": "visual-state-native-boundary",
        "matrix_cell": 0,
        "intervention_name": cell.name,
        "analysis_family": "native-boundary-diagnostic",
        "boundary_source": "admitted-task6-native-comprehension",
        "native_boundary_manifest_sha256": boundary_digest,
        "native_crossing": cell.entry.native_crossing,
        "native_layer": cell.entry.native_layer,
    }
    for key, value in evidence.items():
        if key in record and record[key] != value:
            raise ValueError(f"Task 7 native result attempts to replace immutable evidence: {key}")
        record[key] = value


def validate_task7_native_boundary_result_record(
    record: Mapping[str, object],
    cell: Task7NativeBoundaryCell,
    *,
    fixture_sha256: str,
    native_boundary_manifest_sha256: str,
) -> None:
    """Cross-bind one result to its per-question native-boundary replay."""

    fixture_digest = _require_sha256(fixture_sha256, "fixture checksum")
    boundary_digest = _require_sha256(
        native_boundary_manifest_sha256, "native-boundary manifest checksum"
    )
    if (
        record.get("fixed_page_fixture_sha256") != fixture_digest
        or record.get("fixed_page_provenance") is not True
        or record.get("global_index_loaded") is not False
        or record.get("matrix_kind") != "visual-state-native-boundary"
        or record.get("matrix_cell") != 0
        or record.get("intervention_name") != cell.name
        or record.get("analysis_family") != "native-boundary-diagnostic"
        or record.get("boundary_source") != "admitted-task6-native-comprehension"
        or record.get("native_boundary_manifest_sha256") != boundary_digest
        or record.get("native_crossing") is not cell.entry.native_crossing
        or record.get("native_layer") != cell.entry.native_layer
    ):
        raise ValueError("Task 7 native result has invalid manifest-derived identity")
    trace = record.get("trace")
    if not isinstance(trace, Mapping):
        raise ValueError("Task 7 native result has invalid pruning trace")
    population = trace.get("post_qtp_visual_tokens")
    retained = trace.get("post_ctp_visual_tokens")
    if population != cell.entry.post_qtp_visual_tokens or trace.get("ctp_layer") is not None:
        raise ValueError("Task 7 native result has invalid pruning trace")
    if not cell.entry.native_crossing:
        selection = record.get("policy_selection")
        policy = selection.get("policy") if isinstance(selection, Mapping) else None
        if (
            "forced_intervention" in record
            or retained != population
            or not isinstance(policy, Mapping)
            or policy.get("name") != "btp-qtp-no-ctp"
            or policy.get("family") != "no-ctp"
            or policy.get("selection_kind") != "none"
        ):
            raise ValueError("Task 7 native no-crossing replay is not a no-op")
        return
    forced = record.get("forced_intervention")
    if (
        "policy_selection" in record
        or retained != 0
        or not isinstance(forced, Mapping)
        or forced.get("boundary") != cell.entry.boundary
        or forced.get("mode") != "physical_delete"
        or forced.get("selection_kind") != "forced"
        or forced.get("visual_population") != population
        or forced.get("requested_budget") != 0
        or forced.get("achieved_budget") != 0
        or forced.get("retained_visual_ids") != []
    ):
        raise ValueError("Task 7 native result has invalid all-drop identity")
    logical_ids = forced.get("logical_retained_sequence_ids")
    cache_lengths = forced.get("prefill_cache_lengths")
    position_shape = forced.get("retained_mrope_position_shape")
    position_digest = forced.get("retained_mrope_position_sha256")
    if (
        not isinstance(logical_ids, list)
        or not logical_ids
        or logical_ids != sorted(set(logical_ids))
        or any(type(identifier) is not int or identifier < 0 for identifier in logical_ids)
        or max(logical_ids) >= len(logical_ids) + population
        or not isinstance(cache_lengths, list)
        or len(cache_lengths) != _DECODER_LAYER_COUNT
        or position_shape != [3, 1, len(logical_ids)]
        or not _is_sha256(position_digest)
    ):
        raise ValueError("Task 7 result has invalid physical native-boundary evidence")
    compact_length = len(logical_ids)
    full_length = compact_length + population
    layer = cell.entry.native_layer
    if layer is None:
        raise ValueError("Task 7 native crossing is missing its layer")
    expected_cache_lengths = [full_length] * (layer + 1) + [compact_length] * (
        _DECODER_LAYER_COUNT - layer - 1
    )
    if cache_lengths != expected_cache_lengths:
        raise ValueError("Task 7 result has invalid physical native-boundary evidence")


def publish_task7_native_boundary_manifest(
    *,
    matrix_root: Path,
    qids: Sequence[str],
    fixture_path: Path,
    fixture_sha256: str,
    gate_manifest_path: Path,
    gate_manifest_sha256: str,
    feature_manifest_path: Path,
    feature_manifest_sha256: str,
    config_path: Path,
    config_sha256: str,
    source_runtime_dir: Path,
    source_launcher_path: Path,
    source_launcher_sha256: str,
    admission_analysis_path: Path,
    admission_analysis_sha256: str,
    admission_analysis_internal_sha256: str,
    admission_member_digest_sha256: str,
    source_runtime_commit: str,
    destination: Path,
) -> str:
    """Authenticate a complete Task 6 native matrix and seal only its boundaries."""

    root = Path(matrix_root)
    target = Path(destination)
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ValueError("Task 6 matrix root must be an absolute real directory")
    if not target.is_absolute() or not target.name:
        raise ValueError("native-boundary destination must be a new absolute path")
    ordered_qids = tuple(qids)
    if (
        not ordered_qids
        or any(not isinstance(qid, str) or not qid for qid in ordered_qids)
        or len(set(ordered_qids)) != len(ordered_qids)
    ):
        raise ValueError("native-boundary QIDs must be nonempty and unique")
    fixture_digest = _authenticate_file(fixture_path, fixture_sha256, "fixture")
    gate_digest = _authenticate_file(gate_manifest_path, gate_manifest_sha256, "gate manifest")
    feature_digest = _authenticate_file(
        feature_manifest_path, feature_manifest_sha256, "feature manifest"
    )
    runtime_commit = _require_commit(source_runtime_commit, "source runtime")
    config_digest, launcher_digest, config_content = _authenticate_source_runtime(
        source_runtime_dir,
        source_runtime_commit=runtime_commit,
        config_path=config_path,
        config_sha256=config_sha256,
        launcher_path=source_launcher_path,
        launcher_sha256=source_launcher_sha256,
    )
    expected_cells = _native_cells()
    (
        admission_file_digest,
        admission_internal_digest,
        member_digest,
        admitted_member_contents,
    ) = _authenticate_task6_admission(
        admission_analysis_path,
        expected_sha256=admission_analysis_sha256,
        expected_internal_sha256=admission_analysis_internal_sha256,
        expected_member_digest_sha256=admission_member_digest_sha256,
        matrix_root=root,
        source_runtime_commit=runtime_commit,
        fixture_sha256=fixture_digest,
        gate_sha256=gate_digest,
        qid_count=len(ordered_qids),
    )
    questions: list[dict[str, object]] = []
    for shard_index, qid in enumerate(ordered_qids):
        shard = root / f"shard-{shard_index:04d}"
        if shard.is_symlink() or not shard.is_dir():
            raise ValueError("Task 6 source shard order is incomplete")
        manifest_path = shard / "run_manifest.json"
        manifest_content = admitted_member_contents[manifest_path]
        manifest_sha256 = hashlib.sha256(manifest_content).hexdigest()
        manifest = _load_json_bytes(manifest_content, "Task 6 source run manifest")
        unsigned = dict(manifest)
        supplied = unsigned.pop("run_manifest_sha256", None)
        if supplied != _manifest_digest(unsigned):
            raise ValueError("Task 6 source run manifest digest is invalid")
        if manifest.get("runtime_commit") != runtime_commit:
            raise ValueError("Task 6 source runtime identity drifted")
        if (
            manifest.get("schema_version") != 1
            or manifest.get("status") != "configured-task6-matrix"
            or manifest.get("output") != str(shard)
            or manifest.get("matrix_kind") != "native"
            or manifest.get("shard") != shard_index
            or manifest.get("qid") != qid
            or manifest.get("cell_count") != len(expected_cells)
            or manifest.get("cells") != expected_cells
        ):
            raise ValueError("Task 6 source run manifest has invalid ordered matrix identity")
        if (
            manifest.get("fixture_path") != str(fixture_path)
            or manifest.get("fixture_sha256") != fixture_digest
        ):
            raise ValueError("Task 6 source run manifest fixture identity drifted")
        if (
            manifest.get("gate_manifest_path") != str(gate_manifest_path)
            or manifest.get("gate_manifest_sha256") != gate_digest
            or manifest.get("feature_manifest_path") != str(feature_manifest_path)
            or manifest.get("feature_manifest_sha256") != feature_digest
            or manifest.get("fixed_page_provenance") is not True
        ):
            raise ValueError("Task 6 source run manifest provenance drifted")
        if manifest.get("global_index_loaded") is not False:
            raise ValueError("Task 6 source run manifest permits a global index")
        results_path = shard / "results.jsonl"
        rows, results_sha256 = _ordered_rows(admitted_member_contents[results_path], qid=qid)
        literal_layer, literal_population = _validate_policy_row(
            rows[1], policy_index=1, fixture_sha256=fixture_digest
        )
        aggregate_layer, aggregate_population = _validate_policy_row(
            rows[2], policy_index=2, fixture_sha256=fixture_digest
        )
        if literal_layer != aggregate_layer:
            raise ValueError("literal and aggregate native boundary evidence disagree")
        if literal_population != aggregate_population:
            raise ValueError("literal and aggregate native visual populations disagree")
        literal_pages = rows[1].get("retrieved_pages")
        if (
            not isinstance(literal_pages, list)
            or len(literal_pages) != 4
            or rows[2].get("retrieved_pages") != literal_pages
        ):
            raise ValueError("literal and aggregate native fixed-page identity disagree")
        questions.append(
            {
                "qid": qid,
                "source_shard": shard_index,
                "source_run_manifest_path": str(manifest_path),
                "source_run_manifest_sha256": manifest_sha256,
                "source_results_path": str(results_path),
                "source_results_sha256": results_sha256,
                "native_crossing": literal_layer is not None,
                "native_layer": literal_layer,
                "boundary": None if literal_layer is None else f"B_{literal_layer}",
                "post_qtp_visual_tokens": literal_population,
                "source_literal_policy_name": "literal-native-threshold",
                "source_aggregate_policy_name": "aggregate-native-threshold",
            }
        )
    payload = {
        "schema_version": 1,
        "status": "sealed-task7-native-boundaries",
        "matrix_root": str(root),
        "source_runtime_commit": runtime_commit,
        "source_runtime_dir": str(source_runtime_dir),
        "qids": list(ordered_qids),
        "fixture_path": str(fixture_path),
        "fixture_sha256": fixture_digest,
        "gate_manifest_path": str(gate_manifest_path),
        "gate_manifest_sha256": gate_digest,
        "feature_manifest_path": str(feature_manifest_path),
        "feature_manifest_sha256": feature_digest,
        "config_path": str(config_path),
        "config_sha256": config_digest,
        "source_config_relative_path": str(_SOURCE_CONFIG_RELATIVE_PATH),
        "source_launcher_path": str(source_launcher_path),
        "source_launcher_sha256": launcher_digest,
        "source_launcher_relative_path": str(_SOURCE_LAUNCHER_RELATIVE_PATH),
        "source_admission_path": str(admission_analysis_path),
        "source_admission_sha256": admission_file_digest,
        "source_admission_internal_sha256": admission_internal_digest,
        "source_member_digest_sha256": member_digest,
        "comprehension_threshold": _comprehension_threshold(config_content),
        "fixed_page_provenance": True,
        "global_index_loaded": False,
        "questions": questions,
    }
    content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    digest = hashlib.sha256(content).hexdigest()
    _publish_new_file(content, target)
    return digest


def load_task7_native_boundary_manifest(
    path: Path, *, expected_sha256: str
) -> Task7NativeBoundaryManifest:
    """Load one exact sealed manifest, rejecting schema or identity drift."""

    expected = _require_sha256(expected_sha256, "native-boundary manifest checksum")
    content, _ = _authenticated_file_bytes(path, expected, "native-boundary manifest")
    payload = _load_json_bytes(content, "native-boundary manifest")
    if set(payload) != _MANIFEST_KEYS:
        raise ValueError("native-boundary manifest has an invalid exact schema")
    if (
        payload.get("schema_version") != 1
        or payload.get("status") != "sealed-task7-native-boundaries"
        or payload.get("fixed_page_provenance") is not True
        or payload.get("global_index_loaded") is not False
    ):
        raise ValueError("native-boundary manifest identity is invalid")
    qids = payload.get("qids")
    questions = payload.get("questions")
    threshold = payload.get("comprehension_threshold")
    if (
        not isinstance(qids, list)
        or not qids
        or any(not isinstance(qid, str) or not qid for qid in qids)
        or len(set(qids)) != len(qids)
        or not isinstance(questions, list)
        or len(questions) != len(qids)
        or isinstance(threshold, bool)
        or not isinstance(threshold, int | float)
        or not math.isfinite(threshold)
    ):
        raise ValueError("native-boundary manifest ordered contents are invalid")
    _require_sha256(payload.get("fixture_sha256"), "fixture checksum")
    for field in (
        "gate_manifest_sha256",
        "feature_manifest_sha256",
        "config_sha256",
        "source_launcher_sha256",
        "source_admission_sha256",
        "source_admission_internal_sha256",
        "source_member_digest_sha256",
    ):
        _require_sha256(payload.get(field), field)
    runtime_commit = _require_commit(payload.get("source_runtime_commit"), "source runtime")
    matrix_root_value = payload.get("matrix_root")
    if not isinstance(matrix_root_value, str) or not Path(matrix_root_value).is_absolute():
        raise ValueError("native-boundary manifest matrix root is invalid")
    for field in (
        "source_runtime_dir",
        "fixture_path",
        "gate_manifest_path",
        "feature_manifest_path",
        "config_path",
        "source_launcher_path",
        "source_admission_path",
    ):
        value = payload.get(field)
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ValueError("native-boundary manifest authenticated path is invalid")
    source_runtime_dir = Path(str(payload["source_runtime_dir"]))
    if (
        payload.get("source_config_relative_path") != str(_SOURCE_CONFIG_RELATIVE_PATH)
        or payload.get("source_launcher_relative_path") != str(_SOURCE_LAUNCHER_RELATIVE_PATH)
        or Path(str(payload["config_path"])) != source_runtime_dir / _SOURCE_CONFIG_RELATIVE_PATH
        or Path(str(payload["source_launcher_path"]))
        != source_runtime_dir / _SOURCE_LAUNCHER_RELATIVE_PATH
    ):
        raise ValueError("native-boundary manifest source runtime pins are invalid")
    matrix_root = Path(matrix_root_value)
    entries: list[Task7NativeBoundaryEntry] = []
    for index, (qid, question) in enumerate(zip(qids, questions, strict=True)):
        if not isinstance(question, dict) or set(question) != _QUESTION_KEYS:
            raise ValueError("native-boundary question has an invalid exact schema")
        layer = question.get("native_layer")
        crossing = question.get("native_crossing")
        population = question.get("post_qtp_visual_tokens")
        if (
            question.get("qid") != qid
            or question.get("source_shard") != index
            or type(crossing) is not bool
            or type(population) is not int
            or population <= 0
            or question.get("source_literal_policy_name") != "literal-native-threshold"
            or question.get("source_aggregate_policy_name") != "aggregate-native-threshold"
        ):
            raise ValueError("native-boundary question identity is invalid")
        for field in ("source_run_manifest_sha256", "source_results_sha256"):
            _require_sha256(question.get(field), field)
        expected_shard = matrix_root / f"shard-{index:04d}"
        if question.get("source_run_manifest_path") != str(
            expected_shard / "run_manifest.json"
        ) or question.get("source_results_path") != str(expected_shard / "results.jsonl"):
            raise ValueError("native-boundary question source path identity is invalid")
        if crossing:
            if (
                type(layer) is not int
                or not 0 <= layer < _DECODER_LAYER_COUNT
                or question.get("boundary") != f"B_{layer}"
            ):
                raise ValueError("native-boundary question crossing is invalid")
        elif layer is not None or question.get("boundary") is not None:
            raise ValueError("native-boundary question no-crossing state is invalid")
        entries.append(
            Task7NativeBoundaryEntry(
                qid,
                crossing,
                layer,
                question.get("boundary"),
                population,
                str(question["source_literal_policy_name"]),
                str(question["source_aggregate_policy_name"]),
            )
        )
    return Task7NativeBoundaryManifest(
        tuple(qids),
        float(threshold),
        str(payload["fixture_sha256"]),
        runtime_commit,
        str(payload["source_runtime_dir"]),
        str(payload["source_admission_path"]),
        str(payload["source_admission_sha256"]),
        str(payload["source_admission_internal_sha256"]),
        str(payload["source_member_digest_sha256"]),
        str(payload["config_sha256"]),
        str(payload["source_config_relative_path"]),
        str(payload["source_launcher_sha256"]),
        str(payload["source_launcher_relative_path"]),
        tuple(entries),
    )


__all__ = [
    "Task7NativeBoundaryCell",
    "Task7NativeBoundaryEntry",
    "Task7NativeBoundaryManifest",
    "bind_task7_native_boundary_result_evidence",
    "load_task7_native_boundary_manifest",
    "publish_task7_native_boundary_manifest",
    "task7_native_boundary_cell",
    "validate_task7_native_boundary_result_record",
]
