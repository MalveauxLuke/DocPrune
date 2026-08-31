"""Artifact-only admission and publication for the Task 7 fixed-grid report."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
from collections.abc import Mapping
from pathlib import Path

from docprune.experiment_design import (
    _open_directory_nofollow,
    _renameat2_noreplace,
    compile_task7_visual_state_report,
)
from docprune.task7_artifact_analysis import assemble_task7_analysis_bundle_from_artifacts

_SHARD_COUNT = 64
_SHARD_MEMBERS = ("run_manifest.json", "results.jsonl", "task7-likelihood.jsonl")


def _scheduler_log_names(scheduler_job_id: str | None) -> set[str]:
    if scheduler_job_id is None:
        return set()
    if not scheduler_job_id or any(character not in "0123456789" for character in scheduler_job_id):
        raise ValueError("Task 7 scheduler job ID must contain only decimal digits")
    return {
        f"slurm-docprune-task6-l40s-{scheduler_job_id}_{index}.{suffix}"
        for index in range(_SHARD_COUNT)
        for suffix in ("out", "err")
    }


def _require_bootstrap_manifest(shard_fd: int, label: str) -> None:
    bootstrap_fd: int | None = None
    try:
        bootstrap_fd = os.open(
            "bootstrap",
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=shard_fd,
        )
        if (
            not stat.S_ISDIR(os.fstat(bootstrap_fd).st_mode)
            or set(os.listdir(bootstrap_fd)) != {"run_manifest.json"}
        ):
            raise ValueError(f"{label} bootstrap namespace is invalid")
        manifest_fd = os.open(
            "run_manifest.json",
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=bootstrap_fd,
        )
        try:
            if not stat.S_ISREG(os.fstat(manifest_fd).st_mode):
                raise ValueError(f"{label} bootstrap manifest is not a regular file")
        finally:
            os.close(manifest_fd)
    except OSError as error:
        raise ValueError(f"{label} bootstrap contains a substituted path") from error
    finally:
        if bootstrap_fd is not None:
            os.close(bootstrap_fd)


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


def _json_mapping(content: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(content.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return dict(value)


def _sealed_qids(
    gate_content: bytes,
    *,
    fixture_path: Path,
    fixture_sha256: str,
) -> tuple[str, ...]:
    gate = _json_mapping(gate_content, "Task 7 gate manifest")
    if (
        gate.get("schema_version") != 1
        or gate.get("status") != "sealed-development-gate"
        or gate.get("fixture_path") != str(fixture_path)
        or gate.get("fixture_sha256") != fixture_sha256
        or gate.get("fixed_page_provenance") is not True
        or gate.get("global_index_loaded") is not False
    ):
        raise ValueError("Task 7 gate does not bind the fixed-page authority")
    raw_shards = gate.get("qid_shards")
    if not isinstance(raw_shards, list) or len(raw_shards) != _SHARD_COUNT:
        raise ValueError("Task 7 gate must contain the exact 64-QID shard map")
    qids: list[str] = []
    for index, raw_shard in enumerate(raw_shards):
        if not isinstance(raw_shard, Mapping) or set(raw_shard) != {"shard", "qid"}:
            raise ValueError("Task 7 gate shard schema is invalid")
        qid = raw_shard.get("qid")
        if raw_shard.get("shard") != index or not isinstance(qid, str) or not qid:
            raise ValueError("Task 7 gate shard order or QID is invalid")
        qids.append(qid)
    if len(set(qids)) != _SHARD_COUNT:
        raise ValueError("Task 7 sealed QIDs must be unique")
    return tuple(qids)


def _require_exact_shard_tree(root: Path, *, scheduler_job_id: str | None = None) -> None:
    root_fd = _open_directory_nofollow(root)
    try:
        expected_shards = {f"shard-{index:04d}" for index in range(_SHARD_COUNT)}
        expected_logs = _scheduler_log_names(scheduler_job_id)
        if set(os.listdir(root_fd)) != expected_shards | expected_logs:
            raise ValueError("Task 7 shard root has missing or extra entries")
        for name in sorted(expected_logs):
            try:
                log_fd = os.open(
                    name,
                    os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                    dir_fd=root_fd,
                )
                try:
                    if not stat.S_ISREG(os.fstat(log_fd).st_mode):
                        raise ValueError("Task 7 scheduler log is not a regular file")
                finally:
                    os.close(log_fd)
            except OSError as error:
                raise ValueError("Task 7 scheduler log contains a substituted path") from error
        for name in sorted(expected_shards):
            shard_fd: int | None = None
            try:
                shard_fd = os.open(
                    name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=root_fd,
                )
                if not stat.S_ISDIR(os.fstat(shard_fd).st_mode):
                    raise ValueError("Task 7 shard entry is not a real directory")
                expected_members = set(_SHARD_MEMBERS) | (
                    {"bootstrap"} if scheduler_job_id is not None else set()
                )
                if set(os.listdir(shard_fd)) != expected_members:
                    raise ValueError("Task 7 shard has missing or extra files")
                if scheduler_job_id is not None:
                    _require_bootstrap_manifest(shard_fd, "Task 7 shard")
                for member in _SHARD_MEMBERS:
                    member_fd = os.open(
                        member,
                        os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                        dir_fd=shard_fd,
                    )
                    try:
                        if not stat.S_ISREG(os.fstat(member_fd).st_mode):
                            raise ValueError("Task 7 shard member is not a regular file")
                    finally:
                        os.close(member_fd)
            except OSError as error:
                raise ValueError("Task 7 shard tree contains a substituted path") from error
            finally:
                if shard_fd is not None:
                    os.close(shard_fd)
    finally:
        os.close(root_fd)


def _require_fresh_output(path: Path) -> None:
    parent_fd = _open_directory_nofollow(path.parent)
    try:
        try:
            os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return
        raise FileExistsError(f"Task 7 canonical report already exists: {path}")
    finally:
        os.close(parent_fd)


def _explicit_member_hashes(
    content: bytes,
    *,
    shard_root: Path,
    fixture_sha256: str,
    gate_sha256: str,
    qids: tuple[str, ...],
) -> tuple[dict[str, str], ...]:
    authority = _json_mapping(content, "Task 7 member-hash authority")
    digest = authority.pop("member_manifest_sha256", None)
    if not _is_sha256(digest) or digest != _canonical_sha256(authority):
        raise ValueError("Task 7 member-hash authority identity is invalid")
    if (
        set(authority)
        != {
            "schema_version",
            "status",
            "fixture_sha256",
            "gate_sha256",
            "shard_root",
            "qids",
            "members",
        }
        or authority.get("schema_version") != 1
        or authority.get("status") != "sealed-task7-member-hashes"
        or authority.get("fixture_sha256") != fixture_sha256
        or authority.get("gate_sha256") != gate_sha256
        or authority.get("shard_root") != str(shard_root)
        or authority.get("qids") != list(qids)
    ):
        raise ValueError("Task 7 member-hash authority differs from the sealed inputs")
    raw_members = authority.get("members")
    if not isinstance(raw_members, list) or len(raw_members) != _SHARD_COUNT:
        raise ValueError("Task 7 member-hash authority must cover every shard")
    expected: list[dict[str, str]] = []
    for index, (qid, raw_member) in enumerate(zip(qids, raw_members, strict=True)):
        if (
            not isinstance(raw_member, Mapping)
            or set(raw_member) != {"shard", "qid", "files"}
            or raw_member.get("shard") != index
            or raw_member.get("qid") != qid
        ):
            raise ValueError("Task 7 member-hash authority order is invalid")
        raw_files = raw_member.get("files")
        if not isinstance(raw_files, Mapping) or set(raw_files) != set(_SHARD_MEMBERS):
            raise ValueError("Task 7 member-hash authority file schema is invalid")
        hashes = dict(raw_files)
        if any(not _is_sha256(value) for value in hashes.values()):
            raise ValueError("Task 7 member-hash authority contains an invalid checksum")
        expected.append(hashes)
    return tuple(expected)


def _write_snapshot_file(directory_fd: int, name: str, content: bytes) -> None:
    descriptor = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
        dir_fd=directory_fd,
    )
    try:
        remaining = memoryview(content)
        while remaining:
            written = os.write(descriptor, remaining)
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_snapshot_file(directory_fd: int, name: str, label: str) -> bytes:
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
        dir_fd=directory_fd,
    )
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"{label} snapshot member is not a regular file")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _directory_identity(directory_fd: int) -> tuple[int, int]:
    metadata = os.fstat(directory_fd)
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("Task 7 snapshot identity is not a directory")
    return metadata.st_dev, metadata.st_ino


def _open_verified_directory(
    parent_fd: int,
    name: str,
    expected_identity: tuple[int, int],
    label: str,
) -> int:
    try:
        directory_fd = os.open(
            name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
    except OSError as error:
        raise RuntimeError(f"{label} identity is unavailable; snapshot preserved") from error
    if _directory_identity(directory_fd) != expected_identity:
        os.close(directory_fd)
        raise RuntimeError(f"{label} identity changed; snapshot preserved")
    return directory_fd


def _authenticate_snapshot_tree(
    staging_fd: int,
    *,
    input_hashes: Mapping[str, str],
    member_hashes: tuple[dict[str, str], ...],
    report_sha256: str,
) -> None:
    """Re-read the exact publication tree through directory descriptors."""

    if set(os.listdir(staging_fd)) != {"inputs", "shards", "report.json"}:
        raise ValueError("Task 7 snapshot root schema changed")
    if (
        hashlib.sha256(_read_snapshot_file(staging_fd, "report.json", "Task 7 report")).hexdigest()
        != report_sha256
    ):
        raise ValueError("Task 7 report snapshot checksum mismatch")

    inputs_fd = os.open(
        "inputs",
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        dir_fd=staging_fd,
    )
    try:
        if set(os.listdir(inputs_fd)) != set(input_hashes):
            raise ValueError("Task 7 input snapshot schema changed")
        for name, expected_sha256 in input_hashes.items():
            content = _read_snapshot_file(inputs_fd, name, f"Task 7 {name}")
            if hashlib.sha256(content).hexdigest() != expected_sha256:
                raise ValueError(f"Task 7 {name} snapshot checksum mismatch")
    finally:
        os.close(inputs_fd)

    shards_fd = os.open(
        "shards",
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        dir_fd=staging_fd,
    )
    try:
        expected_shards = {f"shard-{index:04d}" for index in range(_SHARD_COUNT)}
        if set(os.listdir(shards_fd)) != expected_shards:
            raise ValueError("Task 7 shard snapshot schema changed")
        for index, expected_files in enumerate(member_hashes):
            shard_name = f"shard-{index:04d}"
            shard_fd = os.open(
                shard_name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=shards_fd,
            )
            try:
                if set(os.listdir(shard_fd)) != set(expected_files):
                    raise ValueError("Task 7 shard snapshot member schema changed")
                for name, expected_sha256 in expected_files.items():
                    content = _read_snapshot_file(shard_fd, name, f"Task 7 {name}")
                    if hashlib.sha256(content).hexdigest() != expected_sha256:
                        raise ValueError(f"Task 7 {name} snapshot checksum mismatch")
            finally:
                os.close(shard_fd)
    finally:
        os.close(shards_fd)


def _authenticate_named_snapshot(
    parent_fd: int,
    name: str,
    expected_identity: tuple[int, int],
    *,
    input_hashes: Mapping[str, str],
    member_hashes: tuple[dict[str, str], ...],
    report_sha256: str,
) -> None:
    published_fd = _open_verified_directory(
        parent_fd,
        name,
        expected_identity,
        "Task 7 published bundle",
    )
    try:
        _authenticate_snapshot_tree(
            published_fd,
            input_hashes=input_hashes,
            member_hashes=member_hashes,
            report_sha256=report_sha256,
        )
    finally:
        os.close(published_fd)
    verification_fd = _open_verified_directory(
        parent_fd,
        name,
        expected_identity,
        "Task 7 published bundle",
    )
    os.close(verification_fd)


def _require_directory_locator_identity(path: Path, expected_identity: tuple[int, int]) -> None:
    descriptor = _open_directory_nofollow(path)
    try:
        if _directory_identity(descriptor) != expected_identity:
            raise RuntimeError("Task 7 publication parent identity changed; destination preserved")
    finally:
        os.close(descriptor)


def _remove_verified_tree_contents(directory_fd: int) -> None:
    """Delete only entries reached below an already authenticated directory fd."""

    for entry in os.listdir(directory_fd):
        metadata = os.stat(entry, dir_fd=directory_fd, follow_symlinks=False)
        identity = metadata.st_dev, metadata.st_ino
        if stat.S_ISDIR(metadata.st_mode):
            child_fd = os.open(
                entry,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=directory_fd,
            )
            try:
                if _directory_identity(child_fd) != identity:
                    raise RuntimeError("Task 7 snapshot child identity changed; snapshot preserved")
                _remove_verified_tree_contents(child_fd)
            finally:
                os.close(child_fd)
            current = os.stat(entry, dir_fd=directory_fd, follow_symlinks=False)
            if (current.st_dev, current.st_ino) != identity:
                raise RuntimeError("Task 7 snapshot child identity changed; snapshot preserved")
            os.rmdir(entry, dir_fd=directory_fd)
        else:
            current = os.stat(entry, dir_fd=directory_fd, follow_symlinks=False)
            if (current.st_dev, current.st_ino) != identity:
                raise RuntimeError("Task 7 snapshot member identity changed; snapshot preserved")
            os.unlink(entry, dir_fd=directory_fd)


def _discard_verified_snapshot(
    parent_fd: int,
    name: str,
    expected_identity: tuple[int, int],
) -> None:
    """Discard a private snapshot only while its parent locator retains identity."""

    directory_fd = _open_verified_directory(
        parent_fd,
        name,
        expected_identity,
        "Task 7 private staging locator",
    )
    try:
        _remove_verified_tree_contents(directory_fd)
    finally:
        os.close(directory_fd)
    verification_fd = _open_verified_directory(
        parent_fd,
        name,
        expected_identity,
        "Task 7 private staging locator",
    )
    os.close(verification_fd)
    os.rmdir(name, dir_fd=parent_fd)


def _eligible_questions_identity(fixture_content: bytes) -> tuple[Path, str]:
    fixture = _json_mapping(fixture_content, "Task 7 fixture")
    raw_path = fixture.get("eligible_questions_path")
    digest = fixture.get("eligible_questions_sha256")
    if not isinstance(raw_path, str) or not Path(raw_path).is_absolute() or not _is_sha256(digest):
        raise ValueError("Task 7 fixture has invalid eligible-question authority")
    return Path(raw_path), digest


def _publish_regular_file_noreplace(path: Path, content: bytes, label: str) -> None:
    parent_fd = _open_directory_nofollow(path.parent)
    parent_identity = _directory_identity(parent_fd)
    temporary_name = f".{path.name}.{secrets.token_hex(16)}"
    descriptor: int | None = None
    expected_identity: tuple[int, int] | None = None
    try:
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=parent_fd,
        )
        metadata = os.fstat(descriptor)
        expected_identity = metadata.st_dev, metadata.st_ino
        remaining = memoryview(content)
        while remaining:
            written = os.write(descriptor, remaining)
            remaining = remaining[written:]
        os.fsync(descriptor)
        current = os.stat(temporary_name, dir_fd=parent_fd, follow_symlinks=False)
        if (current.st_dev, current.st_ino) != expected_identity:
            raise RuntimeError(f"{label} staging identity changed; file preserved")
        _renameat2_noreplace(parent_fd, temporary_name, parent_fd, path.name)
        try:
            for fsync_parent in (False, True):
                published_fd = os.open(
                    path.name,
                    os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                    dir_fd=parent_fd,
                )
                try:
                    published = os.fstat(published_fd)
                    if (
                        not stat.S_ISREG(published.st_mode)
                        or (published.st_dev, published.st_ino) != expected_identity
                    ):
                        raise RuntimeError(f"{label} published identity changed")
                    chunks: list[bytes] = []
                    while chunk := os.read(published_fd, 1024 * 1024):
                        chunks.append(chunk)
                    if b"".join(chunks) != content:
                        raise RuntimeError(f"{label} published bytes changed")
                finally:
                    os.close(published_fd)
                verification_fd = os.open(
                    path.name,
                    os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                    dir_fd=parent_fd,
                )
                try:
                    verification = os.fstat(verification_fd)
                    if (
                        not stat.S_ISREG(verification.st_mode)
                        or (verification.st_dev, verification.st_ino) != expected_identity
                    ):
                        raise RuntimeError(f"{label} published identity changed")
                finally:
                    os.close(verification_fd)
                if fsync_parent:
                    _require_directory_locator_identity(path.parent, parent_identity)
                else:
                    os.fsync(parent_fd)
        except (OSError, RuntimeError, ValueError) as error:
            raise RuntimeError(f"{label} post-publication gate failed; file preserved") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)


def seal_task7_member_hash_authority(
    *,
    shard_root: Path,
    fixture_path: Path,
    fixture_sha256: str,
    gate_path: Path,
    gate_sha256: str,
    output_path: Path,
    scheduler_job_id: str | None = None,
) -> tuple[dict[str, object], str]:
    """Outcome-blindly pin the exact 192 Task 7 shard-member byte streams."""

    supplied_paths = tuple(
        Path(path) for path in (shard_root, fixture_path, gate_path, output_path)
    )
    if any(not path.is_absolute() for path in supplied_paths):
        raise ValueError("Task 7 member-authority paths must all be absolute")
    shard_root, fixture_path, gate_path, output_path = tuple(
        Path(os.path.abspath(path)) for path in supplied_paths
    )
    if output_path == shard_root or output_path.is_relative_to(shard_root):
        raise ValueError("Task 7 member authority must remain outside the sealed shard root")
    _authenticated_file(fixture_path, fixture_sha256, "Task 7 fixture")
    gate_content = _authenticated_file(gate_path, gate_sha256, "Task 7 gate manifest")
    qids = _sealed_qids(
        gate_content,
        fixture_path=fixture_path,
        fixture_sha256=fixture_sha256,
    )
    _require_exact_shard_tree(shard_root, scheduler_job_id=scheduler_job_id)
    members: list[dict[str, object]] = []
    for index, qid in enumerate(qids):
        shard = shard_root / f"shard-{index:04d}"
        contents = {
            name: _read_regular_file_bytes(shard / name, f"Task 7 {name}")
            for name in _SHARD_MEMBERS
        }
        manifest = _json_mapping(contents["run_manifest.json"], "Task 7 run manifest")
        if (
            manifest.get("shard") != index
            or manifest.get("qid") != qid
            or manifest.get("gate_manifest_path") != str(gate_path)
            or manifest.get("gate_manifest_sha256") != gate_sha256
        ):
            raise ValueError("Task 7 run manifest differs from the sealed shard authority")
        members.append(
            {
                "shard": index,
                "qid": qid,
                "files": {
                    name: hashlib.sha256(content).hexdigest() for name, content in contents.items()
                },
            }
        )
    _require_exact_shard_tree(shard_root, scheduler_job_id=scheduler_job_id)
    authority: dict[str, object] = {
        "schema_version": 1,
        "status": "sealed-task7-member-hashes",
        "fixture_sha256": fixture_sha256,
        "gate_sha256": gate_sha256,
        "shard_root": str(shard_root),
        "qids": list(qids),
        "members": members,
    }
    authority["member_manifest_sha256"] = _canonical_sha256(authority)
    content = (json.dumps(authority, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    file_sha256 = hashlib.sha256(content).hexdigest()
    _publish_regular_file_noreplace(output_path, content, "Task 7 member authority")
    return authority, file_sha256


def compile_task7_report_from_shards(
    *,
    shard_root: Path,
    fixture_path: Path,
    fixture_sha256: str,
    gate_path: Path,
    gate_sha256: str,
    output_path: Path,
    expected_members_path: Path | None = None,
    expected_members_sha256: str | None = None,
    draws: int = 100_000,
    seed: int = 20_260_827,
    validate_only: bool = False,
    scheduler_job_id: str | None = None,
) -> dict[str, object]:
    """Build an authenticated snapshot and atomically publish it, or discard it."""

    supplied_paths = tuple(
        Path(path) for path in (shard_root, fixture_path, gate_path, output_path)
    )
    if any(not path.is_absolute() for path in supplied_paths):
        raise ValueError("Task 7 report paths must all be absolute")
    paths = tuple(Path(os.path.abspath(path)) for path in supplied_paths)
    shard_root, fixture_path, gate_path, output_path = paths
    if output_path == shard_root or output_path.is_relative_to(shard_root):
        raise ValueError("Task 7 canonical report must remain outside the sealed shard root")
    if expected_members_path is None or expected_members_sha256 is None:
        raise ValueError("Task 7 explicit member authority is required")
    expected_members_path = Path(expected_members_path)
    if not expected_members_path.is_absolute():
        raise ValueError("Task 7 expected-member manifest path must be absolute")
    expected_members_path = Path(os.path.abspath(expected_members_path))
    if not validate_only:
        _require_fresh_output(output_path)

    fixture_content = _authenticated_file(fixture_path, fixture_sha256, "Task 7 fixture")
    gate_content = _authenticated_file(gate_path, gate_sha256, "Task 7 gate manifest")
    qids = _sealed_qids(
        gate_content,
        fixture_path=fixture_path,
        fixture_sha256=fixture_sha256,
    )
    _require_exact_shard_tree(shard_root, scheduler_job_id=scheduler_job_id)
    expected_content = _authenticated_file(
        expected_members_path,
        expected_members_sha256,
        "Task 7 member-hash authority",
    )
    expected_members = _explicit_member_hashes(
        expected_content,
        shard_root=shard_root,
        fixture_sha256=fixture_sha256,
        gate_sha256=gate_sha256,
        qids=qids,
    )
    expected_payload = _json_mapping(expected_content, "Task 7 member-hash authority")
    eligible_historical_path, eligible_sha256 = _eligible_questions_identity(fixture_content)
    eligible_content = _authenticated_file(
        eligible_historical_path,
        eligible_sha256,
        "Task 7 eligible questions",
    )

    parent_fd = _open_directory_nofollow(output_path.parent)
    parent_identity = _directory_identity(parent_fd)
    temporary_name = f".{output_path.name}.{secrets.token_hex(16)}"
    temporary_created = False
    staging_fd: int | None = None
    staging_identity: tuple[int, int] | None = None
    inputs_fd: int | None = None
    shards_fd: int | None = None
    try:
        os.mkdir(temporary_name, mode=0o700, dir_fd=parent_fd)
        temporary_created = True
        staging_fd = os.open(
            temporary_name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
        staging_identity = _directory_identity(staging_fd)
        os.mkdir("inputs", mode=0o700, dir_fd=staging_fd)
        os.mkdir("shards", mode=0o700, dir_fd=staging_fd)
        inputs_fd = os.open(
            "inputs", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=staging_fd
        )
        shards_fd = os.open(
            "shards", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=staging_fd
        )
        snapshot_inputs = {
            "fixture.json": fixture_content,
            "gate.json": gate_content,
            "member-authority.json": expected_content,
            "eligible-questions.jsonl": eligible_content,
        }
        for name, content in snapshot_inputs.items():
            _write_snapshot_file(inputs_fd, name, content)
        os.fsync(inputs_fd)
        snapshot_inputs = {
            name: _read_snapshot_file(inputs_fd, name, f"Task 7 {name}") for name in snapshot_inputs
        }
        fixture_content = snapshot_inputs["fixture.json"]
        gate_content = snapshot_inputs["gate.json"]
        expected_content = snapshot_inputs["member-authority.json"]
        eligible_content = snapshot_inputs["eligible-questions.jsonl"]
        qids = _sealed_qids(
            gate_content,
            fixture_path=fixture_path,
            fixture_sha256=fixture_sha256,
        )
        expected_members = _explicit_member_hashes(
            expected_content,
            shard_root=shard_root,
            fixture_sha256=fixture_sha256,
            gate_sha256=gate_sha256,
            qids=qids,
        )
        expected_payload = _json_mapping(expected_content, "Task 7 member-hash authority")

        bundles: list[dict[str, object]] = []
        members: list[dict[str, object]] = []
        for index, qid in enumerate(qids):
            shard_name = f"shard-{index:04d}"
            historical_shard = shard_root / shard_name
            os.mkdir(shard_name, mode=0o700, dir_fd=shards_fd)
            shard_fd = os.open(
                shard_name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=shards_fd,
            )
            try:
                contents = {
                    name: _read_regular_file_bytes(historical_shard / name, f"Task 7 {name}")
                    for name in _SHARD_MEMBERS
                }
                file_hashes = {
                    name: hashlib.sha256(content).hexdigest() for name, content in contents.items()
                }
                if file_hashes != expected_members[index]:
                    raise ValueError("Task 7 shard differs from its explicit member hash authority")
                manifest = _json_mapping(contents["run_manifest.json"], "Task 7 run manifest")
                if (
                    manifest.get("shard") != index
                    or manifest.get("qid") != qid
                    or manifest.get("gate_manifest_path") != str(gate_path)
                    or manifest.get("gate_manifest_sha256") != gate_sha256
                ):
                    raise ValueError("Task 7 run manifest differs from the sealed shard authority")
                for name, content in contents.items():
                    _write_snapshot_file(shard_fd, name, content)
                os.fsync(shard_fd)
                staged_contents = {
                    name: _read_snapshot_file(shard_fd, name, f"Task 7 {name}")
                    for name in _SHARD_MEMBERS
                }
            finally:
                os.close(shard_fd)

            historical_manifest = historical_shard / "run_manifest.json"
            historical_results = historical_shard / "results.jsonl"
            historical_likelihood = historical_shard / "task7-likelihood.jsonl"
            bundle = assemble_task7_analysis_bundle_from_artifacts(
                fixture_path=fixture_path,
                fixture_sha256=fixture_sha256,
                run_manifest_path=historical_manifest,
                run_manifest_file_sha256=file_hashes["run_manifest.json"],
                results_path=historical_results,
                results_sha256=file_hashes["results.jsonl"],
                likelihood_path=historical_likelihood,
                likelihood_sha256=file_hashes["task7-likelihood.jsonl"],
                fixture_content=fixture_content,
                run_manifest_content=staged_contents["run_manifest.json"],
                results_content=staged_contents["results.jsonl"],
                likelihood_content=staged_contents["task7-likelihood.jsonl"],
                eligible_questions_content=eligible_content,
            )
            if bundle.get("qid") != qid:
                raise ValueError("Task 7 admitted bundle QID differs from sealed order")
            bundles.append(bundle)
            members.append(
                {
                    "shard": index,
                    "qid": qid,
                    "path": f"shards/{shard_name}",
                    "historical_locator": str(historical_shard),
                    "files": {
                        name: {
                            "path": f"shards/{shard_name}/{name}",
                            "historical_locator": str(historical_shard / name),
                            "sha256": file_hashes[name],
                        }
                        for name in _SHARD_MEMBERS
                    },
                    "analysis_bundle_sha256": bundle["analysis_bundle_sha256"],
                }
            )
        os.fsync(shards_fd)

        analysis = compile_task7_visual_state_report(bundles, draws=draws, seed=seed)
        report: dict[str, object] = {
            "schema_version": 2,
            "status": "validated-task7-explicit-visual-state-report",
            "snapshot_kind": "atomic-self-contained-task7-report-bundle",
            "locator_semantics": {
                "authoritative": "bundle-relative-paths-with-pinned-sha256",
                "embedded_absolute_paths": "historical-locators-only",
            },
            "fixed_page_provenance": True,
            "global_index_loaded": False,
            "fixture": {
                "path": "inputs/fixture.json",
                "historical_locator": str(fixture_path),
                "sha256": fixture_sha256,
            },
            "gate": {
                "path": "inputs/gate.json",
                "historical_locator": str(gate_path),
                "sha256": gate_sha256,
            },
            "eligible_questions": {
                "path": "inputs/eligible-questions.jsonl",
                "historical_locator": str(eligible_historical_path),
                "sha256": eligible_sha256,
            },
            "historical_shard_root": str(shard_root),
            "member_count": len(members),
            "qids": list(qids),
            "members": members,
            "members_sha256": _canonical_sha256(members),
            "member_hash_authority": {
                "kind": "explicit",
                "path": "inputs/member-authority.json",
                "historical_locator": str(expected_members_path),
                "file_sha256": expected_members_sha256,
                "member_manifest_sha256": expected_payload["member_manifest_sha256"],
            },
            "analysis": analysis,
        }
        report["canonical_report_sha256"] = _canonical_sha256(report)
        report_content = (
            json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
        ).encode("utf-8")
        report_sha256 = hashlib.sha256(report_content).hexdigest()
        _write_snapshot_file(staging_fd, "report.json", report_content)
        os.fsync(staging_fd)
        input_hashes = {
            "fixture.json": fixture_sha256,
            "gate.json": gate_sha256,
            "member-authority.json": expected_members_sha256,
            "eligible-questions.jsonl": eligible_sha256,
        }
        _authenticate_snapshot_tree(
            staging_fd,
            input_hashes=input_hashes,
            member_hashes=expected_members,
            report_sha256=report_sha256,
        )
        verification_fd = _open_verified_directory(
            parent_fd,
            temporary_name,
            staging_identity,
            "Task 7 private staging locator",
        )
        os.close(verification_fd)
        if validate_only:
            return report
        _renameat2_noreplace(parent_fd, temporary_name, parent_fd, output_path.name)
        temporary_created = False
        try:
            _authenticate_named_snapshot(
                parent_fd,
                output_path.name,
                staging_identity,
                input_hashes=input_hashes,
                member_hashes=expected_members,
                report_sha256=report_sha256,
            )
            os.fsync(parent_fd)
            _require_directory_locator_identity(output_path.parent, parent_identity)
            _authenticate_named_snapshot(
                parent_fd,
                output_path.name,
                staging_identity,
                input_hashes=input_hashes,
                member_hashes=expected_members,
                report_sha256=report_sha256,
            )
        except (OSError, RuntimeError, ValueError) as error:
            raise RuntimeError(
                f"Task 7 post-publication gate failed; destination preserved at {output_path}"
            ) from error
        return report
    finally:
        if shards_fd is not None:
            os.close(shards_fd)
        if inputs_fd is not None:
            os.close(inputs_fd)
        try:
            if temporary_created:
                if staging_identity is None:
                    raise RuntimeError(
                        "Task 7 staging identity was not retained; snapshot preserved"
                    )
                _discard_verified_snapshot(parent_fd, temporary_name, staging_identity)
        finally:
            if staging_fd is not None:
                os.close(staging_fd)
            os.close(parent_fd)
