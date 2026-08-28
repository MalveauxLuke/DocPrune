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

_SHARD_COUNT = 64
_SHARD_MEMBERS = ("run_manifest.json", "results.jsonl", "task7-likelihood.jsonl")


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
    if not _is_sha256(expected_sha256):
        raise ValueError(f"{label} checksum must be a lowercase SHA-256")
    content = _read_regular_file_bytes(path, label)
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError(f"{label} checksum mismatch")
    return content


def assemble_task7_analysis_bundle_from_artifacts(**kwargs: object) -> dict[str, object]:
    """Lazily enter the runtime validator only when report compilation begins."""

    from docprune.task7_runtime import assemble_task7_analysis_bundle_from_artifacts as assemble

    return assemble(**kwargs)


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


def _require_exact_shard_tree(root: Path) -> None:
    root_fd = _open_directory_nofollow(root)
    try:
        expected = {f"shard-{index:04d}" for index in range(_SHARD_COUNT)}
        if set(os.listdir(root_fd)) != expected:
            raise ValueError("Task 7 shard root has missing or extra entries")
        for name in sorted(expected):
            shard_fd: int | None = None
            try:
                shard_fd = os.open(
                    name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=root_fd,
                )
                if not stat.S_ISDIR(os.fstat(shard_fd).st_mode):
                    raise ValueError("Task 7 shard entry is not a real directory")
                if set(os.listdir(shard_fd)) != set(_SHARD_MEMBERS):
                    raise ValueError("Task 7 shard has missing or extra files")
                for member in _SHARD_MEMBERS:
                    member_fd = os.open(member, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=shard_fd)
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


def _reauthenticate_admitted_members(
    shard_root: Path,
    members: list[dict[str, object]],
) -> None:
    """Require the live shard tree to retain every byte admitted into the report."""

    _require_exact_shard_tree(shard_root)
    if len(members) != _SHARD_COUNT:
        raise ValueError("Task 7 admitted member set is incomplete")
    for index, member in enumerate(members):
        expected_shard = shard_root / f"shard-{index:04d}"
        if member.get("shard") != index or member.get("path") != str(expected_shard):
            raise ValueError("Task 7 admitted member path changed after admission")
        files = member.get("files")
        if not isinstance(files, Mapping) or set(files) != set(_SHARD_MEMBERS):
            raise ValueError("Task 7 admitted member schema changed after admission")
        for name in _SHARD_MEMBERS:
            identity = files[name]
            expected_path = expected_shard / name
            if (
                not isinstance(identity, Mapping)
                or set(identity) != {"path", "sha256"}
                or identity.get("path") != str(expected_path)
                or not _is_sha256(identity.get("sha256"))
            ):
                raise ValueError("Task 7 admitted member identity changed after admission")
            content = _read_regular_file_bytes(expected_path, f"Task 7 admitted {name}")
            if hashlib.sha256(content).hexdigest() != identity["sha256"]:
                raise ValueError("Task 7 shard member changed after admission")
    _require_exact_shard_tree(shard_root)


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


def _publish_report(path: Path, report: Mapping[str, object]) -> None:
    payload = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    parent_fd = _open_directory_nofollow(path.parent)
    temporary_name = f".{path.name}.{secrets.token_hex(16)}"
    descriptor: int | None = None
    temporary_created = False
    try:
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
        _renameat2_noreplace(parent_fd, temporary_name, parent_fd, path.name)
        temporary_created = False
        os.fsync(parent_fd)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_created:
            try:
                os.unlink(temporary_name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        os.close(parent_fd)


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
) -> dict[str, object]:
    """Admit 64 shards and publish once, or dry-run the same fresh publication."""

    supplied_paths = tuple(
        Path(path) for path in (shard_root, fixture_path, gate_path, output_path)
    )
    if any(not path.is_absolute() for path in supplied_paths):
        raise ValueError("Task 7 report paths must all be absolute")
    paths = tuple(Path(os.path.abspath(path)) for path in supplied_paths)
    shard_root, fixture_path, gate_path, output_path = paths
    if output_path == shard_root or output_path.is_relative_to(shard_root):
        raise ValueError("Task 7 canonical report must remain outside the sealed shard root")
    _require_fresh_output(output_path)
    _authenticated_file(fixture_path, fixture_sha256, "Task 7 fixture")
    gate_content = _authenticated_file(gate_path, gate_sha256, "Task 7 gate manifest")
    qids = _sealed_qids(
        gate_content,
        fixture_path=fixture_path,
        fixture_sha256=fixture_sha256,
    )
    _require_exact_shard_tree(shard_root)
    if (expected_members_path is None) != (expected_members_sha256 is None):
        raise ValueError("Task 7 expected-member path and checksum must be supplied together")
    expected_members: tuple[dict[str, str], ...] | None = None
    member_authority: dict[str, object] = {"kind": "observed-and-reauthenticated"}
    if expected_members_path is not None and expected_members_sha256 is not None:
        expected_members_path = Path(expected_members_path)
        if not expected_members_path.is_absolute():
            raise ValueError("Task 7 expected-member manifest path must be absolute")
        expected_members_path = Path(os.path.abspath(expected_members_path))
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
        member_authority = {
            "kind": "explicit",
            "path": str(expected_members_path),
            "file_sha256": expected_members_sha256,
            "member_manifest_sha256": expected_payload["member_manifest_sha256"],
        }

    bundles: list[dict[str, object]] = []
    members: list[dict[str, object]] = []
    for index, qid in enumerate(qids):
        shard = shard_root / f"shard-{index:04d}"
        manifest_path = shard / "run_manifest.json"
        results_path = shard / "results.jsonl"
        likelihood_path = shard / "task7-likelihood.jsonl"
        manifest_content = _read_regular_file_bytes(manifest_path, "Task 7 run manifest")
        results_content = _read_regular_file_bytes(results_path, "Task 7 results")
        likelihood_content = _read_regular_file_bytes(likelihood_path, "Task 7 likelihoods")
        file_hashes = {
            "run_manifest.json": hashlib.sha256(manifest_content).hexdigest(),
            "results.jsonl": hashlib.sha256(results_content).hexdigest(),
            "task7-likelihood.jsonl": hashlib.sha256(likelihood_content).hexdigest(),
        }
        if expected_members is not None and file_hashes != expected_members[index]:
            raise ValueError("Task 7 shard differs from its explicit member hash authority")
        manifest = _json_mapping(manifest_content, "Task 7 run manifest")
        if (
            manifest.get("shard") != index
            or manifest.get("qid") != qid
            or manifest.get("gate_manifest_path") != str(gate_path)
            or manifest.get("gate_manifest_sha256") != gate_sha256
        ):
            raise ValueError("Task 7 run manifest differs from the sealed shard authority")
        bundle = assemble_task7_analysis_bundle_from_artifacts(
            fixture_path=fixture_path,
            fixture_sha256=fixture_sha256,
            run_manifest_path=manifest_path,
            run_manifest_file_sha256=file_hashes["run_manifest.json"],
            results_path=results_path,
            results_sha256=file_hashes["results.jsonl"],
            likelihood_path=likelihood_path,
            likelihood_sha256=file_hashes["task7-likelihood.jsonl"],
        )
        if bundle.get("qid") != qid:
            raise ValueError("Task 7 admitted bundle QID differs from sealed order")
        bundles.append(bundle)
        members.append(
            {
                "shard": index,
                "qid": qid,
                "path": str(shard),
                "files": {
                    name: {"path": str(shard / name), "sha256": file_hashes[name]}
                    for name in _SHARD_MEMBERS
                },
                "analysis_bundle_sha256": bundle["analysis_bundle_sha256"],
            }
        )

    _reauthenticate_admitted_members(shard_root, members)
    analysis = compile_task7_visual_state_report(bundles, draws=draws, seed=seed)
    report: dict[str, object] = {
        "schema_version": 1,
        "status": "validated-task7-explicit-visual-state-report",
        "fixed_page_provenance": True,
        "global_index_loaded": False,
        "fixture": {"path": str(fixture_path), "sha256": fixture_sha256},
        "gate": {"path": str(gate_path), "sha256": gate_sha256},
        "shard_root": str(shard_root),
        "member_count": len(members),
        "qids": list(qids),
        "members": members,
        "members_sha256": _canonical_sha256(members),
        "member_hash_authority": member_authority,
        "analysis": analysis,
    }
    report["canonical_report_sha256"] = _canonical_sha256(report)
    _reauthenticate_admitted_members(shard_root, members)
    if not validate_only:
        _publish_report(output_path, report)
    return report
