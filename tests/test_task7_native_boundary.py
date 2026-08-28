"""Tests for the separate native-B_l* all-visual-drop replay contract."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

from docprune.cli import _manifest_digest
from docprune.ctp_policy import (
    aggregate_native_threshold_policy,
    literal_native_threshold_policy,
    select_boundary_policy,
)
from docprune.task6_runtime import task6_policy_matrix

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _member_digest(members: list[dict[str, str]]) -> str:
    ordered = [(member["path"], member["sha256"]) for member in members]
    return hashlib.sha256(json.dumps(ordered, separators=(",", ":")).encode("utf-8")).hexdigest()


def _selection(policy_name: str, *, layer: int | None = 14) -> dict[str, object]:
    policy = (
        literal_native_threshold_policy()
        if policy_name == "literal-native-threshold"
        else aggregate_native_threshold_policy()
    )
    if layer is None:
        from docprune.ctp_policy import no_crossing_selection

        selection = no_crossing_selection(policy, 8).to_dict()
        cache_length = 12
    else:
        selection = select_boundary_policy(
            policy,
            literal_scores=(0.8,) * 8,
            aggregate_scores=(0.8,) * 8,
            attention_threshold=0.5,
            boundary=f"B_{layer}",
            native_layer=layer,
        ).to_dict()
        cache_length = 12
    selection.update(
        {
            "geometry_count": 8,
            "geometry_sha256": "9" * 64,
            "prefill_cache_lengths": [cache_length] * 28,
            "retained_mrope_position_shape": [3, 1, cache_length],
            "retained_mrope_position_sha256": "8" * 64,
        }
    )
    return selection


def _source_record(
    qid: str,
    *,
    cell: int,
    policy_name: str,
    fixture_sha256: str,
    layer: int | None = 14,
) -> dict[str, object]:
    return {
        "question_id": qid,
        "question": "Which value is shown?",
        "answers": ["42"],
        "predicted_answer": "outcome-must-not-enter-boundary-manifest",
        "retrieved_pages": [
            {"doc_id": f"doc-{index}", "page_index": index, "score": 1.0 - index / 10}
            for index in range(4)
        ],
        "trace": {
            "original_visual_tokens": 16,
            "post_btp_visual_tokens": 12,
            "post_qtp_visual_tokens": 8,
            "post_ctp_visual_tokens": 8,
            "ctp_layer": layer,
        },
        "timing": {
            "retrieval_seconds": 0.1,
            "page_load_seconds": 0.1,
            "qa_seconds": 0.2,
            "total_sample_seconds": 0.4,
            "encoder_seconds": 0.1,
            "decoder_seconds": 0.1,
            "profiler_enabled": False,
        },
        "policy_selection": _selection(policy_name, layer=layer),
        "fixed_page_fixture_sha256": fixture_sha256,
        "fixed_page_provenance": True,
        "global_index_loaded": False,
        "policy_context": {"experiment_version": None, "repetition": None},
        "matrix_cell": cell,
        "matrix_kind": "native",
        "experiment_version": None,
        "repetition": None,
    }


def _source_tree(tmp_path: Path, *, layer: int | None = 14) -> dict[str, object]:
    root = tmp_path / "source"
    shard = root / "shard-0000"
    shard.mkdir(parents=True)
    fixture = tmp_path / "fixture.json"
    fixture.write_text("sealed fixture\n", encoding="utf-8")
    gate = tmp_path / "gate.json"
    gate.write_text("sealed gate\n", encoding="utf-8")
    feature = tmp_path / "feature.json"
    feature.write_text("sealed features\n", encoding="utf-8")
    runtime = tmp_path / "runtime"
    config = runtime / "configs" / "docprune-m3docvqa.toml"
    config.parent.mkdir(parents=True)
    config.write_text("[paper.top4]\ncomprehension_threshold = 45.0\n", encoding="utf-8")
    launcher = runtime / "examples" / "sbatch" / "35_docprune_task6_l40s_matrix.sbatch"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("#!/usr/bin/env bash\nset -euo pipefail\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(runtime)], check=True)
    subprocess.run(["git", "-C", str(runtime), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(runtime),
            "-c",
            "user.name=Task 7 Test",
            "-c",
            "user.email=task7@example.invalid",
            "commit",
            "-q",
            "-m",
            "sealed source runtime",
        ],
        check=True,
    )
    fixture_sha256 = _sha256(fixture)
    gate_sha256 = _sha256(gate)
    feature_sha256 = _sha256(feature)
    runtime_commit = subprocess.run(
        ["git", "-C", str(runtime), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    qid = "q-1"
    cells = [
        {
            "cell": index,
            "policy": item.policy.to_dict(),
            "experiment_version": (
                "task6-native-v1"
                if item.policy.family in {"random-top-m", "coverage-top-m"}
                else None
            ),
            "repetition": item.repetition,
        }
        for index, item in enumerate(task6_policy_matrix("native"))
    ]
    manifest = {
        "schema_version": 1,
        "status": "configured-task6-matrix",
        "output": str(shard),
        "matrix_kind": "native",
        "shard": 0,
        "qid": qid,
        "cell_count": 45,
        "fixture_path": str(fixture),
        "fixture_sha256": fixture_sha256,
        "gate_manifest_path": str(gate),
        "gate_manifest_sha256": gate_sha256,
        "feature_manifest_path": str(feature),
        "feature_manifest_sha256": feature_sha256,
        "feature_build_source_order_sha256": "2" * 64,
        "fixed_page_provenance": True,
        "global_index_loaded": False,
        "feature_build_runtime_commit": "3" * 40,
        "runtime_commit": runtime_commit,
        "m3docrag_commit": "4" * 40,
        "resources": {
            "qwen": {
                "model": "Qwen/Qwen2-VL-7B-Instruct",
                "revision": "eed13092ef92e448dd6875b2a00151bd3f7db0ac",
            }
        },
        "generation": {
            "do_sample": False,
            "eos_token_ids": [151645, 151643],
            "max_new_tokens": 128,
            "num_beams": 1,
            "prompt": "question: $question\noutput only answer.",
        },
        "cells": cells,
    }
    manifest["run_manifest_sha256"] = _manifest_digest(manifest)
    (shard / "run_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows = [
        {"question_id": qid, "matrix_kind": "native", "matrix_cell": index} for index in range(45)
    ]
    rows[1] = _source_record(
        qid,
        cell=1,
        policy_name="literal-native-threshold",
        fixture_sha256=fixture_sha256,
        layer=layer,
    )
    rows[2] = _source_record(
        qid,
        cell=2,
        policy_name="aggregate-native-threshold",
        fixture_sha256=fixture_sha256,
        layer=layer,
    )
    (shard / "results.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    members = [
        {"path": str(shard / name), "sha256": _sha256(shard / name)}
        for name in ("run_manifest.json", "results.jsonl")
    ]
    member_digest_sha256 = _member_digest(members)
    admission = {
        "schema_version": 1,
        "status": "admitted-development-r10-extension-required",
        "root": str(root),
        "runtime_commit": runtime_commit,
        "fixture_sha256": fixture_sha256,
        "gate_sha256": gate_sha256,
        "admission": {
            "fixed_page_provenance": True,
            "global_index_loaded": False,
            "member_digest_sha256": member_digest_sha256,
            "member_file_count": 2,
            "rows": 45,
            "shards": 1,
        },
        "members": members,
    }
    admission["analysis_sha256"] = _manifest_digest(admission)
    admission_path = tmp_path / "task6-admission.json"
    admission_path.write_text(
        json.dumps(admission, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "root": root,
        "fixture": fixture,
        "fixture_sha256": fixture_sha256,
        "gate": gate,
        "gate_sha256": gate_sha256,
        "feature": feature,
        "feature_sha256": feature_sha256,
        "config": config,
        "config_sha256": _sha256(config),
        "runtime": runtime,
        "launcher": launcher,
        "launcher_sha256": _sha256(launcher),
        "admission": admission_path,
        "admission_sha256": _sha256(admission_path),
        "admission_internal_sha256": admission["analysis_sha256"],
        "member_digest_sha256": member_digest_sha256,
        "runtime_commit": runtime_commit,
        "qid": qid,
    }


def _publish(source: dict[str, object], output: Path) -> str:
    from docprune.task7_native_boundary import publish_task7_native_boundary_manifest

    return publish_task7_native_boundary_manifest(
        matrix_root=source["root"],
        qids=(source["qid"],),
        fixture_path=source["fixture"],
        fixture_sha256=source["fixture_sha256"],
        gate_manifest_path=source["gate"],
        gate_manifest_sha256=source["gate_sha256"],
        feature_manifest_path=source["feature"],
        feature_manifest_sha256=source["feature_sha256"],
        config_path=source["config"],
        config_sha256=source["config_sha256"],
        source_runtime_dir=source["runtime"],
        source_launcher_path=source["launcher"],
        source_launcher_sha256=source["launcher_sha256"],
        admission_analysis_path=source["admission"],
        admission_analysis_sha256=source["admission_sha256"],
        admission_analysis_internal_sha256=source["admission_internal_sha256"],
        admission_member_digest_sha256=source["member_digest_sha256"],
        source_runtime_commit=source["runtime_commit"],
        destination=output,
    )


def _resign_admission_for_semantic_test(source: dict[str, object]) -> None:
    """Let older semantic-drift tests proceed beyond the stronger member-hash gate."""

    admission_path = source["admission"]
    payload = json.loads(admission_path.read_text(encoding="utf-8"))
    for member in payload["members"]:
        member["sha256"] = _sha256(Path(member["path"]))
    member_digest = _member_digest(payload["members"])
    payload["admission"]["member_digest_sha256"] = member_digest
    payload.pop("analysis_sha256")
    payload["analysis_sha256"] = _manifest_digest(payload)
    admission_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    source["admission_sha256"] = _sha256(admission_path)
    source["admission_internal_sha256"] = payload["analysis_sha256"]
    source["member_digest_sha256"] = member_digest


def _replace_after_authenticated_capture(
    monkeypatch: pytest.MonkeyPatch,
    *,
    target: Path,
    replacement: bytes,
    occurrence: int,
) -> None:
    """Replace one path only after its requested descriptor capture completes."""

    import docprune.task7_native_boundary as native

    original = native._read_regular_file_bytes
    observed = 0

    def capture(path: Path, label: str) -> bytes:
        nonlocal observed
        content = original(path, label)
        if Path(path) == target:
            observed += 1
            if observed == occurrence:
                alternate = target.with_name(f".{target.name}.replacement")
                alternate.write_bytes(replacement)
                alternate.replace(target)
        return content

    monkeypatch.setattr(native, "_read_regular_file_bytes", capture)


@pytest.mark.parametrize(
    ("source_name", "occurrence"),
    (("admission", 1), ("config", 1), ("source_manifest", 1), ("source_results", 1)),
)
def test_native_boundary_publisher_parses_the_exact_authenticated_bytes_after_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_name: str,
    occurrence: int,
) -> None:
    """Catch hashing one native input and parsing replacement path bytes."""

    source = _source_tree(tmp_path)
    shard = source["root"] / "shard-0000"
    paths = {
        "admission": source["admission"],
        "config": source["config"],
        "source_manifest": shard / "run_manifest.json",
        "source_results": shard / "results.jsonl",
    }
    target = paths[source_name]
    authenticated_sha256 = _sha256(target)
    _replace_after_authenticated_capture(
        monkeypatch,
        target=target,
        replacement=b"replacement bytes must never be parsed\n",
        occurrence=occurrence,
    )

    output = tmp_path / "native-boundaries.json"
    _publish(source, output)
    payload = json.loads(output.read_text(encoding="utf-8"))

    if source_name == "admission":
        assert payload["source_admission_sha256"] == authenticated_sha256
    elif source_name == "config":
        assert payload["config_sha256"] == authenticated_sha256
        assert payload["comprehension_threshold"] == 45.0
    elif source_name == "source_manifest":
        assert payload["questions"][0]["source_run_manifest_sha256"] == authenticated_sha256
    else:
        assert payload["questions"][0]["source_results_sha256"] == authenticated_sha256


def test_native_boundary_loader_parses_the_exact_authenticated_bytes_after_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch validating one manifest inode and loading a replacement inode."""

    from docprune.task7_native_boundary import load_task7_native_boundary_manifest

    source = _source_tree(tmp_path)
    output = tmp_path / "native-boundaries.json"
    digest = _publish(source, output)
    _replace_after_authenticated_capture(
        monkeypatch,
        target=output,
        replacement=b"replacement bytes must never be parsed\n",
        occurrence=1,
    )

    loaded = load_task7_native_boundary_manifest(output, expected_sha256=digest)

    assert loaded.question("q-1").boundary == "B_14"


def test_native_boundary_authenticated_inputs_reject_symlinked_ancestors(tmp_path: Path) -> None:
    """Catch native inputs being redirected through a non-leaf symlink."""

    from docprune.task7_native_boundary import _authenticate_file

    real = tmp_path / "real" / "nested"
    real.mkdir(parents=True)
    source = real / "source.json"
    source.write_text("sealed bytes\n", encoding="utf-8")
    (tmp_path / "alias").symlink_to(tmp_path / "real", target_is_directory=True)

    with pytest.raises(ValueError, match="symlink|non-directory"):
        _authenticate_file(
            tmp_path / "alias" / "nested" / "source.json",
            _sha256(source),
            "native source",
        )


def test_native_boundary_publisher_rejects_destination_with_symlinked_ancestor(
    tmp_path: Path,
) -> None:
    """Catch publication being redirected through a destination ancestor symlink."""

    source = _source_tree(tmp_path / "inputs")
    real_parent = tmp_path / "real" / "nested"
    real_parent.mkdir(parents=True)
    (tmp_path / "alias").symlink_to(tmp_path / "real", target_is_directory=True)
    destination = tmp_path / "alias" / "nested" / "native-boundaries.json"

    with pytest.raises(ValueError, match="symlink|non-directory"):
        _publish(source, destination)

    assert not destination.exists()
    assert not (real_parent / destination.name).exists()


def test_native_boundary_publisher_rejects_replaced_destination_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch publishing into a retained directory after its named parent is replaced."""

    import docprune.task7_native_boundary as native

    source = _source_tree(tmp_path / "inputs")
    destination_parent = tmp_path / "output"
    destination_parent.mkdir()
    moved_parent = tmp_path / "moved-output"
    destination = destination_parent / "native-boundaries.json"
    original = native._open_directory_nofollow
    replaced = False

    def replace_after_open(path: Path) -> int:
        nonlocal replaced
        descriptor = original(path)
        if Path(path) == destination_parent and not replaced:
            destination_parent.replace(moved_parent)
            destination_parent.mkdir()
            replaced = True
        return descriptor

    monkeypatch.setattr(native, "_open_directory_nofollow", replace_after_open)

    with pytest.raises(ValueError, match="destination parent was replaced"):
        _publish(source, destination)

    assert replaced is True
    assert not destination.exists()
    assert not (moved_parent / destination.name).exists()


def test_native_boundary_cleanup_does_not_delete_replacement_after_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch rollback deleting unrelated bytes swapped into the published name."""

    import docprune.task7_native_boundary as native

    source = _source_tree(tmp_path / "inputs")
    destination = tmp_path / "native-boundaries.json"
    moved_publication = tmp_path / "moved-publication.json"
    unrelated = b"unrelated replacement must survive\n"
    original = native._require_destination_parent_identity
    identity_checks = 0

    def fail_after_name_replacement(path: Path, descriptor: int) -> None:
        nonlocal identity_checks
        original(path, descriptor)
        identity_checks += 1
        if identity_checks == 3:
            destination.replace(moved_publication)
            destination.write_bytes(unrelated)
            raise ValueError("injected post-publication failure")

    monkeypatch.setattr(native, "_require_destination_parent_identity", fail_after_name_replacement)

    with pytest.raises(RuntimeError, match="preserved published destination") as captured:
        _publish(source, destination)

    assert isinstance(captured.value.__cause__, ValueError)
    assert destination.read_bytes() == unrelated
    assert moved_publication.is_file()


def test_native_boundary_post_publish_failure_never_rolls_back_by_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch restoring a stat-then-unlink rollback race after atomic publication."""

    import docprune.task7_native_boundary as native

    source = _source_tree(tmp_path / "inputs")
    destination = tmp_path / "native-boundaries.json"
    original = native._require_destination_parent_identity
    identity_checks = 0

    def fail_after_publish(path: Path, descriptor: int) -> None:
        nonlocal identity_checks
        original(path, descriptor)
        identity_checks += 1
        if identity_checks == 3:
            raise ValueError("injected post-publication failure")

    def forbid_name_unlink(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("post-publication cleanup must never unlink by name")

    monkeypatch.setattr(native, "_require_destination_parent_identity", fail_after_publish)
    monkeypatch.setattr(native.os, "unlink", forbid_name_unlink)

    with pytest.raises(RuntimeError, match="preserved published destination") as captured:
        _publish(source, destination)

    assert str(destination) in str(captured.value)
    assert isinstance(captured.value.__cause__, ValueError)
    assert json.loads(destination.read_text(encoding="utf-8"))["status"] == (
        "sealed-task7-native-boundaries"
    )


def test_native_boundary_post_rename_fsync_failure_reports_preserved_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Report the exact published recovery artifact when durable fsync fails."""

    import docprune.task7_native_boundary as native

    destination = tmp_path / "native-boundaries.json"
    content = b'{"recovery":"post-rename-fsync"}\n'
    original = native.os.fsync
    fsync_calls = 0

    def fail_post_rename_fsync(descriptor: int) -> None:
        nonlocal fsync_calls
        fsync_calls += 1
        if fsync_calls == 3:
            raise OSError("injected post-rename fsync failure")
        original(descriptor)

    monkeypatch.setattr(native.os, "fsync", fail_post_rename_fsync)

    with pytest.raises(RuntimeError, match="preserved published destination") as captured:
        native._publish_new_file(content, destination)

    assert str(destination) in str(captured.value)
    assert destination.read_bytes() == content


def test_native_boundary_final_parent_replacement_reports_preserved_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Report the moved publication through the retained parent after final drift."""

    import docprune.task7_native_boundary as native

    destination_parent = tmp_path / "output"
    destination_parent.mkdir()
    moved_parent = tmp_path / "moved-output"
    destination = destination_parent / "native-boundaries.json"
    moved_destination = moved_parent / destination.name
    content = b'{"recovery":"final-parent-replacement"}\n'
    original = native._require_destination_parent_identity
    identity_checks = 0

    def replace_parent_before_final_check(path: Path, descriptor: int) -> None:
        nonlocal identity_checks
        identity_checks += 1
        if identity_checks == 3:
            destination_parent.rename(moved_parent)
            destination_parent.mkdir()
        original(path, descriptor)

    monkeypatch.setattr(
        native, "_require_destination_parent_identity", replace_parent_before_final_check
    )

    with pytest.raises(RuntimeError, match="preserved published destination") as captured:
        native._publish_new_file(content, destination)

    assert str(moved_destination) in str(captured.value)
    assert moved_destination.read_bytes() == content
    assert not destination.exists()


def test_native_boundary_publisher_stages_in_named_private_temporary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch using unsupported anonymous staging instead of a private recovery name."""

    import docprune.task7_native_boundary as native

    source = _source_tree(tmp_path / "inputs")
    destination = tmp_path / "native-boundaries.json"
    original = native.os.open
    observed_temporary: str | None = None

    def observe_open(
        path: str | bytes | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal observed_temporary
        if (
            isinstance(path, str)
            and path.startswith(f".{destination.name}.")
            and flags & os.O_CREAT
            and flags & os.O_EXCL
        ):
            observed_temporary = path
        return original(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(native.os, "open", observe_open)

    _publish(source, destination)

    assert observed_temporary is not None
    assert not list(tmp_path.glob(f".{destination.name}.*"))


def test_native_boundary_cleanup_does_not_delete_replaced_temporary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch failed-publication cleanup deleting a replacement temporary inode."""

    import docprune.task7_native_boundary as native

    source = _source_tree(tmp_path / "inputs")
    destination = tmp_path / "native-boundaries.json"
    unrelated = b"unrelated temporary replacement must survive\n"
    observed_temporary: str | None = None

    def replace_temporary_then_fail(
        old_directory_fd: int,
        old_name: str,
        new_directory_fd: int,
        new_name: str,
    ) -> None:
        del new_directory_fd, new_name
        nonlocal observed_temporary
        observed_temporary = old_name
        os.rename(
            old_name,
            f"moved-{old_name}",
            src_dir_fd=old_directory_fd,
            dst_dir_fd=old_directory_fd,
        )
        replacement_fd = os.open(
            old_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=old_directory_fd,
        )
        try:
            os.write(replacement_fd, unrelated)
        finally:
            os.close(replacement_fd)
        raise OSError("injected rename failure")

    monkeypatch.setattr(native, "_renameat2_noreplace", replace_temporary_then_fail)

    with pytest.raises(RuntimeError, match="preserved private temporary"):
        _publish(source, destination)

    assert observed_temporary is not None
    assert (tmp_path / observed_temporary).read_bytes() == unrelated
    assert not destination.exists()


def test_native_boundary_failed_rename_preserves_exact_fsynced_temporary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch deleting or failing to report the exact pre-rename recovery artifact."""

    import docprune.task7_native_boundary as native

    source = _source_tree(tmp_path / "inputs")
    destination = tmp_path / "native-boundaries.json"
    original_fsync = native.os.fsync
    directory_fsyncs = 0

    def record_fsync(descriptor: int) -> None:
        nonlocal directory_fsyncs
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            directory_fsyncs += 1
        original_fsync(descriptor)

    def fail_rename(*_args: object) -> None:
        raise OSError("injected rename failure")

    monkeypatch.setattr(native.os, "fsync", record_fsync)
    monkeypatch.setattr(native, "_renameat2_noreplace", fail_rename)

    with pytest.raises(RuntimeError, match="preserved private temporary"):
        _publish(source, destination)

    assert directory_fsyncs == 1
    temporaries = list(tmp_path.glob(f".{destination.name}.*"))
    assert len(temporaries) == 1
    assert json.loads(temporaries[0].read_text(encoding="utf-8"))["status"] == (
        "sealed-task7-native-boundaries"
    )
    assert not destination.exists()


def test_native_boundary_publisher_authenticates_source_and_excludes_answer_outcomes(
    tmp_path: Path,
) -> None:
    """Catch sourcing B_l* from an unauthenticated cell or leaking its answer outcome."""

    from docprune.task7_native_boundary import load_task7_native_boundary_manifest

    source = _source_tree(tmp_path)
    output = tmp_path / "native-boundaries.json"
    digest = _publish(source, output)
    loaded = load_task7_native_boundary_manifest(output, expected_sha256=digest)

    assert loaded.qids == ("q-1",)
    entry = loaded.question("q-1")
    assert entry.native_crossing is True
    assert entry.native_layer == 14
    assert entry.boundary == "B_14"
    assert entry.post_qtp_visual_tokens == 8
    assert entry.source_literal_policy_name == "literal-native-threshold"
    assert entry.source_aggregate_policy_name == "aggregate-native-threshold"
    assert loaded.comprehension_threshold == 45.0
    assert loaded.source_admission_sha256 == source["admission_sha256"]
    assert loaded.source_admission_internal_sha256 == source["admission_internal_sha256"]
    assert loaded.source_member_digest_sha256 == source["member_digest_sha256"]
    assert loaded.source_config_sha256 == source["config_sha256"]
    assert loaded.source_runtime_dir == str(source["runtime"])
    assert loaded.source_config_relative_path == "configs/docprune-m3docvqa.toml"
    assert loaded.source_launcher_sha256 == source["launcher_sha256"]
    assert (
        loaded.source_launcher_relative_path
        == "examples/sbatch/35_docprune_task6_l40s_matrix.sbatch"
    )
    encoded = output.read_text(encoding="utf-8")
    assert "outcome-must-not-enter-boundary-manifest" not in encoded
    assert "predicted_answer" not in encoded


def test_native_boundary_publisher_rejects_member_changed_after_task6_admission(
    tmp_path: Path,
) -> None:
    """Catch sealing a post-admission mutation as admitted native-boundary evidence."""

    source = _source_tree(tmp_path)
    results = source["root"] / "shard-0000" / "results.jsonl"
    results.write_bytes(results.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="admitted member checksum"):
        _publish(source, tmp_path / "native-boundaries.json")


@pytest.mark.parametrize("source_name", ("config", "launcher"))
def test_native_boundary_publisher_rejects_substituted_source_bytes_and_supplied_hash(
    tmp_path: Path, source_name: str
) -> None:
    """Catch substituting source bytes even when the caller also supplies their new checksum."""

    source = _source_tree(tmp_path)
    source[source_name].write_text("substituted bytes\n", encoding="utf-8")
    source[f"{source_name}_sha256"] = _sha256(source[source_name])

    with pytest.raises(ValueError, match="exact clean admitted checkout"):
        _publish(source, tmp_path / "native-boundaries.json")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("literal-layer", "native boundary"),
        ("global-index", "global index"),
        ("fixture", "fixture"),
        ("pages", "page"),
        ("runtime", "runtime"),
        ("cell-order", "ordered"),
    ],
)
def test_native_boundary_publisher_rejects_source_identity_drift(
    tmp_path: Path, mutation: str, message: str
) -> None:
    """Catch silently replaying a boundary from different pages, runtime, or policy evidence."""

    source = _source_tree(tmp_path)
    shard = source["root"] / "shard-0000"
    if mutation in {"literal-layer", "global-index", "fixture", "pages", "cell-order"}:
        rows = [json.loads(line) for line in (shard / "results.jsonl").read_text().splitlines()]
        if mutation == "literal-layer":
            rows[1]["policy_selection"]["boundary"] = "B_13"
            rows[1]["policy_selection"]["native_layer"] = 13
            rows[1]["trace"]["ctp_layer"] = 13
        elif mutation == "global-index":
            rows[2]["global_index_loaded"] = True
        elif mutation == "fixture":
            rows[2]["fixed_page_fixture_sha256"] = "0" * 64
        elif mutation == "pages":
            rows[2]["retrieved_pages"] = list(reversed(rows[2]["retrieved_pages"]))
        else:
            rows[1], rows[2] = rows[2], rows[1]
        (shard / "results.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
        )
    else:
        manifest = json.loads((shard / "run_manifest.json").read_text())
        manifest["runtime_commit"] = "0" * 40
        unsigned = deepcopy(manifest)
        unsigned.pop("run_manifest_sha256")
        manifest["run_manifest_sha256"] = _manifest_digest(unsigned)
        (shard / "run_manifest.json").write_text(json.dumps(manifest) + "\n")

    _resign_admission_for_semantic_test(source)

    with pytest.raises(ValueError, match=message):
        _publish(source, tmp_path / "native-boundaries.json")


def test_native_boundary_manifest_preserves_no_crossing_as_noop(tmp_path: Path) -> None:
    """Catch fabricating a decoder layer when the native comprehension threshold never crosses."""

    from docprune.task7_native_boundary import load_task7_native_boundary_manifest

    source = _source_tree(tmp_path, layer=None)
    output = tmp_path / "native-boundaries.json"
    digest = _publish(source, output)
    entry = load_task7_native_boundary_manifest(output, expected_sha256=digest).question("q-1")

    assert entry.native_crossing is False
    assert entry.native_layer is None
    assert entry.boundary is None


def test_native_boundary_cell_is_separate_forced_replay_or_no_crossing_noop(
    tmp_path: Path,
) -> None:
    """Catch adding B_l* to the fixed grid or relabeling forced replay as native CTP."""

    from docprune.task7_native_boundary import (
        load_task7_native_boundary_manifest,
        task7_native_boundary_cell,
    )

    source = _source_tree(tmp_path)
    output = tmp_path / "native-boundaries.json"
    digest = _publish(source, output)
    crossing = task7_native_boundary_cell(
        load_task7_native_boundary_manifest(output, expected_sha256=digest).question("q-1")
    )
    assert crossing.to_dict()["analysis_family"] == "native-boundary-diagnostic"
    assert crossing.to_dict()["boundary_source"] == "admitted-task6-native-comprehension"
    assert crossing.ctp_policy is None
    assert crossing.forced_intervention.boundary == 14
    assert crossing.forced_intervention.mode == "physical_delete"
    assert crossing.forced_intervention.retained_visual_ids == ()

    no_cross_source = _source_tree(tmp_path / "no-cross", layer=None)
    no_cross_output = tmp_path / "no-cross-boundaries.json"
    no_cross_digest = _publish(no_cross_source, no_cross_output)
    no_cross = task7_native_boundary_cell(
        load_task7_native_boundary_manifest(
            no_cross_output, expected_sha256=no_cross_digest
        ).question("q-1")
    )
    assert no_cross.forced_intervention is None
    assert no_cross.ctp_policy.name == "btp-qtp-no-ctp"
    assert no_cross.to_dict()["native_crossing"] is False


def test_native_boundary_loader_rejects_exact_schema_drift(tmp_path: Path) -> None:
    """Catch accepting a re-signed manifest with injected or internally inconsistent fields."""

    from docprune.task7_native_boundary import load_task7_native_boundary_manifest

    source = _source_tree(tmp_path)
    output = tmp_path / "native-boundaries.json"
    _publish(source, output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["predicted_answer"] = "must never be admitted"
    output.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exact schema"):
        load_task7_native_boundary_manifest(output, expected_sha256=_sha256(output))


def test_native_boundary_loader_rejects_resigned_source_path_drift(tmp_path: Path) -> None:
    """Catch redirecting a sealed boundary entry to a different source artifact."""

    from docprune.task7_native_boundary import load_task7_native_boundary_manifest

    source = _source_tree(tmp_path)
    output = tmp_path / "native-boundaries.json"
    _publish(source, output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["questions"][0]["source_results_path"] = "/tmp/unrelated-results.jsonl"
    output.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="source path"):
        load_task7_native_boundary_manifest(output, expected_sha256=_sha256(output))


def test_native_boundary_result_is_cross_bound_to_manifest_and_physical_drop(
    tmp_path: Path,
) -> None:
    """Catch relabeling a native CTP answer as the separate all-drop replay."""

    from docprune.task7_native_boundary import (
        bind_task7_native_boundary_result_evidence,
        load_task7_native_boundary_manifest,
        task7_native_boundary_cell,
        validate_task7_native_boundary_result_record,
    )

    source = _source_tree(tmp_path)
    output = tmp_path / "native-boundaries.json"
    digest = _publish(source, output)
    entry = load_task7_native_boundary_manifest(output, expected_sha256=digest).question("q-1")
    cell = task7_native_boundary_cell(entry)
    record = {
        "trace": {
            "post_qtp_visual_tokens": 8,
            "post_ctp_visual_tokens": 0,
            "ctp_layer": None,
        },
        "forced_intervention": {
            "boundary": "B_14",
            "mode": "physical_delete",
            "selection_kind": "forced",
            "visual_population": 8,
            "requested_budget": 0,
            "achieved_budget": 0,
            "retained_visual_ids": [],
            "logical_retained_sequence_ids": [0, 1, 10, 11],
            "prefill_cache_lengths": [12] * 15 + [4] * 13,
            "retained_mrope_position_shape": [3, 1, 4],
            "retained_mrope_position_sha256": "a" * 64,
        },
    }
    bind_task7_native_boundary_result_evidence(
        record,
        cell,
        fixture_sha256=source["fixture_sha256"],
        native_boundary_manifest_sha256=digest,
    )
    validate_task7_native_boundary_result_record(
        record,
        cell,
        fixture_sha256=source["fixture_sha256"],
        native_boundary_manifest_sha256=digest,
    )

    drifted = deepcopy(record)
    drifted["forced_intervention"]["prefill_cache_lengths"] = [4] * 28
    with pytest.raises(ValueError, match="physical native-boundary evidence"):
        validate_task7_native_boundary_result_record(
            drifted,
            cell,
            fixture_sha256=source["fixture_sha256"],
            native_boundary_manifest_sha256=digest,
        )


def test_fixed_page_runner_exposes_native_boundary_as_a_separate_one_cell_kind(
    tmp_path: Path,
) -> None:
    """Catch folding per-question B_l* into the shared fixed-boundary grid."""

    from docprune.task7_native_boundary import (
        load_task7_native_boundary_manifest,
        task7_native_boundary_cell,
    )

    source = _source_tree(tmp_path)
    output = tmp_path / "native-boundaries.json"
    digest = _publish(source, output)
    entry = load_task7_native_boundary_manifest(output, expected_sha256=digest).question("q-1")
    cell = task7_native_boundary_cell(entry)
    path = ROOT / "examples" / "run_task6_matrix.py"
    spec = importlib.util.spec_from_file_location("task7_native_boundary_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module._expected_cells(
        {}, "visual-state-native-boundary", native_boundary_cell=cell
    ) == [{"cell": 0, **cell.to_dict()}]
    source_text = path.read_text(encoding="utf-8")
    assert "--native-boundary-manifest" in source_text
    assert "--native-boundary-manifest-sha256" in source_text
    launcher = (ROOT / "examples" / "sbatch" / "35_docprune_task6_l40s_matrix.sbatch").read_text(
        encoding="utf-8"
    )
    assert "visual-state-native-boundary" in launcher
    assert "EXPECTED=1" in launcher
    assert "NATIVE_BOUNDARY_ARGS=(" in launcher


def _write_fake_torch(
    root: Path,
    *,
    available: bool = True,
    count: int = 1,
    current: int = 0,
    name: str = "NVIDIA L40S",
) -> Path:
    modules = root / "modules"
    modules.mkdir()
    (modules / "torch.py").write_text(
        "class Properties:\n"
        f"    name = {name!r}\n"
        "class Cuda:\n"
        "    @staticmethod\n"
        f"    def is_available(): return {available!r}\n"
        "    @staticmethod\n"
        f"    def device_count(): return {count!r}\n"
        "    @staticmethod\n"
        f"    def current_device(): return {current!r}\n"
        "    @staticmethod\n"
        "    def get_device_properties(index): return Properties()\n"
        "cuda = Cuda()\n",
        encoding="utf-8",
    )
    return modules


def test_task7_l40s_probe_uses_exactly_one_cuda_visible_device_without_sigpipe(
    tmp_path: Path,
) -> None:
    """Catch restoring an early-closing nvidia-smi pipeline to the array launcher."""

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_nvidia_smi = fake_bin / "nvidia-smi"
    fake_nvidia_smi.write_text("#!/usr/bin/env bash\nexit 141\n", encoding="utf-8")
    fake_nvidia_smi.chmod(0o755)
    modules = _write_fake_torch(tmp_path)
    environment = dict(os.environ)
    environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
    environment["PYTHONPATH"] = f"{modules}:{environment.get('PYTHONPATH', '')}"
    probe = ROOT / "examples" / "probe_task7_l40s_gpu.sh"

    completed = subprocess.run(
        [str(probe), "/home/lmalveau/mamba-envs/docprune-sol/bin/python"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.stdout.strip() == "NVIDIA L40S"
    launcher = (ROOT / "examples" / "sbatch" / "35_docprune_task6_l40s_matrix.sbatch").read_text(
        encoding="utf-8"
    )
    assert 'probe_task7_l40s_gpu.sh" "$PYTHON"' in launcher
    assert "head -n 1" not in launcher


@pytest.mark.parametrize(
    ("available", "count", "current", "name"),
    (
        (False, 1, 0, "NVIDIA L40S"),
        (True, 2, 0, "NVIDIA L40S"),
        (True, 1, 1, "NVIDIA L40S"),
        (True, 1, 0, ""),
        (True, 1, 0, "NVIDIA A100-SXM4-40GB"),
    ),
)
def test_task7_l40s_probe_rejects_invalid_cuda_visibility_or_identity(
    tmp_path: Path,
    available: bool,
    count: int,
    current: int,
    name: str,
) -> None:
    """Catch admitting an unavailable, ambiguous, nonzero, or non-L40S device."""

    modules = _write_fake_torch(
        tmp_path,
        available=available,
        count=count,
        current=current,
        name=name,
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = f"{modules}:{environment.get('PYTHONPATH', '')}"
    probe = ROOT / "examples" / "probe_task7_l40s_gpu.sh"

    completed = subprocess.run(
        [str(probe), "/home/lmalveau/mamba-envs/docprune-sol/bin/python"],
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode != 0
