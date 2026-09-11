"""Tests for authenticated admission of the Task 7 native-boundary replay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from docprune.task7_native_boundary import Task7NativeBoundaryEntry
from docprune.task7_native_report import (
    _require_exact_native_shard_tree,
    _summarize_native_records,
    _validate_run_manifest,
)

ROOT = Path(__file__).resolve().parents[1]


def _manifest_digest(value: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _native_tree(tmp_path: Path) -> Path:
    root = tmp_path / "native"
    for index in range(64):
        shard = root / f"shard-{index:04d}"
        shard.mkdir(parents=True)
        (shard / "run_manifest.json").write_text("{}\n", encoding="utf-8")
        (shard / "results.jsonl").write_text("{}\n", encoding="utf-8")
    return root


def _scheduler_logs(root: Path, job_id: str) -> None:
    for index in range(64):
        for suffix in ("out", "err"):
            (root / f"slurm-docprune-task6-l40s-{job_id}_{index}.{suffix}").write_text(
                "scheduler record\n", encoding="utf-8"
            )


def _bootstrap_manifests(root: Path) -> None:
    for index in range(64):
        bootstrap = root / f"shard-{index:04d}" / "bootstrap"
        bootstrap.mkdir()
        (bootstrap / "run_manifest.json").write_text("{}\n", encoding="utf-8")


def test_native_tree_requires_exact_64_by_two_regular_files(tmp_path: Path) -> None:
    root = _native_tree(tmp_path)
    _require_exact_native_shard_tree(root)

    (root / "shard-0063" / "extra").write_text("drift", encoding="utf-8")
    with pytest.raises(ValueError, match="missing or extra files"):
        _require_exact_native_shard_tree(root)


def test_native_tree_rejects_symlink_member(tmp_path: Path) -> None:
    root = _native_tree(tmp_path)
    member = root / "shard-0000" / "results.jsonl"
    member.unlink()
    member.symlink_to(root / "shard-0001" / "results.jsonl")

    with pytest.raises(ValueError, match="substituted path|regular file"):
        _require_exact_native_shard_tree(root)


def test_native_tree_accepts_only_exact_job_bound_scheduler_logs(tmp_path: Path) -> None:
    root = _native_tree(tmp_path)
    _scheduler_logs(root, "62314817")
    _bootstrap_manifests(root)

    _require_exact_native_shard_tree(root, scheduler_job_id="62314817")

    (root / "slurm-docprune-task6-l40s-OTHER_0.out").write_text(
        "unrelated\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="missing or extra entries"):
        _require_exact_native_shard_tree(root, scheduler_job_id="62314817")


def test_native_run_manifest_is_bound_to_exact_shard_and_cell(tmp_path: Path) -> None:
    root = tmp_path / "root"
    shard = root / "shard-0007"
    fixture = tmp_path / "fixture.json"
    gate = tmp_path / "gate.json"
    boundary = tmp_path / "native.json"
    cell = {
        "cell": 0,
        "name": "all-visual-drop-native-B_14",
        "analysis_family": "native-boundary-diagnostic",
        "boundary_source": "admitted-task6-native-comprehension",
        "native_crossing": True,
        "native_layer": 14,
        "boundary": "B_14",
        "post_qtp_visual_tokens": 8,
        "ctp_policy": None,
        "forced_intervention": {
            "boundary": "B_14",
            "mode": "physical_delete",
            "retained_visual_ids": [],
        },
    }
    manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "configured-task7-native-boundary",
        "output": str(shard),
        "matrix_kind": "visual-state-native-boundary",
        "shard": 7,
        "qid": "q-7",
        "cell_count": 1,
        "fixture_path": str(fixture),
        "fixture_sha256": "1" * 64,
        "gate_manifest_path": str(gate),
        "gate_manifest_sha256": "2" * 64,
        "feature_manifest_path": str(tmp_path / "features.json"),
        "feature_manifest_sha256": "3" * 64,
        "feature_build_source_order_sha256": "4" * 64,
        "fixed_page_provenance": True,
        "global_index_loaded": False,
        "feature_build_runtime_commit": "5" * 40,
        "runtime_commit": "6" * 40,
        "m3docrag_commit": "7" * 40,
        "resources": {},
        "generation": {},
        "cells": [cell],
        "native_boundary_manifest_path": str(boundary),
        "native_boundary_manifest_sha256": "8" * 64,
    }
    manifest["run_manifest_sha256"] = _manifest_digest(manifest)

    _validate_run_manifest(
        manifest,
        shard_root=root,
        shard_index=7,
        qid="q-7",
        fixture_path=fixture,
        fixture_sha256="1" * 64,
        gate_path=gate,
        gate_sha256="2" * 64,
        native_boundary_path=boundary,
        native_boundary_sha256="8" * 64,
        runtime_commit="6" * 40,
        expected_cell=cell,
    )

    drifted = dict(manifest)
    drifted["qid"] = "other"
    unsigned = dict(drifted)
    unsigned.pop("run_manifest_sha256")
    drifted["run_manifest_sha256"] = _manifest_digest(unsigned)
    with pytest.raises(ValueError, match="sealed native replay"):
        _validate_run_manifest(
            drifted,
            shard_root=root,
            shard_index=7,
            qid="q-7",
            fixture_path=fixture,
            fixture_sha256="1" * 64,
            gate_path=gate,
            gate_sha256="2" * 64,
            native_boundary_path=boundary,
            native_boundary_sha256="8" * 64,
            runtime_commit="6" * 40,
            expected_cell=cell,
        )


def test_native_summary_is_descriptive_and_preserves_no_crossing(tmp_path: Path) -> None:
    del tmp_path
    entries = (
        Task7NativeBoundaryEntry(
            "q-a", True, 14, "B_14", 8, "literal-native-threshold", "aggregate-native-threshold"
        ),
        Task7NativeBoundaryEntry(
            "q-b", False, None, None, 7, "literal-native-threshold", "aggregate-native-threshold"
        ),
    )
    records = (
        {"question_id": "q-a", "predicted_answer": "42", "answers": ["42"]},
        {"question_id": "q-b", "predicted_answer": "wrong", "answers": ["right"]},
    )

    summary = _summarize_native_records(records, entries)

    assert summary == {
        "interpretation": "descriptive-native-boundary-diagnostic-not-information-horizon",
        "qid_count": 2,
        "native_crossing_count": 1,
        "native_no_crossing_noop_count": 1,
        "native_layer_counts": {"14": 1},
        "overall_em": 50.0,
        "overall_f1": 50.0,
        "crossing_only_em": 100.0,
        "crossing_only_f1": 100.0,
    }


def test_native_admission_cli_exposes_all_immutable_pins() -> None:
    path = ROOT / "examples" / "admit_task7_native_boundary.py"
    completed = __import__("subprocess").run(
        [__import__("sys").executable, str(path), "--help"],
        check=True,
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")},
    )
    for option in (
        "--shard-root",
        "--fixture",
        "--fixture-sha256",
        "--gate-manifest",
        "--gate-manifest-sha256",
        "--native-boundary-manifest",
        "--native-boundary-manifest-sha256",
        "--runtime-commit",
        "--scheduler-job-id",
        "--output",
    ):
        assert option in completed.stdout
