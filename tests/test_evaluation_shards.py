"""Deterministic planning and fail-closed merging for evaluation shards."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from docprune.evaluation import ValidationReport, _validate_execution_shards
from docprune.evaluation_shards import (
    build_shard_plan,
    compare_paired_checkpoint,
    merge_evaluation_shards,
    write_shard_plan,
)


def _digest(payload: dict[str, object], field: str) -> str:
    unsigned = dict(payload)
    unsigned.pop(field, None)
    return hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _rows() -> list[dict[str, object]]:
    return [
        {"qid": f"q{number:02d}", "metadata": {"type": "TextQ" if number < 8 else "ImageQ"}}
        for number in range(12)
    ]


def test_build_shard_plan_makes_checkpoint_reusable_and_partitions_full_source() -> None:
    plan = build_shard_plan(
        _rows(),
        source_questions_sha256="a" * 64,
        shard_size=2,
        checkpoint_size=6,
        seed_label="paired-checkpoint-v1",
    )

    assert plan["source_question_ids"] == [f"q{number:02d}" for number in range(12)]
    assert plan["source_questions"] == 12
    assert plan["shard_count"] == 6
    assert plan["checkpoint_shard_count"] == 3
    assert plan["checkpoint_type_counts"] == {"ImageQ": 2, "TextQ": 4}
    shards = plan["shards"]
    assert all(len(shard["question_ids"]) == 2 for shard in shards)
    assert [shard["checkpoint"] for shard in shards] == [True, True, True, False, False, False]
    flattened = [qid for shard in shards for qid in shard["question_ids"]]
    assert len(flattened) == len(set(flattened)) == 12
    assert set(flattened) == set(plan["source_question_ids"])
    positions = {qid: index for index, qid in enumerate(plan["source_question_ids"])}
    for shard in shards:
        assert shard["question_ids"] == sorted(shard["question_ids"], key=positions.__getitem__)
        assert shard["question_ids_sha256"] == hashlib.sha256(
            "\n".join(shard["question_ids"]).encode()
        ).hexdigest()
    assert plan["plan_sha256"] == _digest(plan, "plan_sha256")


def test_build_shard_plan_is_deterministic_and_rejects_bad_source() -> None:
    first = build_shard_plan(
        _rows(), source_questions_sha256="a" * 64, shard_size=2, checkpoint_size=6, seed_label="seed"
    )
    second = build_shard_plan(
        _rows(), source_questions_sha256="a" * 64, shard_size=2, checkpoint_size=6, seed_label="seed"
    )
    assert first == second

    duplicate = _rows()
    duplicate[-1]["qid"] = "q00"
    with pytest.raises(ValueError, match="duplicate"):
        build_shard_plan(
            duplicate,
            source_questions_sha256="a" * 64,
            shard_size=2,
            checkpoint_size=6,
            seed_label="seed",
        )


def test_write_shard_plan_persists_full_and_checkpoint_qid_files(tmp_path: Path) -> None:
    output = tmp_path / "plan"
    plan = write_shard_plan(
        _rows(),
        output,
        source_questions_sha256="a" * 64,
        expected_source_questions=12,
        shard_size=2,
        checkpoint_size=6,
        seed_label="seed",
    )

    assert json.loads((output / "plan.json").read_text()) == plan
    checkpoint = json.loads((output / "checkpoint-plan.json").read_text())
    assert checkpoint["source_questions"] == 6
    assert checkpoint["shard_count"] == 3
    assert checkpoint["source_question_ids"] == [
        qid for shard in plan["shards"][:3] for qid in shard["question_ids"]
    ]
    for shard in plan["shards"]:
        qid_file = output / f"shard-{shard['shard_id']:04d}.qids"
        assert qid_file.read_text().splitlines() == shard["question_ids"]
    with pytest.raises(FileExistsError):
        write_shard_plan(
            _rows(),
            output,
            source_questions_sha256="a" * 64,
            expected_source_questions=12,
            shard_size=2,
            checkpoint_size=6,
            seed_label="seed",
        )


def _write_shard(root: Path, shard_id: int, qids: list[str]) -> Path:
    run = root / f"shard-{shard_id:04d}" / "run"
    run.mkdir(parents=True)
    manifest: dict[str, object] = {
        "schema_version": 2,
        "status": "configured",
        "operation": "evaluate",
        "output": str(run.resolve()),
        "mode": "all-kept",
        "page_count": 4,
        "selection": {
            "requested_sample_ids": qids,
            "limit": None,
            "resolved_question_ids": qids,
            "count": len(qids),
        },
        "measurement": {
            "hardware": {"gpu_model": "NVIDIA A100-SXM4-80GB"},
            "warmup": {
                "count": 1,
                "sample_id": qids[0],
                "sample_identity_sha256": hashlib.sha256("\n".join(qids).encode()).hexdigest(),
            }
        },
    }
    manifest["run_manifest_sha256"] = _digest(manifest, "run_manifest_sha256")
    (run / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run / "results.jsonl").write_text(
        "".join(json.dumps({"question_id": qid}) + "\n" for qid in qids), encoding="utf-8"
    )
    (run / "summary.json").write_text(json.dumps({"shard": shard_id}), encoding="utf-8")
    validation = {
        "valid": True,
        "question_count": len(qids),
        "qids": qids,
        "errors": [],
    }
    (run.parent / "validation.json").write_text(json.dumps(validation), encoding="utf-8")
    return run


def test_merge_evaluation_shards_restores_source_order_and_records_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = build_shard_plan(
        _rows(), source_questions_sha256="a" * 64, shard_size=2, checkpoint_size=6, seed_label="seed"
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    shard_root = tmp_path / "shards"
    for shard in plan["shards"]:
        _write_shard(shard_root, shard["shard_id"], shard["question_ids"])

    def valid(run: Path, expected_questions: int, *, allow_fixture: bool = False):
        del allow_fixture
        qids = tuple(
            json.loads(line)["question_id"]
            for line in (run / "results.jsonl").read_text().splitlines()
        )
        return ValidationReport(True, (), expected_questions, qids)

    monkeypatch.setattr("docprune.evaluation_shards.validate_benchmark_run", valid)
    monkeypatch.setattr(
        "docprune.evaluation_shards.summarize_benchmark_run",
        lambda path, source_rows, **kwargs: {
            "qids": [json.loads(line)["question_id"] for line in path.read_text().splitlines()]
        },
    )
    output = tmp_path / "merged"
    merged = merge_evaluation_shards(
        plan_path=plan_path,
        shard_root=shard_root,
        output_dir=output,
        expected_questions=12,
        allow_fixture=True,
    )

    observed = [json.loads(line)["question_id"] for line in (output / "results.jsonl").read_text().splitlines()]
    assert observed == plan["source_question_ids"]
    assert merged["selection"]["resolved_question_ids"] == plan["source_question_ids"]
    assert merged["measurement"]["warmup"]["count"] == 6
    assert merged["measurement"]["warmup"]["sample_id"] == "q00"
    assert merged["execution_shards"]["count"] == 6
    assert len(merged["execution_shards"]["shards"]) == 6
    assert merged["run_manifest_sha256"] == _digest(merged, "run_manifest_sha256")
    assert json.loads((output / "summary.json").read_text())["qids"] == plan[
        "source_question_ids"
    ]


def test_execution_shard_validator_authenticates_partition_and_digests(tmp_path: Path) -> None:
    plan = build_shard_plan(
        _rows(), source_questions_sha256="a" * 64, shard_size=2, checkpoint_size=6, seed_label="seed"
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    shard_root = tmp_path / "shards"
    entries = []
    for shard in plan["shards"]:
        run = _write_shard(shard_root, shard["shard_id"], shard["question_ids"])
        entry = {"shard_id": shard["shard_id"], "run_path": str(run.resolve())}
        for name, relative in (
            ("manifest_sha256", "run_manifest.json"),
            ("results_sha256", "results.jsonl"),
            ("summary_sha256", "summary.json"),
            ("validation_sha256", "../validation.json"),
        ):
            path = run / relative
            entry[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append(entry)
    manifest = {
        "mode": "all-kept",
        "page_count": 4,
        "selection": {"resolved_question_ids": plan["source_question_ids"]},
        "execution_shards": {
            "schema_version": 1,
            "count": len(entries),
            "plan_path": str(plan_path.resolve()),
            "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
            "shards": entries,
        },
    }
    errors: list[str] = []
    assert _validate_execution_shards(manifest, tmp_path, errors) == len(entries)
    assert errors == []

    entries[0]["results_sha256"] = "0" * 64
    errors = []
    _validate_execution_shards(manifest, tmp_path, errors)
    assert any("results digest mismatch" in error for error in errors)


def test_compare_paired_checkpoint_reports_quality_ci_and_speedup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qids = ("q1", "q2", "q3", "q4")
    measurement = {"hardware": {"gpu_model": "NVIDIA A100-SXM4-80GB"}}
    summaries = {
        "all-kept": {
            "quality": {
                "overall": {"list_em": 25.0, "list_f1": 25.0},
                "per_question": {qid: {"list_em": 0.0, "list_f1": 0.0} for qid in qids},
            },
            "efficiency": {
                "encoder_samples_per_second": 1.0,
                "decoder_samples_per_second": 2.0,
            },
        },
        "docprune": {
            "quality": {
                "overall": {"list_em": 50.0, "list_f1": 75.0},
                "per_question": {qid: {"list_em": 1.0, "list_f1": 0.5} for qid in qids},
            },
            "efficiency": {
                "encoder_samples_per_second": 2.5,
                "decoder_samples_per_second": 5.0,
            },
        },
    }
    runs = {}
    for mode in ("all-kept", "docprune"):
        run = tmp_path / mode
        run.mkdir()
        (run / "run_manifest.json").write_text(
            json.dumps({"measurement": measurement, "mode": mode, "page_count": 4})
        )
        runs[mode] = run

    def valid(run: Path, expected_questions: int):
        assert expected_questions == 4
        return ValidationReport(True, (), 4, qids, summary=summaries[run.name])

    monkeypatch.setattr("docprune.evaluation_shards.validate_benchmark_run", valid)
    report = compare_paired_checkpoint(
        runs["all-kept"], runs["docprune"], expected_questions=4, bootstrap_samples=100
    )

    assert report["delta_percentage_points"] == {"list_em": 25.0, "list_f1": 50.0}
    assert report["speedup"] == {"encoder": 2.5, "decoder": 2.5}
    assert report["paired_f1_bootstrap_95_ci_percentage_points"] == [50.0, 50.0]
    assert report["trend_consistent"] is True
    assert report["strong_paired_f1_evidence"] is True
    assert report["checkpoint_sha256"] == _digest(report, "checkpoint_sha256")

    wrong = {"measurement": measurement, "mode": "docprune", "page_count": 4}
    (runs["all-kept"] / "run_manifest.json").write_text(json.dumps(wrong))
    with pytest.raises(ValueError, match="wrong mode"):
        compare_paired_checkpoint(
            runs["all-kept"],
            runs["docprune"],
            expected_questions=4,
            bootstrap_samples=10,
        )
