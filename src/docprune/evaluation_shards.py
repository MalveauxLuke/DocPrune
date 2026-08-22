"""Deterministic, reusable HTC shard plans and strict evaluation-run merging."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

from docprune.evaluation import (
    _canonical_digest,
    _load_result_rows,
    source_rows_from_manifest,
    summarize_benchmark_run,
    validate_benchmark_run,
)


def _canonical_sha256(payload: Mapping[str, object], digest_field: str) -> str:
    unsigned = dict(payload)
    unsigned.pop(digest_field, None)
    return hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def _qid_digest(qids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(qids).encode("utf-8")).hexdigest()


def _allocation(counts: Mapping[str, int], requested: int) -> dict[str, int]:
    total = sum(counts.values())
    exact = {name: requested * count / total for name, count in counts.items()}
    allocated = {name: int(value) for name, value in exact.items()}
    remaining = requested - sum(allocated.values())
    order = sorted(counts, key=lambda name: (-(exact[name] - allocated[name]), name))
    for name in order[:remaining]:
        allocated[name] += 1
    return allocated


def build_shard_plan(
    source_rows: Sequence[Mapping[str, object]],
    *,
    source_questions_sha256: str,
    shard_size: int = 64,
    checkpoint_size: int = 256,
    seed_label: str = "docprune-paired256-v1",
) -> dict[str, object]:
    """Build one full-corpus plan whose leading shards form a stratified checkpoint."""

    if not isinstance(shard_size, int) or isinstance(shard_size, bool) or shard_size < 1:
        raise ValueError("shard_size must be a positive integer")
    if (
        not isinstance(checkpoint_size, int)
        or isinstance(checkpoint_size, bool)
        or checkpoint_size < 1
    ):
        raise ValueError("checkpoint_size must be a positive integer")
    if not isinstance(seed_label, str) or not seed_label:
        raise ValueError("seed_label must be a non-empty string")
    if (
        not isinstance(source_questions_sha256, str)
        or len(source_questions_sha256) != 64
        or any(character not in "0123456789abcdef" for character in source_questions_sha256)
    ):
        raise ValueError("source_questions_sha256 must be a lowercase SHA-256")
    rows = list(source_rows)
    if not rows or checkpoint_size > len(rows) or checkpoint_size % shard_size:
        raise ValueError("checkpoint_size must fit the source and be divisible by shard_size")
    qids: list[str] = []
    type_by_qid: dict[str, str] = {}
    strata: dict[str, list[str]] = defaultdict(list)
    for position, row in enumerate(rows):
        qid = row.get("qid")
        metadata = row.get("metadata")
        question_type = metadata.get("type") if isinstance(metadata, Mapping) else None
        if not isinstance(qid, str) or not qid:
            raise ValueError(f"source row {position} has an invalid qid")
        if not isinstance(question_type, str) or not question_type:
            raise ValueError(f"source row {position} has an invalid metadata.type")
        qids.append(qid)
        type_by_qid[qid] = question_type
        strata[question_type].append(qid)
    if len(qids) != len(set(qids)):
        raise ValueError("source rows contain duplicate qids")
    counts = {name: len(values) for name, values in sorted(strata.items())}
    allocation = _allocation(counts, checkpoint_size)
    checkpoint_set: set[str] = set()
    for name, values in strata.items():
        ranked = sorted(
            values,
            key=lambda qid: (
                hashlib.sha256(f"{seed_label}\0{name}\0{qid}".encode()).hexdigest(),
                qid,
            ),
        )
        checkpoint_set.update(ranked[: allocation[name]])
    checkpoint_qids = [qid for qid in qids if qid in checkpoint_set]
    remainder_qids = [qid for qid in qids if qid not in checkpoint_set]
    execution_order = checkpoint_qids + remainder_qids
    shards: list[dict[str, object]] = []
    checkpoint_shards = checkpoint_size // shard_size
    for start in range(0, len(execution_order), shard_size):
        shard_qids = execution_order[start : start + shard_size]
        shard_id = len(shards)
        shards.append(
            {
                "shard_id": shard_id,
                "name": f"shard-{shard_id:04d}",
                "checkpoint": shard_id < checkpoint_shards,
                "question_ids": shard_qids,
                "question_ids_sha256": _qid_digest(shard_qids),
            }
        )
    plan: dict[str, object] = {
        "schema_version": 1,
        "method": (
            "proportional allocation by exact metadata.type; SHA-256 deterministic "
            "within-stratum checkpoint choice; source order within every shard"
        ),
        "seed_label": seed_label,
        "source_questions": len(qids),
        "source_questions_sha256": source_questions_sha256,
        "source_question_ids": qids,
        "source_order_sha256": _qid_digest(qids),
        "type_counts": counts,
        "checkpoint_size": checkpoint_size,
        "checkpoint_shard_count": checkpoint_shards,
        "checkpoint_type_counts": dict(
            sorted(Counter(type_by_qid[qid] for qid in checkpoint_qids).items())
        ),
        "shard_size": shard_size,
        "shard_count": len(shards),
        "shards": shards,
    }
    plan["plan_sha256"] = _canonical_sha256(plan, "plan_sha256")
    return plan


def load_shard_plan(path: Path) -> dict[str, object]:
    """Load and validate a plan before it can select work or drive a merge."""

    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"shard plan must be a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("shard plan schema is invalid")
    if value.get("plan_sha256") != _canonical_sha256(value, "plan_sha256"):
        raise ValueError("shard plan canonical digest mismatch")
    source_qids = value.get("source_question_ids")
    shards = value.get("shards")
    if (
        not isinstance(source_qids, list)
        or not source_qids
        or not all(isinstance(qid, str) and qid for qid in source_qids)
        or len(source_qids) != len(set(source_qids))
    ):
        raise ValueError("shard plan source question IDs are invalid")
    if not isinstance(shards, list) or value.get("shard_count") != len(shards):
        raise ValueError("shard plan count is invalid")
    observed: list[str] = []
    positions = {qid: index for index, qid in enumerate(source_qids)}
    for shard_id, shard in enumerate(shards):
        if not isinstance(shard, dict) or shard.get("shard_id") != shard_id:
            raise ValueError(f"shard plan entry {shard_id} identity is invalid")
        qids = shard.get("question_ids")
        if not isinstance(qids, list) or not qids:
            raise ValueError(f"shard plan entry {shard_id} question IDs are invalid")
        if any(qid not in positions for qid in qids):
            raise ValueError(f"shard plan entry {shard_id} contains an unknown qid")
        if qids != sorted(qids, key=positions.__getitem__):
            raise ValueError(f"shard plan entry {shard_id} does not preserve source order")
        if shard.get("question_ids_sha256") != _qid_digest(qids):
            raise ValueError(f"shard plan entry {shard_id} digest mismatch")
        observed.extend(qids)
    if len(observed) != len(set(observed)) or set(observed) != set(source_qids):
        raise ValueError("shard plan does not partition the full source")
    return value


def write_shard_plan(
    source_rows: Sequence[Mapping[str, object]],
    output_dir: Path,
    *,
    source_questions_sha256: str,
    expected_source_questions: int,
    shard_size: int = 64,
    checkpoint_size: int = 256,
    seed_label: str = "docprune-paired256-v1",
) -> dict[str, object]:
    """Atomically persist the full plan, checkpoint view, and argv-safe QID files."""

    output_dir = Path(output_dir).resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"shard plan output already exists: {output_dir}")
    plan = build_shard_plan(
        source_rows,
        source_questions_sha256=source_questions_sha256,
        shard_size=shard_size,
        checkpoint_size=checkpoint_size,
        seed_label=seed_label,
    )
    if plan["source_questions"] != expected_source_questions:
        raise ValueError(
            f"expected {expected_source_questions} source questions, got {plan['source_questions']}"
        )
    checkpoint_count = plan["checkpoint_shard_count"]
    checkpoint_shards = copy.deepcopy(plan["shards"][:checkpoint_count])
    checkpoint_qids = [qid for shard in checkpoint_shards for qid in shard["question_ids"]]
    checkpoint_plan = copy.deepcopy(plan)
    checkpoint_plan["source_questions"] = len(checkpoint_qids)
    checkpoint_plan["source_question_ids"] = checkpoint_qids
    checkpoint_plan["source_order_sha256"] = _qid_digest(checkpoint_qids)
    checkpoint_plan["type_counts"] = copy.deepcopy(plan["checkpoint_type_counts"])
    checkpoint_plan["shard_count"] = len(checkpoint_shards)
    checkpoint_plan["shards"] = checkpoint_shards
    checkpoint_plan["plan_sha256"] = _canonical_sha256(checkpoint_plan, "plan_sha256")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        (temporary / "plan.json").write_text(
            json.dumps(plan, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        (temporary / "checkpoint-plan.json").write_text(
            json.dumps(checkpoint_plan, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        for shard in plan["shards"]:
            (temporary / f"shard-{shard['shard_id']:04d}.qids").write_text(
                "\n".join(shard["question_ids"]) + "\n", encoding="utf-8"
            )
        os.rename(temporary, output_dir)
    except BaseException:
        for child in temporary.iterdir():
            child.unlink()
        temporary.rmdir()
        raise
    return plan


def shard_question_ids(plan_path: Path, shard_id: int) -> tuple[str, ...]:
    """Resolve a single immutable shard's QIDs for a Slurm array task."""

    plan = load_shard_plan(plan_path)
    shards = plan["shards"]
    if not isinstance(shard_id, int) or isinstance(shard_id, bool) or not 0 <= shard_id < len(shards):
        raise ValueError(f"shard_id must be between 0 and {len(shards) - 1}")
    return tuple(shards[shard_id]["question_ids"])


def _file_sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required shard artifact must be a regular file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def merge_evaluation_shards(
    *,
    plan_path: Path,
    shard_root: Path,
    output_dir: Path,
    expected_questions: int = 2441,
    allow_fixture: bool = False,
) -> dict[str, object]:
    """Validate every shard, merge exactly once, and record byte-level provenance."""

    plan_path = Path(plan_path).resolve()
    shard_root = Path(shard_root).resolve()
    output_dir = Path(output_dir).resolve()
    plan = load_shard_plan(plan_path)
    source_qids = plan["source_question_ids"]
    shards = plan["shards"]
    if expected_questions != len(source_qids):
        raise ValueError(
            f"expected_questions is {expected_questions}, but the plan contains {len(source_qids)}"
        )
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"merged output already exists: {output_dir}")
    manifests: list[dict[str, object]] = []
    rows_by_qid: dict[str, dict[str, object]] = {}
    provenance: list[dict[str, object]] = []
    shared_identity: dict[str, object] | None = None
    shared_measurement: dict[str, object] | None = None
    excluded = {
        "output",
        "selection",
        "measurement",
        "run_manifest_sha256",
        "sample_ids",
        "limit",
    }
    for shard in shards:
        shard_id = shard["shard_id"]
        expected_qids = shard["question_ids"]
        run = shard_root / f"shard-{shard_id:04d}" / "run"
        report = validate_benchmark_run(
            run, expected_questions=len(expected_qids), allow_fixture=allow_fixture
        )
        if not report.valid or list(report.qids) != expected_qids:
            raise ValueError(f"shard {shard_id} failed validation: {report.errors!r}")
        manifest_path = run / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("run_manifest_sha256") != _canonical_digest(
            manifest
        ):
            raise ValueError(f"shard {shard_id} manifest digest is invalid")
        identity = {key: value for key, value in manifest.items() if key not in excluded}
        if shared_identity is None:
            shared_identity = identity
        elif identity != shared_identity:
            raise ValueError(f"shard {shard_id} immutable identity differs from earlier shards")
        measurement = manifest.get("measurement")
        if not isinstance(measurement, dict) or not isinstance(measurement.get("warmup"), dict):
            raise ValueError(f"shard {shard_id} measurement identity is invalid")
        measurement_base = copy.deepcopy(measurement)
        warmup = measurement_base.pop("warmup")
        hardware = measurement_base.get("hardware")
        gpu_model = hardware.get("gpu_model") if isinstance(hardware, Mapping) else None
        if not isinstance(gpu_model, str) or "A100" not in gpu_model or "80GB" not in gpu_model:
            raise ValueError(f"shard {shard_id} requires A100 80GB, got {gpu_model!r}")
        if warmup != {
            "count": 1,
            "sample_id": expected_qids[0],
            "sample_identity_sha256": _qid_digest(expected_qids),
        }:
            raise ValueError(f"shard {shard_id} warmup identity does not match its selection")
        if shared_measurement is None:
            shared_measurement = measurement_base
        elif measurement_base != shared_measurement:
            raise ValueError(f"shard {shard_id} measurement identity differs from earlier shards")
        records = _load_result_rows(run / "results.jsonl")
        observed_qids = [str(record.get("question_id")) for record in records]
        if observed_qids != expected_qids:
            raise ValueError(f"shard {shard_id} results do not match the plan")
        for qid, record in zip(observed_qids, records):
            if qid in rows_by_qid:
                raise ValueError(f"duplicate merged question ID: {qid}")
            rows_by_qid[qid] = record
        validation_path = run.parent / "validation.json"
        provenance.append(
            {
                "shard_id": shard_id,
                "run_path": str(run),
                "manifest_sha256": _file_sha256(manifest_path),
                "results_sha256": _file_sha256(run / "results.jsonl"),
                "summary_sha256": _file_sha256(run / "summary.json"),
                "validation_sha256": _file_sha256(validation_path),
            }
        )
        manifests.append(manifest)
    if set(rows_by_qid) != set(source_qids):
        raise ValueError("validated shards do not cover the complete plan")
    merged = copy.deepcopy(manifests[0])
    merged["output"] = str(output_dir)
    selection = copy.deepcopy(merged["selection"])
    selection["requested_sample_ids"] = list(source_qids)
    selection["limit"] = None
    selection["resolved_question_ids"] = list(source_qids)
    selection["count"] = len(source_qids)
    merged["selection"] = selection
    if "sample_ids" in merged:
        merged["sample_ids"] = list(source_qids)
    if "limit" in merged:
        merged["limit"] = None
    measurement = copy.deepcopy(merged["measurement"])
    warmup = copy.deepcopy(measurement["warmup"])
    warmup["count"] = len(shards)
    warmup["sample_id"] = source_qids[0]
    warmup["sample_identity_sha256"] = _qid_digest(source_qids)
    measurement["warmup"] = warmup
    merged["measurement"] = measurement
    merged["execution_shards"] = {
        "schema_version": 1,
        "count": len(shards),
        "plan_path": str(plan_path),
        "plan_sha256": _file_sha256(plan_path),
        "shards": provenance,
    }
    merged["run_manifest_sha256"] = _canonical_digest(merged)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        (temporary / "results.jsonl").write_text(
            "".join(json.dumps(rows_by_qid[qid], sort_keys=True) + "\n" for qid in source_qids),
            encoding="utf-8",
        )
        (temporary / "run_manifest.json").write_text(
            json.dumps(merged, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        source_rows = source_rows_from_manifest(merged, output_dir)
        summary = summarize_benchmark_run(
            temporary / "results.jsonl",
            source_rows,
            require_positive=not allow_fixture,
            result_classification=measurement.get("result_classification"),
        )
        (temporary / "summary.json").write_text(
            json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        os.rename(temporary, output_dir)
    except BaseException:
        for child in temporary.iterdir():
            child.unlink()
        temporary.rmdir()
        raise
    final = validate_benchmark_run(
        output_dir, expected_questions=expected_questions, allow_fixture=allow_fixture
    )
    if not final.valid:
        raise ValueError(f"merged run failed validation: {final.errors!r}")
    return merged


def compare_paired_checkpoint(
    all_kept_run: Path,
    docprune_run: Path,
    *,
    expected_questions: int = 256,
    bootstrap_samples: int = 20_000,
    seed: int = 260422281,
) -> dict[str, object]:
    """Publish paired quality uncertainty and efficiency trends for a checkpoint."""

    runs = {"all-kept": Path(all_kept_run), "docprune": Path(docprune_run)}
    reports = {
        mode: validate_benchmark_run(path, expected_questions=expected_questions)
        for mode, path in runs.items()
    }
    for mode, report in reports.items():
        if not report.valid:
            raise ValueError(f"{mode} checkpoint is invalid: {report.errors!r}")
    if reports["all-kept"].qids != reports["docprune"].qids:
        raise ValueError("checkpoint runs do not contain identical ordered question IDs")
    manifests = {
        mode: json.loads((path / "run_manifest.json").read_text(encoding="utf-8"))
        for mode, path in runs.items()
    }
    if manifests["all-kept"].get("mode") != "all-kept":
        raise ValueError("all-kept checkpoint manifest has the wrong mode")
    if manifests["docprune"].get("mode") != "docprune":
        raise ValueError("DocPrune checkpoint manifest has the wrong mode")
    if manifests["all-kept"].get("page_count") != manifests["docprune"].get("page_count"):
        raise ValueError("checkpoint runs have different page counts")
    measurements = {mode: manifest.get("measurement") for mode, manifest in manifests.items()}
    if measurements["all-kept"] != measurements["docprune"]:
        raise ValueError("checkpoint measurement identities are not paired")
    hardware = measurements["all-kept"].get("hardware", {})
    gpu_model = hardware.get("gpu_model") if isinstance(hardware, Mapping) else None
    if not isinstance(gpu_model, str) or "A100" not in gpu_model or "80GB" not in gpu_model:
        raise ValueError(f"checkpoint requires A100 80GB measurements, got {gpu_model!r}")
    summaries = {mode: reports[mode].summary for mode in runs}
    if any(not isinstance(summary, Mapping) for summary in summaries.values()):
        raise ValueError("checkpoint summary is missing")
    qualities = {mode: summaries[mode]["quality"] for mode in runs}
    efficiencies = {mode: summaries[mode]["efficiency"] for mode in runs}
    qids = reports["all-kept"].qids
    paired_f1 = [
        float(qualities["docprune"]["per_question"][qid]["list_f1"])
        - float(qualities["all-kept"]["per_question"][qid]["list_f1"])
        for qid in qids
    ]
    rng = random.Random(seed)
    bootstrapped = sorted(
        sum(paired_f1[rng.randrange(len(paired_f1))] for _ in paired_f1)
        / len(paired_f1)
        * 100.0
        for _ in range(bootstrap_samples)
    )

    def percentile(fraction: float) -> float:
        return bootstrapped[round((len(bootstrapped) - 1) * fraction)]

    baseline = qualities["all-kept"]["overall"]
    pruned = qualities["docprune"]["overall"]
    encoder_speedup = float(efficiencies["docprune"]["encoder_samples_per_second"]) / float(
        efficiencies["all-kept"]["encoder_samples_per_second"]
    )
    decoder_speedup = float(efficiencies["docprune"]["decoder_samples_per_second"]) / float(
        efficiencies["all-kept"]["decoder_samples_per_second"]
    )
    f1_delta = float(pruned["list_f1"]) - float(baseline["list_f1"])
    report: dict[str, object] = {
        "schema_version": 1,
        "question_count": len(qids),
        "question_ids": list(qids),
        "question_ids_sha256": _qid_digest(qids),
        "hardware": hardware,
        "all_kept": {
            "quality": baseline,
            "efficiency": efficiencies["all-kept"],
        },
        "docprune": {
            "quality": pruned,
            "efficiency": efficiencies["docprune"],
        },
        "delta_percentage_points": {
            "list_em": float(pruned["list_em"]) - float(baseline["list_em"]),
            "list_f1": f1_delta,
        },
        "paired_f1_bootstrap_95_ci_percentage_points": [percentile(0.025), percentile(0.975)],
        "paired_f1_one_sided_95_lower_percentage_points": percentile(0.05),
        "speedup": {"encoder": encoder_speedup, "decoder": decoder_speedup},
        "trend_consistent": f1_delta > 0 and encoder_speedup > 1 and decoder_speedup > 1,
        "strong_paired_f1_evidence": percentile(0.05) > 0,
        "bootstrap": {"samples": bootstrap_samples, "seed": seed},
    }
    report["checkpoint_sha256"] = _canonical_sha256(report, "checkpoint_sha256")
    return report
