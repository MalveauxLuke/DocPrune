"""Authenticated combination of Task 6 native and extension matrices."""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path

from docprune.evaluation import list_em, list_f1
from docprune.experiment_design import (
    classify_f1_result,
    nested_f1_inference,
)
from docprune.task6_runtime import (
    Task6ResultIdentity,
    task6_policy_matrix,
    validate_task6_result_record,
)

_RANDOM_POLICIES = (
    "global-uniform-random",
    "page-stratified-random",
    "grid-stratified-random",
    "coverage-matched-identity-shuffle",
)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _member_bytes(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Task 6 regular member is missing or invalid: {path}")
    return path.read_bytes()


def _json_member(path: Path) -> tuple[dict[str, object], str]:
    data = _member_bytes(path)
    try:
        value = json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"Task 6 JSON member is invalid: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Task 6 JSON member is not an object: {path}")
    return value, hashlib.sha256(data).hexdigest()


def _jsonl_member(path: Path) -> tuple[list[dict[str, object]], str]:
    data = _member_bytes(path)
    try:
        values = [json.loads(line) for line in data.decode("utf-8").splitlines() if line]
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"Task 6 JSONL member is invalid: {path}") from error
    if any(not isinstance(value, dict) for value in values):
        raise ValueError(f"Task 6 JSONL member contains a non-object: {path}")
    return values, hashlib.sha256(data).hexdigest()


def _gate_qids(gate: Mapping[str, object], fixture_sha256: str) -> tuple[str, ...]:
    if (
        gate.get("schema_version") != 1
        or gate.get("status") != "sealed-development-gate"
        or gate.get("fixture_sha256") != fixture_sha256
        or gate.get("fixed_page_provenance") is not True
        or gate.get("global_index_loaded") is not False
    ):
        raise ValueError("Task 6 gate identity is invalid")
    entries = gate.get("qid_shards")
    if not isinstance(entries, Sequence) or isinstance(entries, str | bytes) or not entries:
        raise ValueError("Task 6 gate lacks QID shards")
    qids: list[str] = []
    for index, entry in enumerate(entries):
        if (
            not isinstance(entry, Mapping)
            or entry.get("shard") != index
            or not isinstance(entry.get("qid"), str)
            or not entry["qid"]
        ):
            raise ValueError("Task 6 gate QID shard order is invalid")
        qids.append(str(entry["qid"]))
    if len(qids) != len(set(qids)):
        raise ValueError("Task 6 gate contains duplicate QIDs")
    return tuple(qids)


def _load_root(
    root: Path,
    *,
    kind: str,
    qids: Sequence[str],
    fixture_sha256: str,
    gate_sha256: str,
) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"Task 6 matrix root is not a regular directory: {root}")
    expected_cells = task6_policy_matrix(kind)
    all_rows: list[dict[str, object]] = []
    members: list[dict[str, str]] = []
    expected_names = {f"shard-{index:04d}" for index in range(len(qids))}
    observed_names = {
        path.name for path in root.iterdir() if path.is_dir() and not path.is_symlink()
    }
    if observed_names != expected_names:
        raise ValueError(f"Task 6 {kind} shard set is incomplete or unexpected")
    for shard, qid in enumerate(qids):
        shard_dir = root / f"shard-{shard:04d}"
        manifest_path = shard_dir / "run_manifest.json"
        results_path = shard_dir / "results.jsonl"
        manifest, manifest_sha = _json_member(manifest_path)
        rows, results_sha = _jsonl_member(results_path)
        members.extend(
            (
                {"path": str(manifest_path.resolve()), "sha256": manifest_sha},
                {"path": str(results_path.resolve()), "sha256": results_sha},
            )
        )
        if (
            manifest.get("matrix_kind") != kind
            or manifest.get("qid") != qid
            or manifest.get("cell_count") != len(expected_cells)
            or manifest.get("fixture_sha256") != fixture_sha256
            or manifest.get("gate_sha256", manifest.get("gate_manifest_sha256")) != gate_sha256
        ):
            raise ValueError(f"Task 6 {kind} run manifest identity drift at shard {shard}")
        supplied_manifest_digest = manifest.get("run_manifest_sha256")
        unsigned_manifest = dict(manifest)
        unsigned_manifest.pop("run_manifest_sha256", None)
        if supplied_manifest_digest != _canonical_sha256(unsigned_manifest):
            raise ValueError(f"Task 6 {kind} run manifest digest drift at shard {shard}")
        cells = manifest.get("cells")
        expected_manifest_cells = [
            {
                "cell": index,
                "policy": cell.policy.to_dict(),
                "experiment_version": (
                    f"task6-{kind}-v1"
                    if cell.policy.family in {"random-top-m", "coverage-top-m"}
                    else None
                ),
                "repetition": cell.repetition,
            }
            for index, cell in enumerate(expected_cells)
        ]
        if cells != expected_manifest_cells:
            raise ValueError(f"Task 6 {kind} run manifest cells drift at shard {shard}")
        if len(rows) != len(expected_cells):
            raise ValueError(f"Task 6 {kind} result count drift at shard {shard}")
        for index, (cell, row) in enumerate(zip(expected_cells, rows, strict=True)):
            random = cell.policy.family in {"random-top-m", "coverage-top-m"}
            experiment_version = f"task6-{kind}-v1" if random else None
            if (
                row.get("matrix_cell") != index
                or row.get("matrix_kind") != kind
                or row.get("question_id") != qid
            ):
                raise ValueError(f"Task 6 {kind} result ordering drift at shard {shard}")
            validate_task6_result_record(
                row,
                Task6ResultIdentity(
                    fixture_sha256=fixture_sha256,
                    policy_name=cell.policy.name,
                    experiment_version=experiment_version,
                    repetition=cell.repetition,
                ),
            )
        all_rows.extend(rows)
    return all_rows, members


def _score(row: Mapping[str, object]) -> tuple[float, float]:
    answers = row.get("answers")
    if not isinstance(answers, Sequence) or isinstance(answers, str | bytes) or not answers:
        raise ValueError("Task 6 result lacks answers")
    predicted = row.get("predicted_answer", "")
    return 100.0 * list_em(predicted, answers), 100.0 * list_f1(predicted, answers)


def _policy_name(row: Mapping[str, object]) -> str:
    selection = row.get("policy_selection")
    policy = selection.get("policy") if isinstance(selection, Mapping) else None
    name = policy.get("name") if isinstance(policy, Mapping) else None
    if not isinstance(name, str) or not name:
        raise ValueError("Task 6 result lacks policy name")
    return name


def _policy_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[_policy_name(row)].append(row)
    summary: dict[str, dict[str, object]] = {}
    for name in sorted(grouped):
        values = grouped[name]
        scores = [_score(row) for row in values]
        retentions = []
        for row in values:
            selection = row["policy_selection"]
            population = selection["visual_population"]
            achieved = selection["achieved_budget"]
            if type(population) is not int or population <= 0 or type(achieved) is not int:
                raise ValueError("Task 6 result has invalid retention evidence")
            retentions.append(100.0 * achieved / population)
        summary[name] = {
            "rows": len(values),
            "em": sum(score[0] for score in scores) / len(scores),
            "f1": sum(score[1] for score in scores) / len(scores),
            "conditional_retention_pct": sum(retentions) / len(retentions),
        }
    return summary


def _r20_calibration(values: Mapping[str, Sequence[float]]) -> dict[str, object]:
    qids = sorted(values)
    if not qids or any(len(values[qid]) != 20 for qid in qids):
        raise ValueError("Task 6 r20 calibration requires exactly 20 masks per QID")
    variance_terms = [
        statistics.variance(float(value) for value in values[qid]) / 20 for qid in qids
    ]
    mcse = math.sqrt(sum(variance_terms)) / len(qids)
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "passed",
        "method": "sqrt(sum_i(sample_variance_i / R_i)) / N",
        "qid_count": len(qids),
        "qid_sha256": hashlib.sha256("\n".join(qids).encode("utf-8")).hexdigest(),
        "initial_repetitions": 10,
        "observed_repetitions": 20,
        "mcse_f1": mcse,
        "threshold_f1": 0.25,
        "decision_rule": "20 masks frozen by the admitted r10 trigger; no further extension",
        "frozen_repetitions": 20,
    }
    payload["calibration_sha256"] = _canonical_sha256(payload)
    return payload


def analyze_task6_native_extension(
    *,
    native_root: Path,
    extension_root: Path,
    gate: Mapping[str, object],
    gate_sha256: str,
    fixture_sha256: str,
    native_job_id: str,
    extension_job_id: str,
    draws: int = 100_000,
    seed: int = 20_260_827,
) -> dict[str, object]:
    """Authenticate both matrices and recompute the canonical 20-mask analysis."""

    if not all(
        isinstance(value, str) and len(value) == 64 for value in (gate_sha256, fixture_sha256)
    ):
        raise ValueError("Task 6 gate and fixture SHA-256 values are invalid")
    qids = _gate_qids(gate, fixture_sha256)
    native_rows, native_members = _load_root(
        native_root,
        kind="native",
        qids=qids,
        fixture_sha256=fixture_sha256,
        gate_sha256=gate_sha256,
    )
    extension_rows, extension_members = _load_root(
        extension_root,
        kind="native-extension",
        qids=qids,
        fixture_sha256=fixture_sha256,
        gate_sha256=gate_sha256,
    )
    base_by_qid = {qid: native_rows[index * 45] for index, qid in enumerate(qids)}
    for index, qid in enumerate(qids):
        base = base_by_qid[qid]
        for row in extension_rows[index * 40 : (index + 1) * 40]:
            for key in ("question", "answers", "retrieved_pages"):
                if row.get(key) != base.get(key):
                    raise ValueError(f"Task 6 extension input drift for {qid}: {key}")

    all_rows = native_rows + extension_rows
    rows_by_qid_policy: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in all_rows:
        rows_by_qid_policy[(str(row["question_id"]), _policy_name(row))].append(row)
    calibrations: dict[str, object] = {}
    for policy in _RANDOM_POLICIES:
        values = {
            qid: [_score(row)[1] for row in rows_by_qid_policy[(qid, policy)]] for qid in qids
        }
        if any(len(scores) != 20 for scores in values.values()):
            raise ValueError(f"Task 6 combined {policy} does not have exactly 20 masks per QID")
        calibrations[policy] = _r20_calibration(values)

    statistical_rows = []
    differences = []
    for qid in qids:
        aggregate_rows = rows_by_qid_policy[(qid, "aggregate-score-top-m")]
        if len(aggregate_rows) != 1:
            raise ValueError("Task 6 aggregate score arm is not unique per QID")
        aggregate_f1 = _score(aggregate_rows[0])[1]
        random_f1 = [_score(row)[1] for row in rows_by_qid_policy[(qid, "global-uniform-random")]]
        statistical_rows.append(
            {"qid": qid, "aggregate_f1": aggregate_f1, "global_random_f1": random_f1}
        )
        differences.append(aggregate_f1 - sum(random_f1) / len(random_f1))
    inference = nested_f1_inference(
        statistical_rows,
        components=[(qid,) for qid in qids],
        draws=draws,
        seed=seed,
    )
    bootstrap_draws = inference.pop("bootstrap_draws")
    del bootstrap_draws
    primary = {
        "estimand": inference["estimand"],
        "point_estimate": inference["point_estimate"],
        "draws": inference["draw_count"],
        "seed": inference["seed"],
        "tost_90_interval": inference["tost_90_interval"],
        "superiority_95_interval": inference["superiority_95_interval"],
        "classification": classify_f1_result(
            tost_90_interval=inference["tost_90_interval"],
            superiority_95_interval=inference["superiority_95_interval"],
        ),
        "developmental_sd_f1": statistics.stdev(differences),
    }
    members = native_members + extension_members
    member_digest = _canonical_sha256(members)
    report: dict[str, object] = {
        "schema_version": 1,
        "status": "admitted-development-r20",
        "job_ids": {"native": native_job_id, "extension": extension_job_id},
        "roots": {
            "native": str(Path(native_root).resolve()),
            "extension": str(Path(extension_root).resolve()),
        },
        "fixture_sha256": fixture_sha256,
        "gate_sha256": gate_sha256,
        "admission": {
            "shards": len(qids),
            "native_rows": len(native_rows),
            "extension_rows": len(extension_rows),
            "member_file_count": len(members),
            "member_digest_sha256": member_digest,
            "fixed_page_provenance": True,
            "global_index_loaded": False,
        },
        "members": members,
        "policy_summary": _policy_summary(all_rows),
        "random_calibrations": calibrations,
        "primary_r20": primary,
    }
    if not all(math.isfinite(float(value)) for value in differences):
        raise ValueError("Task 6 combined differences are nonfinite")
    report["analysis_sha256"] = _canonical_sha256(report)
    return report


def _validate_r20_trigger(
    trigger: Mapping[str, object],
    *,
    gate_sha256: str,
    fixture_sha256: str,
) -> tuple[str, str, int]:
    if not isinstance(trigger, Mapping):
        raise ValueError("Task 6 fixed sensitivity trigger is invalid")
    unsigned = dict(trigger)
    observed_sha = unsigned.pop("analysis_sha256", None)
    jobs = trigger.get("job_ids")
    calibrations = trigger.get("random_calibrations")
    expected_calibrations = set(_RANDOM_POLICIES)
    if (
        trigger.get("schema_version") != 1
        or trigger.get("status") != "admitted-development-r20"
        or trigger.get("gate_sha256") != gate_sha256
        or trigger.get("fixture_sha256") != fixture_sha256
        or observed_sha != _canonical_sha256(unsigned)
        or not isinstance(jobs, Mapping)
        or not isinstance(jobs.get("extension"), str)
        or not jobs["extension"]
        or not isinstance(calibrations, Mapping)
        or set(calibrations) != expected_calibrations
        or any(
            not isinstance(calibrations[name], Mapping)
            or calibrations[name].get("frozen_repetitions") != 20
            for name in expected_calibrations
        )
    ):
        raise ValueError("Task 6 fixed sensitivity trigger is invalid")
    return str(observed_sha), str(jobs["extension"]), 20


def analyze_task6_fixed_sensitivity(
    *,
    fixed_root: Path,
    gate: Mapping[str, object],
    gate_sha256: str,
    fixture_sha256: str,
    fixed_job_id: str,
    trigger_analysis: Mapping[str, object],
    trigger_file_sha256: str,
) -> dict[str, object]:
    """Authenticate and summarize the frozen descriptive retention curve."""

    if not all(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
        for value in (gate_sha256, fixture_sha256, trigger_file_sha256)
    ):
        raise ValueError("Task 6 fixed sensitivity SHA-256 identity is invalid")
    if not isinstance(fixed_job_id, str) or not fixed_job_id:
        raise ValueError("Task 6 fixed sensitivity job ID is invalid")
    trigger_sha, extension_job_id, frozen_repetitions = _validate_r20_trigger(
        trigger_analysis,
        gate_sha256=gate_sha256,
        fixture_sha256=fixture_sha256,
    )
    qids = _gate_qids(gate, fixture_sha256)
    rows, members = _load_root(
        fixed_root,
        kind="fixed",
        qids=qids,
        fixture_sha256=fixture_sha256,
        gate_sha256=gate_sha256,
    )

    retentions = {
        "retain-55": Fraction(11, 20),
        "retain-65": Fraction(13, 20),
        "retain-80": Fraction(4, 5),
    }
    for qid in qids:
        qid_rows = [row for row in rows if row.get("question_id") == qid]
        for label, fraction in retentions.items():
            suffix = f"-{label}"
            candidates = [row for row in qid_rows if _policy_name(row).endswith(suffix)]
            if len(candidates) != 14:
                raise ValueError("Task 6 fixed sensitivity cell count drift")
            observed: set[tuple[int, int, int]] = set()
            for row in candidates:
                selection = row["policy_selection"]
                population = selection.get("visual_population")
                requested = selection.get("requested_budget")
                achieved = selection.get("achieved_budget")
                if any(type(value) is not int for value in (population, requested, achieved)):
                    raise ValueError("Task 6 fixed sensitivity matched budget is invalid")
                expected = min(
                    population,
                    max(
                        0,
                        (fraction.numerator * population + fraction.denominator // 2)
                        // fraction.denominator,
                    ),
                )
                observed.add((requested, achieved, expected))
            if len(observed) != 1:
                raise ValueError("Task 6 fixed sensitivity matched budget drift")
            requested, achieved, expected = next(iter(observed))
            if requested != expected or achieved != expected:
                raise ValueError("Task 6 fixed sensitivity matched budget drift")

    report: dict[str, object] = {
        "schema_version": 1,
        "status": "admitted-development-fixed-sensitivity",
        "interpretation": "descriptive-development-only",
        "job_id": fixed_job_id,
        "root": str(Path(fixed_root).resolve()),
        "fixture_sha256": fixture_sha256,
        "gate_sha256": gate_sha256,
        "trigger": {
            "file_sha256": trigger_file_sha256,
            "analysis_sha256": trigger_sha,
            "extension_job_id": extension_job_id,
            "frozen_random_repetitions": frozen_repetitions,
        },
        "admission": {
            "shards": len(qids),
            "rows": len(rows),
            "member_file_count": len(members),
            "member_digest_sha256": _canonical_sha256(members),
            "fixed_page_provenance": True,
            "global_index_loaded": False,
        },
        "members": members,
        "budget_audit": {
            label: {"fraction": str(fraction), "matched_all_policies": True}
            for label, fraction in retentions.items()
        },
        "policy_summary": _policy_summary(rows),
    }
    report["analysis_sha256"] = _canonical_sha256(report)
    return report


def publish_task6_analysis(report: Mapping[str, object], output: Path) -> str:
    """Atomically publish one canonical analysis without replacing any path."""

    if report.get("schema_version") != 1 or report.get("status") != "admitted-development-r20":
        raise ValueError("Task 6 analysis status is not publishable")
    output = Path(output)
    if not output.is_absolute():
        raise ValueError("Task 6 analysis output must be absolute")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Task 6 analysis output exists: {output}")
    payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, staging_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    staging = Path(staging_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(staging, output, follow_symlinks=False)
        except FileExistsError as error:
            raise FileExistsError(f"Task 6 analysis output exists: {output}") from error
        directory = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        staging.unlink(missing_ok=True)
    return hashlib.sha256(payload).hexdigest()


def publish_task6_fixed_analysis(report: Mapping[str, object], output: Path) -> str:
    """Publish one admitted descriptive sensitivity analysis without replacement."""

    if (
        report.get("schema_version") != 1
        or report.get("status") != "admitted-development-fixed-sensitivity"
    ):
        raise ValueError("Task 6 fixed sensitivity analysis status is not publishable")
    output = Path(output)
    if not output.is_absolute():
        raise ValueError("Task 6 fixed sensitivity output must be absolute")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Task 6 fixed sensitivity output exists: {output}")
    payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, staging_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    staging = Path(staging_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(staging, output, follow_symlinks=False)
        except FileExistsError as error:
            raise FileExistsError(f"Task 6 fixed sensitivity output exists: {output}") from error
        directory = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        staging.unlink(missing_ok=True)
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "analyze_task6_fixed_sensitivity",
    "analyze_task6_native_extension",
    "publish_task6_analysis",
    "publish_task6_fixed_analysis",
]
