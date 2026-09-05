"""H200-specific preparation for the authenticated Task 9 MinerU smoke."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path

from docprune.task6_runtime import FixedPageFixture, FixedPageQuestion, load_fixed_page_fixture
from docprune.task8_runtime import (
    TASK8_MINERU_MODEL_REPOSITORY_ID,
    TASK8_MINERU_MODEL_REVISION,
    TASK8_MINERU_REPOSITORY_ID,
    TASK8_MINERU_REPOSITORY_REVISION,
    TASK8_MINERU_VERSION,
    TASK8_SMOKE_SCHEMA_VERSION,
    finalize_task8_mineru_smoke,
    load_task8_smoke_inputs,
    prepare_task8_mineru_smoke,
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_exclusive(path: Path, value: object) -> str:
    raw = _canonical_bytes(value)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(raw).hexdigest()


def _materialize_snapshot(source: Path, destination: Path) -> None:
    if not source.is_dir() or source.is_symlink():
        raise ValueError("MinerU source snapshot must be a real directory")
    if destination.is_symlink():
        raise ValueError("materialized MinerU model must be a real directory")
    if destination.exists():
        if not destination.is_dir():
            raise ValueError("materialized MinerU model must be a real directory")
        source_entries = sorted(source.iterdir(), key=lambda path: path.name)
        destination_entries = sorted(destination.iterdir(), key=lambda path: path.name)
        if not source_entries or [path.name for path in source_entries] != [
            path.name for path in destination_entries
        ]:
            raise ValueError("materialized MinerU model does not match source snapshot")
        for source_entry, destination_entry in zip(source_entries, destination_entries):
            resolved = source_entry.resolve(strict=True)
            if (
                not resolved.is_file()
                or not destination_entry.is_file()
                or destination_entry.is_symlink()
                or _sha256(resolved) != _sha256(destination_entry)
            ):
                raise ValueError("materialized MinerU model does not match source snapshot")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        entries = sorted(source.iterdir(), key=lambda path: path.name)
        if not entries:
            raise ValueError("MinerU source snapshot is empty")
        for entry in entries:
            resolved = entry.resolve(strict=True)
            if not resolved.is_file():
                raise ValueError("MinerU source snapshot contains a non-file")
            os.link(resolved, stage / entry.name)
        os.replace(stage, destination)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def prepare_task9_one_page_mineru_smoke(
    *,
    source_fixture_path: Path,
    source_fixture_sha256: str,
    source_smoke_manifest_path: Path,
    source_model_snapshot: Path,
    model_dir: Path,
    job_root: Path,
    runtime_commit: str,
) -> dict[str, object]:
    """Prepare ordinal-zero, rank-zero MinerU input and authentication manifests."""

    source_fixture_path = Path(source_fixture_path)
    source_smoke_manifest_path = Path(source_smoke_manifest_path)
    source_model_snapshot = Path(source_model_snapshot)
    model_dir = Path(model_dir)
    job_root = Path(job_root)
    if any(
        not path.is_absolute()
        for path in (
            source_fixture_path,
            source_smoke_manifest_path,
            source_model_snapshot,
            model_dir,
            job_root,
        )
    ):
        raise ValueError("Task 9 MinerU preparation paths must be absolute")
    if job_root.exists() or job_root.is_symlink():
        raise FileExistsError(f"Task 9 MinerU smoke root already exists: {job_root}")

    source_fixture = load_fixed_page_fixture(
        source_fixture_path,
        expected_sha256=source_fixture_sha256,
        validate_external_bytes=False,
    )
    source_smoke = load_task8_smoke_inputs(source_smoke_manifest_path)
    qid = source_smoke["qid"]
    source_question = source_fixture.question(qid)
    if (
        source_smoke["fixed_page_fixture_path"] != str(source_fixture_path)
        or source_smoke["fixed_page_fixture_sha256"] != source_fixture_sha256
        or source_smoke["pages"][0]["fixture_rank"] != 0
    ):
        raise ValueError("source smoke identity does not match the fixed fixture")

    _materialize_snapshot(source_model_snapshot, model_dir)
    job_root.parent.mkdir(parents=True, exist_ok=True)
    job_root.mkdir()
    input_root = job_root / "input"
    input_root.mkdir()
    input_path = input_root / "page-00.png"
    shutil.copyfile(Path(source_smoke["pages"][0]["mineru_input_path"]), input_path)

    derived_fixture = FixedPageFixture(
        fixture_version="task9-h200-one-page-smoke-v1",
        reference_path=source_fixture.reference_path,
        reference_sha256=source_fixture.reference_sha256,
        eligible_questions_path=source_fixture.eligible_questions_path,
        eligible_questions_sha256=source_fixture.eligible_questions_sha256,
        feature_manifest_path=source_fixture.feature_manifest_path,
        feature_manifest_sha256=source_fixture.feature_manifest_sha256,
        completion_ledger_path=source_fixture.completion_ledger_path,
        completion_ledger_sha256=source_fixture.completion_ledger_sha256,
        questions=(
            FixedPageQuestion(
                qid=qid,
                question_sha256=source_question.question_sha256,
                pages=(source_question.pages[0],),
            ),
        ),
    )
    fixture_path = job_root / "fixture.json"
    fixture_sha256 = _write_exclusive(fixture_path, derived_fixture.to_dict())

    source_row = dict(source_smoke["pages"][0])
    source_row["mineru_input_path"] = str(input_path)
    source_row["mineru_input_sha256"] = _sha256(input_path)
    smoke_unsigned = {
        "schema_version": TASK8_SMOKE_SCHEMA_VERSION,
        "status": "sealed-fixed-pages",
        "runtime_commit": runtime_commit,
        "fixed_page_fixture_path": str(fixture_path),
        "fixed_page_fixture_sha256": fixture_sha256,
        "qid": qid,
        "global_index_loaded": False,
        "pages": [source_row],
    }
    smoke = {
        **smoke_unsigned,
        "manifest_sha256": hashlib.sha256(_canonical_bytes(smoke_unsigned)).hexdigest(),
    }
    smoke_path = input_root / "smoke-input-manifest.json"
    smoke_sha256 = _write_exclusive(smoke_path, smoke)
    load_task8_smoke_inputs(smoke_path)

    config_path = job_root / "mineru-config.json"
    config_sha256 = _write_exclusive(config_path, {"models-dir": {"vlm": str(model_dir)}})
    tool_path = job_root / "mineru-tool.json"
    tool_sha256 = _write_exclusive(
        tool_path,
        {
            "schema_version": 1,
            "status": "pinned-mineru-tool",
            "backend": "vlm",
            "version": TASK8_MINERU_VERSION,
            "repository_id": TASK8_MINERU_REPOSITORY_ID,
            "repository_revision": TASK8_MINERU_REPOSITORY_REVISION,
            "model_repository_id": TASK8_MINERU_MODEL_REPOSITORY_ID,
            "model_revision": TASK8_MINERU_MODEL_REVISION,
            "configuration_path": str(config_path),
            "configuration_sha256": config_sha256,
        },
    )
    inventory_path = job_root / "model-sha256.txt"
    inventory_bytes = "".join(
        f"{_sha256(path)}  {path}\n" for path in sorted(model_dir.iterdir(), key=str)
    ).encode()
    descriptor = os.open(
        inventory_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(inventory_bytes)
        stream.flush()
        os.fsync(stream.fileno())
    inventory_sha256 = hashlib.sha256(inventory_bytes).hexdigest()
    weights_path = model_dir / "model.safetensors"
    weights_sha256 = _sha256(weights_path)

    prepare_task8_mineru_smoke(
        smoke_input_manifest_path=smoke_path,
        smoke_input_manifest_sha256=smoke_sha256,
        configuration_path=config_path,
        configuration_sha256=config_sha256,
        tool_manifest_path=tool_path,
        tool_manifest_sha256=tool_sha256,
        model_weights_path=weights_path,
        model_weights_sha256=weights_sha256,
        model_inventory_path=inventory_path,
        model_inventory_sha256=inventory_sha256,
        job_root=job_root,
        output_dir=job_root / "mineru-output",
        runtime_commit=runtime_commit,
    )
    result = {
        "status": "prepared",
        "source_fixture_path": str(source_fixture_path),
        "source_fixture_sha256": source_fixture_sha256,
        "source_smoke_manifest_path": str(source_smoke_manifest_path),
        "source_smoke_manifest_sha256": _sha256(source_smoke_manifest_path),
        "selected_qid": qid,
        "selected_fixture_rank": 0,
        "smoke_input_manifest_path": str(smoke_path),
        "smoke_input_manifest_sha256": smoke_sha256,
        "configuration_path": str(config_path),
        "configuration_sha256": config_sha256,
        "tool_manifest_path": str(tool_path),
        "tool_manifest_sha256": tool_sha256,
        "model_dir": str(model_dir),
        "model_inventory_path": str(inventory_path),
        "model_inventory_sha256": inventory_sha256,
        "model_weights_path": str(weights_path),
        "model_weights_sha256": weights_sha256,
        "output_dir": str(job_root / "mineru-output"),
        "global_index_loaded": False,
        "retrieval_run": False,
    }
    signed = {
        **result,
        "manifest_sha256": hashlib.sha256(_canonical_bytes(result)).hexdigest(),
    }
    _write_exclusive(job_root / "preparation-manifest.json", signed)
    return signed


def require_completely_idle_gpu(
    gpu_row: str,
    process_rows: str,
    *,
    expected_gpu_id: int,
    expected_gpu_uuid: str,
) -> dict[str, object]:
    """Require an unused exact physical GPU before launch."""

    lines = [line.strip() for line in gpu_row.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError("approved GPU state is missing or ambiguous")
    fields = [field.strip() for field in lines[0].split(",")]
    if len(fields) != 8:
        raise RuntimeError("approved GPU state has an invalid schema")
    try:
        gpu_id = int(fields[0])
        memory_total_mib = int(fields[3])
        memory_used_mib = int(fields[4])
        gpu_utilization = int(fields[5])
        memory_utilization = int(fields[6])
    except ValueError as error:
        raise RuntimeError("approved GPU state contains a non-integer metric") from error
    if (
        gpu_id != expected_gpu_id
        or fields[1] != expected_gpu_uuid
        or not fields[2]
        or memory_total_mib <= 0
        or not fields[7]
    ):
        raise RuntimeError("approved GPU identity changed")
    foreign_process = any(
        line.strip().split(",", 1)[0].strip() == expected_gpu_uuid
        for line in process_rows.splitlines()
        if line.strip()
    )
    if memory_used_mib != 0 or gpu_utilization != 0 or memory_utilization != 0 or foreign_process:
        raise RuntimeError("approved GPU is not completely idle")
    return {
        "name": fields[2],
        "memory_total_mib": memory_total_mib,
        "driver_version": fields[7],
    }


def _gpu_query(gpu_id: int) -> tuple[str, str]:
    gpu = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu_id}",
            "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,"
            "utilization.memory,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    processes = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return gpu, processes


def _is_descendant(pid: int, ancestor_pid: int) -> bool:
    while pid > 1:
        if pid == ancestor_pid:
            return True
        try:
            status = (Path("/proc") / str(pid) / "status").read_text().splitlines()
            ppid_line = next(line for line in status if line.startswith("PPid:"))
            pid = int(ppid_line.split()[1])
        except (FileNotFoundError, IndexError, PermissionError, StopIteration, ValueError):
            return False
    return pid == ancestor_pid


def _foreign_gpu_pids(process_rows: str, gpu_uuid: str, allowed_ancestor: int) -> list[int]:
    foreign: list[int] = []
    for line in process_rows.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 2 or fields[0] != gpu_uuid:
            continue
        try:
            pid = int(fields[1])
        except ValueError:
            foreign.append(-1)
            continue
        if not _is_descendant(pid, allowed_ancestor):
            foreign.append(pid)
    return foreign


def run_task9_one_page_mineru_smoke(
    *,
    job_root: Path,
    mineru_executable: Path,
    gpu_id: int,
    gpu_uuid: str,
) -> dict[str, object]:
    """Run and finalize the prepared smoke on one completely idle approved GPU."""

    job_root = Path(job_root)
    mineru_executable = Path(mineru_executable)
    if gpu_id != 5:
        raise ValueError("this approved Task 9 smoke is restricted to physical GPU 5")
    if gpu_uuid != "GPU-f9077b23-7301-08eb-4b68-59345b46c22c":
        raise ValueError("physical GPU 5 UUID does not match the approved identity")
    if not mineru_executable.is_file() or not os.access(mineru_executable, os.X_OK):
        raise ValueError("MinerU executable is missing")

    preparation_path = job_root / "preparation-manifest.json"
    preparation_raw = preparation_path.read_bytes()
    preparation = json.loads(preparation_raw)
    supplied_sha256 = preparation.pop("manifest_sha256", None)
    if (
        preparation_raw != _canonical_bytes({**preparation, "manifest_sha256": supplied_sha256})
        or supplied_sha256 != hashlib.sha256(_canonical_bytes(preparation)).hexdigest()
    ):
        raise ValueError("Task 9 MinerU preparation manifest authentication failed")
    if (
        preparation.get("status") != "prepared"
        or preparation.get("selected_fixture_rank") != 0
        or preparation.get("global_index_loaded") is not False
        or preparation.get("retrieval_run") is not False
    ):
        raise ValueError("Task 9 MinerU preparation manifest is invalid")

    gpu_row, process_rows = _gpu_query(gpu_id)
    gpu_manifest = require_completely_idle_gpu(
        gpu_row,
        process_rows,
        expected_gpu_id=gpu_id,
        expected_gpu_uuid=gpu_uuid,
    )
    gpu_path = job_root / "gpu.json"
    gpu_sha256 = _write_exclusive(gpu_path, gpu_manifest)

    command = [
        str(mineru_executable),
        "--path",
        str(Path(preparation["smoke_input_manifest_path"]).parent / "page-00.png"),
        "--output",
        preparation["output_dir"],
        "--backend",
        "vlm-auto-engine",
        "--formula",
        "true",
        "--table",
        "true",
    ]
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    environment["MINERU_TOOLS_CONFIG_JSON"] = preparation["configuration_path"]
    process = subprocess.Popen(command, env=environment, start_new_session=True)
    try:
        while process.poll() is None:
            time.sleep(1)
            _gpu, current_processes = _gpu_query(gpu_id)
            foreign = _foreign_gpu_pids(current_processes, gpu_uuid, process.pid)
            if foreign:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise RuntimeError(
                    f"foreign process appeared on approved GPU 5; smoke stopped: {foreign}"
                )
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command)
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait()
        raise

    return finalize_task8_mineru_smoke(
        job_root=job_root,
        output_dir=Path(preparation["output_dir"]),
        gpu_manifest_path=gpu_path,
        gpu_manifest_sha256=gpu_sha256,
    )


__all__ = [
    "prepare_task9_one_page_mineru_smoke",
    "require_completely_idle_gpu",
    "run_task9_one_page_mineru_smoke",
]
