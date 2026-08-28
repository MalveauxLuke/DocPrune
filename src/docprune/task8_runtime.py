"""Sealed fixed-page inputs for the bounded Task 8 MinerU smoke."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path

from PIL import Image

from docprune.task6_runtime import load_fixed_page_fixture, render_task6_pdf_page

TASK8_SMOKE_SCHEMA_VERSION = "docprune-task8-mineru-smoke-input-v1"
TASK8_MINERU_RUN_SCHEMA_VERSION = "docprune-task8-mineru-run-v1"
TASK8_MINERU_COMPLETION_SCHEMA_VERSION = "docprune-task8-mineru-completion-v1"
TASK8_MINERU_VERSION = "3.0.9"
TASK8_MINERU_REPOSITORY_ID = "https://github.com/opendatalab/MinerU"
TASK8_MINERU_REPOSITORY_REVISION = "d9cd58add047c2364c1198eefcb1ee9cd63a971a"
TASK8_MINERU_MODEL_REPOSITORY_ID = "opendatalab/MinerU2.5-Pro-2604-1.2B"
TASK8_MINERU_MODEL_REVISION = "d3f5e08d073c21466bbabe21c71bb1e9c2e595da"


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _read_regular(path: Path, label: str) -> bytes:
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"{label} must be a regular file")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        return b"".join(chunks)
    except OSError as error:
        raise ValueError(f"{label} is missing or not a regular file") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _reject_symlink_components(path: Path, label: str) -> None:
    current = path
    while True:
        if current.is_symlink():
            raise ValueError(f"{label} path contains a symlink")
        if current == current.parent:
            break
        current = current.parent


def _publish_staged_directory(stage: Path, destination: Path) -> None:
    """Claim a new sibling directory through a pinned no-follow parent handle."""

    parent_descriptor: int | None = None
    output_descriptor: int | None = None
    try:
        parent_descriptor = os.open(
            destination.parent,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        os.mkdir(destination.name, dir_fd=parent_descriptor)
        output_descriptor = os.open(
            destination.name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_descriptor,
        )
        for staged_path in sorted(stage.iterdir(), key=lambda path: path.name):
            if staged_path.name != "smoke-input-manifest.json":
                os.replace(staged_path, staged_path.name, dst_dir_fd=output_descriptor)
        os.replace(
            stage / "smoke-input-manifest.json",
            "smoke-input-manifest.json",
            dst_dir_fd=output_descriptor,
        )
        stage.rmdir()
    finally:
        if output_descriptor is not None:
            os.close(output_descriptor)
        if parent_descriptor is not None:
            os.close(parent_descriptor)


def _require_commit(value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("Task 8 runtime commit must be a lowercase 40-character revision")
    return value


def _record_sha256(record: object) -> str:
    return _sha256_bytes(_canonical_bytes(record))


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _authenticated_bytes(path: Path, expected_sha256: str, label: str) -> bytes:
    _require_sha256(expected_sha256, f"{label} SHA-256")
    _reject_symlink_components(Path(path).parent, label)
    raw = _read_regular(path, label)
    if _sha256_bytes(raw) != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch")
    return raw


def _load_canonical_json(path: Path, expected_sha256: str, label: str) -> dict[str, object]:
    raw = _authenticated_bytes(path, expected_sha256, label)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid JSON") from error
    canonical = _canonical_bytes(value)
    if not isinstance(value, Mapping) or raw not in {canonical, canonical + b"\n"}:
        raise ValueError(f"{label} must be a canonical JSON object")
    return dict(value)


def _publish_json_exclusive(path: Path, value: Mapping[str, object]) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o644,
        )
        raw = _canonical_bytes(value)
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise OSError("short write while publishing JSON")
            offset += written
        os.fsync(descriptor)
    except FileExistsError:
        raise
    except OSError as error:
        raise ValueError(f"cannot publish {path.name} safely") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _validate_tool_contract(
    *,
    configuration_path: Path,
    configuration_sha256: str,
    tool_manifest_path: Path,
    tool_manifest_sha256: str,
    model_weights_path: Path,
    model_weights_sha256: str,
    model_inventory_path: Path,
    model_inventory_sha256: str,
) -> tuple[dict[str, object], dict[str, object], Path, int]:
    config_path = Path(configuration_path)
    tool_path = Path(tool_manifest_path)
    weights_path = Path(model_weights_path)
    inventory_path = Path(model_inventory_path)
    if not all(
        path.is_absolute() for path in (config_path, tool_path, weights_path, inventory_path)
    ):
        raise ValueError("MinerU tool provenance paths must be absolute")
    if config_path == tool_path:
        raise ValueError("MinerU configuration and tool manifest must be distinct")
    config = _load_canonical_json(config_path, configuration_sha256, "MinerU configuration")
    tool = _load_canonical_json(tool_path, tool_manifest_sha256, "MinerU tool manifest")
    expected_tool_fields = {
        "schema_version",
        "status",
        "backend",
        "version",
        "repository_id",
        "repository_revision",
        "model_repository_id",
        "model_revision",
        "configuration_path",
        "configuration_sha256",
    }
    if (
        set(tool) != expected_tool_fields
        or tool["schema_version"] != 1
        or tool["status"] != "pinned-mineru-tool"
        or tool["backend"] != "vlm"
        or tool["version"] != TASK8_MINERU_VERSION
        or tool["repository_id"] != TASK8_MINERU_REPOSITORY_ID
        or tool["repository_revision"] != TASK8_MINERU_REPOSITORY_REVISION
        or tool["model_repository_id"] != TASK8_MINERU_MODEL_REPOSITORY_ID
        or tool["model_revision"] != TASK8_MINERU_MODEL_REVISION
        or tool["configuration_path"] != str(config_path)
        or tool["configuration_sha256"] != configuration_sha256
    ):
        raise ValueError("MinerU tool manifest contradicts the smoke contract")
    _require_commit(tool["repository_revision"])
    _require_commit(tool["model_revision"])
    if set(config) != {"models-dir"} or not isinstance(config["models-dir"], Mapping):
        raise ValueError("MinerU configuration must contain only a models-dir mapping")
    models = config["models-dir"]
    if set(models) != {"vlm"} or not isinstance(models["vlm"], str):
        raise ValueError("MinerU configuration must pin exactly one local VLM directory")
    model_dir = Path(models["vlm"])
    if not model_dir.is_absolute() or not model_dir.is_dir() or model_dir.is_symlink():
        raise ValueError("MinerU VLM directory must be an absolute real directory")
    _reject_symlink_components(model_dir, "MinerU VLM directory")
    resolved_weights = weights_path.resolve(strict=True)
    configured_weights = (model_dir / "model.safetensors").resolve(strict=True)
    if resolved_weights != configured_weights:
        raise ValueError("MinerU weights do not belong to the configured model snapshot")
    _authenticated_bytes(resolved_weights, model_weights_sha256, "MinerU model weights")
    inventory_raw = _authenticated_bytes(
        inventory_path, model_inventory_sha256, "MinerU model inventory"
    )
    try:
        inventory_text = inventory_raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("MinerU model inventory is not UTF-8") from error
    inventory: list[tuple[str, Path]] = []
    for line in inventory_text.splitlines():
        if len(line) < 67 or line[64:66] != "  ":
            raise ValueError("MinerU model inventory line is invalid")
        expected_sha256 = _require_sha256(line[:64], "MinerU snapshot file SHA-256")
        listed_path = Path(line[66:])
        if not listed_path.is_absolute() or listed_path.parent != model_dir:
            raise ValueError("MinerU model inventory path is outside the configured snapshot")
        inventory.append((expected_sha256, listed_path))
    if (
        not inventory
        or len({path for _, path in inventory}) != len(inventory)
        or inventory != sorted(inventory, key=lambda row: str(row[1]))
        or inventory_raw
        != "".join(f"{digest}  {path}\n" for digest, path in inventory).encode("utf-8")
    ):
        raise ValueError("MinerU model inventory is not canonical and unique")
    observed_paths = set(model_dir.iterdir())
    expected_paths = {path for _, path in inventory}
    if observed_paths != expected_paths:
        raise ValueError("MinerU model snapshot file set does not match its inventory")
    for expected_sha256, listed_path in inventory:
        resolved = listed_path.resolve(strict=True)
        if not resolved.is_file():
            raise ValueError("MinerU model inventory entry is not a regular file")
        _authenticated_bytes(resolved, expected_sha256, "MinerU snapshot file")
    return config, tool, resolved_weights, len(inventory)


def _signed_manifest(unsigned: dict[str, object]) -> dict[str, object]:
    value = dict(unsigned)
    value["manifest_sha256"] = _sha256_bytes(_canonical_bytes(unsigned))
    return value


def prepare_task8_mineru_smoke(
    *,
    smoke_input_manifest_path: Path,
    smoke_input_manifest_sha256: str,
    configuration_path: Path,
    configuration_sha256: str,
    tool_manifest_path: Path,
    tool_manifest_sha256: str,
    model_weights_path: Path,
    model_weights_sha256: str,
    model_inventory_path: Path,
    model_inventory_sha256: str,
    job_root: Path,
    output_dir: Path,
    runtime_commit: str,
) -> dict[str, object]:
    """Authenticate a fixed-page MinerU run and publish its run manifest."""

    root = Path(job_root)
    output = Path(output_dir)
    smoke_path = Path(smoke_input_manifest_path)
    if not root.is_absolute() or not output.is_absolute() or not smoke_path.is_absolute():
        raise ValueError("Task 8 MinerU run paths must be absolute")
    _reject_symlink_components(root, "Task 8 MinerU job root")
    if not root.is_dir() or root.is_symlink():
        raise ValueError("Task 8 MinerU job root must be an existing real directory")
    if output.parent != root or output.exists() or output.is_symlink():
        raise FileExistsError("Task 8 MinerU output must be an absent direct child of job root")
    _require_commit(runtime_commit)
    _authenticated_bytes(smoke_path, smoke_input_manifest_sha256, "smoke input manifest")
    smoke = load_task8_smoke_inputs(smoke_path)
    _config, tool, resolved_weights, inventory_count = _validate_tool_contract(
        configuration_path=configuration_path,
        configuration_sha256=configuration_sha256,
        tool_manifest_path=tool_manifest_path,
        tool_manifest_sha256=tool_manifest_sha256,
        model_weights_path=model_weights_path,
        model_weights_sha256=model_weights_sha256,
        model_inventory_path=model_inventory_path,
        model_inventory_sha256=model_inventory_sha256,
    )
    inputs = [
        {"path": row["mineru_input_path"], "sha256": row["mineru_input_sha256"]}
        for row in smoke["pages"]
    ]
    unsigned: dict[str, object] = {
        "schema_version": TASK8_MINERU_RUN_SCHEMA_VERSION,
        "status": "prepared",
        "runtime_commit": runtime_commit,
        "qid": smoke["qid"],
        "smoke_input_manifest_path": str(smoke_path),
        "smoke_input_manifest_sha256": smoke_input_manifest_sha256,
        "configuration_path": str(configuration_path),
        "configuration_sha256": configuration_sha256,
        "tool_manifest_path": str(tool_manifest_path),
        "tool_manifest_sha256": tool_manifest_sha256,
        "repository_revision": tool["repository_revision"],
        "model_revision": tool["model_revision"],
        "model_weights_path": str(resolved_weights),
        "model_weights_sha256": model_weights_sha256,
        "model_inventory_path": str(model_inventory_path),
        "model_inventory_sha256": model_inventory_sha256,
        "model_inventory_count": inventory_count,
        "job_root": str(root),
        "output_dir": str(output),
        "backend_cli": "vlm-auto-engine",
        "engine_expected": "transformers",
        "formula_enable": True,
        "table_enable": True,
        "inputs": inputs,
        "global_index_loaded": False,
        "retrieval_run": False,
    }
    manifest = _signed_manifest(unsigned)
    _publish_json_exclusive(root / "run-manifest.json", manifest)
    return manifest


def _load_signed_run_manifest(path: Path) -> dict[str, object]:
    raw = _read_regular(path, "Task 8 MinerU run manifest")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Task 8 MinerU run manifest is invalid JSON") from error
    if not isinstance(value, Mapping) or raw != _canonical_bytes(value):
        raise ValueError("Task 8 MinerU run manifest must be canonical JSON")
    payload = dict(value)
    supplied = payload.pop("manifest_sha256", None)
    if supplied != _sha256_bytes(_canonical_bytes(payload)):
        raise ValueError("Task 8 MinerU run manifest digest is invalid")
    required = {
        "schema_version",
        "status",
        "runtime_commit",
        "qid",
        "smoke_input_manifest_path",
        "smoke_input_manifest_sha256",
        "configuration_path",
        "configuration_sha256",
        "tool_manifest_path",
        "tool_manifest_sha256",
        "repository_revision",
        "model_revision",
        "model_weights_path",
        "model_weights_sha256",
        "model_inventory_path",
        "model_inventory_sha256",
        "model_inventory_count",
        "job_root",
        "output_dir",
        "backend_cli",
        "engine_expected",
        "formula_enable",
        "table_enable",
        "inputs",
        "global_index_loaded",
        "retrieval_run",
        "manifest_sha256",
    }
    if (
        set(value) != required
        or value.get("schema_version") != TASK8_MINERU_RUN_SCHEMA_VERSION
        or value.get("status") != "prepared"
        or value.get("backend_cli") != "vlm-auto-engine"
        or value.get("engine_expected") != "transformers"
        or value.get("repository_revision") != TASK8_MINERU_REPOSITORY_REVISION
        or value.get("model_revision") != TASK8_MINERU_MODEL_REVISION
        or value.get("formula_enable") is not True
        or value.get("table_enable") is not True
        or value.get("global_index_loaded") is not False
        or value.get("retrieval_run") is not False
    ):
        raise ValueError("Task 8 MinerU run manifest permits an invalid execution")
    _require_commit(value["runtime_commit"])
    if not isinstance(value["qid"], str) or not value["qid"]:
        raise ValueError("Task 8 MinerU run manifest QID is invalid")
    if type(value["model_inventory_count"]) is not int or value["model_inventory_count"] <= 0:
        raise ValueError("Task 8 MinerU run manifest inventory count is invalid")
    inputs = value["inputs"]
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("Task 8 MinerU run manifest inputs are invalid")
    for row in inputs:
        if (
            not isinstance(row, Mapping)
            or set(row) != {"path", "sha256"}
            or not isinstance(row["path"], str)
            or not Path(row["path"]).is_absolute()
        ):
            raise ValueError("Task 8 MinerU run manifest input row is invalid")
        _require_sha256(row["sha256"], "Task 8 MinerU input SHA-256")
    return dict(value)


def _input_page_identity(smoke: Mapping[str, object], row: Mapping[str, object]):
    from docprune.segmentation import InputPageIdentity

    record = row["fixed_page_record"]
    return InputPageIdentity(
        row["input_page_index"],
        row["mineru_page_index"],
        row["document_id"],
        row["source_page_index"],
        Path(record["source_pdf_path"]),
        record["source_pdf_sha256"],
        smoke["qid"],
        row["fixture_rank"],
        Path(smoke["fixed_page_fixture_path"]),
        smoke["fixed_page_fixture_sha256"],
        row["fixed_page_record_sha256"],
        row["rendered_rgb_width"],
        row["rendered_rgb_height"],
        row["rendered_rgb_sha256"],
        record["renderer_contract"],
    )


def _inventory_regular_tree(root: Path) -> list[dict[str, object]]:
    if not root.is_dir() or root.is_symlink():
        raise ValueError("MinerU output directory is missing or unsafe")
    rows: list[dict[str, object]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            raise ValueError("MinerU output contains a symlink")
        if path.is_dir():
            continue
        raw = _read_regular(path, "MinerU output file")
        rows.append(
            {
                "path": str(path),
                "relative_path": path.relative_to(root).as_posix(),
                "sha256": _sha256_bytes(raw),
                "size_bytes": len(raw),
            }
        )
    if not rows:
        raise ValueError("MinerU output tree is empty")
    return rows


def finalize_task8_mineru_smoke(
    *,
    job_root: Path,
    output_dir: Path,
    gpu_manifest_path: Path,
    gpu_manifest_sha256: str,
) -> dict[str, object]:
    """Validate exact MinerU outputs and publish the completion manifest last."""

    from docprune.segmentation import MinerUArtifactIdentity, mineru_regions_from_middle_json

    root = Path(job_root)
    output = Path(output_dir)
    completion_path = root / "completion-manifest.json"
    if completion_path.exists() or completion_path.is_symlink():
        raise FileExistsError("Task 8 MinerU completion manifest already exists")
    if not root.is_absolute() or not output.is_absolute() or output.parent != root:
        raise ValueError("Task 8 MinerU output must be a direct child of the absolute job root")
    _reject_symlink_components(root, "Task 8 MinerU job root")
    if Path(gpu_manifest_path) != root / "gpu.json":
        raise ValueError("GPU manifest must be the fixed job-root gpu.json")
    run_path = root / "run-manifest.json"
    run = _load_signed_run_manifest(run_path)
    if run["job_root"] != str(root) or run["output_dir"] != str(output):
        raise ValueError("Task 8 MinerU run manifest path identity drifted")
    smoke_path = Path(run["smoke_input_manifest_path"])
    _authenticated_bytes(smoke_path, run["smoke_input_manifest_sha256"], "smoke input manifest")
    smoke = load_task8_smoke_inputs(smoke_path)
    _config, tool, resolved_weights, inventory_count = _validate_tool_contract(
        configuration_path=Path(run["configuration_path"]),
        configuration_sha256=run["configuration_sha256"],
        tool_manifest_path=Path(run["tool_manifest_path"]),
        tool_manifest_sha256=run["tool_manifest_sha256"],
        model_weights_path=Path(run["model_weights_path"]),
        model_weights_sha256=run["model_weights_sha256"],
        model_inventory_path=Path(run["model_inventory_path"]),
        model_inventory_sha256=run["model_inventory_sha256"],
    )
    expected_inputs = [
        {"path": row["mineru_input_path"], "sha256": row["mineru_input_sha256"]}
        for row in smoke["pages"]
    ]
    if (
        run["qid"] != smoke["qid"]
        or run["inputs"] != expected_inputs
        or run["repository_revision"] != tool["repository_revision"]
        or run["model_revision"] != tool["model_revision"]
        or run["model_weights_path"] != str(resolved_weights)
        or run["model_inventory_count"] != inventory_count
    ):
        raise ValueError("Task 8 MinerU run manifest identity does not replay")
    gpu = _load_canonical_json(Path(gpu_manifest_path), gpu_manifest_sha256, "GPU manifest")
    if set(gpu) != {"name", "memory_total_mib", "driver_version"}:
        raise ValueError("GPU manifest schema is invalid")
    if (
        not isinstance(gpu["name"], str)
        or not gpu["name"]
        or type(gpu["memory_total_mib"]) is not int
        or gpu["memory_total_mib"] <= 0
        or not isinstance(gpu["driver_version"], str)
        or not gpu["driver_version"]
    ):
        raise ValueError("GPU manifest values are invalid")
    inventory = _inventory_regular_tree(output)
    middle_paths = [Path(row["path"]) for row in inventory if row["path"].endswith("_middle.json")]
    expected_stems = [Path(row["mineru_input_path"]).stem for row in smoke["pages"]]
    observed_by_stem: dict[str, Path] = {}
    for path in middle_paths:
        matches = [stem for stem in expected_stems if path.name == f"{stem}_middle.json"]
        if len(matches) != 1 or matches[0] in observed_by_stem:
            raise ValueError("MinerU output does not contain the exact expected middle JSON set")
        observed_by_stem[matches[0]] = path
    if set(observed_by_stem) != set(expected_stems) or len(middle_paths) != len(expected_stems):
        raise ValueError("MinerU output does not contain the exact expected middle JSON set")

    artifacts: list[dict[str, object]] = []
    for row, stem in zip(smoke["pages"], expected_stems, strict=True):
        raw_path = observed_by_stem[stem]
        raw = _read_regular(raw_path, "raw MinerU middle JSON")
        page = _input_page_identity(smoke, row)
        artifact = MinerUArtifactIdentity(
            raw_middle_json_path=raw_path,
            raw_middle_json_sha256=_sha256_bytes(raw),
            smoke_input_manifest_path=smoke_path,
            smoke_input_manifest_sha256=run["smoke_input_manifest_sha256"],
            mineru_input_path=Path(row["mineru_input_path"]),
            mineru_input_sha256=row["mineru_input_sha256"],
            backend="vlm",
            version=TASK8_MINERU_VERSION,
            repository_id=TASK8_MINERU_REPOSITORY_ID,
            repository_revision=run["repository_revision"],
            model_repository_id=TASK8_MINERU_MODEL_REPOSITORY_ID,
            model_revision=run["model_revision"],
            configuration_path=Path(run["configuration_path"]),
            configuration_sha256=run["configuration_sha256"],
            tool_manifest_path=Path(run["tool_manifest_path"]),
            tool_manifest_sha256=run["tool_manifest_sha256"],
            pages=(page,),
        )
        regions = mineru_regions_from_middle_json(raw, artifact)
        artifacts.append(
            {
                "identity": artifact.to_dict(),
                "region_count": len(regions),
                "region_types": sorted({region.region_type for region in regions}),
            }
        )
    unsigned: dict[str, object] = {
        "schema_version": TASK8_MINERU_COMPLETION_SCHEMA_VERSION,
        "status": "complete",
        "run_manifest_path": str(run_path),
        "run_manifest_sha256": _sha256_bytes(_read_regular(run_path, "run manifest")),
        "gpu_manifest_path": str(gpu_manifest_path),
        "gpu_manifest_sha256": gpu_manifest_sha256,
        "gpu": gpu,
        "output_dir": str(output),
        "raw_output_files": inventory,
        "raw_middle_json_count": len(middle_paths),
        "artifacts": artifacts,
        "global_index_loaded": False,
        "retrieval_run": False,
    }
    completion = _signed_manifest(unsigned)
    _publish_json_exclusive(completion_path, completion)
    return completion


def _validate_manifest(value: object, manifest_path: Path) -> dict[str, object]:
    required = {
        "schema_version",
        "status",
        "runtime_commit",
        "fixed_page_fixture_path",
        "fixed_page_fixture_sha256",
        "qid",
        "global_index_loaded",
        "pages",
        "manifest_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("Task 8 smoke input manifest has an invalid schema")
    payload = dict(value)
    supplied = payload.pop("manifest_sha256")
    if supplied != _sha256_bytes(_canonical_bytes(payload)):
        raise ValueError("Task 8 smoke input manifest digest is invalid")
    if (
        value["schema_version"] != TASK8_SMOKE_SCHEMA_VERSION
        or value["status"] != "sealed-fixed-pages"
        or value["global_index_loaded"] is not False
    ):
        raise ValueError("Task 8 smoke input manifest permits an invalid execution mode")
    _require_commit(value["runtime_commit"])
    fixture_path = Path(value["fixed_page_fixture_path"])
    if not fixture_path.is_absolute():
        raise ValueError("Task 8 fixed-page fixture path must be absolute")
    fixture = load_fixed_page_fixture(
        fixture_path,
        expected_sha256=value["fixed_page_fixture_sha256"],
        validate_external_bytes=False,
    )
    qid = value["qid"]
    if not isinstance(qid, str) or not qid:
        raise ValueError("Task 8 smoke QID must be nonempty")
    fixture.validate_external_bytes(selected_qids=(qid,))
    fixed = fixture.question(qid)
    pages = value["pages"]
    if not isinstance(pages, list) or len(pages) != len(fixed.pages) or not pages:
        raise ValueError("Task 8 smoke pages do not match the fixed-page fixture")
    expected_page_fields = {
        "input_page_index",
        "mineru_page_index",
        "document_id",
        "source_page_index",
        "fixture_rank",
        "fixed_page_record",
        "fixed_page_record_sha256",
        "mineru_input_path",
        "mineru_input_sha256",
        "rendered_rgb_width",
        "rendered_rgb_height",
        "rendered_rgb_sha256",
    }
    root = manifest_path.parent.resolve()
    for rank, (row, record) in enumerate(zip(pages, fixed.pages, strict=True)):
        if not isinstance(row, Mapping) or set(row) != expected_page_fields:
            raise ValueError("Task 8 smoke page row has an invalid schema")
        record_value = record.to_dict()
        input_path = Path(row["mineru_input_path"])
        expected_path = root / f"page-{rank:02d}.png"
        if (
            row["input_page_index"] != rank
            or row["mineru_page_index"] != 0
            or row["document_id"] != record.doc_id
            or row["source_page_index"] != record.page_index
            or row["fixture_rank"] != rank
            or row["fixed_page_record"] != record_value
            or row["fixed_page_record_sha256"] != _record_sha256(record_value)
            or input_path != expected_path
            or row["rendered_rgb_width"] != record.rendered_rgb_width
            or row["rendered_rgb_height"] != record.rendered_rgb_height
            or row["rendered_rgb_sha256"] != record.rendered_rgb_sha256
        ):
            raise ValueError("Task 8 smoke page row contradicts the fixed-page fixture")
        raw = _read_regular(input_path, "MinerU input")
        if _sha256_bytes(raw) != row["mineru_input_sha256"]:
            raise ValueError("MinerU input SHA-256 mismatch")
        try:
            with Image.open(input_path) as opened:
                image = opened.convert("RGB")
                image.load()
        except OSError as error:
            raise ValueError("MinerU input is not a valid image") from error
        if image.size != (record.rendered_rgb_width, record.rendered_rgb_height) or (
            _sha256_bytes(image.tobytes()) != record.rendered_rgb_sha256
        ):
            raise ValueError("MinerU input decoded RGB does not match the fixed page")
    return dict(value)


def load_task8_smoke_inputs(path: Path) -> dict[str, object]:
    """Load and reauthenticate a completed smoke-input seal."""

    manifest_path = Path(path)
    if not manifest_path.is_absolute():
        raise ValueError("Task 8 smoke input manifest path must be absolute")
    _reject_symlink_components(manifest_path.parent, "Task 8 smoke input manifest")
    raw = _read_regular(manifest_path, "Task 8 smoke input manifest")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Task 8 smoke input manifest is invalid JSON") from error
    if raw != _canonical_bytes(value):
        raise ValueError("Task 8 smoke input manifest bytes are not canonical")
    return _validate_manifest(value, manifest_path)


def seal_task8_smoke_inputs(
    *,
    fixture_path: Path,
    fixture_sha256: str,
    qid: str,
    output_root: Path,
    runtime_commit: str,
    render_page: Callable[[Path, int], Image.Image] = render_task6_pdf_page,
) -> dict[str, object]:
    """Render exact fixture pages to immutable per-page MinerU image inputs."""

    requested_root = Path(output_root)
    if not requested_root.is_absolute():
        raise ValueError("Task 8 smoke output root must be absolute")
    _reject_symlink_components(requested_root.parent, "Task 8 smoke output")
    if requested_root.exists() or requested_root.is_symlink():
        raise FileExistsError(f"Task 8 smoke output already exists: {requested_root}")
    root = requested_root.resolve()
    fixture_path = Path(fixture_path)
    if not fixture_path.is_absolute():
        raise ValueError("Task 8 fixed-page fixture path must be absolute")
    _reject_symlink_components(fixture_path.parent, "Task 8 fixed-page fixture")
    _require_commit(runtime_commit)
    fixture = load_fixed_page_fixture(
        fixture_path,
        expected_sha256=fixture_sha256,
        validate_external_bytes=False,
    )
    fixture.validate_external_bytes(selected_qids=(qid,))
    fixed = fixture.question(qid)
    root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    try:
        pages: list[dict[str, object]] = []
        for rank, record in enumerate(fixed.pages):
            image = render_page(record.source_pdf_path, record.page_index).convert("RGB")
            if image.size != (record.rendered_rgb_width, record.rendered_rgb_height) or (
                _sha256_bytes(image.tobytes()) != record.rendered_rgb_sha256
            ):
                raise ValueError("rendered RGB does not match the fixed-page fixture")
            staged_path = stage / f"page-{rank:02d}.png"
            image.save(staged_path, format="PNG", optimize=False, compress_level=9)
            final_path = root / staged_path.name
            record_value = record.to_dict()
            pages.append(
                {
                    "input_page_index": rank,
                    "mineru_page_index": 0,
                    "document_id": record.doc_id,
                    "source_page_index": record.page_index,
                    "fixture_rank": rank,
                    "fixed_page_record": record_value,
                    "fixed_page_record_sha256": _record_sha256(record_value),
                    "mineru_input_path": str(final_path),
                    "mineru_input_sha256": _sha256_bytes(staged_path.read_bytes()),
                    "rendered_rgb_width": record.rendered_rgb_width,
                    "rendered_rgb_height": record.rendered_rgb_height,
                    "rendered_rgb_sha256": record.rendered_rgb_sha256,
                }
            )
        unsigned: dict[str, object] = {
            "schema_version": TASK8_SMOKE_SCHEMA_VERSION,
            "status": "sealed-fixed-pages",
            "runtime_commit": runtime_commit,
            "fixed_page_fixture_path": str(fixture_path),
            "fixed_page_fixture_sha256": fixture_sha256,
            "qid": qid,
            "global_index_loaded": False,
            "pages": pages,
        }
        manifest = {**unsigned, "manifest_sha256": _sha256_bytes(_canonical_bytes(unsigned))}
        (stage / "smoke-input-manifest.json").write_bytes(_canonical_bytes(manifest))
        _publish_staged_directory(stage, root)
        return load_task8_smoke_inputs(root / "smoke-input-manifest.json")
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise


__all__ = [
    "TASK8_MINERU_COMPLETION_SCHEMA_VERSION",
    "TASK8_MINERU_RUN_SCHEMA_VERSION",
    "TASK8_SMOKE_SCHEMA_VERSION",
    "finalize_task8_mineru_smoke",
    "load_task8_smoke_inputs",
    "prepare_task8_mineru_smoke",
    "seal_task8_smoke_inputs",
]
