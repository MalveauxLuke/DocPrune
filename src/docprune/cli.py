"""Command-line entry point for local inspection and SOL evaluation."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import os
import sys
import tempfile
from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from docprune.benchmark_config import (
    COLPALI_BACKBONE_MODEL,
    COLPALI_BACKBONE_REVISION,
    COLPALI_MODEL,
    COLPALI_REVISION,
    M3DOCRAG_COMMIT,
    QWEN_MODEL,
    QWEN_REVISION,
    sha256_file,
)
from docprune.config import DocPruneConfig, load_config
from docprune.m3docrag import DocPruneM3DocRAG, SampleInput
from docprune.metrics import MEASUREMENT_DEFINITION, append_result_jsonl, summarize_jsonl

DEFAULT_FACTORY = "docprune.m3docvqa_factory:build_workload"


def _manifest_digest(payload: dict[str, object]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


_COMMON_COMPLETE_MANIFEST_FIELDS = frozenset(
    {
        "output",
        "operation",
        "mode",
        "page_count",
        "runtime_commit",
        "m3docrag_commit",
        "resources",
        "processor_contract_path",
        "processor_contract_sha256",
        "processor_contract",
        "run_config_source_path",
        "run_config_source_sha256",
        "index_manifest_source_path",
        "index_manifest_source_sha256",
        "corpus",
        "generation",
        "pruning_config",
        "selection",
    }
)


def _validate_resume_manifest(existing: dict[str, object], requested: dict[str, object]) -> None:
    _validate_complete_run_manifest(existing)
    if existing.get("operation") != requested.get("command"):
        raise ValueError("resume manifest operation does not match the requested command")
    for key, value in requested.items():
        if key == "index_manifest" and isinstance(existing.get(key), dict):
            continue
        if existing.get(key) != value:
            raise ValueError("resume manifest does not exactly match the requested run")


def _validate_complete_run_manifest(existing: dict[str, object]) -> None:
    if existing.get("schema_version") != 2 or existing.get("status") != "configured":
        raise ValueError("resume requires a complete run manifest")
    required = set(_COMMON_COMPLETE_MANIFEST_FIELDS)
    operation = existing.get("operation")
    if operation not in {"embed", "evaluate"}:
        raise ValueError("resume requires a complete run manifest")
    if operation == "evaluate":
        required.add("index_manifest")
    if not required <= set(existing):
        raise ValueError("resume requires a complete run manifest")
    supplied_digest = existing.get("run_manifest_sha256")
    unsigned = dict(existing)
    unsigned.pop("run_manifest_sha256", None)
    if supplied_digest != _manifest_digest(unsigned):
        raise ValueError("resume manifest digest is invalid")


def _validate_evaluation_workload_manifest(
    workload: Mapping[str, object],
    *,
    existing: dict[str, object],
    requested: dict[str, object],
    config: DocPruneConfig,
    authoritative: Mapping[str, object],
) -> None:
    from docprune.m3docvqa_factory import _is_cli_placeholder

    required = set(_COMMON_COMPLETE_MANIFEST_FIELDS)
    required.update({"schema_version", "status", "index_manifest", "run_manifest_sha256"})
    if not required <= set(workload):
        raise ValueError("evaluate workload manifest is incomplete")
    if workload.get("schema_version") != 2 or workload.get("status") != "configured":
        raise ValueError("evaluate workload manifest schema/status is invalid")
    unsigned_workload = dict(workload)
    supplied_workload_digest = unsigned_workload.pop("run_manifest_sha256", None)
    if supplied_workload_digest != _manifest_digest(unsigned_workload):
        raise ValueError("evaluate workload manifest digest is invalid")
    if workload.get("operation") != "evaluate":
        raise ValueError("evaluate workload manifest operation does not match")
    if "command" in workload and workload.get("command") != "evaluate":
        raise ValueError("evaluate workload manifest command does not match")
    for key in ("output", "mode", "page_count"):
        if workload.get(key) != requested.get(key):
            raise ValueError(f"evaluate workload manifest {key} does not match the invocation")
    selection = workload.get("selection")
    if not isinstance(selection, Mapping) or not {
        "requested_sample_ids",
        "limit",
        "resolved_question_ids",
        "count",
    } <= set(selection):
        raise ValueError("evaluate workload selection identity is incomplete")
    if selection.get("requested_sample_ids") != requested.get("sample_ids"):
        raise ValueError("evaluate workload sample selection does not match the invocation")
    if selection.get("limit") != requested.get("limit"):
        raise ValueError("evaluate workload sample limit does not match the invocation")
    expected_pruning = {
        "enabled": requested.get("mode") == "docprune",
        "page_settings": asdict(config.for_pages(int(requested["page_count"]))),
        "reconstruction_defaults": asdict(config.reconstruction_defaults),
        "siglip_patch_size": 14,
    }
    if workload.get("pruning_config") != expected_pruning:
        raise ValueError("evaluate workload pruning identity does not match the invocation")
    placeholder_payload = {
        "operation": "evaluate",
        "mode": requested["mode"],
        "page_count": requested["page_count"],
        "output": requested["output"],
    }
    placeholder = _is_cli_placeholder(existing, placeholder_payload, invocation=requested)
    if not placeholder:
        # A completed manifest retains the CLI selectors from the original
        # invocation.  Check them before merging any custom-factory payload so
        # a factory cannot rewrite the command/config/factory or raw source
        # selectors while preserving the deeper benchmark identity.
        for key in (
            "schema_version",
            "status",
            "command",
            "config",
            "factory",
            "output",
            "mode",
            "run_config",
            "run_config_sha256",
            "index_manifest_sha256",
            "limit",
            "sample_ids",
            "page_count",
            "upstream",
            "paper_values",
            "reconstruction_defaults",
        ):
            if existing.get(key) != requested.get(key):
                raise ValueError(f"existing run invocation field {key} does not match")
        for path_key, digest_key, requested_path_key, requested_digest_key in (
            (
                "run_config_source_path",
                "run_config_source_sha256",
                "run_config",
                "run_config_sha256",
            ),
            (
                "index_manifest_source_path",
                "index_manifest_source_sha256",
                "index_manifest",
                "index_manifest_sha256",
            ),
        ):
            if existing.get(path_key) != requested.get(requested_path_key):
                raise ValueError(f"existing run invocation field {path_key} does not match")
            if existing.get(digest_key) != requested.get(requested_digest_key):
                raise ValueError(f"existing run invocation field {digest_key} does not match")
    identity_keys = set(_COMMON_COMPLETE_MANIFEST_FIELDS) | {
        "schema_version",
        "status",
        "index_manifest",
    }
    for key in identity_keys:
        if workload.get(key) != authoritative.get(key):
            raise ValueError(
                f"evaluate workload identity does not match the authoritative run: {key}"
            )
    generic_keys = set(requested) - {"command", "index_manifest"}
    for key in generic_keys:
        if key in workload and workload.get(key) != requested.get(key):
            raise ValueError(f"evaluate workload manifest {key} does not match the invocation")
    if placeholder:
        return
    _validate_complete_run_manifest(existing)
    for key in identity_keys:
        if workload.get(key) != existing.get(key):
            raise ValueError(f"evaluate workload identity changed: {key}")


def _validate_final_evaluation_manifest(
    manifest: Mapping[str, object], *, requested: Mapping[str, object]
) -> None:
    """Check the merged manifest after a factory has returned its workload."""

    if manifest.get("operation") != "evaluate":
        raise ValueError("final evaluation manifest operation does not match")
    if manifest.get("command") != "evaluate":
        raise ValueError("final evaluation manifest command does not match")
    for key in (
        "schema_version",
        "status",
        "command",
        "config",
        "factory",
        "output",
        "mode",
        "run_config",
        "run_config_sha256",
        "index_manifest_sha256",
        "limit",
        "sample_ids",
        "page_count",
        "upstream",
        "paper_values",
        "reconstruction_defaults",
    ):
        if manifest.get(key) != requested.get(key):
            raise ValueError(f"final evaluation invocation field {key} does not match")
    for source_key, selector_key in (
        ("run_config_source_path", "run_config"),
        ("index_manifest_source_path", "index_manifest"),
    ):
        if manifest.get(source_key) != requested.get(selector_key):
            raise ValueError(f"final evaluation source field {source_key} does not match")
    for source_key, digest_key in (
        ("run_config_source_sha256", "run_config_sha256"),
        ("index_manifest_source_sha256", "index_manifest_sha256"),
    ):
        if manifest.get(source_key) != requested.get(digest_key):
            raise ValueError(f"final evaluation source field {source_key} does not match")
    _validate_complete_run_manifest(dict(manifest))


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_json(path: Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw_temp)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        descriptor = -1
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if descriptor != -1:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class EvaluationWorkload:
    runner: DocPruneM3DocRAG
    samples: Iterable[SampleInput | dict[str, Any]]
    manifest: dict[str, object] | None = None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="docprune-m3docvqa")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect = subparsers.add_parser("inspect", help="validate and print resolved parameters")
    inspect.add_argument("--config", type=Path, required=True)
    inspect.add_argument("--pages", type=int, choices=(1, 2, 4), required=True)

    summarize = subparsers.add_parser("summarize", help="aggregate an immutable result JSONL")
    summarize.add_argument("--results", type=Path, required=True)

    validate_run = subparsers.add_parser(
        "validate-run", help="independently validate a completed benchmark run"
    )
    validate_run.add_argument("--run", type=Path, required=True)
    validate_run.add_argument("--expected-questions", type=int, default=2441)

    probe = subparsers.add_parser(
        "probe-processors", help="record the pinned Qwen and ColPali processor contract"
    )
    probe.add_argument("--page-image", type=Path, required=True)
    probe.add_argument("--qwen-model", required=True)
    probe.add_argument("--qwen-revision", required=True)
    probe.add_argument("--colpali-model", required=True)
    probe.add_argument("--colpali-revision", required=True)
    probe.add_argument("--colpali-backbone-model", required=True)
    probe.add_argument("--colpali-backbone-revision", required=True)
    probe.add_argument("--output", type=Path, required=True)

    for name in ("embed", "evaluate"):
        run = subparsers.add_parser(name, help=f"run the external {name} boundary")
        run.add_argument("--config", type=Path, required=True)
        run.add_argument("--pages", type=int, choices=(1, 2, 4), required=True)
        run.add_argument("--output", type=Path, required=True)
        run.add_argument(
            "--factory",
            default=DEFAULT_FACTORY,
            help="Python module:function integration factory",
        )
        run.add_argument("--mode", choices=("all-kept", "docprune"), default="all-kept")
        run.add_argument("--run-config", type=Path)
        run.add_argument("--index-manifest", type=Path)
        run.add_argument("--limit", type=int)
        run.add_argument(
            "--sample-ids",
            nargs="+",
            help="qids selected in immutable source order (comma-separated or repeated)",
        )
        run.add_argument("--dry-run", action="store_true")
        run.add_argument("--resume", action="store_true")
    return parser


def _resolved_config(config: DocPruneConfig, pages: int) -> dict[str, object]:
    return {
        "page_count": pages,
        "upstream": {"m3docrag_commit": config.m3docrag_commit},
        "paper_values": asdict(config.for_pages(pages)),
        "reconstruction_defaults": asdict(config.reconstruction_defaults),
    }


def _manifest(
    command: str,
    config_path: Path,
    config: DocPruneConfig,
    pages: int,
    factory: str,
    *,
    output: Path,
    mode: str,
    run_config: Path | None,
    index_manifest: Path | None,
    limit: int | None,
    sample_ids: tuple[str, ...] | None,
):
    return {
        "schema_version": 2,
        "status": "configured",
        "command": command,
        "config": str(config_path.resolve()),
        "factory": factory,
        "output": str(output.resolve()),
        "mode": mode,
        "run_config": None if run_config is None else str(run_config.resolve()),
        "run_config_sha256": (
            sha256_file(run_config)
            if run_config is not None and run_config.is_file() and not run_config.is_symlink()
            else None
        ),
        "index_manifest": None if index_manifest is None else str(index_manifest.resolve()),
        "index_manifest_sha256": (
            sha256_file(index_manifest)
            if index_manifest is not None
            and index_manifest.is_file()
            and not index_manifest.is_symlink()
            else None
        ),
        "limit": limit,
        "sample_ids": None if sample_ids is None else list(sample_ids),
        "measurement": {
            "definition": MEASUREMENT_DEFINITION,
            "warmup_required": True,
            "profiler_enabled": False,
        },
        **_resolved_config(config, pages),
    }


def _load_factory(spec: str) -> Callable[..., object]:
    module_name, separator, function_name = spec.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError("factory must use module:function syntax")
    factory = getattr(importlib.import_module(module_name), function_name)
    if not callable(factory):
        raise ValueError(f"factory {spec} is not callable")
    return factory


def _prepare_output(output: Path, manifest: dict[str, object], resume: bool) -> None:
    manifest_path = output / "run_manifest.json"
    if resume and not output.exists():
        raise FileNotFoundError("resume requires an existing output directory")
    if output.exists():
        if output.is_symlink() or not output.is_dir():
            raise ValueError(f"output must be a regular directory: {output}")
        if not resume:
            raise FileExistsError(f"output directory already exists: {output}")
        if not manifest_path.is_file():
            raise ValueError("resume manifest does not exactly match the requested run")
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError("resume manifest is not valid JSON") from error
        if not isinstance(existing, dict):
            raise ValueError("resume requires a complete run manifest")
        _validate_resume_manifest(existing, manifest)
        return
    output.mkdir(parents=True)
    _atomic_write_json(manifest_path, manifest)


def _resolve_evaluate_authority(
    config: DocPruneConfig,
    pages: int,
    output: Path,
    *,
    mode: str,
    run_config: Path | None,
    index_manifest: Path | None,
    limit: int | None,
    sample_ids: tuple[str, ...] | None,
) -> dict[str, object]:
    """Resolve every evaluate identity without constructing a model or processor."""

    from docprune.m3docvqa_factory import (
        _expected_pruning_identity,
        _load_index_manifest,
        _make_dataset,
        _require_source_order_sha256,
        _resolve_run_config,
        _selection_identity,
        _source_file_identity,
        _validate_m3docrag_checkout,
        _validate_run_identity,
        filter_samples,
    )

    if index_manifest is None:
        raise ValueError("evaluate requires --index-manifest for the selected mode")
    resolved_run = _resolve_run_config(run_config, mode=mode, page_count=pages)
    identity = _validate_run_identity(resolved_run, mode=mode, page_count=pages)
    _validate_m3docrag_checkout(resolved_run)
    run_source_path, run_source_sha256 = _source_file_identity(
        run_config, label="run configuration"
    )
    index_source_path, index_source_sha256 = _source_file_identity(
        index_manifest, label="index manifest"
    )
    loaded_index = _load_index_manifest(index_manifest)
    if loaded_index.mode != mode or loaded_index.page_count != pages:
        raise ValueError("index manifest mode/page_count does not match the requested run")
    if loaded_index.m3docrag_commit != identity["m3docrag_commit"]:
        raise ValueError("index manifest uses the wrong M3DocRAG commit")
    if loaded_index.runtime_commit != identity["runtime_commit"]:
        raise ValueError("index manifest runtime commit does not match the run configuration")
    if loaded_index.processor_contract_sha256 != identity["processor_contract_sha256"]:
        raise ValueError("index manifest processor contract does not match the run configuration")
    if (
        loaded_index.processor_contract_path.resolve()
        != Path(str(identity["processor_contract_path"])).resolve()
    ):
        raise ValueError(
            "index manifest processor contract path does not match the run configuration"
        )
    if loaded_index.to_dict()["resources"] != identity["resources"]:
        raise ValueError("index manifest model resources do not match the run configuration")
    pruning_config = _expected_pruning_identity(config, mode=mode, page_count=pages)
    if loaded_index.pruning_config != pruning_config:
        raise ValueError(
            "index manifest pruning configuration does not match the run configuration"
        )
    corpus = identity["corpus"]
    if loaded_index.corpus_integrity_sha256 != corpus["integrity_sha256"]:
        raise ValueError("index manifest corpus identity does not match the run configuration")
    dataset = _make_dataset(resolved_run)
    source_order = _require_source_order_sha256(dataset)
    if loaded_index.source_order_sha256 != source_order:
        raise ValueError("index manifest source order does not match the corpus")
    samples = filter_samples(dataset, limit=limit, sample_ids=sample_ids)
    authority = {
        "schema_version": 2,
        "status": "configured",
        "operation": "evaluate",
        "output": str(output.resolve()),
        **identity,
        "run_config_source_path": run_source_path,
        "run_config_source_sha256": run_source_sha256,
        "index_manifest_source_path": index_source_path,
        "index_manifest_source_sha256": index_source_sha256,
        "pruning_config": pruning_config,
        "selection": _selection_identity(samples, limit=limit, sample_ids=sample_ids),
        "index_manifest": loaded_index.to_dict(),
    }
    # Keep the authoritative source rows alongside the in-memory authority.
    # This is deliberately not part of the persisted manifest: it lets the
    # CLI validate a custom factory's sample payload against the immutable
    # corpus rows before any runner or model work begins.
    authority["_authoritative_samples"] = samples
    return authority


def _run_evaluate(
    factory: Callable[..., object],
    config: DocPruneConfig,
    pages: int,
    output: Path,
    *,
    resume: bool,
    mode: str,
    run_config: Path | None,
    index_manifest: Path | None,
    limit: int | None,
    sample_ids: tuple[str, ...] | None,
    requested_manifest: dict[str, object],
    authoritative_manifest: dict[str, object],
) -> None:
    # Keep an invocation snapshot outside the factory's trust boundary.  The
    # factory receives a separate deep copy so nested JSON values cannot be
    # mutated in place and then reused as the authority for publication.
    trusted_requested_manifest = deepcopy(requested_manifest)
    factory_invocation_manifest = deepcopy(trusted_requested_manifest)
    workload = _invoke_factory(
        factory,
        operation="evaluate",
        config=config,
        page_count=pages,
        output=output,
        mode=mode,
        run_config=run_config,
        index_manifest=index_manifest,
        limit=limit,
        sample_ids=sample_ids,
        resume=resume,
        invocation_manifest=factory_invocation_manifest,
    )
    if not isinstance(workload, EvaluationWorkload):
        raise TypeError("evaluate factory must return EvaluationWorkload")
    if workload.manifest is None:
        raise ValueError("evaluate workload must include a complete run manifest")
    manifest_path = output / "run_manifest.json"
    existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(existing_manifest, dict):
        raise ValueError("run manifest must be a JSON object")
    _validate_evaluation_workload_manifest(
        workload.manifest,
        existing=existing_manifest,
        requested=trusted_requested_manifest,
        config=config,
        authoritative=authoritative_manifest,
    )
    samples = tuple(
        sample if isinstance(sample, SampleInput) else SampleInput.from_mapping(sample)
        for sample in workload.samples
    )
    qids = tuple(sample.question_id for sample in samples)
    if len(qids) != len(set(qids)):
        raise ValueError("evaluation workload contains duplicate question IDs")
    selection = workload.manifest.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("evaluation workload selection identity is incomplete")
    expected_qids = selection.get("resolved_question_ids")
    if not isinstance(expected_qids, list) or tuple(expected_qids) != qids:
        raise ValueError("evaluation workload sample order does not match its manifest")
    authoritative_samples = authoritative_manifest.get("_authoritative_samples")
    if authoritative_samples is not None:
        expected_samples = tuple(authoritative_samples)
        if len(expected_samples) != len(samples):
            raise ValueError("evaluation workload sample count does not match the corpus")
        for actual, expected in zip(samples, expected_samples):
            if (
                actual.question_id != expected.question_id
                or actual.question != expected.question
                or actual.answers != expected.answers
            ):
                raise ValueError(
                    f"evaluation workload sample does not match the corpus: {actual.question_id}"
                )
    existing_manifest.update(workload.manifest)
    existing_manifest.pop("run_manifest_sha256", None)
    existing_manifest["run_manifest_sha256"] = _manifest_digest(existing_manifest)
    _validate_final_evaluation_manifest(
        existing_manifest,
        requested=trusted_requested_manifest,
    )
    _atomic_write_json(manifest_path, existing_manifest)
    results_path = output / "results.jsonl"
    completed: set[str] = set()
    if resume:
        from docprune.m3docvqa_factory import (
            _canonicalize_result_record,
            _validate_result_record,
            load_completed_qids,
        )

        completed = load_completed_qids(
            results_path,
            expected_samples=samples,
            expected_page_count=pages,
        )
        existing_order = []
        if results_path.exists():
            with results_path.open(encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        record = json.loads(line)
                        if not isinstance(record, dict):
                            raise ValueError("resume results contain a non-object record")
                        canonical = _canonicalize_result_record(record, line_number=1)
                        existing_order.append(canonical.get("question_id"))
        if tuple(existing_order) != qids[: len(existing_order)]:
            raise ValueError("resume results must be an exact source-order prefix")
    append_mode = resume and results_path.exists()
    pending_samples = tuple(sample for sample in samples if sample.question_id not in completed)
    if pending_samples:
        warmup = getattr(workload.runner, "warmup", None)
        if callable(warmup):
            warmup(pending_samples[0])
    for sample in samples:
        if sample.question_id in completed:
            continue
        result = workload.runner.run_sample(sample)
        record = result.to_dict()
        from docprune.m3docvqa_factory import (
            _canonicalize_result_record,
            _validate_result_record,
        )

        record = _canonicalize_result_record(record, line_number=1)
        _validate_result_record(record, line_number=1, expected_page_count=pages)
        if (
            record.get("question_id") != sample.question_id
            or record.get("question") != sample.question
            or record.get("answers") != list(sample.answers)
        ):
            raise ValueError(f"result does not match source sample {sample.question_id}")
        append_result_jsonl(results_path, record, resume=append_mode)
        append_mode = True
    from docprune.evaluation import source_rows_from_manifest, summarize_benchmark_run

    source_rows = source_rows_from_manifest(existing_manifest, output)
    _atomic_write_json(
        output / "summary.json",
        summarize_benchmark_run(results_path, source_rows),
    )


def _invoke_factory(factory: Callable[..., object], **kwargs: object) -> object:
    """Pass the expanded concrete-factory API while keeping old test bridges usable."""

    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return factory(**kwargs)
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return factory(**kwargs)
    accepted = {name: value for name, value in kwargs.items() if name in signature.parameters}
    return factory(**accepted)


def _validate_embed_result(
    result: object,
    *,
    config: DocPruneConfig,
    pages: int,
    mode: str,
    output: Path,
    run_config: object | None = None,
    expected_source_order_sha256: str | None = None,
) -> None:
    from docprune.artifacts import IndexManifest
    from docprune.indexing import IndexBuildResult

    if type(result) is not IndexBuildResult:
        raise TypeError("embed factory must return IndexBuildResult")
    manifest = result.manifest
    if type(manifest) is not IndexManifest:
        raise TypeError("embed result manifest must be an IndexManifest")
    if manifest.mode != mode or manifest.page_count != pages:
        raise ValueError("embed result mode/page_count does not match the requested run")
    if manifest.m3docrag_commit != M3DOCRAG_COMMIT:
        raise ValueError("embed result uses the wrong M3DocRAG commit")
    if run_config is not None:
        from docprune.m3docvqa_factory import _resolve_run_config, _validate_run_identity

        if isinstance(run_config, Path):
            resolved_run = _resolve_run_config(run_config, mode=mode, page_count=pages)
        else:
            resolved_run = run_config
        if resolved_run is None:
            raise ValueError("embed result validation requires an authoritative run config")
        expected_identity = _validate_run_identity(resolved_run, mode=mode, page_count=pages)
        if manifest.runtime_commit != expected_identity["runtime_commit"]:
            raise ValueError("embed result runtime commit does not match the requested run")
        if (
            manifest.processor_contract_path.resolve()
            != Path(str(expected_identity["processor_contract_path"])).resolve()
        ):
            raise ValueError(
                "embed result processor contract path does not match the requested run"
            )
        if manifest.processor_contract_sha256 != expected_identity["processor_contract_sha256"]:
            raise ValueError("embed result processor contract does not match the requested run")
        expected_corpus = expected_identity["corpus"]
        if manifest.corpus_integrity_sha256 != expected_corpus["integrity_sha256"]:
            raise ValueError("embed result corpus does not match the requested run")
    resources = {
        "qwen_model": QWEN_MODEL,
        "qwen_revision": QWEN_REVISION,
        "colpali_model": COLPALI_MODEL,
        "colpali_revision": COLPALI_REVISION,
        "colpali_backbone_model": COLPALI_BACKBONE_MODEL,
        "colpali_backbone_revision": COLPALI_BACKBONE_REVISION,
    }
    if any(getattr(manifest, name) != value for name, value in resources.items()):
        raise ValueError("embed result model resources do not match the requested run")
    expected_pruning = {
        "enabled": mode == "docprune",
        "page_settings": asdict(config.for_pages(pages)),
        "reconstruction_defaults": asdict(config.reconstruction_defaults),
        "siglip_patch_size": 14,
    }
    if manifest.pruning_config != expected_pruning:
        raise ValueError("embed result pruning configuration does not match the requested run")
    output_root = output.resolve()
    artifact_root = manifest.artifact_root.resolve()
    try:
        artifact_root.relative_to(output_root)
    except ValueError as error:
        raise ValueError("embed result artifacts are outside the requested output") from error
    if result.mode_root.resolve() != artifact_root:
        raise ValueError("embed result mode root does not match its manifest artifact root")
    for digest_name in ("corpus_integrity_sha256", "source_order_sha256"):
        digest = getattr(manifest, digest_name)
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest.lower()):
            raise ValueError(f"embed result {digest_name} is not a SHA-256")
    if expected_source_order_sha256 is not None:
        if manifest.source_order_sha256 != expected_source_order_sha256:
            raise ValueError("embed result source order does not match the authoritative dataset")
    if (
        manifest.processor_contract_path.is_file()
        and sha256_file(manifest.processor_contract_path) != manifest.processor_contract_sha256
    ):
        raise ValueError("embed result processor contract checksum mismatch")
    result_manifest_path = Path(result.manifest_path)
    if result_manifest_path.is_symlink() or not result_manifest_path.is_file():
        raise ValueError("embed result manifest must be a regular file")
    if result_manifest_path.resolve() != artifact_root / "manifest.json":
        raise ValueError("embed result manifest path does not match its artifact root")
    from docprune.m3docvqa_factory import _load_index_manifest

    persisted_manifest = _load_index_manifest(result_manifest_path)
    if persisted_manifest.to_dict() != manifest.to_dict():
        raise ValueError("embed result manifest does not equal its IndexManifest payload")


def _run_external(command: str, args: argparse.Namespace) -> int:
    config = load_config(args.config)
    config.for_pages(args.pages)
    sample_ids = None
    if args.sample_ids is not None:
        sample_ids = tuple(
            value.strip()
            for group in args.sample_ids
            for value in group.split(",")
            if value.strip()
        )
        if not sample_ids:
            raise ValueError("--sample-ids must contain at least one qid")
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("--sample-ids contains a duplicate qid")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if not args.dry_run:
        for path, label in (
            (args.run_config, "run configuration"),
            (args.index_manifest, "index manifest"),
        ):
            if path is not None and (path.is_symlink() or not path.is_file()):
                raise ValueError(f"{label} must be a regular file: {path}")
    resolved_embed_run = None
    resolved_embed_identity = None
    resolved_embed_source_order_sha256 = None
    resolved_evaluate_authority = None
    if command == "embed" and not args.dry_run:
        from docprune.m3docvqa_factory import (
            _resolve_run_config,
            _validate_m3docrag_checkout,
            _validate_run_identity,
        )

        resolved_embed_run = _resolve_run_config(
            args.run_config,
            mode=args.mode,
            page_count=args.pages,
        )
        resolved_embed_identity = _validate_run_identity(
            resolved_embed_run,
            mode=args.mode,
            page_count=args.pages,
        )
        _validate_m3docrag_checkout(resolved_embed_run)
        from docprune.m3docvqa_factory import _make_dataset, _require_source_order_sha256

        resolved_embed_source_order_sha256 = _require_source_order_sha256(
            _make_dataset(resolved_embed_run)
        )
    manifest = _manifest(
        command,
        args.config,
        config,
        args.pages,
        args.factory,
        output=args.output,
        mode=args.mode,
        run_config=args.run_config,
        index_manifest=args.index_manifest,
        limit=args.limit,
        sample_ids=sample_ids,
    )
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"output directory already exists: {args.output}")
    if args.dry_run:
        print(json.dumps({**manifest, "status": "dry_run"}, indent=2, sort_keys=True))
        return 0
    if args.resume:
        _prepare_output(args.output, manifest, resume=True)
    if command == "evaluate":
        resolved_evaluate_authority = _resolve_evaluate_authority(
            config,
            args.pages,
            args.output,
            mode=args.mode,
            run_config=args.run_config,
            index_manifest=args.index_manifest,
            limit=args.limit,
            sample_ids=sample_ids,
        )
    if not args.resume:
        _prepare_output(args.output, manifest, resume=False)
    factory = _load_factory(args.factory)
    if command == "evaluate":
        _run_evaluate(
            factory,
            config,
            args.pages,
            args.output,
            resume=args.resume,
            mode=args.mode,
            run_config=args.run_config,
            index_manifest=args.index_manifest,
            limit=args.limit,
            sample_ids=sample_ids,
            requested_manifest=manifest,
            authoritative_manifest=resolved_evaluate_authority,
        )
    else:
        trusted_manifest = deepcopy(manifest)
        result = _invoke_factory(
            factory,
            operation="embed",
            config=config,
            page_count=args.pages,
            output=args.output,
            mode=args.mode,
            run_config=args.run_config,
            index_manifest=args.index_manifest,
            resume=args.resume,
            invocation_manifest=deepcopy(trusted_manifest),
        )
        _validate_embed_result(
            result,
            config=config,
            pages=args.pages,
            mode=args.mode,
            output=args.output,
            run_config=resolved_embed_run,
            expected_source_order_sha256=resolved_embed_source_order_sha256,
        )
        complete = result.manifest.to_dict()
        manifest_path = args.output / "run_manifest.json"
        if resolved_embed_identity is None:
            raise ValueError("embed run identity was not resolved")
        base = dict(trusted_manifest)
        trusted_identity_keys = (
            "mode",
            "page_count",
            "runtime_commit",
            "m3docrag_commit",
            "resources",
            "processor_contract_path",
            "processor_contract_sha256",
            "processor_contract",
            "corpus",
            "generation",
        )
        for key in trusted_identity_keys:
            if key not in resolved_embed_identity:
                raise ValueError(f"embed identity is missing {key}")
            base[key] = resolved_embed_identity[key]
        base["run_config_source_path"] = trusted_manifest.get("run_config")
        base["run_config_source_sha256"] = trusted_manifest.get("run_config_sha256")
        base["index_manifest_source_path"] = trusted_manifest.get("index_manifest")
        base["index_manifest_source_sha256"] = trusted_manifest.get("index_manifest_sha256")
        base["operation"] = "embed"
        base["selection"] = {
            "requested_sample_ids": None,
            "limit": None,
            "resolved_question_ids": [],
            "count": 0,
        }
        base["pruning_config"] = {
            "enabled": args.mode == "docprune",
            "page_settings": asdict(config.for_pages(args.pages)),
            "reconstruction_defaults": asdict(config.reconstruction_defaults),
            "siglip_patch_size": 14,
        }
        base["index_manifest"] = complete
        base.pop("run_manifest_sha256", None)
        base["run_manifest_sha256"] = _manifest_digest(base)
        for key in (
            "schema_version",
            "status",
            "command",
            "config",
            "factory",
            "output",
            "mode",
            "run_config",
            "run_config_sha256",
            "index_manifest_sha256",
            "limit",
            "sample_ids",
            "page_count",
            "upstream",
            "paper_values",
            "reconstruction_defaults",
        ):
            if base.get(key) != trusted_manifest.get(key):
                raise ValueError(f"final embed invocation field {key} changed")
        for source_key, selector_key in (
            ("run_config_source_path", "run_config"),
            ("index_manifest_source_path", "index_manifest"),
        ):
            if base.get(source_key) != trusted_manifest.get(selector_key):
                raise ValueError(f"final embed source field {source_key} changed")
        for source_key, digest_key in (
            ("run_config_source_sha256", "run_config_sha256"),
            ("index_manifest_source_sha256", "index_manifest_sha256"),
        ):
            if base.get(source_key) != trusted_manifest.get(digest_key):
                raise ValueError(f"final embed source field {source_key} changed")
        _validate_complete_run_manifest(base)
        _atomic_write_json(manifest_path, base)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "inspect":
            config = load_config(args.config)
            print(json.dumps(_resolved_config(config, args.pages), indent=2, sort_keys=True))
            return 0
        if args.command == "summarize":
            print(json.dumps(summarize_jsonl(args.results), indent=2, sort_keys=True))
            return 0
        if args.command == "validate-run":
            from docprune.evaluation import validate_benchmark_run

            report = validate_benchmark_run(args.run, args.expected_questions)
            print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
            return 0 if report.valid else 2
        if args.command == "probe-processors":
            from docprune import processor_probe

            payload = processor_probe.run_processor_probe(
                page_image=args.page_image,
                qwen_model=args.qwen_model,
                qwen_revision=args.qwen_revision,
                colpali_model=args.colpali_model,
                colpali_revision=args.colpali_revision,
                colpali_backbone_model=args.colpali_backbone_model,
                colpali_backbone_revision=args.colpali_backbone_revision,
                output=args.output,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        return _run_external(args.command, args)
    except (FileExistsError, ImportError, OSError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
