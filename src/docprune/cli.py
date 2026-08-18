"""Command-line entry point for local inspection and SOL evaluation."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import sys
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from docprune.config import DocPruneConfig, load_config
from docprune.m3docrag import DocPruneM3DocRAG, SampleInput
from docprune.metrics import append_result_jsonl, summarize_jsonl

DEFAULT_FACTORY = "docprune.m3docvqa_factory:build_workload"


def _manifest_digest(payload: dict[str, object]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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
        "mode": mode,
        "run_config": None if run_config is None else str(run_config.resolve()),
        "index_manifest": None if index_manifest is None else str(index_manifest.resolve()),
        "limit": limit,
        "sample_ids": None if sample_ids is None else list(sample_ids),
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
        if not isinstance(existing, dict) or any(
            existing.get(key) != value for key, value in manifest.items()
        ):
            raise ValueError("resume manifest does not exactly match the requested run")
        return
    output.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


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
) -> None:
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
    )
    if not isinstance(workload, EvaluationWorkload):
        raise TypeError("evaluate factory must return EvaluationWorkload")
    if workload.manifest is not None:
        manifest_path = output / "run_manifest.json"
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(existing_manifest, dict):
            raise ValueError("run manifest must be a JSON object")
        existing_manifest.update(workload.manifest)
        existing_manifest.pop("run_manifest_sha256", None)
        existing_manifest["run_manifest_sha256"] = _manifest_digest(existing_manifest)
        manifest_path.write_text(json.dumps(existing_manifest, indent=2, sort_keys=True) + "\n")
    samples = tuple(
        sample if isinstance(sample, SampleInput) else SampleInput.from_mapping(sample)
        for sample in workload.samples
    )
    qids = tuple(sample.question_id for sample in samples)
    if len(qids) != len(set(qids)):
        raise ValueError("evaluation workload contains duplicate question IDs")
    results_path = output / "results.jsonl"
    completed: set[str] = set()
    if resume:
        from docprune.m3docvqa_factory import load_completed_qids

        completed = load_completed_qids(results_path, expected_samples=samples)
        existing_order = []
        if results_path.exists():
            with results_path.open(encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        record = json.loads(line)
                        existing_order.append(record.get("question_id", record.get("qid")))
        if tuple(existing_order) != qids[: len(existing_order)]:
            raise ValueError("resume results must be an exact source-order prefix")
    append_mode = resume and results_path.exists()
    for sample in samples:
        if sample.question_id in completed:
            continue
        result = workload.runner.run_sample(sample)
        append_result_jsonl(results_path, result.to_dict(), resume=append_mode)
        append_mode = True
    (output / "summary.json").write_text(
        json.dumps(summarize_jsonl(results_path), indent=2, sort_keys=True) + "\n"
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
    manifest = _manifest(
        command,
        args.config,
        config,
        args.pages,
        args.factory,
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
    _prepare_output(args.output, manifest, args.resume)
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
        )
    else:
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
        )
        from docprune.indexing import IndexBuildResult

        if not isinstance(result, IndexBuildResult):
            raise TypeError("embed factory must return IndexBuildResult")
        complete = result.manifest.to_dict()
        manifest_path = args.output / "run_manifest.json"
        base = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(base, dict):
            raise ValueError("run manifest must be a JSON object")
        base["index_manifest"] = complete
        base["status"] = "configured"
        base.pop("run_manifest_sha256", None)
        base["run_manifest_sha256"] = _manifest_digest(base)
        manifest_path.write_text(json.dumps(base, indent=2, sort_keys=True) + "\n")
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
