"""Command-line entry point for local inspection and SOL evaluation."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from docprune.config import DocPruneConfig, load_config
from docprune.m3docrag import DocPruneM3DocRAG, SampleInput
from docprune.metrics import append_result_jsonl, summarize_jsonl


@dataclass(frozen=True)
class EvaluationWorkload:
    runner: DocPruneM3DocRAG
    samples: Iterable[SampleInput | dict[str, Any]]


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
        run.add_argument("--factory", required=True, help="Python module:function integration factory")
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


def _manifest(command: str, config_path: Path, config: DocPruneConfig, pages: int, factory: str):
    return {
        "schema_version": 1,
        "status": "configured",
        "command": command,
        "config": str(config_path.resolve()),
        "factory": factory,
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
        if not resume:
            raise FileExistsError(f"output directory already exists: {output}")
        if not manifest_path.is_file() or json.loads(manifest_path.read_text()) != manifest:
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
) -> None:
    workload = factory(operation="evaluate", config=config, page_count=pages, output=output)
    if not isinstance(workload, EvaluationWorkload):
        raise TypeError("evaluate factory must return EvaluationWorkload")
    results_path = output / "results.jsonl"
    append_mode = resume and results_path.exists()
    for sample in workload.samples:
        result = workload.runner.run_sample(sample)
        append_result_jsonl(results_path, result.to_dict(), resume=append_mode)
        append_mode = True
    (output / "summary.json").write_text(
        json.dumps(summarize_jsonl(results_path), indent=2, sort_keys=True) + "\n"
    )


def _run_external(command: str, args: argparse.Namespace) -> int:
    config = load_config(args.config)
    config.for_pages(args.pages)
    manifest = _manifest(command, args.config, config, args.pages, args.factory)
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"output directory already exists: {args.output}")
    if args.dry_run:
        print(json.dumps({**manifest, "status": "dry_run"}, indent=2, sort_keys=True))
        return 0
    _prepare_output(args.output, manifest, args.resume)
    factory = _load_factory(args.factory)
    if command == "evaluate":
        _run_evaluate(factory, config, args.pages, args.output, resume=args.resume)
    else:
        factory(operation="embed", config=config, page_count=args.pages, output=args.output)
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
