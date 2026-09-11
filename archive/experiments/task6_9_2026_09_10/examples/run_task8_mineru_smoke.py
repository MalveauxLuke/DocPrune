#!/usr/bin/env python3
"""Prepare or finalize the bounded fixed-page Task 8 MinerU smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from docprune.task8_runtime import (
    finalize_task8_mineru_smoke,
    prepare_task8_mineru_smoke,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "finalize"))
    parser.add_argument("--job-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke-input-manifest", type=Path)
    parser.add_argument("--smoke-input-manifest-sha256")
    parser.add_argument("--configuration", type=Path)
    parser.add_argument("--configuration-sha256")
    parser.add_argument("--tool-manifest", type=Path)
    parser.add_argument("--tool-manifest-sha256")
    parser.add_argument("--model-weights", type=Path)
    parser.add_argument("--model-weights-sha256")
    parser.add_argument("--model-inventory", type=Path)
    parser.add_argument("--model-inventory-sha256")
    parser.add_argument("--runtime-commit")
    parser.add_argument("--gpu-manifest", type=Path)
    parser.add_argument("--gpu-manifest-sha256")
    return parser


def _require(args: argparse.Namespace, names: tuple[str, ...]) -> None:
    missing = [name.replace("_", "-") for name in names if getattr(args, name) is None]
    if missing:
        raise SystemExit(f"missing required arguments for {args.phase}: {', '.join(missing)}")


def main() -> None:
    args = _parser().parse_args()
    if args.phase == "prepare":
        required = (
            "smoke_input_manifest",
            "smoke_input_manifest_sha256",
            "configuration",
            "configuration_sha256",
            "tool_manifest",
            "tool_manifest_sha256",
            "model_weights",
            "model_weights_sha256",
            "model_inventory",
            "model_inventory_sha256",
            "runtime_commit",
        )
        _require(args, required)
        if args.gpu_manifest is not None or args.gpu_manifest_sha256 is not None:
            raise SystemExit("GPU manifest arguments are finalize-only")
        result = prepare_task8_mineru_smoke(
            smoke_input_manifest_path=args.smoke_input_manifest,
            smoke_input_manifest_sha256=args.smoke_input_manifest_sha256,
            configuration_path=args.configuration,
            configuration_sha256=args.configuration_sha256,
            tool_manifest_path=args.tool_manifest,
            tool_manifest_sha256=args.tool_manifest_sha256,
            model_weights_path=args.model_weights,
            model_weights_sha256=args.model_weights_sha256,
            model_inventory_path=args.model_inventory,
            model_inventory_sha256=args.model_inventory_sha256,
            job_root=args.job_root,
            output_dir=args.output_dir,
            runtime_commit=args.runtime_commit,
        )
    else:
        _require(args, ("gpu_manifest", "gpu_manifest_sha256"))
        forbidden = (
            "smoke_input_manifest",
            "smoke_input_manifest_sha256",
            "configuration",
            "configuration_sha256",
            "tool_manifest",
            "tool_manifest_sha256",
            "model_weights",
            "model_weights_sha256",
            "model_inventory",
            "model_inventory_sha256",
            "runtime_commit",
        )
        if any(getattr(args, name) is not None for name in forbidden):
            raise SystemExit("prepare-only arguments cannot be used for finalize")
        result = finalize_task8_mineru_smoke(
            job_root=args.job_root,
            output_dir=args.output_dir,
            gpu_manifest_path=args.gpu_manifest,
            gpu_manifest_sha256=args.gpu_manifest_sha256,
        )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
