#!/usr/bin/env python3
"""Seal the Task 6 confirmatory gate from an already validated method holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from docprune.experiment_design import validate_holdout_for_launch
from docprune.task6_runtime import (
    build_task6_holdout_gate_manifest,
    load_fixed_page_fixture,
    publish_task6_gate_manifest,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--fixture-sha256", required=True)
    parser.add_argument("--holdout-root", type=Path, required=True)
    parser.add_argument("--development-registry", type=Path, required=True)
    parser.add_argument("--required-label", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    holdout = validate_holdout_for_launch(
        args.holdout_root,
        required_registry_path=args.development_registry,
        required_registry_labels=tuple(args.required_label),
    )
    holdout_manifest_path = (args.holdout_root / "manifest.json").resolve()
    fixture = load_fixed_page_fixture(args.fixture, expected_sha256=args.fixture_sha256)
    manifest = build_task6_holdout_gate_manifest(
        fixture,
        fixture_path=args.fixture.resolve(),
        fixture_sha256=args.fixture_sha256,
        holdout_manifest=holdout,
        holdout_manifest_path=holdout_manifest_path,
        holdout_manifest_file_sha256=_sha256(holdout_manifest_path),
    )
    digest = publish_task6_gate_manifest(manifest, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "sha256": digest,
                "qid_shards": len(manifest["qid_shards"]),
                "holdout_primary_cells": len(manifest["holdout_primary_cells"]),
                "fixed_page_provenance": True,
                "global_index_loaded": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
