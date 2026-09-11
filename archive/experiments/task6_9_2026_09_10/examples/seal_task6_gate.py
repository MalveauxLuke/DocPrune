#!/usr/bin/env python3
"""Seal the Task 6 QID shards and closed policy matrices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from docprune.task6_runtime import (
    build_task6_gate_manifest,
    load_fixed_page_fixture,
    publish_task6_gate_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--fixture-sha256", required=True)
    parser.add_argument("--smoke-qid", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixture = load_fixed_page_fixture(args.fixture, expected_sha256=args.fixture_sha256)
    manifest = build_task6_gate_manifest(
        fixture,
        fixture_path=args.fixture.resolve(),
        fixture_sha256=args.fixture_sha256,
        smoke_qid=args.smoke_qid,
    )
    digest = publish_task6_gate_manifest(manifest, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "sha256": digest,
                "qid_shards": len(manifest["qid_shards"]),
                "native_cells": len(manifest["native_cells"]),
                "fixed_cells": len(manifest["fixed_cells"]),
                "smoke_cells": len(manifest["smoke_cells"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
