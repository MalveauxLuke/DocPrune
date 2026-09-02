#!/usr/bin/env python3
"""Relocate a verified Task 9 transfer bundle onto H200-local storage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from docprune.task9_transfer import relocate_task9_fixed_inputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--source-fixed-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = relocate_task9_fixed_inputs(
        bundle_root=args.bundle_root,
        source_fixed_root=args.source_fixed_root,
        output_root=args.output_root,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
