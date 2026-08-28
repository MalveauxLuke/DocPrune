#!/usr/bin/env python3
"""Render authenticated Task 8 region and post-QTP token-mask overlays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from docprune.task8_overlay import publish_task8_region_overlays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = publish_task8_region_overlays(args.mapping, args.output_dir)
    print(json.dumps(manifest, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
