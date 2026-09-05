#!/usr/bin/env python3
"""Compose admitted Task 8 artifacts into one cached region/token mapping."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from docprune.task8_mapping import publish_task8_region_mapping

EXPECTED_QID = "e1e6ed53f9ad11813845088f4cf2f6b1"
EXPECTED_GEOMETRY_COUNT = 3586
EXPECTED_GEOMETRY_SHA256 = "47b32cf6156dcee41da7eab1686219c760dd73803a135534546c1a50519086e8"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mineru-completion", type=Path, required=True)
    parser.add_argument("--mineru-completion-sha256", required=True)
    parser.add_argument("--geometry-capture", type=Path, required=True)
    parser.add_argument("--geometry-capture-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    mapping = publish_task8_region_mapping(
        mineru_completion_path=args.mineru_completion,
        mineru_completion_sha256=args.mineru_completion_sha256,
        geometry_capture_path=args.geometry_capture,
        geometry_capture_sha256=args.geometry_capture_sha256,
        output_path=args.output,
        expected_qid=EXPECTED_QID,
        expected_geometry_count=EXPECTED_GEOMETRY_COUNT,
        expected_geometry_sha256=EXPECTED_GEOMETRY_SHA256,
    )
    print(
        json.dumps(
            {
                "geometry_count": mapping.geometry_count,
                "mapping_sha256": mapping.sha256,
                "source_count": len(mapping.sources),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
