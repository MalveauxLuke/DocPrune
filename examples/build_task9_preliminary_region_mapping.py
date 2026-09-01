#!/usr/bin/env python3
"""Compose one Task 9 preliminary mapping from authenticated preprocessing artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from docprune.task8_mapping import publish_task8_region_mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mineru-completion", type=Path, required=True)
    parser.add_argument("--mineru-completion-sha256", required=True)
    parser.add_argument("--geometry-capture", type=Path, required=True)
    parser.add_argument("--geometry-capture-sha256", required=True)
    parser.add_argument("--qid", required=True)
    parser.add_argument("--expected-geometry-count", type=int, required=True)
    parser.add_argument("--expected-geometry-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    mapping = publish_task8_region_mapping(
        mineru_completion_path=args.mineru_completion,
        mineru_completion_sha256=args.mineru_completion_sha256,
        geometry_capture_path=args.geometry_capture,
        geometry_capture_sha256=args.geometry_capture_sha256,
        output_path=args.output,
        expected_qid=args.qid,
        expected_geometry_count=args.expected_geometry_count,
        expected_geometry_sha256=args.expected_geometry_sha256,
        required_geometry_gpu_substring=None,
    )
    print(
        json.dumps(
            {
                "geometry_count": mapping.geometry_count,
                "geometry_sha256": mapping.geometry_sha256,
                "mapping_sha256": mapping.sha256,
                "source_count": len(mapping.sources),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
