#!/usr/bin/env python3
"""Seal an authenticated Task 6 fixed-page fixture without retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from docprune.task6_runtime import build_fixed_page_fixture, publish_fixed_page_fixture


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--eligible-questions", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--pdf-dir", type=Path, required=True)
    parser.add_argument("--fixture-version", required=True)
    parser.add_argument("--pages", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixture = build_fixed_page_fixture(
        reference_path=args.reference,
        eligible_questions_path=args.eligible_questions,
        feature_manifest_path=args.feature_manifest,
        pdf_dir=args.pdf_dir,
        fixture_version=args.fixture_version,
        page_count=args.pages,
    )
    digest = publish_fixed_page_fixture(fixture, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "sha256": digest,
                "question_count": len(fixture.questions),
                "page_count": args.pages,
                "fixed_page_provenance": True,
                "global_index_loaded": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
