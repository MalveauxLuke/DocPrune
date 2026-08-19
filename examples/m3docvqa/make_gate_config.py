#!/usr/bin/env python3
"""Write the complete pre-gate config whose contract is produced by the gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from docprune.benchmark_config import (
    COLPALI_BACKBONE_MODEL,
    COLPALI_BACKBONE_REVISION,
    COLPALI_MODEL,
    COLPALI_REVISION,
    M3DOCRAG_COMMIT,
    MAX_NEW_TOKENS,
    QWEN_MODEL,
    QWEN_REVISION,
    SHORT_ANSWER_TEMPLATE,
    CorpusIdentity,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--processor-contract", type=Path, required=True)
    parser.add_argument("--m3docrag-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    corpus = CorpusIdentity.from_root(args.corpus_root.resolve())
    payload = {
        "mode": "all-kept",
        "page_count": 1,
        "runtime_commit": "d5cefb33f7ca97ce0ef2104fa5e63bd3ad8a5761",
        "m3docrag_commit": M3DOCRAG_COMMIT,
        "m3docrag_root": str(args.m3docrag_root.resolve()),
        "qwen_model": QWEN_MODEL,
        "qwen_revision": QWEN_REVISION,
        "colpali_model": COLPALI_MODEL,
        "colpali_revision": COLPALI_REVISION,
        "colpali_backbone_model": COLPALI_BACKBONE_MODEL,
        "colpali_backbone_revision": COLPALI_BACKBONE_REVISION,
        "processor_contract_path": str(args.processor_contract.resolve()),
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "num_beams": 1,
        "prompt": SHORT_ANSWER_TEMPLATE,
        "corpus": {
            "root": str(corpus.root.resolve()),
            "questions_path": str(corpus.questions_path.resolve()),
            "document_ids_path": str(corpus.document_ids_path.resolve()),
            "pdf_dir": str(corpus.pdf_dir.resolve()),
            "integrity_report_path": str(corpus.integrity_report_path.resolve()),
            "archive_checksum_manifest_path": str(corpus.archive_checksum_manifest_path.resolve()),
            "integrity_sha256": corpus.integrity_sha256,
            "archive_checksum_manifest_sha256": corpus.archive_checksum_manifest_sha256,
            "questions_sha256": corpus.questions_sha256,
            "document_ids_sha256": corpus.document_ids_sha256,
            "expected_question_count": corpus.expected_question_count,
            "expected_pdf_count": corpus.expected_pdf_count,
            "expected_page_count": corpus.expected_page_count,
            "is_fixture": corpus.is_fixture,
            "archive_hashes": dict(corpus.archive_hashes),
        },
    }
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing gate config: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
