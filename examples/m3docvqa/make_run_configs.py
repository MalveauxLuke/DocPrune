#!/usr/bin/env python3
"""Create and validate the six immutable production M3DocVQA run configs."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from docprune.benchmark_config import (
    COLPALI_BACKBONE_MODEL,
    COLPALI_BACKBONE_REVISION,
    COLPALI_MODEL,
    COLPALI_REVISION,
    M3DOCRAG_COMMIT,
    MAX_NEW_TOKENS,
    PAGE_COUNTS,
    QWEN_MODEL,
    QWEN_REVISION,
    SHORT_ANSWER_TEMPLATE,
    CorpusIdentity,
)
from docprune.m3docvqa_factory import _normalise_run_config_mapping, _validate_run_identity
from docprune.processor_probe import validate_processor_contract

RUNTIME_COMMIT = "d5cefb33f7ca97ce0ef2104fa5e63bd3ad8a5761"
MODES = ("all-kept", "docprune")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _corpus_payload(corpus: CorpusIdentity) -> dict[str, object]:
    return {
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
    }


def _control_record(path: Path) -> dict[str, object]:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"invalid sealed control record: {path}") from error
    if not isinstance(record, dict) or record.get("schema_version") != 1:
        raise SystemExit("sealed control record must have schema_version=1")
    if not isinstance(record.get("control_commit"), str) or len(record["control_commit"]) != 40:
        raise SystemExit("sealed control record lacks the full control commit")
    if not isinstance(record.get("tree_sha"), str) or len(record["tree_sha"]) != 40:
        raise SystemExit("sealed control record lacks the full Git tree SHA")
    return record


def _validate_m3docrag_checkout(path: Path) -> None:
    try:
        revision = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"unable to inspect M3DocRAG checkout: {path}") from error
    if revision != M3DOCRAG_COMMIT:
        raise SystemExit(f"M3DocRAG checkout revision mismatch: {revision}")
    unexpected = []
    for line in dirty.splitlines():
        relative = line[3:] if line.startswith("?? ") else ""
        if not (line.startswith("?? ") and "__pycache__/" in relative and relative.endswith((".pyc", ".pyo"))):
            unexpected.append(line)
    if unexpected:
        raise SystemExit(f"M3DocRAG checkout has non-bytecode changes: {unexpected[0]}")


def _payload(
    *,
    mode: str,
    page_count: int,
    corpus: CorpusIdentity,
    contract: Path,
    contract_sha256: str,
    m3docrag_root: Path,
    control_record: Path,
    control: dict[str, object],
) -> dict[str, object]:
    return {
        "mode": mode,
        "page_count": page_count,
        "runtime_commit": RUNTIME_COMMIT,
        "m3docrag_commit": M3DOCRAG_COMMIT,
        "m3docrag_root": str(m3docrag_root.resolve()),
        "qwen_model": QWEN_MODEL,
        "qwen_revision": QWEN_REVISION,
        "colpali_model": COLPALI_MODEL,
        "colpali_revision": COLPALI_REVISION,
        "colpali_backbone_model": COLPALI_BACKBONE_MODEL,
        "colpali_backbone_revision": COLPALI_BACKBONE_REVISION,
        "processor_contract_path": str(contract.resolve()),
        "processor_contract_sha256": contract_sha256,
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "num_beams": 1,
        "prompt": SHORT_ANSWER_TEMPLATE,
        "corpus": _corpus_payload(corpus),
        "control_record": str(control_record.resolve()),
        "control_commit": control["control_commit"],
        "control_tree_sha": control["tree_sha"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--processor-contract", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--control-record", type=Path, required=True)
    parser.add_argument("--m3docrag-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    corpus = CorpusIdentity.from_root(args.corpus_root.resolve())
    corpus.validate()
    contract = args.processor_contract.resolve()
    contract_payload = json.loads(contract.read_text(encoding="utf-8"))
    if not isinstance(contract_payload, dict):
        raise SystemExit("processor contract must be a JSON object")
    validate_processor_contract(contract_payload)
    contract_sha256 = _sha256(contract)
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    if not isinstance(gate, dict) or gate.get("status") != "passed":
        raise SystemExit("run configs require a passed semantic gate")
    if Path(str(gate.get("processor_contract_path", ""))).resolve() != contract:
        raise SystemExit("gate and processor contract paths differ")
    if gate.get("processor_contract_sha256") != contract_sha256:
        raise SystemExit("gate and processor contract digests differ")
    if contract_payload.get("schema_version") != 2:
        raise SystemExit("processor contract schema is not 2")
    control = _control_record(args.control_record.resolve())
    if not args.m3docrag_root.is_dir():
        raise SystemExit(f"M3DocRAG checkout is missing: {args.m3docrag_root}")
    _validate_m3docrag_checkout(args.m3docrag_root.resolve())

    args.output_root.mkdir(parents=True, exist_ok=True)
    for mode in MODES:
        for page_count in PAGE_COUNTS:
            payload = _payload(
                mode=mode,
                page_count=page_count,
                corpus=corpus,
                contract=contract,
                contract_sha256=contract_sha256,
                m3docrag_root=args.m3docrag_root,
                control_record=args.control_record,
                control=control,
            )
            resolved = _normalise_run_config_mapping(payload)
            if str(resolved.mode) != mode or int(resolved.page_count) != page_count:
                raise SystemExit("run config selector normalization changed mode/page")
            _validate_run_identity(resolved, mode=mode, page_count=page_count)
            output = args.output_root / f"{mode}-top{page_count}.json"
            if output.exists():
                raise SystemExit(f"refusing to overwrite existing run config: {output}")
            output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            # Re-read the exact bytes written and validate the production loader path.
            _validate_run_identity(
                _normalise_run_config_mapping(json.loads(output.read_text(encoding="utf-8"))),
                mode=mode,
                page_count=page_count,
            )
    print(json.dumps({"status": "passed", "configs": 6, "output_root": str(args.output_root.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
