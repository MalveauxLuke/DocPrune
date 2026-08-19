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
from docprune.m3docvqa_factory import (
    _normalise_run_config_mapping,
    _validate_run_identity,
    validate_processor_contract_file,
)

RUNTIME_COMMIT = "d5cefb33f7ca97ce0ef2104fa5e63bd3ad8a5761"
MODES = ("all-kept", "docprune")
FIXED_GATE_SAMPLE_IDS = (
    "a33985b1e8b2502fc18cc8147dc27db8",
    "710a6d2254076ea58756c6c7cc211f1e",
    "0d8f2779137fb47db953c4af5247ffe5",
    "e240f5fe65b39eee70d3576cff88fe5a",
    "18ecd2ac6c0ac69993b92dc4b30137e8",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


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


def validate_gate_evidence(gate_path: Path, contract_path: Path) -> dict[str, object]:
    """Authenticate the complete gate evidence before producing final configs."""

    gate_path = Path(gate_path).resolve()
    contract_path = Path(contract_path).resolve()
    if gate_path.is_symlink() or not gate_path.is_file():
        raise ValueError(f"gate JSON is missing or not regular: {gate_path}")
    try:
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("gate JSON is invalid") from error
    if not isinstance(gate, dict):
        raise ValueError("gate JSON must be an object")
    supplied_gate_sha = gate.get("gate_sha256")
    unsigned_gate = dict(gate)
    unsigned_gate.pop("gate_sha256", None)
    if supplied_gate_sha != _sha256_json(unsigned_gate):
        raise ValueError("gate canonical SHA-256 is invalid")
    if gate.get("schema_version") != 1 or gate.get("status") != "passed":
        raise ValueError("gate schema/status is not passed")
    if gate.get("runtime_commit") != RUNTIME_COMMIT:
        raise ValueError("gate runtime commit is not pinned")
    if gate.get("m3docrag_commit") != M3DOCRAG_COMMIT:
        raise ValueError("gate M3DocRAG commit is not pinned")
    if gate.get("page_counts") != [1, 2, 4]:
        raise ValueError("gate page-count set is not exactly [1, 2, 4]")
    if gate.get("sample_ids") != list(FIXED_GATE_SAMPLE_IDS):
        raise ValueError("gate sample IDs are not the exact fixed source-order tuple")
    declared_contract = Path(str(gate.get("processor_contract_path", ""))).resolve()
    if declared_contract != contract_path:
        raise ValueError("gate and processor contract paths differ")
    if contract_path.is_symlink() or not contract_path.is_file():
        raise ValueError("processor contract is missing or not regular")
    validate_processor_contract_file(contract_path)
    contract_sha = _sha256(contract_path)
    if gate.get("processor_contract_sha256") != contract_sha:
        raise ValueError("gate processor-contract digest is invalid")

    semantic_path = Path(str(gate.get("semantic_samples_path", ""))).resolve()
    gate_root = gate_path.parent
    if gate_root not in semantic_path.parents or semantic_path == gate_root:
        raise ValueError("semantic evidence is outside the gate root")
    if semantic_path.is_symlink() or not semantic_path.is_file():
        raise ValueError("semantic evidence is missing or not regular")
    if gate.get("semantic_samples_sha256") != _sha256(semantic_path):
        raise ValueError("semantic evidence digest is invalid")
    try:
        semantic = json.loads(semantic_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("semantic evidence is invalid JSON") from error
    if not isinstance(semantic, dict):
        raise ValueError("semantic evidence must be an object")
    if semantic.get("status") != "passed" or semantic.get("baseline_equivalence") is not True:
        raise ValueError("semantic evidence did not pass baseline equivalence")
    if semantic.get("equivalence_qids") != list(FIXED_GATE_SAMPLE_IDS):
        raise ValueError("semantic evidence qids are not the exact fixed tuple")
    supporting = semantic.get("supporting_documents")
    if (
        not isinstance(supporting, dict)
        or list(supporting) != list(FIXED_GATE_SAMPLE_IDS)
        or any(not isinstance(value, str) or not value for value in supporting.values())
    ):
        raise ValueError("semantic evidence supporting documents are incomplete")
    traces = semantic.get("docprune_traces")
    if not isinstance(traces, dict) or set(traces) != {"1", "2", "4"}:
        raise ValueError("semantic evidence lacks the exact page traces")
    for counts in traces.values():
        if (
            not isinstance(counts, list)
            or len(counts) != 4
            or any(type(value) is not int or value <= 0 for value in counts)
            or any(left < right for left, right in zip(counts, counts[1:]))
        ):
            raise ValueError("semantic evidence has invalid or non-monotonic traces")
    return gate


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
    gate: dict[str, object],
    gate_path: Path,
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
        "gate_path": str(gate_path.resolve()),
        "gate_sha256": gate["gate_sha256"],
        "semantic_samples_path": gate["semantic_samples_path"],
        "semantic_samples_sha256": gate["semantic_samples_sha256"],
        "gate_sample_ids": list(FIXED_GATE_SAMPLE_IDS),
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

    contract = args.processor_contract.resolve()
    gate = validate_gate_evidence(args.gate.resolve(), contract)
    contract_sha256 = _sha256(contract)
    corpus = CorpusIdentity.from_root(args.corpus_root.resolve())
    corpus.validate()
    control = _control_record(args.control_record.resolve())
    if not args.m3docrag_root.is_dir():
        raise SystemExit(f"M3DocRAG checkout is missing: {args.m3docrag_root}")
    _validate_m3docrag_checkout(args.m3docrag_root.resolve())

    if args.output_root.exists():
        raise SystemExit(f"refusing to overwrite existing run-config root: {args.output_root}")
    args.output_root.mkdir(parents=True)
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
                gate=gate,
                gate_path=args.gate,
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
