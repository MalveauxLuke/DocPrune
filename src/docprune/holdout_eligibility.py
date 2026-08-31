"""Project sealed cached DocPrune pages into outcome-blind holdout inputs."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from collections.abc import Mapping
from pathlib import Path

from docprune.experiment_design import (
    _authenticate_eligibility_records,
    _canonical_json_sha256,
    build_eligibility_records,
)

_SINGLE_HOP_TYPES = {"TextQ", "TableQ", "ImageQ", "ImageListQ"}
_DEVELOPMENT_PROJECTION_POLICY = "explicit QID fields only; outcome fields forbidden"


def _read_regular(path: Path, expected_sha256: str, label: str) -> bytes:
    file_path = Path(path)
    if not file_path.is_absolute() or file_path.is_symlink():
        raise ValueError(f"{label} must be an absolute regular file")
    descriptor: int | None = None
    try:
        descriptor = os.open(file_path, os.O_RDONLY | os.O_NOFOLLOW)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"{label} must be an absolute regular file")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        raw = b"".join(chunks)
    except OSError as error:
        raise ValueError(f"{label} must be an absolute regular file") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch")
    return raw


def _json_object(raw: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _jsonl_objects(raw: bytes, label: str) -> list[dict[str, object]]:
    try:
        values = [json.loads(line) for line in raw.decode().splitlines() if line]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid JSONL") from error
    if any(not isinstance(value, dict) for value in values):
        raise ValueError(f"{label} contains a non-object")
    return values


def _sha256_regular(path: Path, label: str) -> str:
    file_path = Path(path)
    if not file_path.is_absolute() or file_path.is_symlink() or not file_path.is_file():
        raise ValueError(f"{label} must be an absolute regular file")
    digest = hashlib.sha256()
    with file_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _development_qids(
    path: Path,
    expected_file_sha256: str,
) -> tuple[dict[str, object], tuple[str, ...]]:
    registry = _json_object(
        _read_regular(path, expected_file_sha256, "development registry"),
        "development registry",
    )
    observed_registry_sha256 = registry.get("registry_sha256")
    unsigned = dict(registry)
    unsigned.pop("registry_sha256", None)
    qids = registry.get("union_qids")
    if (
        registry.get("schema_version") != 1
        or registry.get("status") != "complete"
        or registry.get("projection_policy") != _DEVELOPMENT_PROJECTION_POLICY
        or not isinstance(qids, list)
        or not qids
        or any(not isinstance(qid, str) or not qid for qid in qids)
        or len(qids) != len(set(qids))
        or registry.get("union_count") != len(qids)
        or observed_registry_sha256 != _canonical_json_sha256(unsigned)
    ):
        raise ValueError("development registry identity is invalid")
    return registry, tuple(qids)


def project_cached_holdout_eligibility(
    *,
    questions_path: Path,
    questions_sha256: str,
    cached_results_path: Path,
    cached_results_sha256: str,
    quality_manifest_path: Path,
    quality_manifest_sha256: str,
    index_manifest_path: Path,
    index_manifest_sha256: str,
    completion_ledger_path: Path,
    completion_ledger_sha256: str,
    development_registry_path: Path,
    development_registry_file_sha256: str,
    pdf_dir: Path,
) -> dict[str, object]:
    """Use only cached page identities, never result outcomes or retrieval scores."""

    questions_raw = _read_regular(questions_path, questions_sha256, "question source")
    results_raw = _read_regular(cached_results_path, cached_results_sha256, "cached result")
    quality = _json_object(
        _read_regular(quality_manifest_path, quality_manifest_sha256, "quality manifest"),
        "quality manifest",
    )
    index = _json_object(
        _read_regular(index_manifest_path, index_manifest_sha256, "index manifest"),
        "index manifest",
    )
    completion_raw = _read_regular(
        completion_ledger_path,
        completion_ledger_sha256,
        "completion ledger",
    )
    development_registry, development_qids = _development_qids(
        development_registry_path,
        development_registry_file_sha256,
    )
    if (
        quality.get("schema_version") != 1
        or quality.get("scope") != "quality-only"
        or quality.get("mode") != "docprune"
        or quality.get("page_count") != 4
        or quality.get("results_sha256") != cached_results_sha256
    ):
        raise ValueError("quality manifest is not the sealed DocPrune top-4 result")
    if (
        index.get("mode") != "docprune"
        or index.get("page_count") != 4
        or index.get("completion_ledger_path") != str(completion_ledger_path)
        or index.get("completion_ledger_sha256") != completion_ledger_sha256
    ):
        raise ValueError("index manifest is not the persisted DocPrune top-4 feature source")
    pdf_root = Path(pdf_dir)
    if not pdf_root.is_absolute() or pdf_root.is_symlink() or not pdf_root.is_dir():
        raise ValueError("PDF root must be an absolute real directory")

    questions = _jsonl_objects(questions_raw, "question source")
    results = _jsonl_objects(results_raw, "cached results")
    try:
        completion = json.loads(completion_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("completion ledger is invalid JSON") from error
    if not isinstance(completion, list) or not completion:
        raise ValueError("completion ledger must be a nonempty list")

    quality_qids = quality.get("question_ids")
    if (
        not isinstance(quality_qids, list)
        or quality.get("question_count") != len(quality_qids)
        or len(results) != len(quality_qids)
        or [row.get("question_id") for row in results] != quality_qids
    ):
        raise ValueError("cached result QIDs do not match the quality manifest")
    question_by_qid: dict[str, dict[str, object]] = {}
    for row in questions:
        qid = row.get("qid")
        if not isinstance(qid, str) or not qid or qid in question_by_qid:
            raise ValueError("question source QIDs are invalid")
        question_by_qid[qid] = row
    if set(quality_qids) != set(question_by_qid):
        raise ValueError("cached result QIDs do not match the question source")

    feature_by_page: dict[tuple[str, int], tuple[str, str]] = {}
    for entry in completion:
        if not isinstance(entry, Mapping):
            raise ValueError("completion ledger entry is invalid")
        feature_path = entry.get("document_path")
        feature_sha = entry.get("sha256")
        pages = entry.get("pages")
        if (
            not isinstance(feature_path, str)
            or not feature_path.startswith("/")
            or not isinstance(feature_sha, str)
            or len(feature_sha) != 64
            or not isinstance(pages, list)
            or not pages
        ):
            raise ValueError("completion ledger feature identity is invalid")
        for page in pages:
            if not isinstance(page, Mapping):
                raise ValueError("completion ledger page identity is invalid")
            identity = (page.get("doc_id"), page.get("page_index"))
            if (
                not isinstance(identity[0], str)
                or not identity[0]
                or type(identity[1]) is not int
                or identity in feature_by_page
            ):
                raise ValueError("completion ledger page identity is invalid")
            feature_by_page[identity] = (feature_path, feature_sha)

    source_rows: list[dict[str, object]] = []
    pdf_hashes: dict[str, str] = {}
    for result in results:
        qid = result["question_id"]
        source = question_by_qid[qid]
        metadata = source.get("metadata")
        supports = source.get("supporting_context")
        cached_pages = result.get("retrieved_pages")
        if (
            not isinstance(metadata, Mapping)
            or not isinstance(supports, list)
            or not supports
            or not isinstance(cached_pages, list)
            or len(cached_pages) != 4
        ):
            raise ValueError("holdout source metadata or cached pages are invalid")
        support_ids = []
        for support in supports:
            doc_id = support.get("doc_id") if isinstance(support, Mapping) else None
            if not isinstance(doc_id, str) or not doc_id:
                raise ValueError("holdout support-document identity is invalid")
            if doc_id not in support_ids:
                support_ids.append(doc_id)
        pages: list[dict[str, object]] = []
        features: list[dict[str, object]] = []
        for page in cached_pages:
            doc_id = page.get("doc_id") if isinstance(page, Mapping) else None
            page_index = page.get("page_index") if isinstance(page, Mapping) else None
            identity = (doc_id, page_index)
            if not isinstance(doc_id, str) or type(page_index) is not int:
                raise ValueError("cached DocPrune page identity is invalid")
            try:
                feature_path, feature_sha = feature_by_page[identity]
            except KeyError as error:
                raise ValueError("cached DocPrune page lacks persisted features") from error
            pdf_path = pdf_root / f"{doc_id}.pdf"
            if doc_id not in pdf_hashes:
                pdf_hashes[doc_id] = _sha256_regular(pdf_path, "cached page PDF")
            pages.append(
                {
                    "doc_id": doc_id,
                    "page_index": page_index,
                    "source_path": str(pdf_path),
                    "source_sha256": pdf_hashes[doc_id],
                }
            )
            features.append(
                {
                    "doc_id": doc_id,
                    "page_index": page_index,
                    "feature_path": feature_path,
                    "feature_sha256": feature_sha,
                    "index_manifest_path": str(index_manifest_path),
                    "index_manifest_sha256": index_manifest_sha256,
                }
            )
        source_rows.append(
            {
                "qid": qid,
                "metadata": {
                    "type": "single_hop" if metadata.get("type") in _SINGLE_HOP_TYPES else "other",
                    "supporting_document_ids": support_ids,
                    "source_path": str(questions_path),
                    "source_sha256": questions_sha256,
                },
                "cached_pages": pages,
                "persisted_features": features,
            }
        )
    eligible = build_eligibility_records(source_rows, development_qids=development_qids)
    _authenticate_eligibility_records(eligible)
    development_set = set(development_qids)
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "outcome-blind-cached-page-eligibility",
        "cached_page_source": "sealed-historical-docprune-top4-quality-run",
        "projection_fields": [
            "qid",
            "source/support metadata",
            "cached ordered page identities",
            "persisted feature identities",
        ],
        "forbidden_fields": [
            "question",
            "answers",
            "predicted_answer",
            "retrieval score",
            "timing",
            "trace",
            "quality metrics",
        ],
        "global_index_loaded": False,
        "retrieval_run": False,
        "development_qids_registered": len(development_set),
        "development_qids_excluded": sum(qid in development_set for qid in quality_qids),
        "eligible_count": len(eligible),
        "source_files": {
            "questions": {"path": str(questions_path), "sha256": questions_sha256},
            "cached_results": {
                "path": str(cached_results_path),
                "sha256": cached_results_sha256,
            },
            "quality_manifest": {
                "path": str(quality_manifest_path),
                "sha256": quality_manifest_sha256,
            },
            "index_manifest": {
                "path": str(index_manifest_path),
                "sha256": index_manifest_sha256,
            },
            "completion_ledger": {
                "path": str(completion_ledger_path),
                "sha256": completion_ledger_sha256,
            },
            "development_registry": {
                "path": str(development_registry_path),
                "file_sha256": development_registry_file_sha256,
                "registry_sha256": development_registry["registry_sha256"],
            },
        },
        "eligible_records": list(eligible),
    }
    payload["projection_sha256"] = _canonical_json_sha256(payload)
    return payload


def publish_cached_holdout_eligibility(projection: Mapping[str, object], destination: Path) -> str:
    """Publish one internally authenticated projection without replacement."""

    unsigned = dict(projection)
    observed = unsigned.pop("projection_sha256", None)
    if (
        projection.get("schema_version") != 1
        or projection.get("status") != "outcome-blind-cached-page-eligibility"
        or observed != _canonical_json_sha256(unsigned)
    ):
        raise ValueError("cached holdout eligibility projection identity is invalid")
    output = Path(destination)
    if not output.is_absolute():
        raise ValueError("cached holdout eligibility output must be absolute")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"cached holdout eligibility output exists: {output}")
    payload = (json.dumps(projection, indent=2, sort_keys=True) + "\n").encode()
    descriptor, staging_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    staging = Path(staging_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(staging, output, follow_symlinks=False)
        except FileExistsError as error:
            raise FileExistsError(f"cached holdout eligibility output exists: {output}") from error
        directory = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        staging.unlink(missing_ok=True)
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "project_cached_holdout_eligibility",
    "publish_cached_holdout_eligibility",
]
