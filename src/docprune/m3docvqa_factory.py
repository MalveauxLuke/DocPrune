"""Concrete, fail-closed integration factory for the pinned M3DocVQA run."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from docprune.answerers import AllKeptQwenAnswerer, DocPruneQwenAnswerer
from docprune.artifacts import IndexManifest
from docprune.benchmark_config import (
    COLPALI_BACKBONE_MODEL,
    COLPALI_BACKBONE_REVISION,
    COLPALI_MODEL,
    COLPALI_REVISION,
    M3DOCRAG_COMMIT,
    MODES,
    PAGE_COUNTS,
    QWEN_MODEL,
    QWEN_REVISION,
    BenchmarkRunConfig,
    sha256_file,
)
from docprune.cli import EvaluationWorkload
from docprune.colpali.compat import assert_supported_colpali
from docprune.config import DocPruneConfig
from docprune.indexing import IndexBuildConfig, IndexBuildResult, build_index
from docprune.m3docrag import DocPruneM3DocRAG, OfficialM3DocRAGBoundary, SampleInput
from docprune.processor_probe import validate_processor_contract

DEFAULT_FACTORY = "docprune.m3docvqa_factory:build_workload"


def _value(container: object, name: str, default: object = None) -> Any:
    if isinstance(container, Mapping):
        return container.get(name, default)
    return getattr(container, name, default)


def _required(container: object, name: str) -> Any:
    value = _value(container, name)
    if value is None:
        raise ValueError(f"run configuration is missing {name}")
    return value


def _sha256_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_processor_contract_file(path: Path) -> dict[str, object]:
    """Read and validate the immutable processor-probe contract."""

    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"processor contract is missing or not regular: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"processor contract is not JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError("processor contract must be a JSON object")
    if payload.get("schema_version") != 2:
        raise ValueError("processor contract schema_version must equal 2")
    validate_processor_contract(payload)
    resources = payload.get("resources")
    if not isinstance(resources, Mapping):
        raise ValueError("processor contract is missing resources")
    return payload


def filter_samples(
    samples: Iterable[SampleInput],
    *,
    limit: int | None = None,
    sample_ids: Sequence[str] | None = None,
) -> tuple[SampleInput, ...]:
    """Select source-order samples without changing the immutable corpus order."""

    values = tuple(samples)
    source_ids = tuple(sample.question_id for sample in values)
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("source samples contain duplicate question IDs")
    requested = None if sample_ids is None else tuple(str(value) for value in sample_ids)
    if requested is not None and len(requested) != len(set(requested)):
        raise ValueError("sample IDs contain a duplicate")
    requested_set = None if requested is None else set(requested)
    if requested_set is not None:
        unknown = requested_set - set(source_ids)
        if unknown:
            raise ValueError(
                f"sample IDs are not present in the source corpus: {sorted(unknown)!r}"
            )
    selected = tuple(
        sample for sample in values if requested_set is None or sample.question_id in requested_set
    )
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        selected = selected[:limit]
    return selected


def load_completed_qids(
    path: Path,
    *,
    expected_qids: Sequence[str] | None = None,
    expected_samples: Sequence[SampleInput] | None = None,
) -> set[str]:
    """Validate a result JSONL's qid set before permitting resume."""

    path = Path(path)
    if not path.exists():
        return set()
    if expected_qids is None and expected_samples is None:
        raise ValueError("resume validation requires expected question IDs or samples")
    expected_by_qid = (
        {sample.question_id: sample for sample in expected_samples}
        if expected_samples is not None
        else {}
    )
    expected = set(expected_qids or expected_by_qid)
    completed: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"results JSONL contains a blank line at {line_number}")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"results JSONL is invalid at line {line_number}") from error
            if not isinstance(record, Mapping):
                raise ValueError(f"results JSONL record {line_number} is not an object")
            qid = record.get("question_id", record.get("qid"))
            if not isinstance(qid, str) or not qid:
                raise ValueError(f"results JSONL record {line_number} is missing question_id")
            if qid not in expected:
                raise ValueError(f"results JSONL contains unexpected question ID: {qid}")
            if qid in completed:
                raise ValueError(f"results JSONL contains duplicate question ID: {qid}")
            if qid in expected_by_qid:
                sample = expected_by_qid[qid]
                if record.get("question") != sample.question:
                    raise ValueError(f"results JSONL question drift for question ID: {qid}")
                answers = record.get("answers")
                if (
                    not isinstance(answers, list)
                    or tuple(str(value) for value in answers) != sample.answers
                ):
                    raise ValueError(f"results JSONL answers drift for question ID: {qid}")
            completed.add(qid)
    return completed


def _resolve_mode(mode: str | None, run_config: object | None) -> str:
    resolved = mode or _value(run_config, "mode") or os.environ.get("DOCPRUNE_MODE", "all-kept")
    if resolved not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}")
    return str(resolved)


def _resolve_run_config(
    run_config: BenchmarkRunConfig | Path | Mapping[str, object] | None,
    *,
    mode: str,
    page_count: int,
) -> object:
    if isinstance(run_config, BenchmarkRunConfig):
        resolved: object = run_config
    elif run_config is None:
        resolved = BenchmarkRunConfig.from_env(mode, page_count)
    elif isinstance(run_config, str | Path):
        path = Path(run_config)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"run configuration is missing: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError("run configuration file must be JSON") from error
        if not isinstance(payload, Mapping):
            raise ValueError("run configuration file must contain a JSON object")
        # The run-config file is an identity assertion; production construction
        # remains centralized in BenchmarkRunConfig.from_env so corpus checks
        # cannot be accidentally bypassed by a partial JSON file.
        resolved = BenchmarkRunConfig.from_env(mode, page_count)
        for name in ("runtime_commit", "m3docrag_commit", "qwen_revision", "colpali_revision"):
            if name in payload and str(payload[name]) != str(_required(resolved, name)):
                raise ValueError(f"run configuration {name} does not match the pinned value")
    elif isinstance(run_config, Mapping):
        resolved = SimpleNamespace(**run_config)
    else:
        resolved = run_config
    if str(_required(resolved, "mode")) != mode:
        raise ValueError("run configuration mode does not match requested mode")
    if int(_required(resolved, "page_count")) != page_count:
        raise ValueError("run configuration page_count does not match requested page_count")
    return resolved


def _validate_run_identity(run_config: object, *, mode: str, page_count: int) -> dict[str, object]:
    if str(_required(run_config, "m3docrag_commit")) != M3DOCRAG_COMMIT:
        raise ValueError(f"m3docrag checkout must use pinned commit {M3DOCRAG_COMMIT}")
    runtime_commit = str(_required(run_config, "runtime_commit"))
    if len(runtime_commit) != 40 or any(
        char not in "0123456789abcdefABCDEF" for char in runtime_commit
    ):
        raise ValueError("runtime_commit must be a 40-character hexadecimal commit hash")
    expected_resources = {
        "qwen_model": QWEN_MODEL,
        "qwen_revision": QWEN_REVISION,
        "colpali_model": COLPALI_MODEL,
        "colpali_revision": COLPALI_REVISION,
        "colpali_backbone_model": COLPALI_BACKBONE_MODEL,
        "colpali_backbone_revision": COLPALI_BACKBONE_REVISION,
    }
    for name, expected in expected_resources.items():
        if str(_required(run_config, name)) != expected:
            raise ValueError(f"{name} does not match its pinned revision/resource")
    contract = Path(_required(run_config, "processor_contract_path"))
    contract_payload = validate_processor_contract_file(contract)
    corpus = _value(run_config, "corpus")
    if corpus is None:
        raise ValueError("run configuration is missing corpus identity")
    validate = getattr(corpus, "validate", None)
    if callable(validate):
        validate()
    return {
        "mode": mode,
        "page_count": page_count,
        "runtime_commit": runtime_commit.lower(),
        "m3docrag_commit": M3DOCRAG_COMMIT,
        "resources": {
            "qwen": {"model": QWEN_MODEL, "revision": QWEN_REVISION},
            "colpali": {"model": COLPALI_MODEL, "revision": COLPALI_REVISION},
            "colpali_backbone": {
                "model": COLPALI_BACKBONE_MODEL,
                "revision": COLPALI_BACKBONE_REVISION,
            },
        },
        "processor_contract_path": str(contract.resolve()),
        "processor_contract_sha256": sha256_file(contract),
        "processor_contract": contract_payload,
        "corpus": {
            "root": str(Path(_required(corpus, "root")).resolve()),
            "integrity_sha256": str(_required(corpus, "integrity_sha256")),
            "questions_sha256": str(_required(corpus, "questions_sha256")),
            "document_ids_sha256": str(_required(corpus, "document_ids_sha256")),
            "expected_question_count": int(_required(corpus, "expected_question_count")),
            "expected_pdf_count": int(_required(corpus, "expected_pdf_count")),
            "expected_page_count": int(_required(corpus, "expected_page_count")),
        },
        "generation": {
            "max_new_tokens": int(_value(run_config, "max_new_tokens", 128)),
            "do_sample": bool(_value(run_config, "do_sample", False)),
            "num_beams": int(_value(run_config, "num_beams", 1)),
            "prompt": str(_value(run_config, "prompt", "question: $question\noutput only answer.")),
        },
    }


def _validate_m3docrag_checkout(run_config: object) -> Path:
    root_value = (
        _value(run_config, "m3docrag_root")
        or _value(run_config, "upstream_root")
        or os.environ.get("DOCPRUNE_M3DOCRAG_ROOT")
        or os.environ.get("M3DOCRAG_ROOT")
        or "/home/lmalveau/src/m3docrag-runtime-29e6ac2"
    )
    root = Path(root_value)
    if not root.is_dir():
        raise FileNotFoundError(f"pinned M3DocRAG checkout is missing: {root}")
    try:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"unable to inspect pinned M3DocRAG checkout: {root}") from error
    if revision.lower() != M3DOCRAG_COMMIT:
        raise ValueError(
            f"M3DocRAG checkout revision mismatch: expected {M3DOCRAG_COMMIT}, got {revision}"
        )
    if dirty:
        raise ValueError(f"M3DocRAG checkout is dirty: {root}")
    return root


def _cached_snapshot(repo_id: str, revision: str) -> Path:
    """Resolve only an already-cached immutable HF snapshot."""

    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(repo_id=repo_id, revision=revision, local_files_only=True)
        return Path(path)
    except Exception as error:
        raise FileNotFoundError(
            f"cached model snapshot is missing for {repo_id}@{revision}"
        ) from error


def _load_colpali(run_config: object) -> tuple[object, object]:
    from colpali_engine.models import ColPali, ColPaliProcessor

    backbone = _cached_snapshot(COLPALI_BACKBONE_MODEL, COLPALI_BACKBONE_REVISION)
    adapter = _cached_snapshot(COLPALI_MODEL, COLPALI_REVISION)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = ColPali.from_pretrained(str(backbone), torch_dtype=dtype, low_cpu_mem_usage=True)
    model.load_adapter(str(adapter))
    model.eval()
    processor = ColPaliProcessor.from_pretrained(str(adapter))
    assert_supported_colpali(model, processor)
    return model, processor


def _load_qwen(run_config: object) -> tuple[object, object]:
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    snapshot = _cached_snapshot(QWEN_MODEL, QWEN_REVISION)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        str(snapshot), torch_dtype=dtype, low_cpu_mem_usage=True
    ).eval()
    processor = AutoProcessor.from_pretrained(str(snapshot))
    return model, processor


def _make_dataset(run_config: object) -> object:
    make_dataset = getattr(run_config, "make_dataset", None)
    if callable(make_dataset):
        return make_dataset()
    from docprune.m3docvqa_dataset import M3DocVQADevDataset

    return M3DocVQADevDataset(_required(run_config, "corpus"), expected_question_count=2441)


def _load_index_manifest(value: IndexManifest | Path | str) -> IndexManifest:
    if isinstance(value, IndexManifest):
        manifest = value
    elif not isinstance(value, str | Path) and callable(getattr(value, "validate_files", None)):
        manifest = value
    else:
        path = Path(value)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"index manifest is missing: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError("index manifest must be valid JSON") from error
        if not isinstance(payload, Mapping):
            raise ValueError("index manifest must be a JSON object")
        resources = payload.get("resources")
        if not isinstance(resources, Mapping):
            raise ValueError("index manifest is missing resources")
        qwen = resources.get("qwen", {})
        colpali = resources.get("colpali", {})
        backbone = resources.get("colpali_backbone", {})
        if not all(isinstance(item, Mapping) for item in (qwen, colpali, backbone)):
            raise ValueError("index manifest resources are invalid")
        try:
            manifest = IndexManifest(
                mode=str(payload["mode"]),
                page_count=int(payload["page_count"]),
                corpus_integrity_sha256=str(payload["corpus_integrity_sha256"]),
                source_order_sha256=str(payload["source_order_sha256"]),
                runtime_commit=str(payload["runtime_commit"]),
                m3docrag_commit=str(payload["m3docrag_commit"]),
                qwen_model=str(qwen["model"]),
                qwen_revision=str(qwen["revision"]),
                colpali_model=str(colpali["model"]),
                colpali_revision=str(colpali["revision"]),
                colpali_backbone_model=str(backbone["model"]),
                colpali_backbone_revision=str(backbone["revision"]),
                processor_contract_path=Path(payload["processor_contract_path"]),
                processor_contract_sha256=str(payload["processor_contract_sha256"]),
                pruning_config=payload["pruning_config"],
                artifact_root=Path(payload["artifact_root"]),
                embeddings_path=Path(payload["embeddings_path"]),
                embedding_metadata_path=Path(payload["embedding_metadata_path"]),
                embedding_shape=tuple(payload["embeddings"]["shape"]),
                embedding_dtype=str(payload["embeddings"]["dtype"]),
                embeddings_sha256=str(payload["embeddings_sha256"]),
                token2pageuid_path=Path(payload["token2pageuid_path"]),
                token2pageuid_sha256=str(payload["token2pageuid_sha256"]),
                completion_ledger_path=Path(payload["completion_ledger_path"]),
                completion_ledger_sha256=str(payload["completion_ledger_sha256"]),
                index_path=Path(payload["index_path"]),
                index_sha256=str(payload["index_sha256"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("index manifest is missing required immutable fields") from error
    manifest.validate_files()
    return manifest


class _ColPaliQueryAdapter:
    def __init__(self, model: object, processor: object) -> None:
        self.model = model
        self.processor = processor

    def encode_queries(self, queries: Sequence[str]) -> list[torch.Tensor]:
        process = getattr(self.processor, "process_queries", None)
        if not callable(process):
            raise ValueError("ColPali processor must expose process_queries")
        batch = process(list(queries))
        device = next(self.model.parameters()).device if hasattr(self.model, "parameters") else None
        moved = {
            key: value.to(device)
            if isinstance(value, torch.Tensor) and device is not None
            else value
            for key, value in batch.items()
        }
        with torch.no_grad():
            output = self.model(**moved)
        output = torch.as_tensor(output)
        if output.ndim != 3 or output.shape[0] != len(queries):
            raise ValueError("ColPali query embeddings must have shape [batch, tokens, width]")
        return list(output)


def _run_manifest(
    identity: Mapping[str, object],
    *,
    operation: str,
    output: Path,
    index_manifest: IndexManifest | None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 2,
        "status": "configured",
        "operation": operation,
        "output": str(Path(output).resolve()),
        **dict(identity),
    }
    if index_manifest is not None:
        payload["index_manifest"] = index_manifest.to_dict()
    payload["run_manifest_sha256"] = _sha256_json(payload)
    return payload


def _write_run_manifest(
    output: Path,
    payload: Mapping[str, object],
    *,
    resume: bool,
) -> dict[str, object]:
    """Publish the complete identity before loading any model weights."""

    if output.is_symlink():
        raise ValueError("output must not be a symbolic link")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "run_manifest.json"
    existing: dict[str, object] = {}
    if manifest_path.exists():
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ValueError("run manifest must be a regular file")
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError("run manifest is not valid JSON") from error
        if not isinstance(raw, dict):
            raise ValueError("run manifest must be a JSON object")
        existing = raw
        identity_keys = (
            "mode",
            "page_count",
            "runtime_commit",
            "m3docrag_commit",
            "resources",
            "processor_contract_path",
            "processor_contract_sha256",
            "corpus",
            "generation",
            "index_manifest",
        )
        if resume:
            if any(key not in existing for key in identity_keys if key in payload):
                raise ValueError("resume manifest is missing an immutable identity field")
            supplied_digest = existing.get("run_manifest_sha256")
            unsigned_existing = dict(existing)
            unsigned_existing.pop("run_manifest_sha256", None)
            if supplied_digest != _sha256_json(unsigned_existing):
                raise ValueError("resume manifest digest is invalid")
        compare_keys = tuple(
            key
            for key in identity_keys
            if key in payload
            and key in existing
            and not (key == "index_manifest" and isinstance(existing[key], str) and not resume)
        )
        if compare_keys:
            if any(existing.get(key) != payload.get(key) for key in compare_keys):
                raise ValueError("resume manifest does not exactly match the requested run")
        elif not resume and not ({"command", "factory"} <= set(existing)):
            raise FileExistsError(f"output directory already contains an unrelated run: {output}")
    elif any(output.iterdir()) and not resume:
        raise FileExistsError(f"output directory already contains artifacts: {output}")
    merged = {**existing, **dict(payload)}
    merged.pop("run_manifest_sha256", None)
    merged["run_manifest_sha256"] = _sha256_json(merged)
    descriptor, raw_temp = tempfile.mkstemp(prefix=".run_manifest.", dir=output)
    temporary = Path(raw_temp)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(merged, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, manifest_path)
        descriptor = -1
    finally:
        if descriptor != -1:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
    return merged


def build_workload(
    *,
    operation: str,
    config: DocPruneConfig,
    page_count: int,
    output: Path,
    mode: str | None = None,
    run_config: BenchmarkRunConfig | Path | Mapping[str, object] | None = None,
    index_manifest: IndexManifest | Path | str | None = None,
    limit: int | None = None,
    sample_ids: Sequence[str] | None = None,
    resume: bool = False,
) -> EvaluationWorkload | IndexBuildResult:
    """Build the concrete index operation or lazy end-to-end evaluation workload."""

    if operation not in {"embed", "evaluate"}:
        raise ValueError("operation must be 'embed' or 'evaluate'")
    if not isinstance(config, DocPruneConfig):
        raise TypeError("config must be a DocPruneConfig")
    if page_count not in PAGE_COUNTS:
        raise ValueError("page_count must be 1, 2, or 4")
    if config.m3docrag_commit != M3DOCRAG_COMMIT:
        raise ValueError("config must use the pinned M3DocRAG commit")
    resolved_mode = _resolve_mode(mode, run_config)
    resolved_run = _resolve_run_config(run_config, mode=resolved_mode, page_count=page_count)
    identity = _validate_run_identity(resolved_run, mode=resolved_mode, page_count=page_count)
    _validate_m3docrag_checkout(resolved_run)
    output = Path(output)
    if output.is_symlink():
        raise ValueError("output must not be a symbolic link")
    if output.exists() and output.is_file():
        raise FileExistsError(f"output must be a directory: {output}")
    if operation == "evaluate" and output.exists() and not output.is_dir():
        raise ValueError(f"output must be a directory: {output}")
    if operation == "evaluate" and output.exists() and not resume:
        entries = tuple(output.iterdir())
        if any(entry.name != "run_manifest.json" for entry in entries):
            raise FileExistsError(f"evaluation output already exists: {output}")
        manifest_path = output / "run_manifest.json"
        if manifest_path.is_file():
            try:
                existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                raise ValueError("existing run manifest is not valid JSON") from error
            if isinstance(existing_manifest, Mapping) and "runtime_commit" in existing_manifest:
                raise FileExistsError(f"evaluation output already exists: {output}")
    dataset = _make_dataset(resolved_run)

    if operation == "embed":
        _write_run_manifest(
            output,
            _run_manifest(identity, operation=operation, output=output, index_manifest=None),
            resume=resume,
        )
        model, processor = _load_colpali(resolved_run)
        result = build_index(
            IndexBuildConfig(
                run_config=resolved_run,
                dataset=dataset,
                model=model,
                processor=processor,
                pruning_config=config,
            ),
            resolved_mode,
            output,
        )
        if not isinstance(result, IndexBuildResult):
            raise TypeError("embed factory must return IndexBuildResult")
        return result

    if index_manifest is None:
        raise ValueError("evaluate requires --index-manifest for the selected mode")
    manifest = _load_index_manifest(index_manifest)
    if manifest.mode != resolved_mode or manifest.page_count != page_count:
        raise ValueError("index manifest mode/page_count does not match the requested run")
    if manifest.m3docrag_commit != M3DOCRAG_COMMIT:
        raise ValueError("index manifest uses the wrong M3DocRAG commit")
    if manifest.runtime_commit != str(_required(resolved_run, "runtime_commit")).lower():
        raise ValueError("index manifest runtime commit does not match the run configuration")
    if manifest.processor_contract_sha256 != str(identity["processor_contract_sha256"]):
        raise ValueError("index manifest processor contract does not match the run configuration")
    if manifest.runtime_commit != str(identity["runtime_commit"]):
        raise ValueError("index manifest runtime commit does not match the run configuration")
    if manifest.to_dict()["resources"] != identity["resources"]:
        raise ValueError("index manifest model resources do not match the run configuration")
    corpus = identity["corpus"]
    if manifest.corpus_integrity_sha256 != corpus["integrity_sha256"]:
        raise ValueError("index manifest corpus identity does not match the run configuration")
    dataset_source_order = _value(dataset, "source_order_sha256")
    if dataset_source_order is not None and manifest.source_order_sha256 != dataset_source_order:
        raise ValueError("index manifest source order does not match the corpus")

    complete_manifest = _write_run_manifest(
        output,
        _run_manifest(identity, operation=operation, output=output, index_manifest=manifest),
        resume=resume,
    )

    import faiss
    from safetensors.torch import load_file

    embeddings = load_file(manifest.embeddings_path, device="cpu")["embeddings"]
    token_map = json.loads(manifest.token2pageuid_path.read_text(encoding="utf-8"))
    if not isinstance(token_map, list) or len(token_map) != len(embeddings):
        raise ValueError("index token2pageuid rows do not match embeddings")
    index = faiss.read_index(str(manifest.index_path))
    colpali_model, colpali_processor = _load_colpali(resolved_run)
    query_adapter = _ColPaliQueryAdapter(colpali_model, colpali_processor)
    rag_model = SimpleNamespace(retrieval_model=query_adapter)
    boundary = OfficialM3DocRAGBoundary(
        rag_model=rag_model,
        dataset=dataset,
        docid2embs={},
        index=index,
        token2pageuid=token_map,
        all_token_embeddings=embeddings,
    )
    qwen_model, qwen_processor = _load_qwen(resolved_run)
    if resolved_mode == "all-kept":
        answerer = AllKeptQwenAnswerer(qwen_model, qwen_processor)
    else:
        answerer = DocPruneQwenAnswerer(
            qwen_model,
            qwen_processor,
            colpali_model=colpali_model,
            colpali_processor=colpali_processor,
            page_config=config.for_pages(page_count),
        )
    runner = DocPruneM3DocRAG(boundary, dataset, answerer, top_k=page_count)
    samples = filter_samples(dataset, limit=limit, sample_ids=sample_ids)
    return EvaluationWorkload(
        runner=runner,
        samples=samples,
        manifest=complete_manifest,
    )


__all__ = [
    "DEFAULT_FACTORY",
    "build_workload",
    "filter_samples",
    "load_completed_qids",
    "validate_processor_contract_file",
]
