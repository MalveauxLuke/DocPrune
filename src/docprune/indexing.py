"""Deterministic, resumable construction of manifest-bound ColPali indexes."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers.image_utils import ChannelDimension

from docprune.artifacts import IndexManifest, canonical_json_sha256
from docprune.benchmark_config import MODES, sha256_file
from docprune.btp import background_scores, threshold_keep_mask
from docprune.colpali.compat import assert_supported_colpali
from docprune.colpali.embedding import ColPaliPageEmbedding, encode_colpali_page
from docprune.config import DocPruneConfig
from docprune.processor_probe import resolve_colpali_visual_mapping

EMBEDDING_WIDTH = 128
SIGLIP_PATCH_SIZE = 14


@dataclass(frozen=True)
class IndexBuildConfig:
    """Runtime objects and immutable inputs needed for one index build."""

    run_config: object
    dataset: object
    model: object
    processor: object
    pruning_config: DocPruneConfig


@dataclass(frozen=True)
class IndexBuildResult:
    manifest: IndexManifest
    manifest_path: Path
    mode_root: Path
    token2pageuid_path: Path
    completion_ledger_path: Path
    documents_built: int
    documents_resumed: int


def _value(container: object, name: str) -> Any:
    if isinstance(container, Mapping):
        if name not in container:
            raise ValueError(f"missing required {name}")
        return container[name]
    if not hasattr(container, name):
        raise ValueError(f"missing required {name}")
    return getattr(container, name)


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("utf-8")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_temp(temp_path: Path, target: Path) -> None:
    try:
        if target.is_symlink():
            raise ValueError(f"immutable artifact target must not be a symbolic link: {target}")
        if target.exists():
            if not target.is_file():
                raise ValueError(f"immutable artifact target must be a regular file: {target}")
            if sha256_file(temp_path) != sha256_file(target):
                raise ValueError(
                    f"immutable artifact already exists with different content: {target}"
                )
            return
        try:
            os.link(temp_path, target)
        except FileExistsError:
            if target.is_symlink():
                raise ValueError(f"immutable artifact target must not be a symbolic link: {target}")
            if not target.is_file():
                raise ValueError(f"immutable artifact target must be a regular file: {target}")
            if sha256_file(temp_path) != sha256_file(target):
                raise ValueError(f"immutable artifact was concurrently changed: {target}")
        _fsync_directory(target.parent)
    finally:
        temp_path.unlink(missing_ok=True)


def _atomic_write_bytes(target: Path, payload: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temp = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temp_path = Path(raw_temp)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _publish_temp(temp_path, target)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def _atomic_write_json(target: Path, value: object) -> None:
    _atomic_write_bytes(target, _json_bytes(value))


def _atomic_save_safetensors(target: Path, tensors: Mapping[str, torch.Tensor]) -> None:
    from safetensors.torch import save_file

    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temp = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    os.close(descriptor)
    temp_path = Path(raw_temp)
    try:
        contiguous = {name: tensor.detach().cpu().contiguous() for name, tensor in tensors.items()}
        save_file(contiguous, temp_path)
        with temp_path.open("rb") as stream:
            os.fsync(stream.fileno())
        _publish_temp(temp_path, target)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def _atomic_write_faiss(target: Path, embeddings: torch.Tensor) -> None:
    import faiss

    array = np.ascontiguousarray(embeddings.detach().cpu().numpy(), dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != EMBEDDING_WIDTH:
        raise ValueError("FAISS embeddings must have shape [tokens, 128]")
    if not np.isfinite(array).all():
        raise ValueError("FAISS embeddings must be finite float32 values")
    index = faiss.IndexFlatIP(EMBEDDING_WIDTH)
    index.add(array)

    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temp = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    os.close(descriptor)
    temp_path = Path(raw_temp)
    try:
        faiss.write_index(index, str(temp_path))
        with temp_path.open("rb") as stream:
            os.fsync(stream.fileno())
        _publish_temp(temp_path, target)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def colpali_uint8_raster(processor: object, image: Image.Image) -> torch.Tensor:
    """Return the exact resized ColPali raster before rescaling and normalization."""

    image_processor = _value(processor, "image_processor")
    output = image_processor(
        images=[image.convert("RGB")],
        return_tensors="pt",
        do_rescale=False,
        do_normalize=False,
        data_format=ChannelDimension.FIRST,
    )
    pixels = torch.as_tensor(_value(output, "pixel_values"))
    if pixels.ndim != 4 or pixels.shape[0] != 1 or pixels.shape[1] != 3:
        raise ValueError("ColPali image processor must return one channels-first RGB raster")
    if not bool(torch.isfinite(pixels).all()):
        raise ValueError("post-resize ColPali raster must contain finite pixels")
    rounded = pixels.round()
    if not torch.equal(pixels, rounded) or float(rounded.min()) < 0 or float(rounded.max()) > 255:
        raise ValueError("post-resize ColPali raster must contain uint8-equivalent intensities")
    return rounded.to(torch.uint8)


def _raster_image(raster: torch.Tensor) -> Image.Image:
    array = raster[0].permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(np.ascontiguousarray(array))


def _model_device(model: object) -> torch.device | None:
    parameters = getattr(model, "parameters", None)
    if parameters is None:
        return None
    try:
        return next(parameters()).device
    except StopIteration:
        return None


def _move_batch(batch: object, device: torch.device | None) -> dict[str, object]:
    if not isinstance(batch, Mapping):
        raise ValueError("ColPali process_images must return a mapping")
    moved: dict[str, object] = {}
    for name, value in batch.items():
        moved[name] = (
            value.to(device) if device is not None and isinstance(value, torch.Tensor) else value
        )
    return moved


def _build_identity(config: IndexBuildConfig, mode: str) -> dict[str, object]:
    run = config.run_config
    page_count = int(_value(run, "page_count"))
    page_settings = config.pruning_config.for_pages(page_count)
    processor_contract_path = Path(_value(run, "processor_contract_path"))
    if not processor_contract_path.is_file():
        raise FileNotFoundError(f"processor contract is missing: {processor_contract_path}")
    payload: dict[str, object] = {
        "schema_version": 1,
        "mode": mode,
        "page_count": page_count,
        "corpus_integrity_sha256": str(_value(_value(run, "corpus"), "integrity_sha256")),
        "source_order_sha256": str(_value(config.dataset, "source_order_sha256")),
        "runtime_commit": str(_value(run, "runtime_commit")),
        "m3docrag_commit": str(_value(run, "m3docrag_commit")),
        "resources": {
            "qwen": {
                "model": str(_value(run, "qwen_model")),
                "revision": str(_value(run, "qwen_revision")),
            },
            "colpali": {
                "model": str(_value(run, "colpali_model")),
                "revision": str(_value(run, "colpali_revision")),
            },
            "colpali_backbone": {
                "model": str(_value(run, "colpali_backbone_model")),
                "revision": str(_value(run, "colpali_backbone_revision")),
            },
        },
        "processor_contract_path": str(processor_contract_path.resolve()),
        "processor_contract_sha256": sha256_file(processor_contract_path),
        "pruning_config": {
            "enabled": mode == "docprune",
            "page_settings": asdict(page_settings),
            "reconstruction_defaults": asdict(config.pruning_config.reconstruction_defaults),
            "siglip_patch_size": SIGLIP_PATCH_SIZE,
        },
        "document_ids": list(_document_ids(config.dataset)),
    }
    payload["build_manifest_sha256"] = canonical_json_sha256(payload)
    return payload


def _document_ids(dataset: object) -> tuple[str, ...]:
    values = tuple(_value(dataset, "document_ids"))
    if not values or not all(isinstance(value, str) and value for value in values):
        raise ValueError("dataset document_ids must contain nonempty strings")
    if len(values) != len(set(values)):
        raise ValueError("dataset document_ids must be unique")
    return values


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON artifact: {path}") from error


def _mode_artifact_root(output: Path, mode: str) -> Path:
    output_root = Path(output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    candidate = output_root / mode
    if candidate.is_symlink():
        raise ValueError("mode artifact root must not be a symbolic link")
    candidate.mkdir(exist_ok=True)
    resolved = candidate.resolve()
    try:
        resolved.relative_to(output_root)
    except ValueError as error:
        raise ValueError("mode artifact root must remain below the requested output") from error
    return resolved


def _scoped_directory(root: Path, name: str) -> Path:
    candidate = root / name
    if candidate.is_symlink():
        raise ValueError(f"artifact directory must not be a symbolic link: {candidate}")
    candidate.mkdir(exist_ok=True)
    if candidate.resolve().parent != root:
        raise ValueError(f"artifact directory must remain below artifact root: {candidate}")
    return candidate


def _load_completed_document(
    *,
    document_path: Path,
    ledger_path: Path,
    ordinal: int,
    doc_id: str,
    build_manifest_sha256: str,
    documents_root: Path,
    ledger_root: Path,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, object]], dict[str, object]] | None:
    _require_document_target(document_path, documents_root, "document")
    _require_document_target(ledger_path, ledger_root, "completion ledger")
    if not ledger_path.exists():
        if document_path.exists():
            raise ValueError(
                f"unledgered document artifact is preserved and cannot be resumed: {document_path}"
            )
        return None
    entry = _read_json(ledger_path)
    if not isinstance(entry, dict):
        raise ValueError(f"invalid completion ledger entry: {ledger_path}")
    expected = {
        "ordinal": ordinal,
        "doc_id": doc_id,
        "build_manifest_sha256": build_manifest_sha256,
        "document_path": str(document_path),
    }
    if any(entry.get(name) != value for name, value in expected.items()):
        raise ValueError("completed document is not manifest-identical")
    if not document_path.is_file() or sha256_file(document_path) != entry.get("sha256"):
        raise ValueError(f"completed document checksum mismatch: {doc_id}")

    from safetensors.torch import load_file

    tensors = load_file(document_path, device="cpu")
    if set(tensors) != {"embeddings", "page_offsets", "raster_indices"}:
        raise ValueError(f"completed document has invalid safetensors keys: {doc_id}")
    embeddings = tensors["embeddings"]
    rasters = tensors["raster_indices"]
    offsets = tensors["page_offsets"]
    if embeddings.ndim != 2 or embeddings.shape[1] != EMBEDDING_WIDTH:
        raise ValueError(f"completed document has invalid embedding shape: {doc_id}")
    if entry.get("shape") != list(embeddings.shape):
        raise ValueError(f"completion ledger entry shape mismatch: {doc_id}")
    if entry.get("dtype") != "float32":
        raise ValueError(f"completion ledger entry dtype mismatch: {doc_id}")
    if embeddings.dtype != torch.float32 or rasters.dtype != torch.int64 or rasters.ndim != 1:
        raise ValueError(f"completed document has invalid tensor dtype: {doc_id}")
    if offsets.dtype != torch.int64 or offsets.ndim != 1 or offsets.numel() < 2:
        raise ValueError(f"completed document has invalid page offsets: {doc_id}")
    if int(offsets[0]) != 0 or int(offsets[-1]) != len(embeddings):
        raise ValueError(f"completed document offsets do not cover embeddings: {doc_id}")
    if entry.get("page_offsets") != offsets.tolist():
        raise ValueError(f"completion ledger entry offsets mismatch: {doc_id}")
    if not bool((offsets[1:] > offsets[:-1]).all()):
        raise ValueError(f"completed document page offsets must be strictly increasing: {doc_id}")
    if len(rasters) != len(embeddings):
        raise ValueError(f"completed document rows are inconsistent: {doc_id}")
    if not bool(((rasters >= 0) & (rasters < 1024)).all()):
        raise ValueError(f"completed document raster indices are out of range: {doc_id}")
    pages = entry.get("pages")
    if not isinstance(pages, list) or len(pages) != offsets.numel() - 1:
        raise ValueError(f"completed document page ledger is invalid: {doc_id}")
    mapping: list[dict[str, object]] = []
    for page_index, page in enumerate(pages):
        identity = {"doc_id": doc_id, "page_index": page_index}
        if page != identity:
            raise ValueError(f"completed document page order drift: {doc_id}")
        count = int(offsets[page_index + 1] - offsets[page_index])
        page_rasters = rasters[int(offsets[page_index]) : int(offsets[page_index + 1])]
        if page_rasters.numel() == 0 or (
            page_rasters.numel() > 1 and not bool((page_rasters[1:] > page_rasters[:-1]).all())
        ):
            raise ValueError(
                f"completed document raster indices are not strictly increasing: {doc_id}"
            )
        mapping.extend([identity] * count)
    return embeddings, rasters, mapping, entry


def _require_document_target(target: Path, root: Path, label: str) -> None:
    """Reject symlink/nonregular targets and path escapes before I/O."""

    root = root.resolve()
    if target.parent.resolve() != root:
        raise ValueError(f"{label} target is outside its artifact root: {target}")
    if target.is_symlink():
        raise ValueError(f"{label} target must not be a symbolic link: {target}")
    try:
        target.resolve(strict=False).relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} target is outside its artifact root: {target}") from error
    if target.exists() and not target.is_file():
        raise ValueError(f"{label} target must be a regular file: {target}")


def _encode_document(
    config: IndexBuildConfig,
    *,
    mode: str,
    doc_id: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, object]]]:
    pages = tuple(config.dataset.load_pages(doc_id))
    if not pages:
        raise ValueError(f"document has no pages: {doc_id}")
    page_setting = config.pruning_config.for_pages(int(_value(config.run_config, "page_count")))
    device = _model_device(config.model)
    page_embeddings: list[torch.Tensor] = []
    page_rasters: list[torch.Tensor] = []
    offsets = [0]
    page_identities: list[dict[str, object]] = []
    image_token_id = int(
        _value(_value(_value(config.model, "model"), "config"), "image_token_index")
    )
    image_seq_length = int(_value(config.processor, "image_seq_length"))

    for page_index, image in enumerate(pages):
        raster = colpali_uint8_raster(config.processor, image)
        batch = _move_batch(config.processor.process_images([_raster_image(raster)]), device)
        input_ids = torch.as_tensor(_value(batch, "input_ids"), dtype=torch.long)
        attention_mask = torch.as_tensor(_value(batch, "attention_mask"))
        mapping = resolve_colpali_visual_mapping(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_token_id=image_token_id,
            image_seq_length=image_seq_length,
        )
        expected_hw = (
            mapping.grid_hw[0] * SIGLIP_PATCH_SIZE,
            mapping.grid_hw[1] * SIGLIP_PATCH_SIZE,
        )
        if tuple(raster.shape[-2:]) != expected_hw:
            raise ValueError("post-resize uint8 raster does not match the ColPali patch grid")
        if mode == "all-kept":
            keep = torch.ones(image_seq_length, dtype=torch.bool, device=input_ids.device)
        else:
            scores = background_scores(
                raster,
                patch_size=SIGLIP_PATCH_SIZE,
                error_tolerance=page_setting.background_error_tolerance,
            )
            keep = threshold_keep_mask(
                scores,
                page_setting.retrieval_background_threshold,
            )[0].to(input_ids.device)
        if not bool(keep.any()):
            raise ValueError(f"retrieval BTP rejected every patch for {doc_id} page {page_index}")
        encoded = encode_colpali_page(config.model, batch, mapping, keep)
        _validate_page_embedding(encoded, keep, doc_id=doc_id, page_index=page_index)
        visual = encoded.visual_embeddings[0].detach().to(device="cpu", dtype=torch.float32)
        rasters = encoded.raster_indices.detach().to(device="cpu", dtype=torch.int64)
        page_embeddings.append(visual)
        page_rasters.append(rasters)
        offsets.append(offsets[-1] + len(visual))
        page_identities.append({"doc_id": doc_id, "page_index": page_index})
    return (
        torch.cat(page_embeddings),
        torch.cat(page_rasters),
        torch.tensor(offsets, dtype=torch.int64),
        page_identities,
    )


def _validate_page_embedding(
    encoded: ColPaliPageEmbedding,
    keep: torch.Tensor,
    *,
    doc_id: str,
    page_index: int,
) -> None:
    visual = encoded.visual_embeddings
    expected_rasters = torch.arange(keep.numel(), device=keep.device, dtype=torch.long)[keep]
    if visual.ndim != 3 or visual.shape[0] != 1 or visual.shape[2] != EMBEDDING_WIDTH:
        raise ValueError(
            f"ColPali visual embedding shape is invalid for {doc_id} page {page_index}"
        )
    if visual.shape[1] != len(expected_rasters):
        raise ValueError(f"ColPali visual row count is invalid for {doc_id} page {page_index}")
    if not torch.equal(encoded.raster_indices.to(expected_rasters.device), expected_rasters):
        raise ValueError(f"ColPali raster identity drift for {doc_id} page {page_index}")
    if not bool(torch.isfinite(visual).all()):
        raise ValueError(f"ColPali visual embeddings are nonfinite for {doc_id} page {page_index}")


def build_index(config: IndexBuildConfig, mode: str, output: Path) -> IndexBuildResult:
    """Build or resume one deterministic mode-scoped ColPali FAISS index."""

    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}")
    if str(_value(config.run_config, "mode")) != mode:
        raise ValueError("build mode must match run configuration mode")
    assert_supported_colpali(config.model, config.processor)
    mode_root = _mode_artifact_root(output, mode)
    documents_root = _scoped_directory(mode_root, "documents")
    ledger_root = _scoped_directory(mode_root, "completion-ledger")

    build_identity = _build_identity(config, mode)
    build_manifest_path = mode_root / "build-manifest.json"
    if build_manifest_path.exists() and _read_json(build_manifest_path) != build_identity:
        raise ValueError("resume requires a manifest-identical build configuration")
    _atomic_write_json(build_manifest_path, build_identity)
    build_sha = str(build_identity["build_manifest_sha256"])

    all_embeddings: list[torch.Tensor] = []
    all_rasters: list[torch.Tensor] = []
    token2pageuid: list[dict[str, object]] = []
    completion_entries: list[dict[str, object]] = []
    documents_built = 0
    documents_resumed = 0
    for ordinal, doc_id in enumerate(_document_ids(config.dataset)):
        document_path = documents_root / f"{ordinal:06d}.safetensors"
        ledger_path = ledger_root / f"{ordinal:06d}.json"
        completed = _load_completed_document(
            document_path=document_path,
            ledger_path=ledger_path,
            ordinal=ordinal,
            doc_id=doc_id,
            build_manifest_sha256=build_sha,
            documents_root=documents_root,
            ledger_root=ledger_root,
        )
        if completed is None:
            embeddings, rasters, offsets, pages = _encode_document(
                config,
                mode=mode,
                doc_id=doc_id,
            )
            _atomic_save_safetensors(
                document_path,
                {
                    "embeddings": embeddings,
                    "raster_indices": rasters,
                    "page_offsets": offsets,
                },
            )
            entry: dict[str, object] = {
                "schema_version": 1,
                "ordinal": ordinal,
                "doc_id": doc_id,
                "build_manifest_sha256": build_sha,
                "document_path": str(document_path),
                "sha256": sha256_file(document_path),
                "shape": list(embeddings.shape),
                "dtype": "float32",
                "pages": pages,
                "page_offsets": offsets.tolist(),
            }
            _atomic_write_json(ledger_path, entry)
            mapping: list[dict[str, object]] = []
            for page_index, page in enumerate(pages):
                count = int(offsets[page_index + 1] - offsets[page_index])
                mapping.extend([page] * count)
            documents_built += 1
        else:
            embeddings, rasters, mapping, entry = completed
            documents_resumed += 1
        all_embeddings.append(embeddings)
        all_rasters.append(rasters)
        token2pageuid.extend(mapping)
        completion_entries.append(entry)

    embeddings = torch.cat(all_embeddings).to(torch.float32).contiguous()
    rasters = torch.cat(all_rasters).to(torch.int64).contiguous()
    if len(token2pageuid) != len(embeddings):
        raise ValueError("token2pageuid rows do not match visual embeddings")

    embeddings_path = mode_root / "embeddings.safetensors"
    token2pageuid_path = mode_root / "token2pageuid.json"
    metadata_path = mode_root / "embeddings.json"
    completion_ledger_path = mode_root / "completion-ledger.json"
    index_path = mode_root / "index.faiss"
    _atomic_save_safetensors(
        embeddings_path,
        {"embeddings": embeddings, "raster_indices": rasters},
    )
    _atomic_write_json(token2pageuid_path, token2pageuid)
    _atomic_write_json(
        metadata_path,
        {
            "shape": list(embeddings.shape),
            "dtype": "float32",
            "document_ids": list(_document_ids(config.dataset)),
            "token2pageuid_sha256": sha256_file(token2pageuid_path),
        },
    )
    _atomic_write_json(completion_ledger_path, completion_entries)
    _atomic_write_faiss(index_path, embeddings)

    run = config.run_config
    pruning_payload = build_identity["pruning_config"]
    manifest = IndexManifest(
        mode=mode,
        page_count=int(_value(run, "page_count")),
        corpus_integrity_sha256=str(build_identity["corpus_integrity_sha256"]),
        source_order_sha256=str(build_identity["source_order_sha256"]),
        runtime_commit=str(_value(run, "runtime_commit")),
        m3docrag_commit=str(_value(run, "m3docrag_commit")),
        qwen_model=str(_value(run, "qwen_model")),
        qwen_revision=str(_value(run, "qwen_revision")),
        colpali_model=str(_value(run, "colpali_model")),
        colpali_revision=str(_value(run, "colpali_revision")),
        colpali_backbone_model=str(_value(run, "colpali_backbone_model")),
        colpali_backbone_revision=str(_value(run, "colpali_backbone_revision")),
        processor_contract_path=Path(_value(run, "processor_contract_path")),
        processor_contract_sha256=str(build_identity["processor_contract_sha256"]),
        pruning_config=pruning_payload if isinstance(pruning_payload, Mapping) else {},
        artifact_root=mode_root,
        embeddings_path=embeddings_path,
        embedding_metadata_path=metadata_path,
        embedding_shape=(len(embeddings), EMBEDDING_WIDTH),
        embedding_dtype="float32",
        embeddings_sha256=sha256_file(embeddings_path),
        token2pageuid_path=token2pageuid_path,
        token2pageuid_sha256=sha256_file(token2pageuid_path),
        completion_ledger_path=completion_ledger_path,
        completion_ledger_sha256=sha256_file(completion_ledger_path),
        index_path=index_path,
        index_sha256=sha256_file(index_path),
    )
    manifest.validate_files()
    manifest_path = mode_root / "manifest.json"
    _atomic_write_json(manifest_path, manifest.to_dict())
    return IndexBuildResult(
        manifest=manifest,
        manifest_path=manifest_path,
        mode_root=mode_root,
        token2pageuid_path=token2pageuid_path,
        completion_ledger_path=completion_ledger_path,
        documents_built=documents_built,
        documents_resumed=documents_resumed,
    )
