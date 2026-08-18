from __future__ import annotations

import hashlib
import json
from pathlib import Path

import faiss
import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from docprune.artifacts import (
    IndexManifest,
    canonical_json_sha256,
    sha256_file,
    validate_index_manifest_pair,
)
from docprune.benchmark_config import (
    COLPALI_BACKBONE_MODEL,
    COLPALI_BACKBONE_REVISION,
    COLPALI_MODEL,
    COLPALI_REVISION,
    M3DOCRAG_COMMIT,
    QWEN_MODEL,
    QWEN_REVISION,
)


def make_manifest(tmp_path: Path, *, mode: str = "docprune") -> IndexManifest:
    root = tmp_path / mode
    root.mkdir()
    embeddings = root / "embeddings.safetensors"
    save_file(
        {
            "embeddings": torch.zeros(12, 128, dtype=torch.float32),
            "raster_indices": torch.arange(12, dtype=torch.int64),
        },
        embeddings,
    )
    metadata = root / "embeddings.json"
    metadata.write_text('{"shape": [12, 128], "dtype": "float32"}')
    contract = root / "processor-contract.json"
    contract.write_text("{}")
    token2pageuid = root / "token2pageuid.json"
    token2pageuid.write_text(
        json.dumps([{"doc_id": "doc", "page_index": 0}] * 12), encoding="utf-8"
    )
    ledger = root / "completion-ledger.json"
    ledger.write_text("[]", encoding="utf-8")
    index = root / "index.faiss"
    faiss_index = faiss.IndexFlatIP(128)
    faiss_index.add(np.zeros((12, 128), dtype=np.float32))
    faiss.write_index(faiss_index, str(index))
    return IndexManifest(
        mode=mode,
        page_count=4,
        corpus_integrity_sha256="a" * 64,
        source_order_sha256="b" * 64,
        runtime_commit="c" * 40,
        m3docrag_commit=M3DOCRAG_COMMIT,
        qwen_model=QWEN_MODEL,
        qwen_revision=QWEN_REVISION,
        colpali_model=COLPALI_MODEL,
        colpali_revision=COLPALI_REVISION,
        colpali_backbone_model=COLPALI_BACKBONE_MODEL,
        colpali_backbone_revision=COLPALI_BACKBONE_REVISION,
        processor_contract_path=contract,
        processor_contract_sha256=sha256_file(contract),
        pruning_config={"attention_threshold": 0.075},
        artifact_root=root,
        embeddings_path=embeddings,
        embedding_metadata_path=metadata,
        embedding_shape=(12, 128),
        embedding_dtype="float32",
        token2pageuid_path=token2pageuid,
        token2pageuid_sha256=sha256_file(token2pageuid),
        completion_ledger_path=ledger,
        completion_ledger_sha256=sha256_file(ledger),
        index_path=index,
        index_sha256=sha256_file(index),
    )


def test_canonical_json_hash_is_order_independent() -> None:
    assert canonical_json_sha256({"b": [2], "a": 1}) == canonical_json_sha256({"a": 1, "b": [2]})


def test_index_manifest_serializes_all_immutable_inputs_and_checksums(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)

    payload = manifest.to_dict()

    assert payload["schema_version"] == 3
    assert payload["mode"] == "docprune"
    assert payload["corpus_integrity_sha256"] == "a" * 64
    assert payload["source_order_sha256"] == "b" * 64
    assert payload["resources"]["qwen"] == {"model": QWEN_MODEL, "revision": QWEN_REVISION}
    assert payload["resources"]["colpali"] == {
        "model": COLPALI_MODEL,
        "revision": COLPALI_REVISION,
    }
    assert payload["processor_contract_sha256"] == sha256_file(manifest.processor_contract_path)
    assert payload["pruning_config"] == {"attention_threshold": 0.075}
    assert payload["artifact_root"] == str(manifest.artifact_root)
    assert payload["embeddings"] == {"shape": [12, 128], "dtype": "float32"}
    assert payload["index_sha256"] == sha256_file(manifest.index_path)
    assert payload["runtime_commit"] == "c" * 40
    assert payload["m3docrag_commit"] == M3DOCRAG_COMMIT
    assert payload["manifest_sha256"] == canonical_json_sha256(
        {key: value for key, value in payload.items() if key != "manifest_sha256"}
    )
    manifest.validate_files()


@pytest.mark.parametrize(
    ("path_name", "contents", "message"),
    [
        ("processor_contract_path", "changed", "processor contract SHA-256 mismatch"),
        ("embedding_metadata_path", '{"shape": [12, 127], "dtype": "float32"}', "shape mismatch"),
        ("embedding_metadata_path", '{"shape": [12, 128], "dtype": "float16"}', "dtype mismatch"),
    ],
)
def test_manifest_validates_contract_and_embedding_metadata_sidecars(
    tmp_path: Path, path_name: str, contents: str, message: str
) -> None:
    manifest = make_manifest(tmp_path)
    getattr(manifest, path_name).write_text(contents)

    with pytest.raises(ValueError, match=message):
        manifest.validate_files()


@pytest.mark.parametrize(
    "tensors",
    [
        {
            "embeddings": torch.zeros(11, 128, dtype=torch.float32),
            "raster_indices": torch.arange(11, dtype=torch.int64),
        },
        {
            "embeddings": torch.zeros(12, 128, dtype=torch.float16),
            "raster_indices": torch.arange(12, dtype=torch.int64),
        },
        {
            "embeddings": torch.zeros(12, 128, dtype=torch.float32),
            "raster_indices": torch.arange(11, dtype=torch.int64),
        },
    ],
)
def test_manifest_validates_physical_safetensors_shape_dtype_and_raster_rows(
    tmp_path: Path, tensors: dict[str, torch.Tensor]
) -> None:
    manifest = make_manifest(tmp_path)
    save_file(tensors, manifest.embeddings_path)

    with pytest.raises(ValueError, match="safetensors"):
        manifest.validate_files()


def test_manifest_validation_rejects_artifact_symlink_escape(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)
    external_root = tmp_path / "shared"
    external_root.mkdir()
    external_embedding = external_root / manifest.embeddings_path.name
    external_embedding.write_bytes(b"outside artifact root")
    manifest.embeddings_path.unlink()
    manifest.embeddings_path.symlink_to(external_embedding)

    with pytest.raises(ValueError, match="artifact root"):
        manifest.validate_files()


def test_index_manifest_rejects_changed_index_or_invalid_embedding_contract(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)
    manifest.index_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="index SHA-256 mismatch"):
        manifest.validate_files()

    with pytest.raises(ValueError, match="width 128"):
        IndexManifest(
            **{
                **manifest.__dict__,
                "embedding_shape": (12, 127),
                "index_sha256": hashlib.sha256(b"changed").hexdigest(),
            }
        )


def test_manifest_pruning_configuration_is_immutable(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)

    with pytest.raises(TypeError):
        manifest.pruning_config["attention_threshold"] = 0.5  # type: ignore[index]


def test_manifest_nested_pruning_configuration_is_immutable(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)
    arguments = dict(manifest.__dict__)
    arguments["pruning_config"] = {"retrieval": {"threshold": 0.9}}
    nested = IndexManifest(**arguments)

    with pytest.raises(TypeError):
        nested.pruning_config["retrieval"]["threshold"] = 1.0  # type: ignore[index]


def test_manifest_pair_requires_distinct_mode_bound_artifacts(tmp_path: Path) -> None:
    all_kept = make_manifest(tmp_path, mode="all-kept")
    docprune = make_manifest(tmp_path, mode="docprune")

    validate_index_manifest_pair(all_kept, docprune)

    object.__setattr__(docprune, "embeddings_path", all_kept.embeddings_path)
    with pytest.raises(ValueError, match="embedding roots"):
        validate_index_manifest_pair(all_kept, docprune)


def test_manifest_pair_rejects_mode_path_mismatch(tmp_path: Path) -> None:
    all_kept = make_manifest(tmp_path, mode="all-kept")
    docprune = make_manifest(tmp_path, mode="docprune")
    object.__setattr__(docprune, "index_path", tmp_path / "all-kept" / "other.faiss")

    with pytest.raises(ValueError, match="artifact root"):
        validate_index_manifest_pair(all_kept, docprune)


@pytest.mark.parametrize("path_name", ["embeddings_path", "embedding_metadata_path", "index_path"])
def test_manifest_rejects_cross_mode_artifact_paths_at_construction(
    tmp_path: Path, path_name: str
) -> None:
    manifest = make_manifest(tmp_path, mode="docprune")
    arguments = dict(manifest.__dict__)
    arguments[path_name] = tmp_path / "all-kept" / getattr(manifest, path_name).name

    with pytest.raises(ValueError, match="artifact root"):
        IndexManifest(**arguments)


@pytest.mark.parametrize("path_name", ["embeddings_path", "embedding_metadata_path", "index_path"])
def test_manifest_rejects_traversal_outside_mode_artifact_root(
    tmp_path: Path, path_name: str
) -> None:
    manifest = make_manifest(tmp_path, mode="docprune")
    arguments = dict(manifest.__dict__)
    arguments[path_name] = (
        tmp_path / "docprune" / ".." / "shared" / getattr(manifest, path_name).name
    )

    with pytest.raises(ValueError, match="artifact root"):
        IndexManifest(**arguments)
