from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from docprune.artifacts import IndexManifest, canonical_json_sha256, sha256_file
from docprune.benchmark_config import (
    COLPALI_BACKBONE_MODEL,
    COLPALI_BACKBONE_REVISION,
    COLPALI_MODEL,
    COLPALI_REVISION,
    M3DOCRAG_COMMIT,
    QWEN_MODEL,
    QWEN_REVISION,
)


def make_manifest(tmp_path: Path) -> IndexManifest:
    embeddings = tmp_path / "embeddings.safetensors"
    embeddings.write_bytes(b"embeddings")
    index = tmp_path / "index.faiss"
    index.write_bytes(b"index")
    return IndexManifest(
        mode="docprune",
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
        processor_contract_sha256="d" * 64,
        pruning_config={"attention_threshold": 0.075},
        embeddings_path=embeddings,
        embedding_shape=(12, 128),
        embedding_dtype="float32",
        index_path=index,
        index_sha256=sha256_file(index),
    )


def test_canonical_json_hash_is_order_independent() -> None:
    assert canonical_json_sha256({"b": [2], "a": 1}) == canonical_json_sha256(
        {"a": 1, "b": [2]}
    )


def test_index_manifest_serializes_all_immutable_inputs_and_checksums(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)

    payload = manifest.to_dict()

    assert payload["mode"] == "docprune"
    assert payload["corpus_integrity_sha256"] == "a" * 64
    assert payload["source_order_sha256"] == "b" * 64
    assert payload["resources"]["qwen"] == {"model": QWEN_MODEL, "revision": QWEN_REVISION}
    assert payload["resources"]["colpali"] == {
        "model": COLPALI_MODEL,
        "revision": COLPALI_REVISION,
    }
    assert payload["processor_contract_sha256"] == "d" * 64
    assert payload["pruning_config"] == {"attention_threshold": 0.075}
    assert payload["embeddings"] == {"shape": [12, 128], "dtype": "float32"}
    assert payload["index_sha256"] == sha256_file(manifest.index_path)
    assert payload["runtime_commit"] == "c" * 40
    assert payload["m3docrag_commit"] == M3DOCRAG_COMMIT
    assert payload["manifest_sha256"] == canonical_json_sha256(
        {key: value for key, value in payload.items() if key != "manifest_sha256"}
    )
    manifest.validate_files()


def test_index_manifest_rejects_changed_index_or_invalid_embedding_contract(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)
    manifest.index_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="index SHA-256 mismatch"):
        manifest.validate_files()

    with pytest.raises(ValueError, match="width 128"):
        IndexManifest(
            **{**manifest.__dict__, "embedding_shape": (12, 127), "index_sha256": hashlib.sha256(b"changed").hexdigest()}
        )


def test_manifest_pruning_configuration_is_immutable(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)

    with pytest.raises(TypeError):
        manifest.pruning_config["attention_threshold"] = 0.5  # type: ignore[index]
