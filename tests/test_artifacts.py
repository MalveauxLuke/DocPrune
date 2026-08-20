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


def make_manifest(
    tmp_path: Path, *, mode: str = "docprune", artifact_root: Path | None = None
) -> IndexManifest:
    root = artifact_root if artifact_root is not None else tmp_path / mode
    root.mkdir(parents=True)
    embeddings = root / "embeddings.safetensors"
    save_file(
        {
            "embeddings": torch.zeros(12, 128, dtype=torch.float32),
            "raster_indices": torch.arange(12, dtype=torch.int64),
        },
        embeddings,
    )
    metadata = root / "embeddings.json"
    metadata.write_text(
        json.dumps(
            {
                "shape": [12, 128],
                "dtype": "float32",
                "document_ids": ["doc"],
            }
        )
    )
    build_manifest = {
        "schema_version": 1,
        "source_order_sha256": "b" * 64,
        "document_ids": ["doc"],
    }
    build_manifest["build_manifest_sha256"] = canonical_json_sha256(build_manifest)
    (root / "build-manifest.json").write_text(json.dumps(build_manifest), encoding="utf-8")
    contract = root / "processor-contract.json"
    contract.write_text("{}")
    token2pageuid = root / "token2pageuid.json"
    token2pageuid.write_text(
        json.dumps([{"doc_id": "doc", "page_index": 0}] * 12), encoding="utf-8"
    )
    ledger = root / "completion-ledger.json"
    ledger.write_text(
        json.dumps(
            [
                {
                    "schema_version": 1,
                    "ordinal": 0,
                    "doc_id": "doc",
                    "build_manifest_sha256": build_manifest["build_manifest_sha256"],
                    "document_path": str(root / "documents" / "000000.safetensors"),
                    "sha256": "d" * 64,
                    "shape": [12, 128],
                    "dtype": "float32",
                    "pages": [{"doc_id": "doc", "page_index": 0}],
                    "page_offsets": [0, 12],
                }
            ]
        ),
        encoding="utf-8",
    )
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
        embeddings_sha256=sha256_file(embeddings),
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

    assert payload["schema_version"] == 4
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
    assert payload["embeddings_sha256"] == sha256_file(manifest.embeddings_path)
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

    with pytest.raises(ValueError, match="(safetensors|embeddings SHA-256)"):
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


def test_manifest_rejects_tampered_embedding_values_even_when_shape_and_dtype_match(
    tmp_path: Path,
) -> None:
    manifest = make_manifest(tmp_path)
    save_file(
        {
            "embeddings": torch.ones(12, 128, dtype=torch.float32),
            "raster_indices": torch.arange(12, dtype=torch.int64),
        },
        manifest.embeddings_path,
    )

    with pytest.raises(ValueError, match="embeddings SHA-256 mismatch"):
        manifest.validate_files()


def test_manifest_checks_faiss_rows_against_embedding_payload(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)
    changed = faiss.IndexFlatIP(128)
    changed.add(np.ones((12, 128), dtype=np.float32))
    faiss.write_index(changed, str(manifest.index_path))
    object.__setattr__(manifest, "index_sha256", sha256_file(manifest.index_path))

    with pytest.raises(ValueError, match="FAISS rows do not match"):
        manifest.validate_files()


def test_manifest_rejects_same_width_l2_index(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)
    changed = faiss.IndexFlatL2(128)
    changed.add(np.zeros((12, 128), dtype=np.float32))
    faiss.write_index(changed, str(manifest.index_path))
    object.__setattr__(manifest, "index_sha256", sha256_file(manifest.index_path))

    with pytest.raises(ValueError, match="IndexFlatIP"):
        manifest.validate_files()


def test_manifest_rejects_token_map_that_disagrees_with_completion_ledger(
    tmp_path: Path,
) -> None:
    manifest = make_manifest(tmp_path)
    rows = json.loads(manifest.token2pageuid_path.read_text(encoding="utf-8"))
    rows[0] = {"doc_id": "wrong", "page_index": 0}
    manifest.token2pageuid_path.write_text(json.dumps(rows), encoding="utf-8")
    object.__setattr__(manifest, "token2pageuid_sha256", sha256_file(manifest.token2pageuid_path))

    with pytest.raises(ValueError, match="completion ledger"):
        manifest.validate_files()


def test_manifest_rejects_duplicate_completion_ledger_ordinal(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)
    entry = json.loads(manifest.completion_ledger_path.read_text(encoding="utf-8"))[0]
    manifest.completion_ledger_path.write_text(json.dumps([entry, entry]), encoding="utf-8")
    object.__setattr__(
        manifest, "completion_ledger_sha256", sha256_file(manifest.completion_ledger_path)
    )

    with pytest.raises(ValueError, match="completion ledger"):
        manifest.validate_files()


@pytest.mark.parametrize(
    "raster_indices",
    [
        torch.tensor([0, 0, *range(2, 12)], dtype=torch.int64),
        torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1024], dtype=torch.int64),
        torch.tensor([1, 0, *range(2, 12)], dtype=torch.int64),
    ],
)
def test_manifest_rejects_invalid_raster_identity_or_order(
    tmp_path: Path, raster_indices: torch.Tensor
) -> None:
    manifest = make_manifest(tmp_path)
    save_file(
        {
            "embeddings": torch.zeros(12, 128, dtype=torch.float32),
            "raster_indices": raster_indices,
        },
        manifest.embeddings_path,
    )
    object.__setattr__(manifest, "embeddings_sha256", sha256_file(manifest.embeddings_path))

    with pytest.raises(ValueError, match="raster indices"):
        manifest.validate_files()


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


def test_manifest_accepts_all_kept_root_below_unrelated_docprune_ancestor(
    tmp_path: Path,
) -> None:
    root = tmp_path / "docprune" / "benchmark-6c19bfc" / "indexes" / "all-kept"

    manifest = make_manifest(tmp_path, mode="all-kept", artifact_root=root)

    manifest.validate_files()


def test_manifest_accepts_docprune_root_below_unrelated_all_kept_ancestor(
    tmp_path: Path,
) -> None:
    root = tmp_path / "all-kept" / "benchmark-6c19bfc" / "indexes" / "docprune"

    manifest = make_manifest(tmp_path, mode="docprune", artifact_root=root)

    manifest.validate_files()


@pytest.mark.parametrize(
    ("mode", "wrong_leaf"),
    [("all-kept", "docprune"), ("docprune", "all-kept")],
)
def test_manifest_rejects_wrong_mode_artifact_root(
    tmp_path: Path, mode: str, wrong_leaf: str
) -> None:
    root = tmp_path / "benchmark-6c19bfc" / "indexes" / wrong_leaf

    with pytest.raises(ValueError, match=f"manifest mode/path mismatch for {mode}"):
        make_manifest(tmp_path, mode=mode, artifact_root=root)


def test_manifest_rejects_nested_cross_mode_artifact_path(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path, mode="docprune")
    arguments = dict(manifest.__dict__)
    arguments["index_path"] = manifest.artifact_root / "all-kept" / "index.faiss"

    with pytest.raises(ValueError, match="manifest mode/path mismatch for docprune"):
        IndexManifest(**arguments)


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
