from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import faiss
import numpy as np
import pytest
import torch
from PIL import Image
from safetensors.torch import load_file, save_file
from transformers.image_utils import ChannelDimension

from docprune.benchmark_config import (
    COLPALI_BACKBONE_MODEL,
    COLPALI_BACKBONE_REVISION,
    COLPALI_MODEL,
    COLPALI_REVISION,
    M3DOCRAG_COMMIT,
    QWEN_MODEL,
    QWEN_REVISION,
)
from docprune.colpali.embedding import ColPaliPageEmbedding
from docprune.config import DocPruneConfig
from docprune.indexing import IndexBuildConfig, build_index, colpali_uint8_raster


class FakeImageProcessor:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        *,
        images,
        return_tensors: str,
        do_rescale: bool,
        do_normalize: bool,
        data_format: ChannelDimension,
    ):
        self.calls.append(
            {
                "return_tensors": return_tensors,
                "do_rescale": do_rescale,
                "do_normalize": do_normalize,
                "data_format": data_format,
            }
        )
        arrays = [np.asarray(image.resize((28, 28)), dtype=np.uint8) for image in images]
        pixels = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).to(torch.float32)
        return {"pixel_values": pixels}


class FakeProcessor:
    image_seq_length = 4

    def __init__(self) -> None:
        self.image_processor = FakeImageProcessor()

    def process_images(self, images: list[Image.Image]) -> dict[str, torch.Tensor]:
        pixels = torch.from_numpy(np.stack([np.asarray(image) for image in images])).permute(
            0, 3, 1, 2
        )
        return {
            "input_ids": torch.tensor([[7, 99, 99, 99, 99, 8, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 0]]),
            "pixel_values": pixels.to(torch.float32),
        }


class FakeDataset:
    source_order_sha256 = "d" * 64

    def __init__(self, pages: dict[str, tuple[Image.Image, ...]]) -> None:
        self.document_ids = tuple(pages)
        self.pages = pages

    def load_pages(self, doc_id: str) -> tuple[Image.Image, ...]:
        return self.pages[doc_id]


def make_build_config(
    tmp_path: Path,
    *,
    mode: str,
    page_count: int,
    dataset: FakeDataset,
) -> IndexBuildConfig:
    contract = tmp_path / f"processor-{mode}-{page_count}.json"
    contract.write_text("{}", encoding="utf-8")
    run = SimpleNamespace(
        mode=mode,
        page_count=page_count,
        corpus=SimpleNamespace(integrity_sha256="a" * 64),
        runtime_commit="c" * 40,
        m3docrag_commit=M3DOCRAG_COMMIT,
        qwen_model=QWEN_MODEL,
        qwen_revision=QWEN_REVISION,
        colpali_model=COLPALI_MODEL,
        colpali_revision=COLPALI_REVISION,
        colpali_backbone_model=COLPALI_BACKBONE_MODEL,
        colpali_backbone_revision=COLPALI_BACKBONE_REVISION,
        processor_contract_path=contract,
    )
    model = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(image_token_index=99)))
    return IndexBuildConfig(
        run_config=run,
        dataset=dataset,
        model=model,
        processor=FakeProcessor(),
        pruning_config=DocPruneConfig.paper_defaults(),
    )


def fake_page_encoder(model, batch, mapping, patch_keep_mask) -> ColPaliPageEmbedding:
    del model
    keep = torch.as_tensor(patch_keep_mask, dtype=torch.bool)
    rasters = torch.arange(4, dtype=torch.long)[keep]
    code = float(batch["pixel_values"][0, 0, 0, 0])
    visual = torch.zeros(1, len(rasters), 128, dtype=torch.float32)
    visual[0, :, 0] = code
    visual[0, :, 1] = rasters.to(torch.float32)
    sequence_keep = torch.ones(batch["input_ids"].shape[1], dtype=torch.bool)
    sequence_keep[mapping.visual_start : mapping.visual_stop] = keep
    compact_ids = batch["input_ids"][:, sequence_keep]
    compact_attention = batch["attention_mask"][:, sequence_keep]
    return ColPaliPageEmbedding(
        embeddings=torch.zeros(1, compact_ids.shape[1], 128),
        visual_embeddings=visual,
        raster_indices=rasters,
        input_ids=compact_ids,
        attention_mask=compact_attention,
    )


def test_colpali_rasterizer_returns_exact_post_resize_uint8_pixels() -> None:
    processor = FakeProcessor()
    image = Image.fromarray(np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3), mode="RGB")

    raster = colpali_uint8_raster(processor, image)

    assert raster.dtype == torch.uint8
    assert raster.shape == (1, 3, 28, 28)
    assert processor.image_processor.calls == [
        {
            "return_tensors": "pt",
            "do_rescale": False,
            "do_normalize": False,
            "data_format": ChannelDimension.FIRST,
        }
    ]
    expected = torch.from_numpy(np.asarray(image.resize((28, 28))).copy()).permute(2, 0, 1)
    assert torch.equal(raster[0], expected)


def test_build_index_persists_source_order_visual_rows_mapping_and_faiss(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("docprune.indexing.assert_supported_colpali", lambda model, processor: None)
    monkeypatch.setattr("docprune.indexing.encode_colpali_page", fake_page_encoder)
    pages = {
        "doc-b": (
            Image.new("RGB", (28, 28), color=(10, 0, 0)),
            Image.new("RGB", (28, 28), color=(20, 0, 0)),
        ),
        "doc-a": (
            Image.new("RGB", (28, 28), color=(30, 0, 0)),
            Image.new("RGB", (28, 28), color=(40, 0, 0)),
        ),
    }
    config = make_build_config(tmp_path, mode="all-kept", page_count=4, dataset=FakeDataset(pages))

    result = build_index(config, "all-kept", tmp_path / "indexes")

    assert result.documents_built == 2
    assert result.documents_resumed == 0
    assert result.manifest.mode == "all-kept"
    assert result.manifest.embedding_shape == (16, 128)
    tensors = load_file(result.manifest.embeddings_path)
    assert tensors["embeddings"][:, 0].tolist() == [10.0] * 4 + [20.0] * 4 + [30.0] * 4 + [40.0] * 4
    assert tensors["raster_indices"].tolist() == [0, 1, 2, 3] * 4
    token_map = json.loads(result.token2pageuid_path.read_text(encoding="utf-8"))
    assert token_map == (
        [{"doc_id": "doc-b", "page_index": 0}] * 4
        + [{"doc_id": "doc-b", "page_index": 1}] * 4
        + [{"doc_id": "doc-a", "page_index": 0}] * 4
        + [{"doc_id": "doc-a", "page_index": 1}] * 4
    )
    index = faiss.read_index(str(result.manifest.index_path))
    assert index.d == 128
    assert index.ntotal == 16
    result.manifest.validate_files()


def test_docprune_uses_page_count_retrieval_threshold_and_separate_mode_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("docprune.indexing.assert_supported_colpali", lambda model, processor: None)
    seen_masks: list[list[bool]] = []

    def recording_encoder(model, batch, mapping, keep):
        seen_masks.append(torch.as_tensor(keep).tolist())
        return fake_page_encoder(model, batch, mapping, keep)

    monkeypatch.setattr("docprune.indexing.encode_colpali_page", recording_encoder)
    pixels = np.full((28, 28, 3), 255, dtype=np.uint8)
    pixels[:14, :14] = 0
    dataset = FakeDataset({"doc": (Image.fromarray(pixels, mode="RGB"),)})
    docprune = make_build_config(tmp_path, mode="docprune", page_count=1, dataset=dataset)
    baseline = make_build_config(tmp_path, mode="all-kept", page_count=1, dataset=dataset)

    pruned_result = build_index(docprune, "docprune", tmp_path / "indexes")
    baseline_result = build_index(baseline, "all-kept", tmp_path / "indexes")

    assert seen_masks == [[True, False, False, False], [True, True, True, True]]
    assert pruned_result.manifest.embedding_shape == (1, 128)
    assert baseline_result.manifest.embedding_shape == (4, 128)
    assert pruned_result.mode_root == tmp_path / "indexes" / "docprune"
    assert baseline_result.mode_root == tmp_path / "indexes" / "all-kept"
    assert (
        pruned_result.manifest.to_dict()["manifest_sha256"]
        != baseline_result.manifest.to_dict()["manifest_sha256"]
    )


def test_resume_skips_only_manifest_identical_atomically_completed_documents(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("docprune.indexing.assert_supported_colpali", lambda model, processor: None)
    pages = {
        "doc-b": (Image.new("RGB", (28, 28), color=(10, 0, 0)),),
        "doc-a": (Image.new("RGB", (28, 28), color=(30, 0, 0)),),
    }
    config = make_build_config(tmp_path, mode="all-kept", page_count=4, dataset=FakeDataset(pages))

    def interrupted_encoder(model, batch, mapping, keep):
        if int(batch["pixel_values"][0, 0, 0, 0]) == 30:
            raise RuntimeError("scheduler interruption")
        return fake_page_encoder(model, batch, mapping, keep)

    monkeypatch.setattr("docprune.indexing.encode_colpali_page", interrupted_encoder)
    with pytest.raises(RuntimeError, match="scheduler interruption"):
        build_index(config, "all-kept", tmp_path / "indexes")

    first_entry_path = tmp_path / "indexes" / "all-kept" / "completion-ledger" / "000000.json"
    first_entry = json.loads(first_entry_path.read_text(encoding="utf-8"))
    first_entry["shape"] = [999, 128]
    first_entry_path.write_text(json.dumps(first_entry), encoding="utf-8")
    with pytest.raises(ValueError, match="completion ledger entry"):
        build_index(config, "all-kept", tmp_path / "indexes")
    first_entry["shape"] = [4, 128]
    first_entry_path.write_text(
        json.dumps(first_entry, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    resumed_codes: list[int] = []

    def resumed_encoder(model, batch, mapping, keep):
        resumed_codes.append(int(batch["pixel_values"][0, 0, 0, 0]))
        return fake_page_encoder(model, batch, mapping, keep)

    monkeypatch.setattr("docprune.indexing.encode_colpali_page", resumed_encoder)
    result = build_index(config, "all-kept", tmp_path / "indexes")

    assert resumed_codes == [30]
    assert result.documents_built == 1
    assert result.documents_resumed == 1
    assert len(tuple((result.mode_root / "completion-ledger").glob("*.json"))) == 2

    changed = make_build_config(tmp_path, mode="all-kept", page_count=2, dataset=FakeDataset(pages))
    with pytest.raises(ValueError, match="manifest-identical"):
        build_index(changed, "all-kept", tmp_path / "indexes")


def test_manifest_binds_mapping_ledger_and_detects_index_checksum_change(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("docprune.indexing.assert_supported_colpali", lambda model, processor: None)
    monkeypatch.setattr("docprune.indexing.encode_colpali_page", fake_page_encoder)
    dataset = FakeDataset({"doc": (Image.new("RGB", (28, 28), color=(10, 0, 0)),)})
    config = make_build_config(tmp_path, mode="all-kept", page_count=4, dataset=dataset)
    result = build_index(config, "all-kept", tmp_path / "indexes")
    payload = result.manifest.to_dict()

    assert (
        payload["token2pageuid_sha256"]
        == hashlib.sha256(result.token2pageuid_path.read_bytes()).hexdigest()
    )
    assert (
        payload["completion_ledger_sha256"]
        == hashlib.sha256(result.completion_ledger_path.read_bytes()).hexdigest()
    )

    result.manifest.index_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="index SHA-256 mismatch"):
        result.manifest.validate_files()


def test_build_rejects_mode_root_symlink_before_writing_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("docprune.indexing.assert_supported_colpali", lambda model, processor: None)
    dataset = FakeDataset({"doc": (Image.new("RGB", (28, 28), color=(10, 0, 0)),)})
    config = make_build_config(tmp_path, mode="all-kept", page_count=4, dataset=dataset)
    output = tmp_path / "indexes"
    output.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    (output / "all-kept").symlink_to(external, target_is_directory=True)

    with pytest.raises(ValueError, match="artifact root"):
        build_index(config, "all-kept", output)

    assert not (external / "documents").exists()


@pytest.mark.parametrize(
    "target_name", ["documents/000000.safetensors", "completion-ledger/000000.json"]
)
def test_build_rejects_per_document_external_symlink_targets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, target_name: str
) -> None:
    monkeypatch.setattr("docprune.indexing.assert_supported_colpali", lambda model, processor: None)
    dataset = FakeDataset({"doc": (Image.new("RGB", (28, 28), color=(10, 0, 0)),)})
    config = make_build_config(tmp_path, mode="all-kept", page_count=4, dataset=dataset)
    mode_root = tmp_path / "indexes" / "all-kept"
    target = mode_root / target_name
    target.parent.mkdir(parents=True)
    external = tmp_path / "external-target"
    external.write_text("external", encoding="utf-8")
    target.symlink_to(external)

    with pytest.raises(ValueError, match="symbolic link"):
        build_index(config, "all-kept", tmp_path / "indexes")

    assert external.read_text(encoding="utf-8") == "external"


def test_resume_rejects_empty_page_and_bad_raster_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("docprune.indexing.assert_supported_colpali", lambda model, processor: None)
    monkeypatch.setattr("docprune.indexing.encode_colpali_page", fake_page_encoder)
    pages = {
        "doc": (
            Image.new("RGB", (28, 28), color=(10, 0, 0)),
            Image.new("RGB", (28, 28), color=(20, 0, 0)),
        )
    }
    config = make_build_config(tmp_path, mode="all-kept", page_count=4, dataset=FakeDataset(pages))
    result = build_index(config, "all-kept", tmp_path / "indexes")
    document_path = result.mode_root / "documents" / "000000.safetensors"
    ledger_path = result.mode_root / "completion-ledger" / "000000.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    tensors = load_file(document_path)

    for offsets, rasters, message in (
        (
            torch.tensor([0, 0, 8], dtype=torch.int64),
            tensors["raster_indices"],
            "strictly increasing",
        ),
        (
            tensors["page_offsets"],
            torch.tensor([0, 0, 2, 3, 0, 1, 2, 3], dtype=torch.int64),
            "raster indices",
        ),
        (
            tensors["page_offsets"],
            torch.tensor([0, 1, 2, 3, 0, 1, 2, 1024], dtype=torch.int64),
            "raster indices",
        ),
    ):
        save_file(
            {
                "embeddings": tensors["embeddings"],
                "raster_indices": rasters,
                "page_offsets": offsets,
            },
            document_path,
        )
        mutated = dict(ledger)
        mutated["sha256"] = hashlib.sha256(document_path.read_bytes()).hexdigest()
        mutated["page_offsets"] = offsets.tolist()
        ledger_path.write_text(json.dumps(mutated, sort_keys=True), encoding="utf-8")

        with pytest.raises(ValueError, match=message):
            build_index(config, "all-kept", tmp_path / "indexes")

        save_file(
            {
                "embeddings": tensors["embeddings"],
                "raster_indices": tensors["raster_indices"],
                "page_offsets": tensors["page_offsets"],
            },
            document_path,
        )
        ledger_path.write_text(
            json.dumps(ledger, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
