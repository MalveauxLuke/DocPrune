from __future__ import annotations

import torch

from docprune.colpali.embedding import encode_colpali_page
from docprune.processor_probe import resolve_colpali_visual_mapping


def make_page_batch() -> tuple[dict[str, torch.Tensor], object]:
    input_ids = torch.tensor([[11, 127, 127, 127, 127, 23, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0]])
    mapping = resolve_colpali_visual_mapping(
        input_ids=input_ids,
        attention_mask=attention_mask,
        image_token_id=127,
        image_seq_length=4,
    )
    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": torch.arange(3 * 28 * 28, dtype=torch.float32).reshape(1, 3, 28, 28)
        / (3 * 28 * 28),
    }
    return batch, mapping


def test_all_kept_page_embedding_numerically_matches_stock_colpali(tiny_colpali) -> None:
    batch, mapping = make_page_batch()

    with torch.no_grad():
        stock = tiny_colpali(**batch)
        encoded = encode_colpali_page(
            tiny_colpali,
            batch,
            mapping,
            torch.ones(4, dtype=torch.bool),
        )

    torch.testing.assert_close(encoded.embeddings, stock, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(encoded.visual_embeddings, stock[:, 1:5], rtol=1e-5, atol=1e-6)
    assert encoded.raster_indices.tolist() == [0, 1, 2, 3]
    assert encoded.input_ids.tolist() == batch["input_ids"].tolist()


def test_all_kept_casts_processor_pixels_to_half_precision_model(tiny_colpali) -> None:
    model = tiny_colpali.half()
    batch, mapping = make_page_batch()
    stock_batch = {**batch, "pixel_values": batch["pixel_values"].half()}

    with torch.no_grad():
        stock = model(**stock_batch)
        encoded = encode_colpali_page(
            model,
            batch,
            mapping,
            torch.ones(4, dtype=torch.bool),
        )

    assert encoded.embeddings.dtype == torch.float16
    assert encoded.embeddings.shape == (1, 7, 128)
    torch.testing.assert_close(encoded.embeddings, stock, rtol=0, atol=0)


def test_sparse_path_casts_processor_pixels_to_half_precision_vision_tower(tiny_colpali) -> None:
    model = tiny_colpali.half()
    batch, mapping = make_page_batch()

    with torch.no_grad():
        encoded = encode_colpali_page(
            model,
            batch,
            mapping,
            torch.tensor([True, False, True, False]),
        )

    assert encoded.embeddings.dtype == torch.float16
    assert encoded.visual_embeddings.shape == (1, 2, 128)


def test_sparse_page_removes_only_rejected_image_placeholders_and_keeps_raster_identity(
    tiny_colpali,
) -> None:
    batch, mapping = make_page_batch()

    with torch.no_grad():
        encoded = encode_colpali_page(
            tiny_colpali,
            batch,
            mapping,
            torch.tensor([True, False, True, False]),
        )

    assert encoded.input_ids.tolist() == [[11, 127, 127, 23, 0]]
    assert encoded.attention_mask.tolist() == [[1, 1, 1, 1, 0]]
    assert encoded.embeddings.shape == (1, 5, 128)
    assert encoded.visual_embeddings.shape == (1, 2, 128)
    assert encoded.raster_indices.tolist() == [0, 2]
    assert torch.isfinite(encoded.embeddings).all()
    torch.testing.assert_close(
        encoded.visual_embeddings,
        encoded.embeddings[:, 1:3],
        rtol=0,
        atol=0,
    )


def test_page_embedding_rejects_mapping_or_mask_drift(tiny_colpali) -> None:
    batch, mapping = make_page_batch()
    bad_batch = {**batch, "input_ids": batch["input_ids"].clone()}
    bad_batch["input_ids"][0, 1] = 12

    for candidate_batch, keep in (
        (bad_batch, torch.ones(4, dtype=torch.bool)),
        (batch, torch.zeros(4, dtype=torch.bool)),
        (batch, torch.ones(3, dtype=torch.bool)),
    ):
        try:
            encode_colpali_page(tiny_colpali, candidate_batch, mapping, keep)
        except ValueError:
            pass
        else:
            raise AssertionError("mapping and mask drift must fail closed")
