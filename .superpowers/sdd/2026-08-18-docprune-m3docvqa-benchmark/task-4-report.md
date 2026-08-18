# Task 4 report: QTP mapping, Qwen preprocessing, and answerers

## RED evidence

The new behavioral tests were written before the implementation. They first
failed during collection because `dense_relevance_from_sparse_tokens`,
`prepare_qwen_page`, and `AllKeptQwenAnswerer` did not exist. A separate
fail-closed attention-mask test then failed because the adapter accepted a
padded mask even though its manual cache path had no padding support.

## GREEN evidence

With the implementation in place:

- targeted Task 4 tests pass (`tests/test_pipeline.py`,
  `tests/test_answerers.py`, `tests/qwen2vl/test_preprocessing.py`, and
  `tests/qwen2vl/test_model.py`); the real-model test skips by default;
- the complete CPU suite passes: **167 passed, 1 skipped**;
- Ruff passes for every changed Python file;
- the opt-in real-model test uses only `local_files_only=True`, requires CUDA,
  the pinned Qwen revision, a local model path, and a fixed local probe page,
  so ordinary CPU tests cannot download or load the 7B model.

## Decisions and implementation

- Sparse ColPali visual rows are scattered by their original raster IDs into a
  source grid. Missing cells are represented as `-inf` by the public helper and
  carried as an explicit validity mask through resize/smoothing; QTP group
  decisions require all fine cells in a rejected group to remain valid.
- Qwen page preprocessing delegates smart-resize, bicubic resampling,
  normalization, patch flattening, and `image_grid_thw` to the pinned
  Transformers 4.46.3 processor. The retained uint8 raster is reconstructed at
  the exact returned geometry, and merge placeholders are counted in the
  processor's page order (`prod(grid) // 2**2`).
- Stock and DocPrune answerers use the exact
  `question: $question\noutput only answer.` prompt, image-first chat payload,
  greedy `max_new_tokens=128`, and suffix-only decoding. DocPrune retrieves
  sparse page embeddings through Task 3's `encode_colpali_page` path and gets
  query embeddings from that same ColPali instance.
- The Qwen adapter now rejects padded attention masks because compact cache
  support is only implemented for unpadded batch-one sequences. It exposes the
  first-step logits needed by the opt-in parity gate while preserving the
  existing tiny-model generation and trace contract.

## Self-review and limitations

- The real-model parity test is intentionally opt-in and was not run here
  because no approved local 7B checkpoint/probe CUDA resource was present.
- ColPali and Qwen resource loading remains an integration/factory concern for
  Task 5; this task does not download models or add CLI/factory behavior.
- Padding support is explicitly fail-closed rather than guessed. Batched or
  padded generation should not use this adapter until compact 2-D mask support
  is implemented and independently tested.
- The all-kept parity oracle supplies the exact M-ROPE positions. A high-level
  Transformers forward without `position_ids` uses linear fallback positions
  in 4.46.3 and is therefore not a valid stock generation oracle.
