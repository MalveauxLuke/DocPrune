# DocPrune Reconstruction Gaps

The paper and supplement are authoritative for reported behavior. The table
prevents local choices from being misreported as author-provided details.

| Topic | Source status | Current handling | Required validation |
|---|---|---|---|
| Grayscale conversion | Not specified | `reconstruction_default`: BT.601, rounded uint8 | Ablate against mean/channel/PIL grayscale |
| QTP Gaussian sigma and border rule | Not specified | sigma 1.0, replicate padding | Sweep and compare masks/quality |
| ColPali special/non-image tokens | Pinned M3DocRAG v1.2 contract | Require exactly `image_seq_length` contiguous, unpadded PaliGemma image placeholders; preserve all non-image tokens | Processor probe records the visual span and fails closed on ambiguity |
| ColPali relevance source grid | Pinned M3DocRAG v1.2 contract | Derive the square visual grid and unique row-major raster indices from the verified placeholder span | Processor probe records the raster mapping before retrieval is authorized |
| Qwen resized pixels used by BTP | Not specified | Caller supplies the exact post-resize uint8 image and patch size | Verify against pinned processor pixels and `grid_thw` |
| 2-by-2 group aggregation | Group pruning specified; reduction rule absent | Keep group if any member passes | Ablate any/all/mean |
| Attention head aggregation | Not specified | Arithmetic mean | Compare mean and max |
| CTP attention-score scale | Not specified by the authors | Author-unspecified `reconstruction_default`: aggregate heads, then multiply by the current visual-token count | Validate with real-gate token traces and paper drop-rate/parity comparison |
| CTP generation timing | Last/output token named; exact step absent | Last prompt token during prefill | Compare documented alternatives if parity misses |
| CTP cache compaction | Not specified | Full cache through selected layer, compact deeper caches | GPU decode equivalence and memory trace |
| Model and dataset revisions | IDs given; revisions absent | Run must supply immutable revisions | Record HF commit hashes before any benchmark |
| M3DocRAG source revision | Repository release not given by paper | Contract pinned to `29e6ac2294d6b87075a1d45b8a8df175b214248a` | Verify baseline reproduction |
| Dependency versions | Not reported | Transformers 4.46.3; candidate SOL environment | Build and archive explicit lock |
| Throughput protocol | Samples/s and hardware reported; warmup/order details absent | No local claim | Freeze order, warmup, synchronization, and sample count |
| FLOP accounting | TFLOPs reported; profiler definition absent | No local claim | Record profiler events and counting formula |

## Hard stops

- Do not silently retain or remove ColPali tokens to make a shape fit.
- Do not interpolate across pages as though they form one image.
- Do not compare speed with different question order, retrieved pages,
  generation settings, precision, attention backend, or hardware.
- Do not call local synthetic-test success a reproduction of paper numbers.
- Do not modify the frozen source dataset or retrieval index in place. BTP
  retrieval embeddings require a derived index with its own manifest.
