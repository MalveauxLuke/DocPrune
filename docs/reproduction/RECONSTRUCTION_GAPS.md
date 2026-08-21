# DocPrune Reconstruction Gaps

The paper and supplement are authoritative for reported behavior. The table
prevents local choices from being misreported as author-provided details.

| Topic | Source status | Current handling | Required validation |
|---|---|---|---|
| Grayscale conversion | Not specified | `reconstruction_default`: BT.601, rounded uint8 | Ablate against mean/channel/PIL grayscale |
| QTP Gaussian sigma and border rule | Not specified | sigma 1.0, replicate padding | Sweep and compare masks/quality |
| ColPali special/non-image tokens | Pinned M3DocRAG v1.2 contract | Persist the complete unpadded ColPali sequence; preserve every nonvisual row and remove only BTP-rejected visual rows | Schema-5 index validator checks active sequence rows, page offsets, and `-1`/row-major raster maps |
| ColPali relevance source grid | Pinned M3DocRAG v1.2 contract | Derive the square `[32, 32]` visual grid and unique row-major raster indices from the verified placeholder span | Processor probe and the real gate record the mapping before retrieval is authorized |
| Qwen resized pixels used by BTP | Not specified | Caller supplies the exact post-resize uint8 image and patch size | Verify against pinned processor pixels and `grid_thw` |
| 2-by-2 group aggregation | Group pruning specified; reduction rule absent | Keep group if any member passes | Ablate any/all/mean |
| Attention head aggregation | Not specified | Arithmetic mean | Compare mean and max |
| CTP attention-score scale | Not specified by the authors | Author-unspecified `reconstruction_default`: aggregate heads, then multiply by the current visual-token count | Validate with real-gate token traces and paper drop-rate/parity comparison |
| CTP generation timing | Last/output token named; exact step absent | Last prompt token during prefill; reduced-query projection at the first threshold crossing | Real CUDA gate and cached all-kept equivalence probe |
| CTP cache compaction | Not specified | Full cache through selected layer, compact deeper caches | GPU decode equivalence and memory trace |
| Model and dataset revisions | IDs given; revisions absent | Run config freezes immutable Qwen, ColPali, backbone, corpus, and upstream revisions | Gate rejects resource or revision drift |
| M3DocRAG source revision | Repository release not given by paper | Contract pinned to `29e6ac2294d6b87075a1d45b8a8df175b214248a` | Verify baseline reproduction |
| Dependency versions | Not reported | Transformers 4.46.3; candidate SOL environment | Build and archive explicit lock |
| Throughput protocol | Samples/s and hardware reported; warmup/order details absent | Freeze source order, one warmup sample, synchronized positive retrieval/page-load/QA/total timing, and positive encoder/decoder boundaries | Independent validation rejects absent/nonfinite/nonpositive fields |
| Measurement identity | Paper does not define an executable identity | Record GPU/compute capability, software, precision, FlashAttention-2, allocator, timer boundaries, warmup, and sample identity; classify A100 as reconstruction | Paired six-cell validator requires exact shared identity |
| FLOP accounting | TFLOPs reported; profiler definition absent | Omit TFLOPs unless profiling is explicitly enabled with one valid definition and positive FLOPs in every cell | Comparator fails closed on partial, mixed, nonfinite, or nonpositive profiler data |

## Hard stops

- Do not silently retain or remove ColPali tokens to make a shape fit.
- Do not interpolate across pages as though they form one image.
- Do not compare speed with different question order, retrieved pages,
  generation settings, precision, attention backend, or hardware.
- Do not call local synthetic-test success a reproduction of paper numbers.
- Do not modify the frozen source dataset or retrieval index in place. BTP
  retrieval embeddings require a derived index with its own manifest.

## Correction attempt history

The prior corrected-control attempt is diagnostic-only and cannot be promoted:
evaluation array `61830411` was canceled before evaluation work, while six
schema-4 indexes (`61830405`–`61830410`) completed with `0:0` but are invalid
for the schema-5 contract. Their scratch artifacts are preserved and must not
be modified or deleted. The fresh active root is
`/scratch/lmalveau/docprune/benchmark-a8d8ca6/attempt-1/` under the immutable
runtime checkout `/home/lmalveau/DocPrune-runtime-a8d8ca6`.
