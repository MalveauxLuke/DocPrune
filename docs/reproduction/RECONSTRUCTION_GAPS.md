> Historical baseline evidence. Commands and status below describe prior experiments; current work is defined in `docs/ExperimentPlan.md`.

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

## Explicit reconstruction policies

CTP uses an arithmetic mean over attention heads, then scales the raw mean by
the current visual-token count. Gate evidence records both semantics as
`mean_head_attention_scores_before_visual_count_scaling` and
`raw_mean_times_current_visual_count`, respectively, and records the last
prompt token during prefill as the timing/query boundary. QTP uses BT.601
rounded-uint8 grayscale, Gaussian sigma 1.0 with replicate padding, and
any-member-keeps 2-by-2 groups. Sparse raster maps retain `-1` holes for
nonvisual rows; only verified visual rows are eligible for QTP and pruning.
These are reconstruction choices, not paper claims; the canonical schema-5
loader performs substantive page-offset and ledger validation.

The acquired reconstruction corpus validates 2,441 questions, 3,366 PDFs, and
44,638 pages. The paper reports 2,441 / 3,368 / 41,005; the discrepancy is
documented rather than corrected in the corpus, so this benchmark must not be
described as paper-identical data.

The six-cell comparison is deliberately only DocPrune versus all-kept at
top-1, top-2, and top-4 retrieval. It is not a reproduction of FastV,
DivPrune, VTW, or the paper's full Table 2. ACC is likewise not added: the
active source metric definitions and comparison contract unambiguously expose
EM, F1, modality F1, and hop F1, but no unambiguous paper ACC definition.

## Hard stops

- Do not silently retain or remove ColPali tokens to make a shape fit.
- Do not interpolate across pages as though they form one image.
- Do not compare speed with different question order, retrieved pages,
  generation settings, precision, attention backend, or hardware.
- Do not call local synthetic-test success a reproduction of paper numbers.
- Do not modify the frozen source dataset or retrieval index in place. BTP
  retrieval embeddings require a derived index with its own manifest.

## Correction attempt history

The prior corrected-control attempt at
`/scratch/lmalveau/docprune/benchmark-6c19bfc/attempt-2/` is diagnostic-only and
cannot be promoted:
evaluation array `61830411` was canceled before evaluation work, while six
schema-4 indexes (`61830405`–`61830410`) completed with `0:0` but are invalid
for the schema-5 contract. Their scratch artifacts are preserved and must not
be modified or deleted. The next failed root
`/scratch/lmalveau/docprune/benchmark-02385b3/attempt-2/` ran gate `61943239`
for 29s on `scg011` and failed before GPU/model work because the runtime
validator hard-coded `attempt-1`; downstream jobs `61943240`–`61943247` were
auto-canceled. Preserve that root and all IDs unchanged. The old-runtime
production graph (`61968793`, `61968794`–`61968797`, `61968799`–`61968800`,
`61968821`, `61968823`) was canceled at `2026-08-21 17:50:21` with no nodes
and elapsed 0; probes `61969352` and `61969614` were canceled with no
node/elapsed 0. L40 attempts `61970394`, `61972695`, `61973090`, and `61974092`,
and A100-40GB hedge `61974173` are immutable failed/canceled history and are
not promotable. The fresh active root is
`/scratch/lmalveau/docprune/benchmark-15301ea/attempt-1/` under the
immutable runtime checkout `/home/lmalveau/DocPrune-runtime-15301ea`.

The corrected Phase-A runtime is
`15301ea557288a4f67fc3c85228bf5e148014d17`; the control identity is sealed in
the fresh attempt's `control.json`. The gate now persists a fixture-marked
schema-5 mini-index through the production indexing path and authenticates it
through the canonical loader. Retrieval returns ordered `(doc_id,page_index)`
rows, and QA image loading/features are asserted against those exact rows.
Actual ColPali processor/model counters prove one query encode per fixed QID
and no QA-time calls. The legacy no-raster production branch remains deferred;
schema-5 production cannot enter it.

The active resource contract is HTC/public QoS with an unconstrained
`gpu:l40:1`, 8 CPUs, 128G, and 1 hour for the semantic gate; the six indexes
use the same L40 resources for 4 hours. The six-cell evaluation array uses
HTC/public QoS, exact `gpu:a100:1` plus `a100_80`, 8 CPUs, 128G, 4 hours, and
`%6` concurrency. Only those exact A100-80 rows support efficiency claims;
the gate/index hardware is non-measurement and the comparator is HTC CPU-only.
