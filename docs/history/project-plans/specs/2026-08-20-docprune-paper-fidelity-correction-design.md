# DocPrune Paper-Fidelity Correction Design

## Decision

Correct the benchmark before any full evaluation. The paper-to-code audit found
that the pruning equations are implemented faithfully, but the retrieval/QTP
dataflow and efficiency protocol are not yet faithful enough for the requested
comparison. Attempt 2 remains preserved as diagnostic evidence. Its pending
evaluation array is canceled, and no result from that attempt may be reported as
the final benchmark.

## Authoritative observations

DocPrune section 3.3 defines QTP over document and question token embeddings
already produced by the retrieval stage. The pinned M3DocRAG index branch uses
the complete ColPali page output for retrieval, approximately 1,030 rows per
page, rather than only the 1,024 image-placeholder rows. The supplement requires
Qwen2-VL's FlashAttention-compatible path and recomputation of attention for
only the final query at the selected CTP layer. Section 4.2 reports vision
encoder and LLM decoder throughput separately.

## Corrected retrieval artifact

Each page contributes its complete unpadded ColPali output sequence to FAISS.
Every persisted row has the existing structured document/page identity. A
parallel signed `raster_indices` tensor uses `-1` for nonvisual rows and the
original row-major `[0, 1024)` raster index for visual rows. Validators require
exactly one increasing visual subsequence per page, permit nonvisual sentinels
only outside that subsequence, and bind the artifact to the pinned 32-by-32
ColPali visual grid. The index schema is bumped so older visual-only indexes
cannot be loaded as corrected indexes.

All-kept page embeddings must numerically equal the complete pinned stock
ColPali output, not merely its visual slice. DocPrune still removes only
BTP-rejected image placeholders and preserves every nonvisual sequence row.

## Corrected retrieval-to-QTP dataflow

Indexed retrieval encodes the question once. It uses exactly `k=top_k` token
neighbors per query token and the pinned upstream MaxSim aggregation; it does
not over-fetch. The retrieval result contains the ordered pages, the already
computed question-token embeddings, and the persisted visual embeddings/raster
indices for each selected page. QTP consumes that context directly. It must not
invoke either ColPali processor or model during QA.

The public result JSON retains only serializable page identities and scores.
Tensor context is an in-memory execution detail validated against the selected
page identities and index row map.

## Qwen backend and timing

The factory loads Qwen2-VL with the pinned upstream
`attn_implementation="flash_attention_2"`, bfloat16 weights, and bfloat16 vision
configuration. CTP recomputes only the final query projection against the full
key sequence at the first comprehension-threshold crossing; it must not create
a full square attention matrix for this recomputation.

Every measured row retains end-to-end retrieval and QA wall time and adds
positive synchronized `encoder_seconds` and `decoder_seconds`. Encoder timing
covers the Qwen vision encoder. Decoder timing covers the language-model
prefill and greedy decode. BTP/QTP preparation remains visible in end-to-end QA
time but is not mislabeled as encoder or decoder execution. Page loading is
recorded explicitly, and a total sample wall time spans retrieval, loading, and
QA so no pipeline interval is mislabeled as end-to-end. Aggregates report
samples per encoder second and samples per decoder second. Visual-token drop is
the arithmetic mean of per-sample drop proportions, matching section 4.2; any
token-weighted aggregate is secondary and labeled as such. Peak allocated GPU
memory remains a whole-QA-path measurement. TFLOPs remain unreported unless a
profiler definition and values are actually recorded.

Each run manifest freezes the measurement identity needed for paired claims:
GPU model and compute capability, CUDA/PyTorch/Transformers versions, dtype,
attention backend, allocator/peak-memory definition, timer synchronization and
stage-boundary definitions, and warmup count and sample identity. Production
validation rejects absent, nonfinite, or nonpositive stage timings and missing
or inconsistent measurement identity.

## Six-cell comparison contract

Final reporting is generated only from a complete matrix containing all-kept
and DocPrune at one, two, and four retrieved pages. Every input run first passes
the independent single-run validator. The matrix validator then requires the
same 2,441 question IDs in source order and exact shared corpus, runtime,
upstream, model, processor, generation, hardware, precision, backend, warmup,
and measurement identities within every paired page-count cell. Mode and index
artifact identities are expected to differ; all other comparison-defining
fields fail closed on disagreement.

For every result row, retrieved document IDs must belong to the pinned corpus
and page indices must be within the actual corresponding PDF page count. A
signed final JSON comparison artifact and human-readable table report the six
absolute cells plus paired deltas for EM/F1, modality/hop F1, retrieval recall,
stage token retention/drop, total and component timing, encoder/decoder
throughput, and peak allocated GPU memory. No aggregate is published from an
incomplete or incomparable matrix.

Because SOL uses A100 80 GB rather than the paper's RTX A6000, absolute
throughput and memory values are observed reconstruction results, not direct
hardware parity. Baseline-versus-DocPrune relative comparisons remain valid
because each pair uses the same node class, order, prompt, generation settings,
backend, and warmup policy.

## Scope ruling

The requested goal is the paper's M3DocRAG all-kept versus DocPrune comparison
at top-1, top-2, and top-4. FastV, DivPrune, VTW, component ablations, alternate
comprehension criteria, VDocRAG, and secondary datasets are paper experiments
outside this goal. Their absence must not be described as missing support for
the requested six-cell benchmark.

## Verification

Tests must fail first for visual-only indexing, repeated ColPali inference in
QA, over-fetching, a non-FlashAttention model load, full-query CTP projection,
missing stage timings, token-weighted paper drop rates, fabricated retrieved
pages, incomplete six-cell matrices, and mismatched paired identities.
Corrected CPU tests, the opt-in cached real-model equivalence probe, Ruff, shell
checks, a fresh semantic gate, six fresh indexes, six 2,441-row evaluations,
independent single-run and paired validation, and signed final reporting are all
required before completion.
