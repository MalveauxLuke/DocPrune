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
time but is not mislabeled as encoder or decoder execution. Aggregates report
samples per encoder second and samples per decoder second. Peak allocated GPU
memory remains a whole-QA-path measurement. TFLOPs remain unreported unless a
profiler definition and values are actually recorded.

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
and missing stage timings. Corrected CPU tests, the opt-in cached real-model
equivalence probe, Ruff, shell checks, a fresh semantic gate, six fresh indexes,
six 2,441-row evaluations, and independent validation are all required before
completion.
