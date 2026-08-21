# DocPrune Reproduction

## Status

This repository contains a locally tested reproduction of the method described
in the [CVPR paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Choi_DocPrune_Efficient_Document_Question_Answering_via_Background_Question_and_Comprehension-aware_CVPR_2026_paper.pdf)
and [supplement](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Choi_DocPrune_Efficient_Document_CVPR_2026_supplemental.pdf).
It is not an official implementation. The supplied sources do not link a
DocPrune code release.

Local validation covers the equations, threshold boundaries, complete
unpadded ColPali sequences and raster maps, exact upstream MaxSim retrieval,
2-by-2 merge groups, sparse Qwen vision execution, original rotary positions,
prompt-token preservation, reduced-query CTP, heterogeneous per-layer KV
caches, greedy generation, the pinned M3DocRAG API boundary, exact
measurement identity, positive stage timing, metrics, immutable manifests,
independent validation, and CLI behavior. The active benchmark runtime is
pinned in
[`sol/handoffs/DOCPRUNE_M3DOCVQA_BENCHMARK_HANDOFF.md`](../../sol/handoffs/DOCPRUNE_M3DOCVQA_BENCHMARK_HANDOFF.md).
No benchmark result is claimed until its gate, index manifests, and 2,441-row
validation report pass.

The six-cell result scope is DocPrune versus all-kept only (top-1/top-2/top-4);
FastV, DivPrune, VTW, and the full paper Table 2 are out of scope. The active
reconstruction corpus is intentionally immutable at 2,441 questions, 3,366
PDFs, and 44,638 pages, even though the paper reports 2,441 / 3,368 / 41,005.

The current Phase-A seal uses `/home/lmalveau/mamba-envs/docprune-sol`, PDF
tools from `/home/lmalveau/mamba-envs/m3docvqa-acquisition`, the acquired
corpus at `/scratch/lmalveau/docprune/datasets/m3docvqa`, and runtime
`02385b3a6fc939f23a8632a7ce58b4cac8bff263` from
`/home/lmalveau/DocPrune-runtime-02385b3`. Its immutable attempt root is
`/scratch/lmalveau/docprune/benchmark-02385b3/attempt-2`; control identity is
sealed in that root's `control.json`. Attempt-2 remains diagnostic-only:
evaluation `61830411` was canceled before work and indexes `61830405`–`61830410`
are schema-4 artifacts at `/scratch/lmalveau/docprune/benchmark-6c19bfc/attempt-2/`
that cannot be promoted. Scheduling-only attempt-1
(61883512, 61883881–61883886, 61883887, 61883888) was canceled before work
and remains immutable history.

## Implemented method

1. BTP converts each resized page to grayscale, finds the page-wide modal
   intensity, scores each patch by its near-mode pixel fraction, and rejects
   patches over the page-count-specific threshold.
2. QTP consumes the already encoded question and complete persisted ColPali
   page context from retrieval. It sums cosine similarity from each visual
   document token to all question tokens, reshapes and bilinearly resizes the
   relevance map, smooths it with a Gaussian kernel, and rejects values below
   the relevance threshold. QA never invokes ColPali.
3. BTP and QTP decisions are lifted to complete 2-by-2 Qwen spatial-merge
   groups before the vision encoder.
4. CTP observes the last prompt token after each decoder layer. At the first L2
   norm threshold crossing, it recomputes last-query attention and removes
   visual tokens below the attention threshold from all deeper layers.
5. Nonvisual tokens are never removed. Selected fine patches keep their
   original Qwen rotary positions. The selected layer retains its full cache;
   deeper layers receive compact caches.

## Paper configurations

Values below are supplement Table B facts. `tau_info` in that table corresponds
to the main paper's comprehension threshold `tau_comp`.

| Pages | retrieval BTP | QA BTP | error | QTP | comprehension | attention |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.9 | 0.9 | 1 | 0.3 | 65 | 0.5 |
| 2 | 1.0 | 1.0 | 1 | 0.3 | 60 | 0.25 |
| 4 | 1.0 | 0.8 | 1 | 0.4 | 45 | 0.075 |

The machine-readable source is
[`configs/docprune-m3docvqa.toml`](../../configs/docprune-m3docvqa.toml).

## Code map

| Concern | Path |
|---|---|
| Configuration and provenance | `src/docprune/benchmark_config.py`, `src/docprune/artifacts.py` |
| BTP/QTP/CTP | `src/docprune/btp.py`, `qtp.py`, `ctp.py` |
| Pagewise BTP/QTP composition | `src/docprune/pipeline.py` |
| Sparse Qwen2-VL | `src/docprune/qwen2vl/` |
| M3DocRAG boundary | `src/docprune/m3docrag.py` |
| Metrics and CLI | `src/docprune/evaluation.py`, `src/docprune/metrics.py`, `src/docprune/cli.py` |

The Qwen adapter intentionally supports batch size one, image pages only, a
2-by-2 spatial merge, greedy decoding, and Transformers 4.46.3. A changed
private Qwen structure fails before generation.

## Trace and outputs

Every result JSONL record contains the question, accepted reference answers,
prediction, ordered retrieved pages, timing, and this trace:

```json
{
  "original_visual_tokens": 4096,
  "post_btp_visual_tokens": 3000,
  "post_qtp_visual_tokens": 1500,
  "post_ctp_visual_tokens": 400,
  "ctp_layer": 17
}
```

Counts are merged visual tokens. They must be monotonically nonincreasing.
Aggregate summaries contain absolute counts, arithmetic-mean drop rates from
the original count, retrieval/page-load/QA/total seconds, positive encoder and
decoder seconds, and samples per encoder/decoder second. Peak allocated GPU
memory is measured over the complete QA path. TFLOPs are omitted unless
profiling is explicitly enabled with a valid definition. A100 values are
reconstruction measurements, not RTX A6000 hardware parity.

## CLI

```bash
docprune-m3docvqa inspect \
  --config configs/docprune-m3docvqa.toml --pages 4

docprune-m3docvqa evaluate \
  --config configs/docprune-m3docvqa.toml --pages 4 \
  --mode docprune \
  --run-config /scratch/lmalveau/docprune/benchmark-02385b3/attempt-2/run-configs/docprune-top4.json \
  --index-manifest /scratch/lmalveau/docprune/benchmark-02385b3/attempt-2/indexes/docprune/top4/docprune/manifest.json \
  --output /scratch/lmalveau/docprune/benchmark-02385b3/attempt-2/eval/docprune/top4/run \
  --factory docprune.m3docvqa_factory:build_workload

docprune-m3docvqa validate-run \
  --run /scratch/lmalveau/docprune/benchmark-02385b3/attempt-2/eval/docprune/top4/run \
  --expected-questions 2441

docprune-m3docvqa summarize \
  --results /scratch/lmalveau/docprune/benchmark-02385b3/attempt-2/eval/docprune/top4/run/results.jsonl

docprune-m3docvqa compare-runs \
  --corpus-root /scratch/lmalveau/docprune/datasets/m3docvqa \
  --all-kept-top1 /scratch/lmalveau/docprune/benchmark-02385b3/attempt-2/eval/all-kept/top1/run \
  --all-kept-top2 /scratch/lmalveau/docprune/benchmark-02385b3/attempt-2/eval/all-kept/top2/run \
  --all-kept-top4 /scratch/lmalveau/docprune/benchmark-02385b3/attempt-2/eval/all-kept/top4/run \
  --docprune-top1 /scratch/lmalveau/docprune/benchmark-02385b3/attempt-2/eval/docprune/top1/run \
  --docprune-top2 /scratch/lmalveau/docprune/benchmark-02385b3/attempt-2/eval/docprune/top2/run \
  --docprune-top4 /scratch/lmalveau/docprune/benchmark-02385b3/attempt-2/eval/docprune/top4/run \
  --json-output /scratch/lmalveau/docprune/benchmark-02385b3/attempt-2/comparison/six-cell.json \
  --markdown-output /scratch/lmalveau/docprune/benchmark-02385b3/attempt-2/comparison/six-cell.md
```

Before benchmarking, SOL must execute the ordered gate and dependency graph in
[`../../sol/handoffs/DOCPRUNE_M3DOCVQA_BENCHMARK_HANDOFF.md`](../../sol/handoffs/DOCPRUNE_M3DOCVQA_BENCHMARK_HANDOFF.md).
The gate records the pinned processor contract and fixed-sample equivalence;
the six index jobs and six evaluation cells remain separate and manifest-bound.

The evaluation factory is an explicit integration boundary. It must return
`docprune.cli.EvaluationWorkload` and must pin the official M3DocRAG checkout,
model revisions, acquired corpus digests, source question order, prompt, and
generation settings. The CLI does not guess these values. Resume is accepted
only when the existing run manifest exactly equals the requested manifest.

## Parity gates

The first SOL task must establish, in order:

1. environment and pinned runtime/source cleanliness;
2. real CUDA/FlashAttention-2 execution and exact measurement identity;
3. complete unpadded ColPali all-kept equivalence, grid, and raster mapping;
4. exact fixed-sample upstream retrieval order and one-time query encoding;
5. all-kept baseline answer equivalence on fixed samples;
6. one-sample positive pruning traces and stage timing probes for pages 1/2/4;
7. six separate schema-5 manifest-bound baseline/DocPrune indexes;
8. frozen baseline versus DocPrune top-1, top-2, and top-4 evaluation;
9. independent 2,441-row result validation followed by the signed six-cell
   JSON and Markdown comparison.

Only after those gates may reports compare EM, F1, throughput, memory, or FLOPs
with the paper.
