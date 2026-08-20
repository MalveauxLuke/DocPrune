# DocPrune Reproduction

## Status

This repository contains a locally tested reproduction of the method described
in the [CVPR paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Choi_DocPrune_Efficient_Document_Question_Answering_via_Background_Question_and_Comprehension-aware_CVPR_2026_paper.pdf)
and [supplement](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Choi_DocPrune_Efficient_Document_CVPR_2026_supplemental.pdf).
It is not an official implementation. The supplied sources do not link a
DocPrune code release.

Local validation covers the equations, threshold boundaries, page layouts,
2-by-2 merge groups, sparse Qwen vision execution, original rotary positions,
prompt-token preservation, one-time CTP, heterogeneous per-layer KV caches,
greedy generation, the pinned M3DocRAG API boundary, metrics, immutable
manifests, independent validation, and CLI behavior. The active benchmark
runtime is pinned in
[`sol/handoffs/DOCPRUNE_M3DOCVQA_BENCHMARK_HANDOFF.md`](../../sol/handoffs/DOCPRUNE_M3DOCVQA_BENCHMARK_HANDOFF.md).
No benchmark result is claimed until its gate, index manifests, and 2,441-row
validation report pass.

## Implemented method

1. BTP converts each resized page to grayscale, finds the page-wide modal
   intensity, scores each patch by its near-mode pixel fraction, and rejects
   patches over the page-count-specific threshold.
2. QTP sums cosine similarity from each visual document token to all question
   tokens, reshapes and bilinearly resizes the relevance map, smooths it with a
   Gaussian kernel, and rejects values below the relevance threshold.
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
Aggregate summaries contain absolute counts, drop rates from the original
count, retrieval/QA seconds, and original visual tokens per end-to-end second.

## CLI

```bash
docprune-m3docvqa inspect \
  --config configs/docprune-m3docvqa.toml --pages 4

docprune-m3docvqa evaluate \
  --config configs/docprune-m3docvqa.toml --pages 4 \
  --mode docprune \
  --run-config /scratch/$USER/docprune/benchmark-3755812/attempt-1/run-configs/docprune-top4.json \
  --index-manifest /scratch/$USER/docprune/benchmark-3755812/attempt-1/indexes/docprune/top4/docprune/manifest.json \
  --output /scratch/$USER/docprune/benchmark-3755812/attempt-1/eval/docprune/top4/run \
  --factory docprune.m3docvqa_factory:build_workload

docprune-m3docvqa validate-run \
  --run /scratch/$USER/docprune/benchmark-3755812/attempt-1/eval/docprune/top4/run \
  --expected-questions 2441

docprune-m3docvqa summarize \
  --results /scratch/$USER/docprune/benchmark-3755812/attempt-1/eval/docprune/top4/run/results.jsonl
```

Before benchmarking, SOL must execute the ordered gate and dependency graph in
[`../../sol/handoffs/DOCPRUNE_M3DOCVQA_BENCHMARK_HANDOFF.md`](../../sol/handoffs/DOCPRUNE_M3DOCVQA_BENCHMARK_HANDOFF.md).
The gate records the pinned processor contract and fixed-sample equivalence;
the six index jobs and six evaluation cells remain separate and manifest-bound.

The evaluation factory is an explicit integration boundary. It must return
`docprune.cli.EvaluationWorkload` and must pin the official M3DocRAG checkout,
model revisions, dataset revision, question order, prompt, and generation
settings. The CLI does not guess these values. Resume is accepted only when
the existing run manifest exactly equals the requested manifest.

## Parity gates

The first SOL task must establish, in order:

1. environment and pinned runtime/source cleanliness;
2. exact ColPali visual-token slice, grid, and raster mapping;
3. exact Qwen processor resize and patch/grid correspondence;
4. all-kept baseline answer equivalence on fixed samples;
5. one-sample pruning traces for pages 1/2/4 with no empty page;
6. six separate manifest-bound baseline/DocPrune indexes;
7. frozen baseline versus DocPrune top-1, top-2, and top-4 evaluation;
8. independent 2,441-row result validation and summary reproduction.

Only after those gates may reports compare EM, F1, throughput, memory, or FLOPs
with the paper.
