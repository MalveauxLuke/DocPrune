# DocPrune M3DocVQA Benchmark Design

## Decision

Complete the first DocPrune benchmark target on SOL using the validated
runtime, the acquired M3DocVQA dev corpus, and the official M3DocRAG contract.
The run must establish processor mappings and baseline equivalence before it
constructs indexes or evaluates DocPrune. It then evaluates frozen baseline
and DocPrune variants at top-1, top-2, and top-4 with identical question order,
prompts, generation settings, and hardware.

## Corrected immutable inputs

The earlier probe handoff named `vidore/colpali-v1`. That repository does not
resolve anonymously and is not the model named by the pinned M3DocRAG source.
M3DocRAG commit `29e6ac2294d6b87075a1d45b8a8df175b214248a`
explicitly uses the ColPali v1.2 adapter and its PaliGemma backbone. The
benchmark therefore corrects the reconstruction pin to the upstream contract:

- DocPrune runtime: `64ea70c66a8f9e3dbce804d1fde265a4a2b8b09d`;
- M3DocRAG: `29e6ac2294d6b87075a1d45b8a8df175b214248a`;
- Qwen: `Qwen/Qwen2-VL-7B-Instruct` at
  `eed13092ef92e448dd6875b2a00151bd3f7db0ac`;
- ColPali adapter: `vidore/colpali-v1.2` at
  `961b51745de3e9adb3468ac5c9ccca0ac626c217`;
- ColPali backbone: `vidore/colpaligemma-3b-pt-448-base` at
  `30ab955d073de4a91dc5a288e8c97226647e3e5a`.

The previously assumed `m3docrag/m3docvqa` Hugging Face dataset repository
also does not resolve. Dataset identity is instead the acquired official MMQA
archives, their recorded SHA-256 values, the pinned M3DocRAG transformation,
and the final integrity report. No dataset content or generated artifact enters
Git.

## Integration boundary

Add a concrete in-package factory for the existing `docprune-m3docvqa` CLI.
The factory owns only construction and adaptation:

1. load the pinned M3DocVQA dev questions and PDF/page data;
2. load the pinned ColPali backbone, adapter, and processor;
3. build or load a manifest-bound page embedding index;
4. expose the official M3DocRAG retrieval API through
   `OfficialM3DocRAGBoundary`;
5. load pinned Qwen2-VL and its processor;
6. select either an all-kept baseline answerer or the DocPrune Qwen adapter;
7. return `EvaluationWorkload` for evaluation or build the requested index for
   embedding.

The factory accepts all paths and immutable revisions through a validated run
configuration or environment contract. It must fail closed on a missing path,
revision mismatch, output collision, processor mapping ambiguity, empty page,
or non-monotonic token trace.

## Indexes and comparison modes

Baseline and DocPrune do not share an index. Baseline embeds the unchanged
ColPali page inputs. DocPrune applies retrieval-stage BTP and builds a separate
derived index with its own manifest. Both indexes record source PDF hashes,
processor/model revisions, page ordering, embedding shapes, and configuration.

The all-kept mode disables BTP, QTP, and CTP without changing the integration
path. Its fixed-sample answers and retrieved-page ordering must match the
official M3DocRAG path before the pruned mode is authorized. The DocPrune mode
uses the paper page-count settings and records every reconstruction default.

## Ordered gates

The benchmark handoff executes these gates in order:

1. processor contract for ColPali v1.2 and pinned Qwen;
2. reviewed ColPali visual-token slice, grid, and raster ordering;
3. reviewed Qwen resize, patch grid, and 2-by-2 merge mapping;
4. all-kept baseline equivalence on deterministic fixed samples;
5. one-sample DocPrune generation for each page count with nonempty pages and
   monotonically nonincreasing visual-token counts;
6. baseline index construction and DocPrune index construction;
7. frozen baseline and DocPrune evaluation for top-1, top-2, and top-4;
8. independent result validation and aggregate reporting.

A failed semantic gate stops downstream jobs. Scheduler interruption,
preemption, or time limit may resume only from a manifest-identical artifact.
Corrupt or mismatched output is preserved and a new attempt directory is used.

## Evaluation contract

Use all 2,441 dev questions in the deterministic source order. Each comparison
pair uses the same retrieved-page count, short-answer prompt, decoding settings,
question order, and A100 80 GB allocation. Record:

- exact match and token F1;
- evidence-modality and question-hop F1 when the source labels permit it;
- retrieval recall at the official cutoffs;
- original, post-BTP, post-QTP, and post-CTP visual-token counts;
- retrieval and QA wall time after documented warmup;
- peak allocated GPU memory;
- profiler-derived FLOPs only when the profiler definition is recorded.

Results are reconstruction evidence, not paper parity, unless the measured
protocol and values independently satisfy the documented parity gates.

## SOL authority and artifacts

Before submission, activate one benchmark handoff naming the exact control and
runtime commits, environment, corpus root, model revisions, factory, Slurm
resources, commands, artifact paths, pass conditions, and recovery rules. The
handoff may authorize processor probing, mapping inspection, fixed-sample
generation, index construction, and the six evaluation cells. It never
authorizes training or fine-tuning.

Large artifacts live under `/scratch/lmalveau/docprune/benchmark-<commit>/`.
Git contains only source, tests, configuration, launch scripts, handoffs, and
small provenance summaries. Secrets, weights, caches, indexes, page images,
predictions, and raw profiles remain outside Git.

## Verification and completion

Implementation is ready for SOL only when the full CPU suite, targeted factory
tests, Ruff, shell syntax, dry-run manifests, link checks, artifact scans, and
clean-worktree checks pass. The benchmark is complete only when all ordered
gates pass, all six evaluation cells finish, result counts equal 2,441 per
cell, manifests match the declared inputs, summaries reproduce from immutable
JSONL, and the final report distinguishes observed measurements from paper
values and reconstruction defaults.
