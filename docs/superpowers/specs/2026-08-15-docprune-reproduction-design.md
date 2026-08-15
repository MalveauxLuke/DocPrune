# DocPrune Reproduction Design

## Decision

Implement the complete locally testable reproduction before handoff. The
handoff begins only after the package, paper configurations, integration
entry points, tests, documentation, and SOL launch examples are complete.
GPU execution, benchmark parity, and performance claims remain explicitly
unvalidated until a SOL-capable worker runs the handoff.

## Objective

Recreate the training-free DocPrune method described in the CVPR 2026 paper:

1. Background Token Pruning (BTP) before the retrieval and QA vision encoders.
2. Question-aware Token Pruning (QTP) before the QA vision encoder, using
   token embeddings already computed by the retrieval stage.
3. Comprehension-aware Token Pruning (CTP) once inside the QA language-model
   decoder, using the last-token hidden-state norm to select the pruning layer
   and last-token attention to select retained visual tokens.

The first benchmark target is M3DocVQA with the M3DocRAG pipeline,
ColPali-v1 retrieval, Qwen2-VL-7B-Instruct QA, and top-1, top-2, and top-4
retrieved-page settings. Secondary paper tables are deferred until this
primary target has been executed successfully.

## Evidence boundary

### Paper-reported facts

- DocPrune is training-free.
- BTP computes the image-wide modal grayscale intensity, measures each
  patch's near-background pixel ratio, and removes patches whose ratio exceeds
  a threshold.
- QTP sums cosine similarity between each document token and every question
  token, resizes the relevance map with bilinear interpolation, applies
  Gaussian smoothing, and thresholds the result.
- Qwen2-VL's 2 by 2 spatial merger requires BTP and QTP to prune complete
  2 by 2 token groups at encoder input.
- CTP activates at the first decoder layer whose final-token hidden-state L2
  norm crosses a comprehension threshold, then retains visual tokens whose
  final-token attention crosses an attention threshold.
- The paper uses page-count-specific hyperparameters for top-1, top-2, and
  top-4 inputs and reports results on one NVIDIA RTX A6000.

### Unresolved source gaps

The paper and supplement do not publish DocPrune code or specify:

- the Gaussian standard deviation used by QTP;
- the grayscale conversion rule;
- how QTP maps or excludes ColPali non-image tokens;
- how 2 by 2 group scores are aggregated;
- how attention is aggregated across heads;
- whether CTP runs during prompt prefill or a later generation step;
- exact hidden-state, mask, position, and KV-cache compaction mechanics;
- exact model, dataset, and dependency revisions;
- profiler warmup, sample ordering, or TFLOP accounting details.

Every chosen default for these gaps must be named `reconstruction_default` in
configuration or provenance, exposed where an ablation is meaningful, and
never described as author-provided.

## Architecture

### Core pruning package

Create a small `docprune` Python package with paper-equation-level components:

- `config.py`: validated top-1/top-2/top-4 paper parameters and separately
  labeled reconstruction defaults.
- `layout.py`: visual-token layout and 2 by 2 group bookkeeping.
- `btp.py`: modal-intensity background scoring and group-safe keep masks.
- `qtp.py`: cosine-sum relevance, bilinear resizing, Gaussian smoothing, and
  group-safe keep masks.
- `ctp.py`: first-crossing comprehension trigger and attention-based visual
  token selection.
- `provenance.py`: source revisions, reconstruction choices, environment, and
  run-manifest serialization.

The core functions consume and return explicit tensors and masks. They must
not download models, mutate global state, or depend on M3DocRAG.

### Qwen2-VL integration

Add a version-gated Qwen2-VL adapter. It will:

1. compact BTP/QTP-selected 2 by 2 visual groups before the vision encoder;
2. retain the original spatial rotary positions for selected groups;
3. preserve all nonvisual prompt tokens;
4. inspect the final prompt token at each decoder layer during prefill;
5. trigger CTP at the first comprehension-threshold crossing;
6. recompute final-token attention at that layer when FlashAttention does not
   expose it;
7. compact visual hidden states, masks, positions, and KV state consistently
   for subsequent layers and generation;
8. emit a per-sample trace containing original, post-BTP, post-QTP, and
   post-CTP visual-token counts plus the selected decoder layer.

Unsupported Transformers structures must fail with an actionable compatibility
error, not silently run an unpruned model.

### M3DocRAG integration

Use the official archived M3DocRAG repository at commit
`29e6ac2294d6b87075a1d45b8a8df175b214248a` as the baseline contract rather
than copying its full source tree. A thin adapter and runner will:

- apply BTP during page-embedding generation and therefore require a new
  pruned retrieval index;
- retain per-page ColPali document embeddings needed by QTP;
- encode the raw retrieval question once and pass its token embeddings to QTP;
- invoke the DocPrune Qwen2-VL adapter for answer generation;
- preserve the baseline short-answer prompt and M3DocVQA evaluator;
- write immutable JSONL predictions and a separate run manifest.

### Evaluation and instrumentation

Local tests validate equations, masks, token order, configuration, and adapter
contracts using synthetic tensors and toy decoder layers. The SOL run will add:

- baseline and DocPrune EM/F1;
- evidence-modality and question-hop F1;
- retrieval recall at the baseline's reported cutoffs;
- visual-token counts and drop rates per stage;
- encoder and decoder throughput after warmup;
- peak allocated GPU memory;
- profiler-derived FLOP estimates with the profiler definition recorded.

Performance comparisons must use the same frozen question order, retrieved
pages, generation settings, and hardware allocation.

## Configuration

Paper values from supplement Table B:

| Pages | RET BTP | QA BTP | error tolerance | QTP | comprehension | attention |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.9 | 0.9 | 1 | 0.3 | 65 | 0.5 |
| 2 | 1.0 | 1.0 | 1 | 0.3 | 60 | 0.25 |
| 4 | 1.0 | 0.8 | 1 | 0.4 | 45 | 0.075 |

The supplement labels the comprehension column `tau_info` in Table B while the
main paper and sensitivity table call it `tau_comp`. The implementation uses
`comprehension_threshold` and records both source symbols in documentation.

Initial reconstruction defaults:

- grayscale: ITU-R BT.601 luma weights, rounded to uint8;
- Gaussian sigma: 1.0 retrieval-map cell;
- 2 by 2 group retention: retain the group when any member passes, favoring
  evidence preservation;
- attention-head aggregation: arithmetic mean;
- CTP timing: first prompt prefill pass, using the last prompt token;
- no forced minimum visual-token count beyond validated nonempty masks.

These defaults are hypotheses to test, not paper facts.

## Test strategy

Tests are written before production behavior and must first fail for the
intended reason. Required coverage includes:

- exact BTP ratios and threshold boundary behavior;
- all-background, no-background, and colored-background images;
- QTP cosine-sum values, interpolation shape, Gaussian impulse response, and
  threshold boundary behavior;
- 2 by 2 grouping without spatial-token reordering;
- CTP first threshold crossing, no crossing, head aggregation, and preservation
  of nonvisual tokens;
- top-1/top-2/top-4 configuration values and invalid page counts;
- adapter rejection of incompatible Transformers versions;
- a toy end-to-end trace with monotonically nonincreasing visual-token counts;
- manifest serialization with paper facts and reconstruction defaults separated.

## Completion and handoff boundary

Local implementation is complete only when:

- the full CPU-testable suite passes from a clean environment;
- static checks pass;
- configs and CLI dry runs validate;
- no model weights, datasets, outputs, or caches are tracked;
- README and navigation explain reproduction status;
- an SOL handoff names the exact commit, environment build, datasets, model
  revisions, commands, resources, output paths, smoke gate, benchmark gates,
  and recovery authority.

The handoff must state that no GPU model execution or numerical paper parity
was performed on this machine.
