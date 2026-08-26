# Fair CTP Comparison Baseline

Date: 2026-08-26

## Purpose

This protocol isolates the CTP implementation as the only intended experimental
variable. It supports a defensible comparison between a new CTP method and two
locally reconstructed DocPrune references. It does not claim identity with the
authors' unpublished implementation.

## Required shared controls

Every paired comparison must use the same:

1. ordered cached top-4 page identities and persisted page features;
2. BTP and QTP masks;
3. pinned Qwen checkpoint and processor revision;
4. page raster inputs, prompt bytes, chat template, and preprocessing outputs;
5. greedy decoding parameters, 128-token cap, and complete EOS set
   `[151645, 151643]`;
6. decoder/cache/position implementation outside the candidate CTP operation;
7. question cohort, reference answers, and evaluator; and
8. per-question pairing and artifact-integrity checks.

Fresh or global retrieval, loading the global retrieval index, rebuilding page
features, or changing BTP/QTP is forbidden for this comparison unless a later
task explicitly authorizes it. Output from a run that violates this rule is
non-canonical and must not be merged or scored as a fixed-page control.

## Generation-path admission gate

Before a CTP comparison is admissible, the manual DocPrune decoder with all
visual tokens retained must reproduce stock Qwen on a fixed cached page:

- stock and DocPrune preprocessing tensors are exactly equal for `input_ids`,
  `attention_mask`, `pixel_values`, and `image_grid_thw`;
- both paths resolve and use EOS IDs `[151645, 151643]`;
- the full generated suffix is exactly equal, through EOS or the shared
  128-token cap; and
- first-step logits agree within the declared BF16 kernel tolerance
  (`rtol=0.02`, `atol=0.02`).

The canonical gate runs on L40S. One shard-zero-equivalent probe on A30,
A100-40GB, and H100 records numerical portability; those probes do not replace
the L40S result. Each job is fixed-page, retrieval-free, and at most 20 minutes.

## Reference CTP implementations

Retain both references in every later paired comparison:

### Literal/current reconstruction

This is closest to the paper's phrase "attention weights": it averages
post-softmax attention probabilities across query heads during prompt prefill.
The local implementation additionally multiplies by the number of post-QTP
visual tokens, a scale that the paper does not specify. On the sealed
245-question fixed-page diagnostic it scored F1 `41.3347`, versus `43.6327` for
BTP+QTP, a paired delta of `-2.2980`.

Historical full-benchmark runtime: `4e2473bdbbc2e4eca0e92c30d4a0633044501ccf`.

### Best-fit aggregate-logit reconstruction

This diagnostic averages query-key logits across heads, applies a visual-only
softmax, and scales by the post-QTP visual-token count. It is not explicitly
specified by the paper, but it best matches the paper-derived top-4 retention
fingerprint among the tested prompt-prefill candidates. On the same sealed
245-question fixed-page cohort it scored F1 `42.4163`: `+1.0816` over the
literal/current reconstruction and `-1.2163` below BTP+QTP.

Canonical historical analysis:
`/scratch/lmalveau/docprune/ctp-aggregate-logit-stage245-fixed-v1/analysis-final.json`
(SHA-256 `0932ff...`; full digest remains in active SOL authority).

These historical quality numbers predate the corrected shared EOS contract.
They are evidence for selecting the two reference semantics, not final numbers
to compare against a new method. Literal/current, aggregate-logit, and the new
method must be rerun together under the corrected runtime and shared controls.

## Promotion and claim boundary

A candidate comparison is valid only when:

- the generation-path admission gate passes;
- all modes use the identical cached ordered pages and shared inputs;
- fixed-page provenance is true and global index search is false;
- every expected question has one result per mode;
- artifact and result digests validate; and
- paired quality is reported with a question-cluster confidence interval.

If a new method beats both references under these controls, the supported claim
is that it outperforms the literal and best-fit local DocPrune reconstructions.
It is not evidence that it outperforms the unpublished official implementation.
