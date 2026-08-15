# MiniVGent Residual-Image Candidate

## Proposal metadata

- Status: `unvalidated_proposal`
- Last updated: 2026-08-11
- Possible consumer: a future MiniVGent evidence-set experiment
- Active-task effect: none

## Idea

Add one permanent bbox-free candidate representing the complement of the
union of every DeepSeek-OCR-2 candidate region on a page:

```text
RESIDUAL = page pixels outside union(all ordinary candidate masks)
```

This candidate is an escape hatch when relevant content exists on the page but
the frozen parser did not produce a usable ordinary candidate. A downstream
answerer would receive a full-page residual view: ordinary-candidate pixels
are masked or dimmed while uncovered pixels retain their original page
coordinates. Do not sum visual features indiscriminately or pack disconnected
regions into a layout-destroying collage.

## Required semantics

`RESIDUAL` and `NONE` are different:

- `RESIDUAL`: relevant content exists on the page outside the ordinary
  candidate universe;
- `NONE`: no relevant content exists on the page.

A partial candidate miss may require ordinary candidates plus `RESIDUAL`.
Whether joint selection is the primary mode or an ablation remains an explicit
unresolved decision.

## Training proposal

- Natural candidate misses: label `RESIDUAL` when accepted target evidence is
  outside all ordinary candidates.
- Counterfactual dropout: remove an otherwise positive ordinary candidate,
  return its pixels to the residual mask, and train the selector to choose
  `RESIDUAL` without generating a new question.
- Fully covered examples: treat unnecessary residual use as negative.
- Partial misses: supervise the represented ordinary evidence and residual
  jointly.
- Ambiguous surrounding context: ignore rather than force a negative.

Answer correctness alone is insufficient because the selector can learn to
include the broad residual candidate as insurance. Training and evaluation
must include an information-cost term or fixed residual cost.

## Evaluation

Report:

- residual-selection rate;
- precision and recall on genuine candidate misses;
- selected pixel and visual-token cost;
- downstream answer recovery conditional on a genuine miss;
- unnecessary residual use when ordinary candidates cover the target; and
- candidate-deletion stress tests.

This mechanism softens the fixed-candidate ceiling for downstream answering,
but it does not provide a precise recovered box. DeepSeek-OCR2-Box remains a
different experiment that can generate explicit new coordinates.
