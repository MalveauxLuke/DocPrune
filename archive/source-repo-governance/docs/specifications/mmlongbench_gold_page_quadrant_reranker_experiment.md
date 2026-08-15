# MMLongBench Gold-Page Quadrant Reranker Baseline

## Status

Planned. The prerequisite SOL acquisition is DeepSeek-OCR-2 on every unique,
valid MMLongBench-Doc gold page. The segmentation-lab implementation remains an
active parallel task and is not replaced by this experiment.

## Research question

Does zero-shot hierarchical quadrant routing improve a fixed answerer's
MMLongBench performance relative to giving that same answerer all annotated
gold pages at once?

This is a practical end-to-end baseline for the combined segmentation and
reranking pipeline. A positive result demonstrates that localized evidence
routing helps under the tested configuration. A negative result does not by
itself distinguish candidate-construction failure, reranker domain mismatch,
threshold error, missing context, or answerer sensitivity.

The experiment does not use a human-selected or answerer-selected oracle
quadrant. Oracle localization is deferred because the first baseline must not
require manual region annotation or a multiplicative answerer-call budget.

## Dataset scope

Use every answerable MMLongBench-Doc question with a nonempty, valid official
gold-page list. Preserve all question, answer, document, evidence-page,
evidence-source, and answer-format fields in the frozen experiment manifest.

OCR each unique `(document, gold page)` once and retain a separate
question-to-page link table. Questions with no valid gold page, including
unanswerable questions, are excluded from this first answer-generation
comparison and recorded in an explicit audit rather than repaired.

The initial SOL source is:

```text
/scratch/lmalveau/agenticdocai/data/MMLongBench-Doc
```

The acquisition task must discover and validate the authoritative dataset file
and page-number convention at that location before freezing counts. It must not
reuse the retired 26-document academic subset or its counts.

## Compared arms

### Arm F: Full gold pages

For each question:

1. Load every official gold page in stable document/page order.
2. Provide all gold-page images and the original question to the answerer in
   one fresh context.
3. Generate one final answer.

### Arm Q: Zero-shot quadrant routing

For the same question:

1. Start every official gold page as automatically positive. Do not ask the
   reranker whether a known gold page is necessary.
2. Apply the frozen semantic-routing strategy and the canonical hierarchical
   evidence-routing rules independently to each gold page.
3. Route through horizontal halves and then vertical quadrant-scale
   candidates. Stop indiscriminate geometry at quadrants.
4. Use the zero-shot `Qwen/Qwen3-VL-Reranker-2B` binary relevance decision for
   child candidates. Use the native `yes` versus `no` argmax without
   MMLongBench fine-tuning or threshold calibration, and retain the raw logits
   for analysis.
5. Preserve every positive branch. Apply the canonical `A`, `B`, `C`,
   exclusivity, fingerprint deduplication, strict-progress, and parent-fallback
   rules exactly.
6. If evidence is distributed within a page, retain multiple positive
   quadrants. If the routing decision cannot preserve a positive child, retain
   the specified parent/page fallback.
7. For cross-page questions, collect retained candidates from every gold page.
   Never collapse a cross-page question to candidates from only one page.
8. Provide the complete retained candidate set and the original question to
   the answerer in one fresh context and generate one final answer.

No fine-grained image-caption, paragraph, figure, or table reranking occurs in
this baseline. Quadrant-scale localization is the terminal experimental level.

## Candidate representation

Routing operates on the segmentation lab's immutable DeepSeek routing atoms
and candidate fingerprints. The eventual evaluation runner must use the same
candidate membership that the segmentation lab displays.

Candidate rendering must preserve every complete member atom and must never
clip an atom at a half or quadrant boundary. The rendering method, margin,
resolution, ordering, and image preprocessing must be frozen in the run
manifest and held constant for all Arm Q candidates. The answerer must not see
OCR text, gold answers, evidence-source annotations, or reranker scores unless
the same information is explicitly provided in both arms.

## Fairness controls

Freeze one answerer checkpoint before answer inference. The exact checkpoint
is a run-level choice, but both arms must use the same:

- answerer checkpoint and revision;
- question and answer instructions;
- page/candidate image preprocessing and resolution policy;
- decoding parameters and maximum output length;
- question order and independent-context policy; and
- official MMLongBench answer evaluator.

Each arm receives one final answerer call per question. Reranker calls are
additional retrieval cost and must be reported separately. Do not carry model
state, conversation history, or candidate decisions between questions.

The different input size is intentional: the experiment tests whether removing
irrelevant page content improves answering. Record the number of input images,
retained routing atoms, retained page-area fraction, and fallback use so that a
nominally segmented arm that usually returns complete pages is not mistaken for
successful localization.

## Required measurements

Report:

- official answer score for Arm F and Arm Q;
- paired per-question win, loss, tie, rescue, and harm counts;
- results separated into single-gold-page and cross-page questions;
- results separated by official evidence-source category where available;
- number of reranker calls and distinct fingerprint-cache hits;
- number of retained candidates and gold pages per question;
- retained page-area fraction;
- page/parent-fallback frequency;
- both-positive, both-negative, and identical-child routing frequency; and
- reranker `yes`/`no` logit distributions and margins by routing level.

Use paired bootstrap confidence intervals over questions for the primary Arm Q
minus Arm F answer-score difference. Keep all failed questions in the paired
analysis unless the failure is a documented infrastructure failure affecting
both arms.

## Interpretation rules

- `Arm Q > Arm F` supports the claim that the combined zero-shot segmentation
  and routing pipeline improves this answerer under gold-page-conditioned
  evaluation.
- `Arm Q = Arm F` does not establish that localization is unnecessary; the
  routed arm may have returned broad fallbacks or the answerer may be
  insensitive to the removed content.
- `Arm Q < Arm F` is not a clean rejection of segmentation because there is no
  oracle-localization arm. Analyze routing depth, score margins, retained area,
  and fallback behavior before deciding whether the next step is calibration,
  reranker training, candidate-rule revision, or manual oracle analysis.
- Because every gold page begins positive, this experiment does not evaluate
  document-wide page retrieval or nongold-page rejection.
- Because routing stops at quadrants, it does not evaluate the planned
  fine-grained image-caption candidate generator.

## Execution sequence

1. Acquire and package pinned DeepSeek-OCR-2 outputs for all unique valid gold
   pages on SOL.
2. Return the Git-safe OCR bundle and frozen question/page mappings locally.
3. Validate the returned inventory and integrate it with the segmentation lab.
4. Freeze the semantic strategy, routing configuration, candidate renderer,
   reranker revision, and answerer revision.
5. Run a small deterministic smoke set containing single-page, cross-page,
   text, table, chart, and image evidence cases.
6. Run both arms over the frozen answerable-question set.
7. Score with the official evaluator and produce the paired analysis above.

The active SOL handoff covers only Steps 1 and 2. Segmentation, reranking,
answer generation, evaluation, and display remain local or require a later,
separately approved compute task.
