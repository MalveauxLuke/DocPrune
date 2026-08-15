# DeepSeek-OCR-2 Generative Evidence Localization

## Proposal metadata

- Status: `unvalidated_proposal`
- Last updated: 2026-08-04
- Possible consumer: future Evidence-DINO-Units model experiments
- Capability tags: question-conditioned representation; generative region
  localization; document-specific pretraining; multi-region output
- Active-task effect: none. This proposal does not alter or authorize work
  beyond the current Evidence-DINO-Units Stage 0–4 contract.

## Source model and verified boundary

Primary sources:

- [DeepSeek-OCR 2 paper, arXiv `2601.20552`](https://arxiv.org/abs/2601.20552)
- [Official DeepSeek-OCR-2 repository](https://github.com/deepseek-ai/DeepSeek-OCR-2)

**Evidence level: `paper_reported`.**

DeepSeek-OCR-2 is a document-reading and parsing model. Its DeepEncoder V2
uses bidirectional visual tokens plus learnable causal-flow query tokens, and
only the resulting causal-flow tokens are passed to the language decoder. The
paper evaluates OCR, document parsing, and reading order. It does **not**
report question-conditioned supporting-evidence localization or establish
that fine-tuning it to emit evidence boxes will work.

## Project hypothesis

**Evidence level: `unvalidated_proposal`.**

Fine-tune DeepSeek-OCR-2 so that the prompt contains a natural document
question and the decoder outputs the supporting location or locations instead
of Markdown.

```text
question + page image
  -> DeepSeek-OCR-2 document encoder and language decoder
  -> constrained list of relevant normalized boxes, or an empty list
  -> optional snapping to fixed DeepSeek semantic units
  -> downstream answerer
```

One possible output contract is:

```json
{"boxes": [[x1, y1, x2, y2]], "no_evidence": false}
```

Coordinates should use one documented normalized coordinate system. The
schema must permit multiple boxes and a true empty result. Output repair must
be measured rather than silently counted as valid localization.

The anticipated training corpus is approximately 60,000 grounding examples.
That number is a project planning estimate, not a verified final Stage 0–4
dataset count. Any experiment must pin the actual accepted rows, supervision
strength, source mixture, and split revision.

## Why it may work

DeepSeek-OCR-2 is pretrained specifically around document text, reading order,
layout, tables, and document parsing. It may therefore understand implicit
question-to-document relationships better than a natural-image phrase
grounder such as Grounding DINO. It also avoids first translating the page
into an unrelated object-detection vocabulary.

This arm gives the document model freedom to place boxes around evidence that
is not already represented by a fixed proposal set. That is valuable when the
parser omits a useful region or when the appropriate evidence boundary cuts
across existing units.

## Central risks

- Autoregressive coordinate generation can be malformed, incomplete,
  duplicated, prematurely terminated, or numerically inconsistent.
- A valid box can still be oversized and retain nearly the whole page.
- A sequence decoder can omit later evidence components after finding one
  salient answer-bearing region.
- Empty-target and multi-target calibration are less natural than parallel
  classification over a fixed candidate universe.
- Box generation does not directly exploit the stable semantic units and
  structural links already produced by DeepSeek-OCR-2 parsing.
- Answer-region annotations may not include all supporting context required by
  a downstream answerer.
- Fine-tuning the decoder for coordinates may disturb its pretrained document
  abilities or teach dataset-specific coordinate conventions.

These are not reasons to reject the arm. They define what the controlled test
must measure.

## Fair controlled arm

Use the same document groups, questions, train/dev/test splits, supervision
revision, downstream answerer, and evaluation prompt as the fixed-unit models.
Compare at least:

1. raw generated boxes;
2. generated boxes snapped to intersecting DeepSeek semantic units;
3. snapped boxes plus deterministic structural closure; and
4. the full-page fallback.

Report:

- strict complete-evidence recall, not only best-box IoU;
- invalid or repaired output rate;
- missing, duplicated, and oversized box rates;
- false localization on no-evidence pages;
- retained units, image area, and actual downstream tokens;
- answer accuracy from the selected evidence alone;
- localization latency and total end-to-end cost; and
- direct, implicit, comparative, and multi-region strata.

The most informative comparison is against the fixed-unit reranker and
mini-VGent using identical evidence and answer evaluation. This determines
whether document-specific generative flexibility buys anything beyond stable
parallel candidate selection.

## Decision boundary

This is a valuable controlled arm, not an assumed replacement for Grounding
DINO, Evidence-DINO-Units, a document reranker, or mini-VGent. It becomes more
compelling if it recovers valid evidence outside the fixed-unit oracle while
maintaining complete-set recall and manageable output failures. If its gains
come only from producing large boxes, the apparent localization improvement
does not establish useful evidence compression.
