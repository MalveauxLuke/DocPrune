# VGOQ: Visual Grounding for Object Questions

## Registry metadata

- Paper: Visual Grounding for Object Questions
- Stable identifier and reviewed version: CVPR 2026 camera-ready/project page
- Last reviewed: 2026-07-28
- Applicability: `adjacent`
- Capability tags: question-conditioned representation; region, unit, or
  patch selection; answer sufficiency and necessity; supervision and
  pseudo-labeling; abstention, recovery, and uncertainty; evaluation and
  calibration
- Possible project consumers: Evidence-DINO-Units; future evidence-supervision
  datasets; future no-evidence calibration
- Evidence levels used: `paper_reported`, `repository_inference`,
  `unvalidated_proposal`
- Registry status: `promising`

## Paper link and factual capsule

- Primary project page and paper links:
  [Visual Grounding for Object Questions](https://martin-ev.github.io/vgoq/)
- CVPR Open Access entry:
  [Visual Grounding for Object Questions, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Everaert_Visual_Grounding_for_Object_Questions_CVPR_2026_paper.html)

**Evidence level: `paper_reported`.**

VGOQ distinguishes direct referring-expression grounding from grounding visual
evidence for a general object question. The relevant visual region need not be
named in the question and may provide evidence or context rather than visibly
contain the answer.

The authors construct two automatically generated datasets:

- VizWiz-VGOQ transforms existing visually grounded questions into more
  general object questions.
- ABO-VGOQ uses multimodal models and grounding systems to generate and
  categorize evidence for customer questions about products.

Their lightweight model uses frozen CLIP image and text encoders, task-specific
FiLM conditioning, three transformer blocks, a spatial segmentation head, and
a separate scalar image-relevance head. It accepts one image and a textual
question and returns both a \(336\times336\) evidence heatmap and a relevance
score in \([0,1]\).

The project reports that existing state-of-the-art grounding performance drops
from approximately 52% gIoU to 37% gIoU when direct visual questions are
rephrased as general object questions requiring evidence grounding. The paper
therefore supplies empirical evidence that ordinary referring-expression
training does not automatically transfer to question-to-evidence grounding.

Read the linked paper for dataset generation, category definitions, training
details, and complete evaluation.

## Why this paper attracted attention

**Evidence level: `repository_inference`.**

VGOQ tests the semantic transition at the center of Evidence-DINO-Units:

```text
phrase or visibly answerable question -> directly named/visible region
```

versus:

```text
general question -> region that supplies evidence or useful context
```

This is closer to our intended task than ordinary referring-expression
grounding. It provides an external reason to expect a Grounding-DINO model
trained mainly on phrases or object descriptions to degrade when the question
requires implicit evidence.

It is also relevant to our learned `[EVIDENCE]` token because VGOQ separates
local spatial grounding from global image relevance. That separation may
support our decision to keep a global no-evidence signal alongside local
candidate scores—or reveal that the global token contributes little beyond
the local selector.

## What our current architecture does

**Evidence level: `repository_inference` based on the canonical
Evidence-DINO-Units specification.**

The current planned architecture does not ask one learned token to perform the
entire grounding task.

### Question and global path

```text
full question
  -> BERT token states t1...tm
  -> append learned [EVIDENCE] token qE
  -> full self-attention
  -> Grounding-DINO multimodal feature enhancer
```

The learned `[EVIDENCE]` token is:

- a global question–image readout;
- an input to the page answerability/no-evidence head;
- an optional coarse candidate guide; and
- an auxiliary contrastive anchor.

The specification explicitly says it is not the only query representation and
not the final evidence bottleneck.

### Local evidence path

Every parser-derived candidate receives:

- fixed geometry;
- visual ROI features;
- candidate OCR text;
- type, order, generator, and hierarchy features; and
- one decoder query anchored to its fixed box.

Candidate queries retain:

- candidate-to-candidate self-attention;
- candidate-to-question-token cross-attention;
- deformable image cross-attention around the fixed candidate box; and
- a scalar evidence-relevance head.

Thus our local evidence decision is made by per-candidate fused states, while
the learned `[EVIDENCE]` token provides page-level context and abstention.

## What VGOQ does

**Evidence level: `paper_reported`.**

VGOQ uses:

```text
image + object question
  -> frozen CLIP image and text encoders
  -> task-specific FiLM conditioning
  -> three lightweight transformer blocks
  -> split global CLS and spatial tokens
     -> spatial evidence heatmap
     -> scalar image-relevance score
```

The global relevance head asks whether the image is useful for answering the
question. The spatial head separately identifies the evidence or related
context inside that image. The global output does not itself perform
localization.

## Direct architecture comparison

**Evidence level: `repository_inference`.**

| Aspect | Evidence-DINO-Units | VGOQ |
|---|---|---|
| Question representation | Full BERT question-token sequence plus learned `[EVIDENCE]` token | Frozen CLIP text representation used through FiLM conditioning |
| Global signal | Optional page answerability/no-evidence head from `[EVIDENCE]` and pooled candidates | Explicit scalar image-relevance head |
| Local output | One evidence score per structured candidate unit | Dense \(336\times336\) evidence heatmap |
| Spatial reasoning | Fixed candidate boxes, deformable attention, and candidate self-attention | Spatial CLIP features processed by lightweight transformer blocks |
| Evidence unit | Parser-derived paragraph, line, cell, figure, caption, or fallback region | Pixels/segments in a natural or product image |
| Structural context | OCR text, type, order, hierarchy, links, and deterministic closure | No document-specific structural representation |
| Main supervision | Visual-CoT answer-bearing boxes mapped to candidates, with weak/incomplete labels distinguished | Synthetic question-to-evidence masks plus image relevance |
| No-evidence behavior | Planned page/no-evidence head and wrong-page calibration | Learned image-relevance score |

## Our analysis and interpretation

**Evidence level: `repository_inference`.**

VGOQ supports the *division of labor* in our architecture more strongly than it
supports any single-token design:

- use a global representation to estimate whether the page or image contains
  useful evidence;
- use spatial representations to identify where that evidence lies.

It would be a mistake to interpret VGOQ as evidence that a single global token
can implicitly learn the complete answer-relevant region. Its relevance head
does not localize; the spatial tokens and grounding head do. This matches the
current specification's decision that `[EVIDENCE]` is auxiliary rather than
the final evidence bottleneck.

The more important result is semantic. VGOQ deliberately changes the input
from a direct visible reference to a general question, and ordinary grounding
models degrade. This strengthens the concern that Visual-CoT answer boxes and
conventional phrase-grounding pretraining may not be sufficient for:

- implicit evidence;
- regions whose contents are not repeated in the question;
- related context rather than directly visible answers; and
- no-evidence or merely relevant-context cases.

VGOQ may therefore serve as:

1. evidence that our direct Grounding-DINO baseline must be stratified by
   semantic distance from the question;
2. a source of auxiliary question-to-evidence supervision for scene/product
   images in a later branch;
3. precedent for explicitly supervising the `[EVIDENCE]` page-relevance head;
   and
4. a diagnostic challenge for whether the local candidate decoder truly uses
   full-question meaning rather than answer-string overlap.

It does not establish that our learned token will work. It makes the token and
page head more testable: the global head should improve page/no-evidence
calibration while local candidate recall remains attributable to the
candidate decoder.

## Reusable mechanisms

**Evidence level: `repository_inference`.**

1. Separate global image relevance from local evidence localization.
2. Train on questions whose evidence is not directly named.
3. Include zero- or low-relevance images rather than forcing every query to
   ground somewhere.
4. Categorize evidence quality, including specific evidence versus related
   context that is not sufficient alone.
5. Evaluate semantic reformulations of the same underlying visual problem.
6. Use a lightweight frozen-encoder model as a diagnostic before scaling a
   larger architecture.

## Possible COLPALI integration points

| Consumer | Possible reuse | Evidence level | Why it may help |
|---|---|---|---|
| Evidence-DINO-Units page head | Add explicit page/image relevance supervision to `[EVIDENCE]` plus pooled candidate states | `unvalidated_proposal` | Tests whether the global token improves abstention rather than merely duplicating local scores |
| Evidence-DINO-Units evaluation | Stratify direct, implicit, related-context, and no-evidence questions | `repository_inference` | Prevents direct answer-box rows from hiding semantic grounding failures |
| Future scene branch | Evaluate VGOQ data as auxiliary question-to-evidence supervision | `unvalidated_proposal` | The task semantics are closer than ordinary phrase grounding, although the domain remains non-document |
| Evidence-label schema | Separate specific evidence, useful context, insufficient context, and no evidence | `unvalidated_proposal` | Avoids treating every overlapping region as equally sufficient |

## Required adaptations

- Replace natural/product-image masks with document candidates or map masks to
  parser-derived units.
- Add OCR text, layout, hierarchy, table relations, and structural closure.
- Support several disconnected evidence components.
- Distinguish answer-bearing boxes from all premises needed to derive an
  answer.
- Extend image relevance to page and cross-page relevance.
- Validate the automatically generated evidence categories before using them
  as selector ground truth.
- Measure actual downstream answer sufficiency and token cost.

## Advantages relative to the current subsystem

VGOQ supplies supervision and evaluation closer to question-to-evidence
semantics than ordinary object or phrase grounding. Its explicit relevance
head also provides a clean precedent for the current planned
`[EVIDENCE]`-based no-evidence head.

The current Evidence-DINO-Units architecture is stronger for documents because
it preserves structured candidates, full question tokens, candidate
interaction, OCR/layout features, and semantic closure. VGOQ does not justify
replacing that architecture with a CLIPSeg-style heatmap model.

## Concerns and failure risks

- Both released datasets are automatically generated.
- Valid evidence for general questions is subjective.
- Related context may be relevant without being sufficient.
- Product-image and VizWiz evidence differ from structured documents.
- A single-image relevance score does not solve multi-page completeness.
- Dense masks may not map cleanly to tables, headers, captions, or logical
  premises.
- The reported semantic gap may partly reflect dataset-generation or wording
  artifacts rather than only grounding difficulty.
- A strong page head could still coexist with weak local evidence selection.

## Open questions

- Does `[EVIDENCE]` improve calibrated no-evidence behavior beyond pooling the
  candidate scores?
- Should the page head predict binary answerability, graded relevance,
  evidence quality, or several outputs?
- Can VGOQ's specific-evidence/context categories be mapped reliably into our
  strong, weak, ignored, and no-evidence label schema?
- Does question-to-evidence training transfer from products and natural images
  to document pages?
- Can a VGOQ-style semantic reformulation test separate lexical shortcuts from
  actual question understanding in Evidence-DINO-Units?
- How should several relevant pages be represented when no single page is
  sufficient?

## Smallest decisive experiment

**Evidence level: `unvalidated_proposal`.**

Hypothesis:

> Explicit supervision of the global `[EVIDENCE]`/page-relevance head improves
> no-evidence calibration without reducing strong-label local candidate recall.

Control:

- canonical full-question unit scorer with page relevance derived only from
  pooled candidate states.

Treatment:

- the same model with the learned `[EVIDENCE]` token explicitly supervised for
  page relevance/no evidence and combined with pooled candidate states.

Freeze:

- candidate revision and cap;
- training rows and grouped split;
- local candidate labels and losses;
- visual/text encoders;
- selection thresholds and downstream answerer.

Measure:

- no-evidence FPR at fixed positive-page recall;
- page-relevance calibration error;
- strong-label local R@1 and R@5;
- strict distributed recall;
- source-specific changes; and
- whether `[EVIDENCE]` adds predictive information beyond pooled local scores.

Failure interpretation:

- remove or demote the learned token if it merely duplicates pooled candidate
  scores, harms local recall, or learns source artifacts rather than semantic
  page relevance.

This experiment is not scheduled or authorized by the current
Evidence-DINO-Units Stage 0–4 task.

## Analysis and interpretation log

- 2026-07-28 — Initial interpretation: VGOQ is adjacent evidence that
  question-to-evidence grounding is materially harder than direct referring
  expressions. Its separate relevance and spatial heads support our global-
  versus-local division, but do not prove that the learned `[EVIDENCE]` token
  is useful.
- 2026-07-28 — Specification comparison: the current `[EVIDENCE]` token is an
  auxiliary global readout and no-evidence input. Full question tokens and
  fixed candidate queries remain responsible for local evidence selection.

## Adoption status and decision history

- 2026-07-28 — `promising`: primary project/paper materials checked; no
  Evidence-DINO-Units change or VGOQ experiment implemented.
