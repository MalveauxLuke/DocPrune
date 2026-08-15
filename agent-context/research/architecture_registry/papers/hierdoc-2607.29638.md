# HierDoc: Hierarchical Page-to-Region Evidence Routing for Long-Document Visual Question Answering

## Registry metadata

- Paper: HierDoc: Hierarchical Page-to-Region Evidence Routing for
  Long-Document Visual Question Answering
- Stable identifier and reviewed version: arXiv `2607.29638v1`
- Last reviewed: 2026-08-14
- Applicability: `direct`
- Capability tags: long-document VQA; page selection; region selection;
  parser-native candidate IDs; structured set prediction; GRPO; grounded
  answering; multi-page evidence
- Possible project consumers: overlap-first V1 answer-anchor POC; future
  full-document evidence routing; downstream evidence-conditioned answering
- Evidence levels used: `paper_reported`, `repository_inference`,
  `unvalidated_proposal`
- Registry status: `promising`

## Paper link and factual capsule

- Primary paper:
  [arXiv abstract, HTML, and PDF (`2607.29638v1`)](https://arxiv.org/abs/2607.29638v1)
- Reviewed appendix: stage-wise GRPO objectives, reward ablations,
  same-document negatives, qualitative examples, and complete inference
  prompts in the primary arXiv version
- Official implementation: not identified in the reviewed paper; no code
  behavior is assumed by this record

**Evidence level: `paper_reported`.**

HierDoc is a long-document VQA evidence-routing pipeline. A generative page
policy selects page IDs from fixed-size windows. Selected pages are parsed by
MinerU2.5-Pro into typed regions with bounding boxes and OCR or table text. A
separate generative region policy sees the question, selected pages with
numeric aliases overlaid, and an `alias | type | OCR hint` candidate list, then
emits one XML-serialized region-ID set. A separate answer model receives the
selected full pages, region crops, and selected OCR/table text.

The page and region policies are initialized from Qwen3-VL-8B-Thinking and
optimized independently for one epoch with GRPO and deterministic set rewards.
The paper evaluates the complete system on five multi-page or long-document
VQA benchmarks. It does not demonstrate transfer to our frozen V1 candidate
revision, answer-anchor labels, Qwen3-VL-Reranker-2B, or a parallel MiniVGent
decoder.

## Paper-reported mechanism

**Evidence level: `paper_reported`.**

The three stages are:

```text
page policy:   (question, all document pages) -> selected page-ID set
region policy: (question, selected pages, parser regions) -> selected region-ID set
answer model:  (question, selected pages, selected crops/text) -> answer
```

Long documents are divided into non-overlapping page windows of at most 16
pages. Window outputs are unioned. If more than eight pages are selected, the
same page policy performs a bounded reflection pass over only the first-pass
pages. Region selection uses an analogous reflection trigger of eight selected
regions.

Parser candidates contain a sample-local alias, page ID, pixel-space box,
semantic type, and OCR or table text. The region policy returns exactly one tag
such as `<evidence_region>3,17</evidence_region>`. The page images are treated
as authoritative and OCR/table text as a hint.

Gold evidence boxes are mapped to parser regions when either box IoU is at
least `0.10` or at least `0.80` of the parser-candidate area lies within the
gold box. The training candidate pool also includes regions from sampled
same-document non-evidence pages.

The region reward is:

```text
0.20 * set recall
+ 0.50 * set F1
+ 0.20 * set precision
+ 0.10 * strict-format compliance
- invalid-alias penalty
```

The invalid-alias penalty is `min(0.50, 0.10 * invalid_count)`. A missing
region tag receives zero reward, a non-strict action is capped at `0.25`, and
an action containing an invalid alias is capped at `0.10`. Both policies use
four rollouts per prompt, learning rate `1e-6`, KL coefficient `0.01`, and
maximum image long edge `1024` in the reported experiments.

The reported region corpus begins with 2,418 ViDoRe-v3 records and retains
1,022 after parser alignment and auditing. In the MMLongBench-Doc incremental
ablation, the complete trained and reflected region stage improves downstream
QA accuracy/F1 by 5.51%/4.82% relative to selected full pages alone. The same
ablation also shows that removing full pages and retaining only selected crops
and OCR reduces performance, supporting complementary global and local views.

## Why this paper attracted attention

**Evidence level: `repository_inference`.**

HierDoc addresses almost exactly the interface this project wants to test:
question-conditioned selection of a compact set of semantically meaningful
document regions from a frozen candidate universe. It also makes three useful
scientific separations explicit:

- page retrieval and within-page localization are different error stages;
- evidence selection can terminate before answer generation; and
- selected local crops should augment, not automatically replace, the full
  page context.

Its region-ID action is the closest external baseline for our proposed
MiniVGent selector. It is much closer than a generic hierarchy parser because
it actually performs question-conditioned document evidence selection.

## Our analysis and interpretation

**Evidence level: `repository_inference`.**

The most relevant transfer is the region-policy action space, not the complete
pipeline. V1 starts from a known question-page record and frozen candidate
revision, so its single-hop POC can bypass page routing and ask whether the
model can select answer-bearing candidate IDs on the supplied page.

HierDoc and MiniVGent solve the same surface decision in fundamentally
different ways:

| Property | HierDoc region policy | Proposed MiniVGent |
|---|---|---|
| Candidate representation | Numeric aliases overlaid on pages plus text list | Explicit content, ROI, geometry, and type tensors |
| Decision | Autoregressive XML ID set | Parallel independent candidate logits after joint decoding |
| Joint reasoning | Implicit in VLM generation context | Explicit candidate self-attention |
| Semantic memory | Ordinary VLM visual/text token sequence | Selected frozen Qwen hidden-state memories |
| Training | GRPO with structured-set reward | Group-aware supervised ranking/multilabel loss |
| Backbone in primary proposal | Qwen3-VL-8B-Thinking | Qwen3-VL-Reranker-2B plus compact decoder |

This makes HierDoc a strong system-level baseline but not a controlled
architecture ablation. A smaller Qwen autoregressive-ID arm using the same V1
page, candidates, and aliases is needed to isolate the output/action design
from model scale and training method.

The supervision mismatch is the main constraint. HierDoc's reward assumes a
gold evidence set against which precision, recall, and F1 are meaningful. V1
provides accepted answer-anchor alternatives. It does not prove that every
unmatched region is unnecessary context or that the anchor alone is a complete
answer-support set. A direct copy of the reward could therefore punish useful
but unlabeled evidence. GRPO is valid only on a derived view with auditable
reward-complete candidate labels, and alternative anchors must be scored with
OR semantics rather than as one mandatory union.

The paper is also directly informative for later phases. Its page-stage error
is irreversible: omitted pages cannot be recovered by region selection. A
future full-document arm must report page oracle/recall separately from region
performance. Its result that full pages and local evidence are complementary
also supports evaluating localization quality and downstream answer quality as
separate outcomes.

## Reusable mechanisms

**Evidence level: `repository_inference`.**

1. Separate page-set and region-set policies with independent metrics.
2. Parser-native numeric aliases instead of free-form coordinate generation.
3. Question plus alias-overlaid page plus typed OCR candidate list.
4. Strict machine-parseable set actions with invalid-ID penalties.
5. Recall/F1/precision reward balancing for compact evidence sets.
6. Bounded second-pass reflection only after abnormally large selections.
7. Same-document distractors for region-policy training.
8. Full-page context augmented by selected crops and OCR rather than replaced.

## Possible COLPALI integration points

**Evidence level: `repository_inference`.**

| Consumer | Possible reuse | Why it may help |
|---|---|---|
| V1 answer-anchor POC | Run only the region-ID policy on the supplied page and frozen candidates | Direct external baseline for answer-anchor localization |
| MiniVGent evaluation | Compare variable selected sets under common anchor-set metrics | Prevents a ranking-only evaluation from favoring parallel logits |
| Future multi-page routing | Add an independently trained page-ID policy before region selection | Tests coarse-to-fine error propagation on real documents |
| Downstream answering | Send selected full pages plus selected crops and OCR | Tests whether compact local emphasis helps without discarding global layout |
| Negative design | Add audited same-document distractor pages or regions | Provides harder document-consistent negatives than unrelated examples |

None of these mappings has been implemented or measured locally.

## Required adaptations

**Evidence level: `repository_inference`.**

A valid V1 adaptation requires:

- replacing MinerU2.5-Pro outputs with the exact frozen V1 candidate revision;
- deterministic alias assignment and auditable overlay rendering;
- a strict prompt and output parser with all failure modes retained;
- OR-aware scoring across accepted answer-location alternatives;
- a reward-completeness audit before using precision-bearing GRPO;
- no page policy in the single-page POC;
- common variable-set metrics for HierDoc and MiniVGent;
- a smaller autoregressive-ID control for architecture interpretation;
- matched source/split/candidate manifests across all arms; and
- explicit cost reporting because an 8B generative policy and a 2B frozen
  reranker plus compact decoder are not compute matched.

## Advantages relative to the current subsystem

**Evidence level: `repository_inference`.**

Relative to independent pairwise reranking, HierDoc produces a single joint
candidate-ID set and can trade coverage against compactness during training.
It uses a document-native action space, preserves the clean boundary between
evidence acquisition and answering, and supports multi-page extension without
free-form box generation.

The current pairwise reranker remains the correct first baseline. It is
cheaper, already authorized, easier to audit, naturally produces a ranking,
and does not require a reward-complete candidate set or policy rollouts.

## Concerns and failure risks

- Page-to-region routing has irreversible upstream error propagation.
- Parser/candidate recall remains a hard ceiling.
- Alias overlays can obscure tiny text or create a shortcut based on rendering.
- Autoregressive output can be malformed, incomplete, duplicated, or invalid.
- Region-set precision is unsafe with incomplete evidence labels.
- The paper's 8B model and GRPO training are not compute matched to MiniVGent.
- Bounded reflection is a heuristic second inference pass and changes cost.
- Answer-anchor success does not demonstrate complete evidence selection.
- Downstream QA gains may depend on the answer model and evidence-composition
  prompt rather than selector quality alone.

## Open questions

- How many V1 rows have reward-complete candidate labels suitable for GRPO?
- Are numeric overlays readable without hiding document evidence?
- Does alias renaming change the generated selection?
- Does direct GRPO learn the strict action reliably on V1, or is an explicitly
  labeled SFT warm-up required as a separately named adaptation?
- At matched selected-set size, does HierDoc improve anchor recall over the
  tuned pairwise reranker?
- Does a small autoregressive-ID control retain the benefit of the 8B policy?
- Does MiniVGent's explicit candidate interaction improve single-anchor
  competition, or only later complete multi-evidence tasks?
- When multi-page supervision exists, how much final error is attributable to
  page misses versus region misses?

## Smallest decisive experiment

**Evidence level: `unvalidated_proposal`.**

Hypothesis:

> On a supplied page and frozen V1 candidate universe, a HierDoc-style
> region-ID policy improves compact answer-anchor set selection over its
> zero-shot initialization, while MiniVGent can match or exceed it at lower
> deployment cost.

Controls and treatments:

- tuned pairwise Qwen ranking;
- zero-shot HierDoc region prompt;
- V1-trained HierDoc region policy if the reward audit passes;
- a matched small-Qwen autoregressive-ID control; and
- MiniVGent without and with candidate interaction.

Freeze:

- V1 split and candidate revision;
- question text, aliases, overlays, and candidate-list serialization;
- answer-anchor alternatives and audited negatives;
- validation threshold and reflection policy; and
- common evaluation code and cost accounting.

Measure:

- candidate oracle;
- best-alternative selected-set precision, recall, and F1;
- answer-anchor hit rate and average selected count;
- ranking metrics where an arm emits scores;
- strict-format, invalid-ID, empty-set, and alias-renaming failures; and
- latency, peak memory, visual/generated tokens, and model parameters.

Failure interpretation:

- retain the tuned pairwise reranker if joint set policies do not improve the
  common metrics;
- reject GRPO on V1 if the precision reward cannot be grounded in audited
  labels; and
- do not attribute an H1/MiniVGent difference to architecture without the
  smaller autoregressive-ID control.

This experiment is not scheduled or authorized.

## Analysis and interpretation log

- 2026-08-14 - Initial review of `2607.29638v1`, including the method,
  implementation details, selector ablations, evidence-composition ablation,
  GRPO appendix, same-document-negative ablation, and complete prompts.
- 2026-08-14 - Corrected the MiniVGent POC baseline from an unrelated global
  hierarchy parser to HierDoc's actual generative page-to-region evidence
  routing. Restricted the V1 POC to the region stage because its records supply
  one page and do not supervise full-document page selection.

## Adoption status and decision history

- 2026-08-14 - `promising` as a direct external baseline and later multi-page
  routing reference. No implementation or training authorization was granted.
