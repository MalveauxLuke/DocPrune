Yes: start with a single-hop POC. Do not begin with synthetic multi-hop or answer-only RL.

The right sequence is:

> answer-anchor grounding → verified same-page multi-evidence → answer-feedback optimization → genuine multi-hop

This isolates failures and makes the Mini‑VGent claim defensible.

## What you actually have now

The completed V1 is substantial:

- 65,787 usable questions over 23,043 page positions.
- 40,963 training, 5,023 validation, and 4,748 internal-test questions.
- InfographicsVQA is isolated as a 15,053-question holdout.
- Every usable question maps to an OCR-backed page, but candidate construction and training have not begun. [Current state](</Users/god/Documents/COLPALI binary classification/agent-context/CURRENT_TASK.md:5>) [Split counts](</Users/god/Documents/COLPALI binary classification/reports/overlap_first_document_corpus.md:122>)

The critical limitation is supervision semantics:

- The boxes identify answer-bearing text.
- They usually do not prove that all headers, qualifiers, operands, legends, or surrounding context needed to understand the answer are included.
- Multiple boxes may merely be separate words of one answer or alternative answer occurrences. They are not automatically multiple evidence roles.
- V1 retains 27,312 conflict groups, including answer, page, and box disagreements; these cannot be flattened into one mandatory union. [Conflict audit](</Users/god/Documents/COLPALI binary classification/reports/overlap_first_document_corpus.md:82>)

So answer boxes are best understood as a high-precision lower bound on relevant evidence—not a noisy but complete evidence annotation.

I also found that several untracked planning documents still describe the older “acquisition in progress” and 90/5/5 design. The committed current task and corpus report now supersede those with the completed 80/10/10 V1.

## What VGent really demonstrates

The [actual VGent paper](https://arxiv.org/html/2512.11099) uses:

1. Detector-generated box proposals.
2. A frozen Qwen2.5‑VL encoder whose layerwise hidden states form semantic memory.
3. Box-coordinate proposal queries.
4. Cross-attention from proposals into those hidden states.
5. Self-attention among proposals.
6. Proposal-level binary classification trained with known spatial labels.

Its QuadThinker RL is separate: the encoder is rewarded against known quadrant counts, total object counts, coordinates, and center points. It is not trained from downstream answer correctness.

Three paper results matter especially for us:

- VGent also performs strongly on single-target grounding—90.1% average REC accuracy—so a single-evidence POC is legitimate.
- Its distinctive advantage appears in larger multi-target regimes. Therefore single-evidence data cannot validate the claimed complementarity benefit of candidate self-attention.
- Jointly unfreezing the encoder and decoder reduced F1 from 60.55 to 45.66 in the paper’s ablation. That strongly supports freezing the document VLM during the first Mini‑VGent experiment.

The original system is also not “mini”: it reports about 15.7B parameters and a large multi-stage A100 training program. Our smaller Qwen-based version is a project adaptation, not something the paper has demonstrated. The proposed [Qwen3‑VL‑Reranker‑2B](https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B) is nevertheless a sensible backbone because it is already trained for multimodal relevance scoring and visual-document retrieval.

I found no linked official VGent implementation on the paper or the [author’s publication page](https://weitaikang.github.io/), so these are paper-grounded architecture conclusions, not code-verified ones.

## Recommended training design

### Phase 0: Freeze and audit the candidate universe

Before training either model:

- Construct DeepSeek‑OCR‑2 semantic candidates: words/lines, paragraphs, table cells and regions, headers, captions, figures, plus controlled fallbacks.
- Give each candidate geometry, OCR text, type, hierarchy, and reading order.
- Map each answer box to candidates.
- Group contiguous word boxes into an answer-anchor component.
- Represent alternative occurrences as OR alternatives, not as “select every box.”
- Treat structural neighbors and unresolved annotation disagreements as `ignore`, not negatives.
- Use only verified wrong regions as negatives.

Do not train until answer-anchor candidate-oracle recall is high. The existing project targets—roughly ≥95% by document source and ≥90% strict multi-box recall—are reasonable POC gates.

### Phase 1: Native single-hop answer-anchor POC

Use a deterministic 15,000–20,000-row pilot from the 40,963 training questions, subject to the measured eligible count:

- `usable_in_v1=true`;
- exclude answer/page disagreements;
- initially exclude unresolved box conflicts;
- preserve document-grouped splits;
- use source/document-balanced sampling.

Train two matched models:

- **Direct scorer:** independently score every question–candidate pair using Qwen3‑VL‑Reranker‑2B.
- **Mini‑VGent:** freeze the same backbone and train only candidate projections, a small two-to-four-layer set decoder, and output heads.

The Mini‑VGent candidate query should include content and structure—not only coordinates:

```text
candidate query =
  geometry
  + candidate text/crop representation
  + unit type
  + hierarchy/reading-order features
```

Use masked or positive-unlabeled supervision:

- positive loss on mapped answer anchors;
- negative loss only on verified negatives;
- zero loss on plausible but unverified context;
- an OR-group loss for alternative valid answer locations.

Do not add a literal evidence-count head yet. Current box count mostly reflects annotation and candidate granularity, not the number of reasoning obligations.

This phase proves:

- the candidate pipeline;
- hidden-state/candidate alignment;
- single-anchor localization;
- latency and memory;
- whether Mini‑VGent beats the simpler scorer on duplicate and distractor competition.

It does not prove multi-evidence sufficiency or multi-hop reasoning.

### Phase 2: Same-page multi-evidence before multi-hop

This is the smallest dataset addition that tests the actual Mini‑VGent hypothesis.

Create a modest proposed pilot—about 1,000–2,000 training examples plus 200–300 separately held-out, heavily audited examples—requiring combinations such as:

- table header + value;
- two rows or values for comparison;
- claim + qualifier or exception;
- chart mark + legend;
- definition + application;
- answer span + disambiguating heading.

For each example, verify that:

1. The full evidence set answers the question.
2. No proper subset is sufficient.
3. Every selected unit has a named role.
4. The question sounds natural.
5. Generation occurs after document split assignment.

This tests candidate complementarity without conflating it with iterative multi-hop reasoning. Your existing staged research specification already points in this direction. [Proposed curriculum](</Users/god/Documents/COLPALI binary classification/docs/specifications/deepseek_ocr2_minivgent_evidence_localization/dataset_specification.md:2194>)

### Phase 3: Answer feedback—first as search/distillation, then RL

Your answer-feedback idea is promising, but I would not begin with online RL.

First use a frozen answerer to search for pseudo-minimal evidence sets:

1. Start from the answer anchor.
2. Add candidate context from a small plausible neighborhood.
3. Score answer correctness and groundedness.
4. Prune candidates with leave-one-out tests.
5. Reject ambiguous examples.
6. Distill the accepted sets into Mini‑VGent with supervised training.

Define “optimal information” explicitly:

\[
S^*=\arg\min_S \mathrm{Cost}(S)
\]

subject to the frozen answerer producing the correct, evidence-grounded answer from \(S\), while proper subsets fail.

Cost should include selected OCR tokens, visual area, and candidate/page count—not just number of boxes.

Only after that action grammar works should you try policy optimization over sampled evidence subsets. A useful reward would combine:

- answer correctness;
- evidence entailment/groundedness;
- improvement over empty or corrupted evidence;
- selected-area/token cost;
- redundancy;
- abstention accuracy.

Freeze the answerer and candidate universe. Otherwise the answerer and selector can co-adapt and reward each other for shortcuts. Also note that GRPO is not directly plug-and-play with Mini‑VGent’s parallel Bernoulli outputs; this would be a new subset-policy objective, unlike the paper’s autoregressive QuadThinker training.

### Phase 4: Genuine multi-hop

Do this only after same-page multi-evidence succeeds.

Synthetic composition is not automatically multi-hop. Genuine multi-hop requires a dependency graph: a later lookup must depend on or be disambiguated by an earlier result.

The preferred later source is real program/evidence data such as the TAT‑QA/TAT‑DQA family. Synthetic examples can augment it, but they should not be the sole evaluation set.

## The decisive experiment

Mini‑VGent is justified only if all of the following hold:

- Candidate oracle coverage is already high.
- The direct scorer systematically under-selects verified complementary evidence.
- Mini‑VGent improves strict complete-set recall or selected-only answer accuracy.
- The gain survives matched backbone, candidates, training rows, selected area/tokens, and compute.
- It does not materially worsen single-anchor grounding or no-evidence calibration.

If Mini‑VGent only improves ordinary single-box localization, the likely contribution is better semantic conditioning—not set reasoning. In that case, the simpler Qwen reranker should remain the default.

So my concrete answer is:

- **Yes, single-hop is appropriate for the first POC.**
- **No, it is not enough for the final Mini‑VGent claim.**
- **Add same-page verified multi-evidence before genuine multi-hop.**
- **Use answer feedback first to generate and distill minimal evidence sets; introduce RL only after supervised multi-evidence grounding is stable.**

Fresh verification: the focused corpus-builder suites pass, 20/20. The remote `/home/lmalveau/.../v1` package is not mounted on this Mac, so I verified the committed builders and repository records, not its artifact hashes live.
