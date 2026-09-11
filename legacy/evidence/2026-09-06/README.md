# Presentation evidence packet: frozen-Qwen document selection

Prepared 2026-09-06 on SOL. This curated copy is tracked in Git and works after cloning. This is source material for a presentation writer,
not a slide deck. It gathers useful completed experiments and their implications,
with optional detail for speaker notes. No model run, training, retrieval or mask
sweep was launched for this packet.

## Suggested central story

**We are testing whether document context can be selected using measured effects on
a frozen answerer, and whether a deployable selector can learn those effects.**
Attention pruning is the starting comparator; answer-conditioned regional attribution
is a diagnostic of possible improvement; learned selection is the next research
question. Evidence verification determines whether improved scores mean true answer
correction.

| Story beat | Evidence to use | Defensible interpretation |
|---|---|---|
| 1. Test the original heuristic | Controlled native-budget DocPrune-versus-random results; larger preliminary follow-up | Attention ranking's advantage must be measured at the same token budget and boundary; keep small-study uncertainty and larger-study status visible |
| 2. Locate what needs improvement | All-kept/BTP/QTP/CTP stage comparison | In this local reconstruction, decoder pruning is a promising stage to improve; this is not a claim about unpublished author code |
| 3. Measure region effects directly | One-question ContextCite-style masks and held-out LDS/RMSE | A simple surrogate can predict answer-conditioned mask responses on this question; selection is no longer based only on attention scores |
| 4. Ask what target and depth matter | Own-boundary B_input/B13 fits for accepted gold and generated response | Gold-conditioned and self-conditioned objectives differ; early/late results must be compared within the same target and mask setup |
| 5. Test beyond one question | 48-question development pilot, native matched budgets | Gold support improves saved wrong-stratum F1 and preserves scored-correct answers; this is development evidence of useful intervention effects |
| 6. Inspect what the gains mean | Real table correction alongside missing evidence, stale gold and a hidden semantic loss | Score changes are not automatically grounded correction; this motivates stronger targets and evidence checks |
| 7. Make effects predictable and deployable | Planned shared probes, own-boundary selector alternatives | Test whether useful information is learnable from available inputs, and what computation/depth it needs; no architecture winner yet |

The chronology matters, but do not imply that every later design decision was
prespecified by the earliest experiment. The visual audit and corpus corrections are
later interpretation of earlier measurements.

## Read in this order

1. [DocPrune versus random and stage localization](01-docprune-random.md).
2. [One-question depth, target and LDS diagnostics](02-one-question-depth.md).
3. [48-question pilot, mask-count experiment and visual audit](03-pilot48.md).
4. [Next questions, corpus correction and architecture context](04-next-questions.md).

Each experiment chapter has a corresponding JSON evidence record. The packet also
includes an [asset manifest](asset-manifest.json), copied case assets under `assets/`,
and source snapshots/provenance in the source manifest. The archived source paths
and runtime revisions are evidence provenance; the repository HEAD at packet creation
is not automatically the revision that ran an experiment.

## Priority for the main presentation

**Core:** matched-random comparison, the two-by-two accepted/self × early/late
one-question table, wrong/correct-stratum pilot table, one verified table-correction
example, and the resulting next research question.

**Useful context:** stage localization; surrogate fidelity and set stability;
comparison against unpruned as well as DocPrune; a single scoring/evidence caveat
example; no test-time gold requirement in the intended learned selector.

**Appendix/speaker notes:** reconstruction alternatives, preliminary larger-random
analysis status, exact-sequence target definitions, 192-mask failure concentration,
cohort overlap and split discrepancies, full case token footprints, and corpus
collection provenance. These should remain available without overwhelming the story.

## Numerical and terminology rules

- Tables in the early work often report **F1 points on 0–100**, while the pilot
  reports **F1 fractions on 0–1**. For example, 0.38125 equals 38.125 points.
  A +0.121875 fraction is +12.1875 points, not +0.121875 points.
- **Native DocPrune**, **post-BTP/QTP without CTP**, and **all visual tokens kept**
  are different references. Always label the one used.
- **LDS** measures agreement between a surrogate and held-out intervention scores;
  it is not QA accuracy, proof of grounding, or proof that a retained set is sufficient.
- The diagnostic is **adapted answer-conditioned regional ContextCite**, with
  physically deleted tokens at a decoder boundary. It is not vanilla ContextCite,
  a token-level oracle, or an already deployable selector.
- **B_input** is before the first decoder block. **B_0** is after zero-based block0.
  The one-question B13 and 48-pilot dynamic B14/B16 results are different designs.
- **Exact rescue**, unless explicitly adjudicated otherwise, means an exact match
  under the historical scorer. Use “document-supported correction” only when the
  retrieved evidence and answer have been checked.
- Mask repeats, questions and documents are different units. Confidence intervals
  and comparative claims must retain the original resampling unit and cohort status.

## Claims to avoid

- “DocPrune is no better than random.” The small study is unresolved; the larger
  preliminary result must also be reported with its status.
- “High LDS proves an early selector works.” It establishes score predictability
  on the measured mask distribution for a particular question/target, not deployment.
- “All four exact rescues are grounded corrections,” or “no harms occurred.” Those
  describe historical scorer outputs, not the later visual audit's semantic findings.
- “192 masks loses four different corrected questions.” The four repeat losses
  concern one question, later flagged for missing evidence.
- “The corpus has 30 incorrect questions,” or “only3of30incorrect questions are valid.”
  The inspected collection omitted an incorrect-baseline filter and was mixed.
- “We have independently evaluated the 48, 100 and 600 collections.” Available
  manifests have overlap discrepancies; confirmation outcomes are not part of this packet.
- “Attached/compact/~2B/hybrid selectors have been compared.” They remain proposed
  alternatives; the shared-probe plan and own-boundary comparison are distinct studies.

## Visual material

- **Q29:** table evidence plus native/support token footprints: a real correction.
- **Q26:** correct portrait and deleted-token illustration: a scored tie can hide harm.
- **Q37:** supplied-page overview beside separately located source poster: label the
  latter **not supplied to the answerer**.
- **Q41/Q43:** appendix details for complete-set scoring and stale-gold problems.

Overlay images illustrate which token locations survived deletion. They are not
literal edited images supplied to the cached decoder. Read each asset's caption and
case JSON; some details were inspected only after outcomes were revealed.

## Git transfer

This directory includes the selected images and source snapshots. Open README.md on GitHub or index.html locally. Absolute SOL paths inside provenance records identify original artifacts; they are not required to read this packet. Full experiment folders and the larger raw result banks remain on SOL.
