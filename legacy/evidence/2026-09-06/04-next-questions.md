# From measured effects to the next research questions

Status: interpretation and future-study context, not new experimental results.
Sources: the canonical regional experiment plan, completed visual audit, correction
corpus record, and this chat's explicit owner decisions. No future study was run to
prepare this packet.

## What the completed work motivates

The central scientific progression is from an attention heuristic to observed
intervention effects, then toward learning a selector that can predict useful effects
without hundreds of test-time interventions. The available results motivate that
progression; they do not establish a deployable selector or choose its architecture.

Three questions should stay separate:

1. **Selection headroom:** can a privileged answer-conditioned intervention improve
   generated answers at a matched token budget? The 48 development observations
   provide positive measured evidence, subject to scoring and evidence limitations.
2. **Predictability:** do pre-answer representations contain distinctions that a
   learned model can use to predict deletion responses on other documents? This is
   the planned shared-probe question. An incorrect final answer does not prove its
   internal states are useless; human recognition does not prove their decodability.
3. **Deployment depth:** can useful selection occur before the first answer-decoder
   block, or does it depend on computation already performed in the decoder? The
   one-question early/late fits do not resolve cross-document answer correction.

A good presentation transition is: “We can change the frozen answerer's behavior
through region selection. We now need to verify genuine correction and learn which
inputs and computation make that selection predictable.” This preserves the positive
measurement without treating every scorer gain as a grounded correction.

## Better corpus: what exists and what went wrong in collection

The corpus work produced a real, inspectable candidate collection and conservative
answer contracts. **It did not produce a 30-question incorrect-only pool.** Inspection
of `outputs/correction-corpus-2026-09-06/gather.py` confirmed that the collection applied
protected-document exclusion and supporting-document presence filters, but omitted
an incorrect-baseline filter. This was a collection/process error, not a finding that
only three questions in an incorrect pool are truly wrong.

| Current artifact | Count | Permitted interpretation |
|---|---:|---|
| Mixed candidate pool | 317 | Supporting document IDs appear in cached top four; exact evidence not guaranteed |
| Matching cached post-BTP/QTP, no-CTP references | 30 | Available references, not 30 wrong answers |
| Visually reviewed original questions | 30 | 3 verified wrong, 10 already correct, 17 quarantined |
| Minimal rewritten questions | 6 | New baseline unknown; not admitted wrong cases |
| Other candidates without that matching reference | 287 | Baseline correctness unestablished under this condition |

The three verified-wrong seeds are the 28 Days Later poster count, Mladen Žganjer's
2. Liga season, and George Hilton's role in The Atlantis Interceptors. They are
agent-reviewed exploratory seeds, not independently adjudicated generalization data.

Do not use 3/30 as an estimate of incorrect-pool quality, prevalence of benchmark
errors, or the probability of finding a correction case. The immediate corpus task
is to identify available **incorrect** answers under a clearly declared reference
condition first, then visually adjudicate evidence and answer contracts. The actual
incorrect pool has not been shown to be exhausted. The 40-question target remains
unfinished; the current matching-reference artifact limit explains the last stopping
point but does not certify an exhaustive search of every existing result source.

The baseline choice must be explicit: post-BTP/QTP without CTP is the reference
before the selector deletion intended here. Native DocPrune includes CTP. All-kept
omits the initial pruning. These are different experimental conditions; a wrong
answer in one cannot label another condition as wrong.

## What rigorous correction evaluation needs

- Verify all required evidence in the actual four supplied pages, not merely the
  presence of a supporting document ID. Check image identity, caption, table header,
  unit, temporal scope and cross-page relations.
- Treat all members of a requested set as one complete target; do not maximize over
  members as if they were interchangeable accepted answers. Use reviewed aliases and
  precision-aware numeric rules; unfamiliar paraphrases remain unresolved.
- Freeze question, document snapshot, complete-answer target and scoring contract
  before examining oracle comparisons. Evaluate generated answers without arm labels
  when obtaining adjudication.
- Keep minimal revisions as new designed development items. Do not transfer the
  original baseline outcome to revised wording or call them untouched confirmation.
- Group splits by all documents visible to the selector, including retrieved
  distractor documents. Questions and masks sharing a document are not independent.

These are evaluation requirements, not evidence that pruning has no useful effect.
A corpus intentionally designed to test particular relationships is appropriate for
exploration if selection rules and its scope are disclosed.

## Two future studies that must not be conflated

### Existing shared-probe plan (design, not reported outcome)

The canonical plan describes 600 questions, 360 train / 120 validation / 60 balanced
primary test / 60 natural-prevalence secondary test. It predicts a **fixed B13 deletion
target** from independently read B0..B13 pre-answer features. Planned inputs include
pooled hidden states, compact QK features, and their combination, with geometry-only,
question-only and native-policy controls. It trains on centered observed G and S
responses to 32 masks per training question. Gold budget-local response R² is the
primary model-selection metric. Conditional local trajectory comparisons test whether
ordered feature history adds information beyond capacity and aggregation controls.

This is not an early-versus-late intervention ablation: changing the feature read
layer while predicting the same B13 teacher target changes the predictor, not the
intervention boundary. The 32-mask shared-learning design also does not supersede
256 masks for reliable per-question oracle refits.

Owner-reported operational state: 600 transferred to H200, segmentation pending.
No current H200 jobs or results were inspected for this packet. The 100 wrong-question
cohort is confirmation and remains outside architecture tuning.

### Proposed own-boundary selector comparison (not executed)

Give each architecture targets generated at its own deployment boundary. For a
small early/late study, use corresponding masks and achieved budgets at B_input
(before the first decoder block) and the relevant intermediate boundary. Generate
outcomes and fit separate 256-mask oracles at each boundary; check all-keep parity.
B_0 means after the first block, not B_input. Cached intermediate states and earlier
KV caches can retain information from regions deleted later.

The early contract is initial BTP/QTP, one shared vision pass, question plus all
remaining visual tokens with region identities available to the selector, and only
selected original visual tokens entering the frozen answer decoder. No second vision
pass, extra answerer crops, full-page fallback, gold answer or fixed-self answer is
available at deployment.

| Candidate | Potential strength | What remains to demonstrate |
|---|---|---|
| Attached | Uses frozen question/visual states after some decoder computation | Useful distinctions are decodable; prefix cost is justified |
| Compact independent | Low overhead; question-conditioned pooling and cross-region interaction | Adequate learned semantics, not just access to all tokens |
| Pretrained approximately 2B | May bring useful semantic priors and relation handling | Shared-Qwen connector compatibility, cost, and selection ability |
| Hybrid | Combines compact token/region processing with pretrained interaction | Alignment and semantic benefit justify added complexity |

No comparative accuracy or latency results for these four architectures are supplied
by the completed experiments. The packet should introduce them as alternatives to
test, not announce a winner. A concrete early candidate could predict region utility
from its own-boundary gold-support oracle, use cross-region transformer interaction,
and choose a whole-region set by token-cost knapsack at the native achieved budget.
This is a hypothesis, not a replacement of the separately approved shared-probe plan.

## Manifest discrepancy to preserve in speaker notes

The available SOL 600 split artifact had 360/120/120 balanced splits rather than the
canonical 360/120/60/60 design. The visual audit also found QID and retrieved-document
overlap across available 48/100/600 manifests, plus cross-split retrieved-document
overlap inside the 600. The plan's requested independence is therefore not established
by those saved manifests. A newer H200 manifest may resolve this; none was verified.
Do not write “three independent cohorts” or imply the audit changed running inputs.

## Source pointers

- `docs/experiments/regional-attribution/EXPERIMENT_PLAN.md`: current probe design.
- `docs/experiments/regional-attribution/analysis/VISUAL_AUDIT_24_2026-09-06.md`:
  boundary/cache interpretation, cohort overlap and architecture alternatives.
- `docs/experiments/regional-attribution/research/CORRECTION_CORPUS_2026-09-06.md`:
  corpus records and scoring contracts; read with the filter-error correction above.
- `outputs/correction-corpus-2026-09-06/round2/summary.json`: cumulative review counts.
- `src/docprune/correction_scoring.py`: conservative opt-in scoring prototype.
- Original protected cohort outcomes were not used in this synthesis.
