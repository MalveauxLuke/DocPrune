# DocPrune Random, Coverage, Attribution, and Visual-State Dependence Experiment

Status: approved amended design; preliminary Task 9 pilot complete; enriched pilot next
Canonical date: 2026-08-26
Revision: Wang semantics amendment approved 2026-08-27; review-driven
amendment approved 2026-08-26; Task 9 oracle-pilot amendment approved
2026-08-31
Execution anchor: [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)
Result ledger: [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md)

## Purpose

This experiment asks whether the two local DocPrune CTP reconstructions select
useful document visual tokens better than chance, whether spatial coverage
explains any random-versus-attention difference, and how dependence on explicit
visual-token states changes with decoder depth. A later exploratory component
tests whether privileged answer-conditioned regional interventions reveal
useful selection structure.

The experiment starts from the completed local DocPrune reproduction and its
admitted all-kept generation path. It does not rerun retrieval. All comparisons
use cached ordered top-4 pages, persisted page features, current BTP and QTP,
the same Qwen checkpoint, prompt, preprocessing, decoder, positions, decoding,
and evaluator.

The supported claims concern local controlled reconstructions. They do not
establish identity with DocPrune's unpublished author implementation.

## Experiment order

The phases are ordered so that the primary result does not depend on a
speculative attribution method:

1. Native-policy controls under the corrected runtime.
2. Ranking-only aggregate/literal scores versus global and coverage-controlled
   random selection.
3. An explicit-visual-state removal curve over fixed decoder boundaries.
4. Exploratory gold-answer-conditioned regional attribution using ContextCite
   machinery and independent MinerU masks.
5. A separately approved, bounded Wang standalone-contribution study only if
   it remains useful after a feasibility audit.

Failure of phases 4 or 5 does not invalidate or delay phases 1–3.

## Questions and hypotheses

1. At the native aggregate-logit CTP layer and budget, is its expected F1
   superior, equivalent, or unresolved relative to uniform random selection?
2. Does literal or aggregate score ranking add value after page and coarse
   spatial occupancy are controlled?
3. At which tested decoder boundaries can explicit visual-token states be
   removed within a prespecified quality margin?
4. On questions where retrieved/post-QTP visual evidence can affect behavior,
   are the random-versus-attention conclusions different?
5. In a controlled developmental pilot, how much generated-answer headroom
   does an accepted-answer-conditioned ContextCite whole-region selector show
   over native dynamic-layer DocPrune, and how much of that gap remains after
   comparison with privilege-matched gold-answer-conditioned attention?

The experiment does not ask whether MinerU regions are evidence annotations,
whether the method provides localization, or whether a privileged gold-answer
ranking is deployable.

## Required literature and source code

Inspect paper-author code before writing an equivalent local component. A local
port may begin only after the relevant source files, commit, behavior, and
Qwen/DocPrune adaptations are recorded in the experiment log. Absence of author
code must be recorded explicitly.

| Work | Complete paper | Paper/author code and pin | Role here |
|---|---|---|---|
| DocPrune | [arXiv:2604.22281](https://arxiv.org/abs/2604.22281) | No author repository is linked by the paper as of 2026-08-26. Linked resources include [Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct), [ColPali-v1](https://huggingface.co/vidore/colpali-v1), and [M3DocVQA](https://huggingface.co/datasets/m3docrag/m3docvqa). | Existing local reconstruction and controls. |
| Wen et al., *Are We Solving the Right Problem?* | [arXiv:2502.11501](https://arxiv.org/abs/2502.11501) | No author implementation is linked in v2. [FastV](https://github.com/pkunlp-icler/FastV) at audited commit `d1659729b5bf1be225e99ee15783deeea80f63b1` is code for a studied baseline, not Wen et al.'s code. | Random and coverage-controlled baselines. |
| Zhang et al., *Beyond Text-Visual Attention* | [arXiv:2412.01818](https://arxiv.org/abs/2412.01818) | [VisPruner](https://github.com/Theia-4869/VisPruner), audited commit `aefa01adc7c7ce6334e880c88225e90cede760d1` | Attention-concentration and diversity interpretation; not an arm. |
| Wang et al., *When Token Pruning Is Worse than Random* | [arXiv:2512.07580](https://arxiv.org/abs/2512.07580) | [Information-Horizon](https://github.com/YahongWang1/Information-Horizon), audited commit `75909b2936a13d7214e83514f0bc0cb9cc91139e` | Uniform random core, layer diagnostics, paper-described zero-mask information, and released-code 2x2-window physical-deletion ranking. |
| Endo et al., *Feather the Throttle* | [arXiv:2412.13180](https://arxiv.org/abs/2412.13180) | [FEATHER](https://github.com/markendo/FEATHER), audited commit `c2a09b2765967601054c7b1fd513ac3e94ee4fc9` | Coverage interpretation and claim limits; not an arm. |
| ContextCite | [arXiv:2409.00729](https://arxiv.org/abs/2409.00729) | [MadryLab/context-cite](https://github.com/MadryLab/context-cite), audited commit `c11f8ace6e68ba0121b2e2f1f5c896da9e4156f4` | Sparse subset-intervention surrogate; intermediate visual-state use is an adaptation. |

MinerU is an independent experiment tool, not a method used by the required
pruning papers. Use [MinerU](https://github.com/opendatalab/MinerU) at tag
`mineru-3.1.0-released` / commit
`d9cd58add047c2364c1198eefcb1ee9cd63a971a` with
[MinerU2.5-Pro-2604-1.2B](https://huggingface.co/opendatalab/MinerU2.5-Pro-2604-1.2B)
revision `d3f5e08d073c21466bbabe21c71bb1e9c2e595da`. Do not silently
substitute MinerU 2605 or current HEAD.

## Verified literature boundaries

- Wen et al. show that random and pooling beat specialized methods in many, not
  all, tested settings and directly test spatial distribution with Window
  FastV. Coverage must therefore be experimentally controlled if it is part of
  the mechanism claim.
- Zhang et al. motivate combining saliency with visual diversity. VisPruner is
  not a DocPrune decoder-stage replacement.
- Wang et al.'s paper describes standalone token information with zero-masked
  hidden states, a no-visual baseline, and first-target-token probability. The
  pinned released `cal_info`/`info_prune` paths instead physically delete all
  visual states or retain one 2x2 visual-token window, rank the first-gold-token
  probability for that window, broadcast its score to the four members, and do
  not use the saved no-visual baseline in ordering. These are distinct
  estimands; neither may be presented as the other.
- Endo et al. show that QA can conceal poor localization. This experiment does
  not make grounding claims without evidence-region annotations.
- ContextCite attributes a specified response to whole sources under subset
  interventions. It does not validate broadcasting a whole-region coefficient
  to arbitrary individual tokens.

## Frozen execution contract

Every paired arm uses the same:

1. cached ordered top-4 pages and persisted per-document page features;
2. current BTP and QTP semantics and, within a comparison, identical masks;
3. `Qwen/Qwen2-VL-7B-Instruct` revision
   `eed13092ef92e448dd6875b2a00151bd3f7db0ac`;
4. processor, raster bytes, prompt bytes, chat template, and preprocessing;
5. BF16 FlashAttention-2, greedy decoding, 128-token cap, and EOS IDs
   `[151645, 151643]`;
6. corrected manual decoder/runtime beginning at
   `dd5f000a909a718826541df816a4b65396764e1c` or a reviewed descendant;
7. original nonvisual tokens, token order, Qwen M-RoPE position values, and
   cache topology outside the declared intervention; and
8. reference answers, evaluator, artifact schema, and paired analysis.

Fresh/global retrieval, global-index loading, feature rebuilding, and
support-document-only retrieval are forbidden. Violating output is
non-canonical and cannot be scored or merged with fixed-page controls.

## Cohorts and information control

### Development data

The existing 16-, 64-, and 245-question cohorts are all developmental. The
aggregate-logit reconstruction and other CTP semantics were selected or
evaluated on the 245 questions. The 64-question prefix and the remaining 181
questions are not confirmatory partitions.

Development data may be used for implementation, mask validation, Monte Carlo
calibration, power analysis, budget sensitivity, attribution feasibility, and
promotion decisions.

### Confirmatory method holdout

Before any confirmatory job:

1. Exclude every QID in the developmental 245 and every additional QID used by
   a CTP semantic diagnostic.
2. Consider only single-hop questions among the other full top-4 benchmark QIDs
   for which canonical cached DocPrune-mode ordered pages and persisted page
   features can be authenticated without retrieval.
3. Compute the required sample size from the developmental paired variance for
   80% power under the locked primary equivalence procedure. Use every eligible
   question if the required size exceeds the pool.
4. Order eligible QIDs by SHA-256 of
   `docprune-random-coverage-v2 || qid` and take the first required count.
5. Seal QIDs, order, source/support-document metadata, page identities, feature
   hashes, and the selection-program hash before opening candidate outcomes.

The holdout is a method holdout, not a pristine benchmark test, because
historical all-kept and literal results exist for the full 2,441 questions.
Selection may not inspect those scores.

Optional 245-question totals remain descriptive and must be labeled as reused
development data.

## Token population, boundaries, and budgets

Let `V` contain all merged visual tokens surviving current BTP+QTP across the
four cached pages. Ranking policies operate globally over `V` unless their
declared coverage constraint says otherwise.

Use `B_K` to mean **after zero-based decoder block `K` and before block `K+1`**.
Blocks `0..K` keep their full pre-intervention caches; later blocks see the
shorter sequence. `B_input` means after prompt embedding and before block 0.

For the native comparison, `l*` is the first block whose output last-token norm
crosses DocPrune's comprehension threshold. If no crossing occurs, native CTP
does not prune; ranking-only comparisons set `M=|V|` and become no-ops for that
question.

For a normal crossing, `M` is the aggregate-logit native threshold policy's
retained visual count at `B_l*`. `M=0` and `M=|V|` are valid edge cases.

The primary ranking-only comparison uses per-question `M`. A developmental
sensitivity curve additionally uses exact conditional retention targets of
55%, 65%, and 80% of `|V|`, with deterministic integer rounding recorded in the
artifact. Only the primary `M` comparison is confirmatory.

## Experiment family A — Native-policy controls

1. **All-kept generation admission.** Reuse the admitted parity gate unless the
   shared decoder path changes.
2. **BTP+QTP, no CTP.** Keeps every token in `V`.
3. **Literal/current native CTP.** Uses its post-softmax mean-head score,
   undocumented `|V|` scaling, native threshold, and native retained count.
4. **Aggregate-logit native CTP.** Uses mean raw head logits, visual-only
   softmax, `|V|` scaling, native threshold, and native retained count.

These arms compare complete local policies. They are not budget matched. Both
CTP reconstructions remain explicitly local and author-unconfirmed.

All quality controls must be rerun under the corrected runtime before final
comparison with a new arm. Historical quality is planning evidence only.

## Experiment family B — Ranking-only comparison

All arms intervene at identical `B_l*` and retain identical per-question `M`:

1. **Aggregate-score top-M.** Primary deterministic reference.
2. **Literal-score top-M.** Secondary deterministic score reference.
3. **Global uniform random top-M.** Uniform without replacement over `V`.
4. **Page-stratified random top-M.** Matches aggregate-score top-M's retained
   count on each page and samples identities uniformly within each page.
5. **Grid-stratified random top-M.** Uses a normalized 4×4 grid per page. If the
   budget is smaller than the number of nonempty cells, sample cells uniformly;
   otherwise give every nonempty cell one token. Allocate any remainder
   proportionally to remaining cell capacity using largest-remainder rounding,
   then sample uniformly within cells.
6. **Coverage-matched identity shuffle top-M.** Matches aggregate-score top-M's
   exact retained count in every page×4×4 cell and randomizes identities only
   within cells.

Because `M` comes from the aggregate-native threshold policy, aggregate-score
top-M must reproduce that native mask whenever no tokens tie at the threshold.
Threshold-tie cases use the frozen deterministic top-M rule and report the tied
set and any resulting identity difference. An unexplained non-tie difference is
an implementation failure, not an experimental contrast.

The exact 4×4 geometry is a local variable-resolution adaptation. The source
audit must compare it with Wen's window algorithm before implementation; any
change must be approved and recorded before QA outcomes.

### Random semantics

Preserve Wang's published core:

- uniform sampling without replacement;
- exact retained cardinality;
- original sequence order after selection; and
- three repetitions reported for author-code fidelity.

For statistical inference, use more repetitions as defined below. The Qwen
adaptation samples the dynamic combined post-BTP+QTP population, preserves
original M-RoPE values and heterogeneous caches, and records seeds keyed by
experiment version, QID, boundary, policy, and repetition. Never copy Wang's
hard-coded LLaVA visual offset, fixed 576-token span, or position renumbering.

## Primary estimand and statistical rules

The primary estimand is:

> Expected paired normalized token-F1 difference, aggregate-score top-M minus
> global uniform-random top-M, at each question's native `B_l*` and
> aggregate-native `M`, averaged over random masks.

Rules fixed before the method holdout:

- Primary metric: normalized M3DocVQA token F1 in percentage points.
- Secondary metric: EM.
- Practical equivalence margin: `δ = 1.0 F1`.
- Equivalence: the paired two-one-sided-test 90% interval lies wholly inside
  `[-1.0, +1.0]`.
- Superiority/inferiority: the paired two-sided 95% interval excludes zero in
  the declared direction.
- Otherwise: no statistically resolved difference; never call it a match.
- Start with 10 random masks per question on development data. Increase to 20
  before holdout sealing if estimated random-mask Monte Carlo standard error is
  greater than `0.25 F1`. Freeze the resulting count for every confirmatory
  question.
- Average random masks within question for the point estimand. Use nested
  resampling to preserve mask uncertainty.
- Form outer bootstrap clusters as connected components of questions sharing
  any `supporting_context` document ID. Report component sizes and fail closed
  if a single component makes the requested inference unidentified.
- Use 100,000 bootstrap draws for terminal intervals.
- The primary contrast is unadjusted. Apply Holm correction within each family
  of secondary coverage, budget, and layer contrasts.
- The method-holdout size must provide at least 80% developmental-estimate
  power for the equivalence procedure; record the calculation before sealing.

## Coverage interpretation

Coverage is experimentally manipulated only in family B. Page coverage,
4×4-cell occupancy, center-of-mass, dispersion, and mask overlap are also
reported descriptively.

- Aggregate versus coverage-matched shuffle tests identity within the same
  coarse occupancy pattern.
- Global versus page/grid-stratified random tests the effect of spatial
  allocation under random identity.
- Observational correlations alone do not support a causal coverage claim.
- MinerU layout coverage does not establish evidence coverage or grounding.

## Evidence-opportunity strata

The full sealed cohort remains primary. Prespecified secondary strata are:

1. **Support-document retrieved:** document recall@4 is 1.0 using existing
   `supporting_context` and cached retrieved document IDs.
2. **No-CTP answerable:** report BTP+QTP/no-CTP EM-correct and F1-positive
   strata separately.
3. **Input-visually sensitive:** `B_input` all-visual removal either changes a
   no-CTP-correct answer to incorrect or lowers the best-reference normalized
   gold sequence log-likelihood by at least `0.1` nat/token.

These are assay-sensitivity proxies. They do not prove that the exact evidence
page or answer-bearing post-QTP token survived.

## Experiment family C — Explicit-visual-state removal curve

This family is not called an information horizon. It physically deletes every
member of `V` at these fixed boundaries:

- `B_input`;
- `B_0`;
- `B_6`;
- `B_13`;
- `B_20`;
- `B_23`; and
- `B_26`.

Native per-question `B_l*` is a separate diagnostic. The fixed sparse grid can
bracket a transition among tested points; it cannot locate an exact earliest
decoder layer or make claims about unsampled intermediate layers.

Reference: BTP+QTP/no CTP on identical inputs. Primary contrast: all-drop minus
reference F1. Use simultaneous one-sided 95% lower bounds across fixed
boundaries from a max-statistic document-cluster bootstrap. A persistent
dependence boundary is the earliest tested `B_K` whose lower bound is greater
than `-1.0 F1` and for which every later tested boundary also passes. If the
curve later fails, do not declare persistence. Report non-monotonicity.

The result concerns dispensability of explicit visual-token states downstream.
Earlier blocks may have transferred visual information into text states. Use
“Wang-style information horizon” only if a separately approved study also
reproduces the near-zero standalone token-information condition.

## Experiment family D — Answer-conditioned causal-selection oracle pilot

This is **not vanilla ContextCite**, **not a token-level oracle**, and **not a
deployable selector**. MinerU is independent tooling. ContextCite sees accepted
answers and is therefore a privileged reference oracle. Its result measures
potential selection headroom; it does not show that a deployable learned
selector can recover that headroom.

### Region masks

MinerU identifies page-space text, table, figure, and other regions. A region is
one binary intervention source, not one decoder token. Post-BTP+QTP tokens retain
page-space footprints; ablating a region physically deletes the surviving
visual tokens assigned to it at `B_K`. Overlap uses the union mask. Every token
maps deterministically to one primary region or an explicit residual source.
Partition large residual background into the fixed page 4×4 cells before
attribution so residual cost is bounded.

Never broadcast a region coefficient to member tokens and never split a region
using original/raster index.

### Targets

For accepted gold answers `A`, the primary target under mask `z` is:

`max_{a in A} mean_t log P(a_t | a_<t, prompt, z)`.

The secondary contributive target is the same normalized full-sequence
log-likelihood for the model's unpruned generated response. Save per-reference
values and target identity. The accepted-answer target is primary for the
oracle pilot. The generated-response target and the completed `B_input`
diagnostic remain mechanistic diagnostics and do not decide pilot eligibility.
First-token probability may be reported only as a Wang comparability
diagnostic.

### Reference surrogate and deployment

1. Use 256 deterministic global fit masks with region keep probability `0.5`.
   Add 32 independent global held-out masks and 32 held-out masks local to the
   primary deployment budget. Keep the fit masks and pinned solver unchanged
   for the reference oracle.
2. Fit the pinned ContextCite Lasso-style surrogate; log adaptations.
3. Transform coefficients back to raw inclusion-feature space and treat each
   signed coefficient as the estimated total value of its entire region.
4. Let each region's cost be its post-QTP token count.
5. Find the maximum attainable whole-region cost `M′ <= M`; among subsets with
   that cost, select the maximum total coefficient. Reverse attribution selects
   the minimum total coefficient at the same `M′`.
6. Use hard physical deletion for both perturbation scoring and deployed-mask
   evaluation. Preserve surviving state order and original M-RoPE positions.
7. Report `M′`, requested `M`, and the gap. Never standardize coefficients a
   second time, broadcast them to tokens, or split a whole region.

### Preliminary stratified-random 48-question pilot

Run a simple discovery/calibration pilot before the distractor-enriched panel.
Its frozen eligible pool is the 245-question single-hop development set with
authenticated BTP+QTP/no-CTP outputs from the completed stage-localization
study. Under the repository's canonical list exact-match evaluator, that pool
contains 90 baseline-correct and 155 baseline-wrong questions. Using a recorded
seed, sample without replacement 24 questions from each stratum. Selection may
read only QID, accepted answers, the already-observed BTP+QTP prediction, and
fixed-page provenance; it may not read attention, ContextCite, or subsequent
arm outcomes.

For each question, first run native aggregate-threshold DocPrune. Before and
independently of ContextCite, it selects its crossing layer `l*_q`, retained
token count `M_q`, and native retained-token set. Freeze `l*_q` for every
regional intervention on that question. At native DocPrune's exact `M_q` and
the closest attainable whole-region cost `M′_q <= M_q`, evaluate:

1. the unpruned post-QTP reference;
2. native DocPrune using its dynamic layer, threshold, and exact retained-token
   set;
3. accepted-answer gold-support ContextCite-region;
4. gold-margin ContextCite-region when the unpruned generated response is a
   distinct non-gold alternative, using gold minus alternative normalized
   sequence log-likelihood as the target; and
5. one deterministic region-size-aware random comparator.

FastV remains excluded. The matched regional random comparator is included only
to make this same-question, same-action-space pilot interpretable; it does not
rerun or replace the 1,213-question Task 6 token-level random study.

Report paired normalized token-F1 and exact-match differences, per-question
win/tie/loss, rescue rate on baseline-wrong questions, preservation on
baseline-correct questions, gold-likelihood change, and gold-versus-alternative
margin change where defined. Report the two strata separately. Any combined
descriptive result is reweighted to the frozen eligible pool's natural
90/245 correct and 155/245 wrong proportions. Resample by question for
uncertainty calculations.
This is a developmental discovery/calibration result, not a precise estimate
of a small average advantage or a dataset-wide population claim.

### Subsequent enriched pilot cohort

Only after the preliminary random pilot is complete, seal the separate
48-question developmental mechanism panel before creating its ContextCite
outcomes. Selection may use fixed-page document
structure, accepted answers, already-observed unpruned answers, and region
contents, but it may not use ContextCite outcomes, attention scores, or any
selector comparison. Sample without replacement within each frozen pool using
a recorded seed, in this order:

1. **Uniform anchor — 16 questions.** Uniformly sample from the authenticated
   eligible fixed-page pool. This is a question-sampling anchor, not a new
   uniform-random pruning arm.
2. **Traceable distractor errors — 16 questions.** The unpruned answer is
   incorrect; its normalized value occurs in a region; an accepted answer
   occurs in a different region; both regions are on the fixed retrieved
   pages; the two values share a frozen semantic type such as date, amount,
   percentage, name, address, or table entry; and a pre-ContextCite audit says
   the question is answerable from the document.
3. **High ambiguity, baseline correct — 8 questions.** Rank eligible correct
   questions using a frozen score comprising the number of same-type candidate
   regions, question-similar non-gold regions, and a cross-page duplication
   indicator, then randomly sample from the prespecified high-ambiguity tier.
4. **Clean controls — 8 questions.** Randomly sample baseline-correct questions
   with one audited evidence region, few competing same-type values, and
   prespecified high unpruned confidence.

Sample the uniform anchor first and exclude its members from the remaining
primary strata. Preserve secondary multi-labels for analysis. If a stratum is
too small under its frozen rule, stop before attribution scoring and amend the
cohort rule; never fill it by inspecting selector outcomes. Attention-conflict
cases may be labeled after cohort sealing as an adversarial diagnostic, but
attention failure cannot define a primary stratum.

This enriched panel estimates mechanism behavior, not population prevalence or
average commercial value. Report every contrast by primary stratum. The
uniform anchor is descriptive at `n=16`; do not combine the four strata into an
unqualified population estimate. The completed 1,213-question Task 6
random-versus-DocPrune holdout remains separate population-level evidence and
is not rerun or reinterpreted as this same-action-space oracle comparison.

### Dynamic layers, matched budgets, and controlled pilot arms

The primary comparison is question-adaptive but outcome-independent. Native
aggregate-threshold DocPrune runs first and selects `l*_q` and exact budget
`M_q` without access to ContextCite or gold-conditioned outcomes. Freeze that
same layer for every comparator. Native DocPrune retains its exact token set;
whole-region arms use the maximum attainable common cost `M′_q <= M_q`.
Report `M_q`, `M′_q`, and their gap. Every arm uses identical cached pages,
BTP/QTP output, post-QTP visual population, physical-deletion implementation,
model, original positions, decoding, and evaluator. The arms are:

1. **Native DocPrune.** Use its normal dynamic layer selection, aggregate
   native threshold, and exact retained-token set. Freeze this selection before
   ContextCite fitting.
2. **Gold-answer-conditioned attention-region.** Teacher-force every accepted
   answer, aggregate answer-token-to-visual attention at `l*_q`, average the
   per-reference region scores, and solve the same knapsack. Sum over member
   token attention is primary; mean-over-member-token aggregation is a
   region-size sensitivity analysis.
3. **Accepted-answer ContextCite-region.** The reference surrogate is the
   canonical fit on all 256 unique masks. The primary robust selector uses the
   elementwise median signed raw-space coefficient from five deterministic
   80%-without-replacement refits of those 256 rows, followed by the same exact
   knapsack. The canonical 256-fit selector is a required sensitivity.
4. **Reverse ContextCite-region.** Use the minimum robust-coefficient solution at the
   identical achieved budget as a manipulation check, not a competing method.
5. **Unpruned.** Preserve all post-QTP tokens as the quality ceiling/reference.
6. **Audited gold/distractor constraint.** On traceable distractor questions
   only, force audited gold regions in, force the mapped wrong-answer regions
   out, and fill the remaining budget by a frozen source-ID rule. This is a
   diagnostic ceiling for whether physical pruning can rescue the case, not a
   learned or deployable method.

FastV is excluded. Uniform random and coverage-matched random are not rerun in
Task 9; the already established random/coverage work remains part of the
broader experiment and may be cited only within its authenticated boundary,
budget, action-space, and cohort limits. The preliminary pilot retains its one
prespecified region-size-aware random comparator at `M′_q`.

### Budget-local validation

The global Bernoulli-0.5 holdout validates interventions near half of the
regions, while deployment uses native DocPrune's question-specific budget. Add
32 deterministic budget-local held-out masks per pilot question whose physical
retained-token counts lie within `M_q ± 5%` of the post-QTP visual population.
These masks and the 32 global
holdouts are validation-only and never enter the canonical 256-mask fit. They
must be unique, outcome-blind, sealed before scoring, and physically delete the
same whole regions at `l*_q`.

Report global and budget-local LDS/Spearman and RMSE against the fit-target-mean
constant separately. Undefined local LDS, a local LDS below `0.5`, or local
RMSE that does not beat the constant is a fidelity warning that must be
reported; it does not retroactively invalidate the already-authorized
developmental pilot or restore the superseded identity-Jaccard gate.

### Functional selection stability

Full-set Jaccard remains a descriptive coefficient/identity diagnostic. It is
not an admission gate because the deployed budget reaches into a weak tail in
which refits can exchange regions without changing answer quality. At every
budget, directly evaluate the robust-median mask, canonical 256-fit mask, and
the five 80%-without-replacement refit-selected masks. For each set, record
accepted-answer normalized log-likelihood, greedy generated-answer F1, EM,
retained tokens, and selected-region identity.

For each refit `b`, report budgeted selection regret

`R_b = M(S_robust) - M(S_b)`.

Use generated-answer token F1 as `M` for the primary functional result and
accepted-answer log-likelihood as a secondary mechanism result. Also report
canonical-versus-robust regret. Identity instability with negligible
downstream regret is benign; material downstream regret is a limitation of the
oracle. Classical with-replacement bootstrap summaries may remain uncertainty
diagnostics, but they do not choose the deployed set.

Before sealing the enriched panel, use all 48 completed preliminary questions
for a mask-count ablation that adds no perturbation scoring. For
`N in {64, 96, 128, 192}`, draw five deterministic independently sampled
without-replacement subsets of the already-scored 256 fitting masks; the
canonical 256 fit is the reference. Fit the same frozen surrogate, evaluate
each fit on the same 32 global and 32 native-budget-local holdouts, and construct
its whole-region retained set at the already-frozen native budget and layer.
The CPU phase reports global/local LDS and RMSE, coefficient agreement,
selected-region and selected-token agreement, and a sealed inventory of unique
reduced-mask selected sets. It must not infer generated-answer equivalence from
coefficient agreement alone.

GPU follow-up directly evaluates the sealed reduced-mask selected sets against
the existing canonical-256 and native DocPrune responses. Report preservation
of the four canonical-256 exact rescues, baseline-correct preservation,
wrong-stratum mean F1, selected-context accepted-answer likelihood, and
per-question win/tie/loss. Prefer 128 or 192 only if repeated subsets preserve
nearly the same selected-arm behavior as 256; otherwise retain 256. Sixty-four
masks are not adopted merely because they are cheaper. Any operational
equivalence threshold must be frozen from the CPU results before reduced-mask
generation outcomes are read.

The CPU screen advances 192 masks first because it is the closest reduced
count to the 256 reference. Reuse any exact canonical selection and generate
only unique noncanonical 192 sets. Repeat 0 is the deterministic primary
subset; the other four repeats measure subset sensitivity. Before reading
those GPU outcomes, freeze this adoption rule: repeat 0 must retain all four
canonical exact rescues, preserve all 24 baseline-correct exact answers,
reduce wrong-stratum mean token-F1 by no more than `0.02`, and reduce mean
selected-context gold log-likelihood by no more than `0.05` nats/token versus
canonical 256. Across all five repeats, at least 18/20 rescue opportunities
and 118/120 baseline-correct outcomes must be preserved, while averaged wrong-
stratum F1 and gold likelihood must satisfy the same `0.02` and `0.05`
tolerances. If 192 fails, retain 256 and stop the reduction study. If 192
passes, evaluate 128 under the identical rule; adopt the smallest passing
count. Intervals and win/tie/loss remain descriptive and cannot relax the rule.

### Estimands and outcomes

Primary outcome is paired generated-answer token F1. Secondary outcomes are
accepted-answer normalized log-likelihood, EM, answer length/invalid-answer
rate, retained tokens, decoder time, and peak memory. At each question's native
budget, report the following separately within every stratum. The selector
headroom estimand is

`H_total = F1(ContextCite-region) - F1(native DocPrune)`.

Required interpretive decompositions are

`H_privilege = F1(gold-attention-region) - F1(native DocPrune)`

and

`H_intervention = F1(ContextCite-region) - F1(gold-attention-region)`.

For traceable distractor errors, also report:

- rescue-rate difference between ContextCite and native DocPrune among
  baseline-wrong cases;
- the change from unpruned in the normalized gold-versus-wrong-answer
  teacher-forced log-likelihood margin;
- whether each selector retained the audited gold region and removed the
  mapped wrong-answer region; and
- whether the audited gold/distractor-constrained control can rescue the case.

For baseline-correct questions, report harm rate: the fraction changed from
correct unpruned to incorrect pruned. Report rescue, harm, likelihood margin,
and the four gold/distractor retention states at the native budget. A result
must not be called distractor removal merely because a signed coefficient is
negative; it requires the mapped region and downstream intervention evidence.

Use paired support-document-component inference and the existing `±1.0 F1`
practical margin. Cluster by document/support component when questions share
pages. Because this is an enriched 48-question developmental pilot, intervals
are descriptive and cannot authorize a population claim, confirmatory claim,
or method-holdout run without a later approved handoff.

### Superseded admission gate and current progression rule

The original gate required all of the following before attribution QA outcomes
could be examined:

- deterministic token mapping covers 100% of tokens;
- held-out LDS and held-out Spearman point estimates are each at least `0.5`;
- each metric's 95% bootstrap lower bound exceeds `0.2`;
- held-out error beats the constant predictor;
- selected-region Jaccard is at least `0.8` across five bootstrap/surrogate
  refits; and
- direct physical-pruning evaluation of deployed top/reverse whole-region masks
  gives the top mask a higher likelihood target on at least 80% of tested
  development questions and a positive mean paired difference whose 95%
  interval excludes zero.

The one-question accepted-answer 256-mask result passed global predictive
fidelity but failed the `0.8` selected-region Jaccard threshold. On 2026-08-31
the user explicitly approved bypassing that identity gate for the controlled
developmental oracle pilot. The LDS and RMSE checks remain fidelity evidence;
the cross-question interval is computed only on the sealed panel; direct
budget-local fidelity and deployed-mask outcomes replace exact-set identity as
the scientifically relevant stability assessment. Top-versus-reverse remains
a manipulation check, not a prerequisite for looking at the other controlled
pilot arms.

A pilot advantage supports only that a privileged accepted-answer-conditioned
regional procedure found a better mask in this local setup. It does not
authorize a deployable selector or method-holdout evaluation.

### 2026-08-31 diagnostic amendment

After the original 64-fit/32-held-out one-question gate failed, the user
approved two outcome-blind diagnostics on that same development question:
(1) the existing `B_13` intervention with 256 fit masks and 64 unseen masks,
and (2) the same 256+64 schedule with those regions ablated at model input.
Only the exact generated-response ContextCite target decided those boundary
diagnostics. Both are now complete. Moving the intervention to `B_input` did
not improve predictive or selection stability, so `B_13` remains the pilot
boundary. The later accepted-answer 256-mask analysis is the evidence used for
the oracle-pilot amendment. Neither completed diagnostic by itself authorizes
a method-holdout evaluation.

### 2026-09-01 native-comparator amendment

The user superseded fixed `B_13`/fixed-percentage pilot comparisons with the
native DocPrune comparison defined above. The B13/input diagnostics remain
historical development evidence. They no longer set the primary pilot layer or
budget. The new choice is scientifically independent of ContextCite because
native DocPrune selects `l*_q` and `M_q` first; those values are then frozen for
the regional arms.

## Conditional Wang standalone-contribution study

Wang's released-code score is not interchangeable with regional attribution
and is not an oracle. If regional attribution fails, the core experiment
continues. Task 10 remains separately approval gated after Tasks 1 and 3 and
requires a cost and transfer audit before activation.

The faithful released-code diagnostic partitions the relevant visual-token
population into 2x2 windows. For each window `w`, it physically deletes every
other visual state at `B_K` and ranks:

`S_K(w) = P(y1 | only the 2x2 window w is retained at B_K)`.

The score is broadcast to the window's four members. The pinned `cal_info` and
`info_prune` paths also generate/save a no-visual result by physically deleting
all visual states, but they do not subtract or otherwise use that baseline in
the released ordering. Inspect and port those paths faithfully, preserve
physical deletion, and explicitly label every Qwen/dynamic-token/M-RoPE
adaptation, including treatment of incomplete windows.

The paper-described zero-mask quantity may be implemented only as a separately
labeled local transfer/estimand diagnostic:

`I_K^zero(v) = P(y1 | only v is nonzero at B_K) - P(y1 | no visual state is nonzero at B_K)`.

This local zero-mask intervention is not Wang released-code implementation
fidelity. If used, compare it with the faithful window-deletion ranking and
with local physical top-M deletion, preserve any divergence, and do not merge
their results. Include normalized full-answer sequence validation on the
bounded sample for either diagnostic.

Before approval:

1. count exact windows, tokens, and continuations;
2. microbenchmark a small continuation batch;
3. estimate GPU-hours and storage;
4. restrict the first study to one boundary and at most eight questions or
   40,000 window-specific continuations, whichever is smaller; and
5. keep any local zero-mask transfer/estimand diagnostic separate from the
   faithful released-code arm.

## Intervention validation

Before any quality run, test:

- all-kept/no-op replay against ordinary generation at every forced boundary;
- physical deletion versus zero masking on a prespecified smoke sample;
- full cache lengths through `B_K` and compact later caches;
- unchanged retained-token M-RoPE values, order, dtype, and device;
- `M=0`, `M=|V|`, empty regions, overlap, and residual mapping;
- native failure to cross the comprehension threshold;
- deterministic masks and seeds; and
- fixed-page/global-index fail-closed provenance.

Divergence between zero masking and deletion is reported rather than hidden. It
invalidates transfer claims, not the separately defined physical-deletion core.

## Staged execution and compute

1. CPU source audits, hashes, mapping, mask generation, power code, and
   synthetic tests on the active lightwork allocation.
2. Short fixed-page GPU smoke jobs for cache, mask, and numerical-drift checks.
3. Developmental random/coverage calibration and attribution feasibility.
4. Seal the method holdout and final random repetition count.
5. Run confirmatory native and ranking-only arms using short independent shards
   on one canonical GPU family.
6. Run the dependence curve with multiplicity-controlled analysis.
7. Run regional attribution only after its feasibility gate; run Wang only
   after separate approval.

Use broad compatible GPU types for smoke/drift coverage. Never combine absolute
timing across GPU families. No 2,441-question expansion or new dataset is
authorized by this plan.

## Required measurements

- paired EM/F1, intervals, wins/ties/losses, and equivalence classification;
- question and random-mask uncertainty;
- original, post-BTP, post-QTP, and retained token counts;
- native `l*`, forced boundary, policy, seed, and exact retained indices;
- mask Jaccard and overlap;
- per-page and per-4×4-cell occupancy, dispersion, and coverage;
- support-document recall and evidence-opportunity strata;
- attribution LDS, errors, coefficient/selection stability, region costs, and
  direct deployed-mask validation;
- zero-mask/deletion comparisons;
- exact fixed-page/feature provenance; and
- runtime commit, environment, GPU, job ID, artifact paths, and hashes.

## Claim language and decision rules

- “Equivalent” requires the powered equivalence procedure to pass. An interval
  containing zero alone means unresolved.
- “Aggregate-score superior to random” refers only to the local reconstruction,
  tested boundary, token population, budget, model, and cohort.
- Coverage causation requires controlled coverage arms; correlations support
  association only.
- A regional-attribution advantage is privileged and answer conditioned. It
  does not show a deployable selector can recover the mask.
- The all-drop result is an explicit-visual-state dependence result, not proof
  that the model no longer contains or uses visual information.
- Wang released-code 2x2-window standalone contribution misses interactions
  and redundancy and is not an optimal-subset bound. The separate local
  zero-mask diagnostic measures a different estimand.
- No result establishes official DocPrune performance, end-to-end retrieval
  quality, BTP/QTP optimality, grounding, multi-hop generalization, another
  model, or another dataset.

## Exclusions

This plan does not add VisPruner, FEATHER, FastV, SparseVLM, pooling, a new
localization benchmark, new retrieval, alternate BTP/QTP semantics, training,
or a new dataset as experiment arms. The papers inform controls and claim
boundaries only.

## Change control

### Approved Wang semantics amendment — 2026-08-27

Authority: on 2026-08-27 the user approved this exact course change before any
Task 2/3 outcome existed:

> Wang's faithful released-code diagnostic uses physical deletion of 2x2
> visual-token windows. Zero masking remains a separate local diagnostic and
> must not be described as Wang's implementation.

Rationale: the completed Task 1 source audit established that the Wang paper's
zero-mask description and the pinned released `cal_info`/`info_prune` behavior
are not the same intervention. This amendment makes the conditional faithful
arm follow the pinned released code while retaining zero masking only as a
separately labeled local transfer/estimand diagnostic. No affected Task 2 or
Task 3 outcome existed or was inspected before approval. Numerical thresholds,
cohorts, unrelated methods, source pins, and all other claim boundaries remain
unchanged.

Scientific changes require an approved dated amendment here before affected
outcomes are inspected. Implementation status belongs in
[`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md); source acquisitions, jobs,
failures, results, and decisions belong in
[`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md). Chat history is not experimental
authority.

### Approved Task 9 baseline-wrong confirmation amendment — 2026-09-02

After the completed stratified random 48-question development pilot showed four
baseline-wrong rescues and no baseline-correct harm, and after the frozen
192-mask functional comparison failed, the user approved a new confirmation
cohort of 100 ordinary baseline-wrong questions using 256 fitting masks.

The cohort is sampled randomly from the authenticated Task 6 holdout failures,
excludes all preliminary-48 QIDs, and prefers one question per support-document
connected component. If 100 independent components are unavailable, one per
component is taken first, the remainder is filled randomly, and inference is
clustered by support component. This cohort is not distractor enriched.

Each question reuses its exact cached ordered top-4 pages. Arms are unpruned,
native dynamic DocPrune, gold-support regional ContextCite, gold-margin regional
ContextCite, and the existing region-size-aware random control, all compared at
the native dynamic layer and budget through whole-region physical deletion.
The primary endpoint is the generated downstream answer. The prior additional
32 global plus 32 budget-local masks are removed: each new question uses 256
fit masks and zero surrogate holdouts. The method, Lasso, mask distribution,
selection, prompt, decoding, model revision, and comparisons are frozen before
GPU outcomes. Fit-only stability remains descriptive and cannot be reported as
held-out LDS/RMSE.

This is a new-question confirmation of an answer-conditioned regional
ContextCite diagnostic. It is not vanilla ContextCite, a token oracle, a
deployable selector, or a prevalence estimate outside ordinary baseline
failures. H200 CoRAL execution policy is authoritative for this phase; SOL
documents are historical implementation context only. Operationally, the
source/SOL computer constructs and validates the complete fixed-input, MinerU,
geometry, and mapping bundle before transfer. H200 authenticates that bundle
and performs only CPU input validation, smoke, production, retry, and
aggregation; it must not reconstruct experiment inputs or mappings.

### Approved Task 9 shared-probe feasibility amendment — 2026-09-03

The user approved an appended Task 9 experiment that asks whether frozen
pre-answer document-VLM features can predict gold- and self-conditioned
post-boundary physical-deletion responses across unseen documents. The complete
scientific motivation and recommended design are preserved verbatim in
[`TASK9_SHARED_PROBE_RESEARCH_REVIEW_2026-09-03.md`](TASK9_SHARED_PROBE_RESEARCH_REVIEW_2026-09-03.md).

This experiment is separate from both existing Task 9 cohorts. The 48-question
pilot remains development data for the target-structure/additivity gate. The
locked 100-question baseline-wrong confirmation remains a confirmation cohort
and may not become probe-training or model-selection data.

The new target is 600 document-disjoint questions from the authenticated
existing corpus, using only their exact cached top-4 pages and persisted
features. No fresh retrieval or global-index access is authorized. The cohort
contains balanced correct/wrong train, validation, and primary-test partitions
plus a separate natural-prevalence secondary test, all frozen before new
intervention outcomes are inspected. Each training question uses the
prespecified 32-mask mixture of singleton, native-budget-local,
less-aggressive, and pair/small-set interventions. Gold and self supervision
come from the same masked computation and training targets the centered mask
outcomes directly rather than per-question LASSO coefficients.

The initial representation experiment is amended by
[`TASK9_SHARED_PROBE_LAYER_TRAJECTORY_AMENDMENT_2026-09-03.md`](TASK9_SHARED_PROBE_LAYER_TRAJECTORY_AMENDMENT_2026-09-03.md).
It trains matched independent probes at every zero-based read block `0..13`
for metadata/DocPrune, pooled-region-state, pre-answer QK, and combined feature
families. Validation selects the best single-layer snapshot before a small
controlled local-trajectory linear model is compared with it. Full-prefix
low-rank layer aggregation is conditional on the local trajectory passing its
frozen validation gate. Rank-4 question-region bilinear probing remains a
conditional within-layer expressivity diagnostic rather than the automatic
first escalation.

The intervention target is fixed across the entire new cohort: semantic-region
positions are physically deleted at the already validated decoder boundary
`B13`, preserving the established positional-identity semantics. Dynamic
per-question DocPrune layers are not teacher-label boundaries for this study.
The QK feature read layer is a separate variable and may be swept across
prespecified pre-answer layers while every label continues to measure deletion
at B13. This cleanly tests where the fixed B13 deletion response becomes
decodable without changing the response being predicted.

The read-layer sweep and any later deletion-boundary sweep are distinct
experiments. The initial study varies only the read block while holding the
teacher deletion target at B13. A later multi-boundary causal diagnostic is not
authorized without its own outcome-blind cohort, exact boundaries, compute
envelope, and execution handoff. Multi-layer predictive success alone cannot
be interpreted as evidence that distractors progressively balance correct and
incorrect beliefs; that mechanism requires explicit layerwise gold/self
divergence and later causal-boundary evidence.

SOL owns cohort sealing, implementation, CPU Gate 0 analysis, focused tests,
and authenticated packaging. CoRAL H200 owns every new model/GPU job. To
parallelize safely, SOL first pushes the sealed cohort and minimum admitted
teacher-data scaffolding; H200 begins resumable data generation while SOL
finishes the probe implementation, then pulls a second finalized commit and
handoff before training or evaluation. No new MinerU smoke is required unless
the pinned MinerU implementation, model, command semantics, or output contract
changes materially.
