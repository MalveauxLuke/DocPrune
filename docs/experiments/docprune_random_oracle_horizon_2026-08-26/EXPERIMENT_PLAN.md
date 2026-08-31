# DocPrune Random, Coverage, Attribution, and Visual-State Dependence Experiment

Status: approved amended design; Tasks 2–3 authorized, with Task 2 active
Canonical date: 2026-08-26
Revision: Wang semantics amendment approved 2026-08-27; review-driven
amendment approved 2026-08-26
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
5. Exploratorily, can a validated answer-conditioned regional attribution
   procedure select a better whole-region mask at a matched achieved budget?

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

## Experiment family D — Exploratory regional attribution

This is **not vanilla ContextCite**, **not a token-level oracle**, and **not a
deployable selector**. MinerU is independent tooling.

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
values and target identity. First-token probability may be reported only as a
Wang comparability diagnostic.

### Surrogate and deployment

1. Use 64 deterministic fit masks with region keep probability `0.5` and an
   independent 32-mask held-out set.
2. Fit the pinned ContextCite Lasso-style surrogate; log adaptations.
3. Treat each coefficient as the estimated total value of its entire region.
4. Let each region's cost be its post-QTP token count.
5. Find the maximum attainable whole-region cost `M′ <= M`; among subsets with
   that cost, select the maximum total coefficient. Reverse attribution selects
   the minimum total coefficient at the same `M′`.
6. Compare aggregate-score, literal-score, and random masks at the identical
   achieved `M′`. Report `M′`, `M`, and the gap.

### Admission gate

Before attribution QA outcomes are examined, all must pass:

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

These are experiment admission thresholds, not universal ContextCite standards.
Failure is a valid negative feasibility result. A passing advantage supports
only that a privileged answer-conditioned regional procedure found a better
mask in this local setup.

### 2026-08-31 diagnostic amendment

After the original 64-fit/32-held-out one-question gate failed, the user
approved two outcome-blind diagnostics on that same development question:
(1) the existing `B_13` intervention with 256 fit masks and 64 unseen masks,
and (2) the same 256+64 schedule with those regions ablated at model input.
Only the exact generated-response ContextCite target decides these diagnostics.
Run B13 first; input-level requires a mechanical smoke and separate handoff.
Neither diagnostic authorizes another question or method-holdout evaluation.

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
  does not show a query-only selector can recover the mask.
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
