# DocPrune Random/Coverage/Attribution Implementation Plan

> **Execution rule:** Read this file, [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md),
> and [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) before acting. Complete one
> checked step at a time, update the log with evidence, and stop at every
> approval or SOL-handoff gate.

Status: Tasks 1–5 and Task 8 are complete. Task 9's four-mask smoke, original
64+32 run, paired 256+64 B13/input diagnostics, and accepted-answer analysis
are complete. The original identity-Jaccard gate is superseded. Task 9 is now
authorized for implementation and sealing of a 48-question answer-conditioned
causal-selection oracle pilot. A preliminary 24-correct/24-wrong stratified-
random pilot ran before the enriched panel and is complete. Unique accepted Task 6–9 Git
work is consolidated on the active Task 9 branch; no method holdout is yet
authorized.
Baseline runtime: `dd5f000a909a718826541df816a4b65396764e1c`
Canonical revision: approved Wang semantics amendment, 2026-08-27; approved
review amendment, 2026-08-26; approved Task 9 oracle-pilot amendment,
2026-08-31

## Goal and implementation shape

Add controlled policies to the existing DocPrune Qwen decoder so that native
CTP, fixed-budget score ranking, global/coverage-controlled random selection,
and all-visual deletion can run at the same explicit boundary without changing
the generation contract. Add regional attribution later as a separate
exploratory path using whole MinerU regions. Keep Wang standalone contribution
conditional on a separate cost/transfer approval.

Expected implementation surfaces are:

- `src/docprune/qwen2vl/decoder.py`: forced boundary, physical deletion,
  zero-mask diagnostic, and no-op replay;
- `src/docprune/ctp_controls.py`: score top-M, global random, page-stratified,
  grid-stratified, and occupancy-matched policies;
- `src/docprune/experiment_design.py`: cohort sealing, seed contracts, power,
  equivalence, nested/document-cluster bootstrap, and multiplicity;
- `src/docprune/attribution.py`: full-sequence targets, subset designs,
  ContextCite surrogate validation, whole-region knapsack, and conditional Wang
  score utilities;
- `src/docprune/segmentation.py`: MinerU region schema and region-to-token masks;
- matching focused tests under `tests/` and `tests/qwen2vl/`;
- versioned launchers under `examples/sbatch/`; and
- immutable runtime artifacts under `/scratch/lmalveau/docprune/`.

File names may change only to follow a stronger existing convention; record the
mapping in the experiment log before implementation.

## Non-negotiable constraints

- Inspect and pin the directly relevant paper-author source before writing a
  local equivalent.
- Never install into `/home/lmalveau/mamba-envs/docprune-sol`; use isolated tool
  environments.
- Use cached ordered pages and persisted page features. Never run retrieval,
  load the global index, or rebuild features without new explicit authority.
- Preserve original Qwen M-RoPE values, nonvisual tokens, token order, and the
  admitted all-kept generation contract.
- Treat the existing 16/64/245 cohorts as development data. Do not call the
  remaining 181 questions confirmatory.
- Keep native-threshold policies distinct from score top-M policies in names,
  artifacts, and analyses.
- Use the lightwork CPU allocation for source inspection, hashing, cohort and
  mask construction, power analysis, synthetic tests, and result analysis.
- GPU model work requires a sealed SOL handoff composed of short independent
  jobs. Never overwrite an artifact root.
- Historical pre-`dd5f000` quality is planning evidence, not a final control.

## Progress key

- `[x]` verified and complete
- `[ ]` not complete
- `[~]` conditional; execute only when its gate activates

## Starting state

- [x] Historical full top-4 all-kept and literal DocPrune benchmark complete.
- [x] Historical 245-question stage diagnostic complete.
- [x] Aggregate-logit CTP selected as the best-fit local reference using
  development data.
- [x] Current BTP and QTP semantics retained after targeted audits.
- [x] Stock-versus-manual all-kept generation admitted on L40S, with A30,
  A100-40GB, and H100 drift coverage.
- [x] Corrected runtime `dd5f000a909a718826541df816a4b65396764e1c`
  admitted for controlled comparisons.
- [x] Review-driven redesign approved: no 245 confirmation, no token-level
  ContextCite oracle, formal equivalence, coverage controls, and renamed
  visual-state dependence curve.
- [x] Wang semantics amendment approved before Task 2/3 outcomes: faithful
  released code means 2x2-window physical deletion; zero masking is a separate
  local diagnostic and not Wang implementation fidelity.

## Task 1 — Acquire, pin, and audit every source/tool

This remains the first implementation task. No local intervention code may be
written until its source-audit acceptance criteria pass.

### 1.1 Immutable external source inventory

- [x] On the lightwork CPU allocation, create a fresh versioned root such as
  `/scratch/lmalveau/docprune/paper-code/random-coverage-attribution-v2/`.
- [x] Clone and detach these repositories:

  | Repository | Required revision | Role |
  |---|---|---|
  | [Information-Horizon](https://github.com/YahongWang1/Information-Horizon) | `75909b2936a13d7214e83514f0bc0cb9cc91139e` | Direct random and conditional standalone-deletion source. The Task 1 audit found no zero-mask implementation in the relevant paths. |
  | [context-cite](https://github.com/MadryLab/context-cite) | `c11f8ace6e68ba0121b2e2f1f5c896da9e4156f4` | Sparse surrogate source. |
  | [MinerU](https://github.com/opendatalab/MinerU) | tag `mineru-3.1.0-released`, commit `d9cd58add047c2364c1198eefcb1ee9cd63a971a` | Independent region tool. |
  | [VisPruner](https://github.com/Theia-4869/VisPruner) | `aefa01adc7c7ce6334e880c88225e90cede760d1` | Read-only diversity reference. |
  | [FEATHER](https://github.com/markendo/FEATHER) | `c2a09b2765967601054c7b1fd513ac3e94ee4fc9` | Read-only coverage reference. |
  | [FastV](https://github.com/pkunlp-icler/FastV) | `d1659729b5bf1be225e99ee15783deeea80f63b1` | Read-only Wen-studied baseline. |

- [x] Record repository URL, detached commit, license, tree hash, clean status,
  and SHA-256 of every directly ported file in the experiment log.
- [x] Record that DocPrune and Wen do not link author implementations; do not
  substitute adjacent repositories as their code.

### 1.2 Exact behavior audit

- [x] Inspect and hash Information-Horizon's random branch, wrapper repetition,
  `cal_info`, and `info_prune` paths. Map sampling, visual population, exact
  count, ordering, layer boundary, physical deletion, cache, positions, and
  target. Record the audited contradiction with the earlier zero-mask reading.
- [x] Confirm uniform `random.sample` without replacement and three wrapper
  repetitions, while identifying every LLaVA-only constant.
- [x] Inspect Wen's window/random appendix and FastV source. Compare its spatial
  window semantics with the plan's normalized 4×4 Qwen-page grid. Record why the
  final local geometry is compatible or propose a pre-outcome amendment.
- [x] Inspect and hash `context_cite/context_citer.py`, `solver.py`, and
  `utils.py`. Record 64 ablations, keep probability `0.5`, Lasso `0.01`,
  intercept behavior, random state, response target, and the absence of an
  upstream fidelity/LDS implementation.
- [x] Identify the pinned MinerU snapshot's output fields for page, region type,
  polygon/box, reading order, and model/config provenance. Record that it
  provides rectangular region geometry from which the local experiment may
  construct masks, not deep-token masks, token scores, or evidence labels.

### 1.3 Isolated tool setup

- [x] Create versioned ContextCite and MinerU environments outside the
  production environment.
- [x] Install from detached source snapshots, not unpinned package releases.
- [x] Pin MinerU2.5-Pro-2604-1.2B revision
  `d3f5e08d073c21466bbabe21c71bb1e9c2e595da` in scratch cache.
- [x] Run the smallest CPU-capable import/unit tests and record environment
  manifests. GPU-only tool checks wait for a short-job handoff.

Acceptance: directly used sources and tool revisions are reproducible; the
source-to-local mapping is logged; and production code/environment is untouched.

## Task 2 — Seal statistical, cohort, and artifact contracts

- [x] Add a manifest listing every developmental QID used by the 16-, 64-,
  245-question, threshold, timing, grouped-head, and other CTP diagnostics.
- [x] Build eligibility from single-hop questions using only QID,
  source/support metadata, cached DocPrune-mode top-4 page identities, and
  persisted-feature availability. Candidate/outcome scores are forbidden
  inputs.
- [x] Implement deterministic holdout order:
  SHA-256(`docprune-random-coverage-v2 || qid`).
- [x] Implement support-document connected components for clustered inference
  and report the component-size distribution.
- [x] Implement the primary aggregate-score-minus-global-random F1 estimand,
  `δ=1.0`, TOST 90% interval, two-sided 95% superiority interval, nested mask
  resampling, 100,000 terminal draws, and Holm-adjusted secondary families.
- [x] Implement developmental power analysis targeting 80% equivalence power.
- [x] Implement random-repetition calibration: 10 masks initially; freeze 20
  instead if developmental Monte Carlo standard error exceeds `0.25 F1`.
- [x] Implement and test the fail-closed atomic seal/launch contract for QIDs,
  order, required `N`, selection hash, pages, features, source metadata, and all
  file hashes. Final holdout membership and sealing remain deferred to Task 6,
  after development random results provide variance and repetition-count
  inputs.
- [x] Define result terminology in code: `equivalent`, `superior`, `inferior`,
  or `unresolved`. Equivalence requires the 90% TOST interval wholly within
  `[-1,+1]`, even when that interval contains zero. A zero-containing 95%
  superiority interval alone is `unresolved`.

Acceptance: synthetic fixtures reproduce known equivalence/superiority cases;
cohort selection cannot read outcomes; and no holdout job can launch without a
sealed power calculation and manifest.

## Task 3 — Freeze decoder intervention contracts as tests

- [x] Define `B_input` and `B_K = after block K, before block K+1` in tests and
  artifacts.
- [x] Add an explicit forced-boundary hook while preserving native threshold
  selection as the existing default.
- [x] Test that blocks through `K` retain full caches and only later blocks see
  compact inputs/caches after physical deletion.
- [x] Test retained nonvisual tokens, original order, M-RoPE position values,
  dtype, and device.
- [x] Add an opt-in real-model all-kept/no-op test for `B_input`, `B_0`, `B_6`,
  `B_13`, `B_20`, `B_23`, and `B_26`.
  - [x] Execute and admit the seven-boundary live all-kept parity gate. Clean
    runtime L40S job `62265662` passed and is admitted under
    `sol/handoffs/DOCPRUNE_QWEN_FORCED_BOUNDARY_PARITY_2026-08-27.md`; no
    further Task 3 submission is authorized.
- [x] Add zero-mask mode for diagnosis and compare it with physical deletion on
  a prespecified synthetic/smoke sample; persist rather than hide divergence.
- [x] Test `M=0`, `M=|V|`, empty visual populations, and native comprehension
  threshold failure-to-cross. Failure-to-cross must leave native CTP unpruned
  and make ranking-only `M=|V|` a no-op.
- [x] Define an artifact schema containing QID, cohort classification,
  page/feature hashes, policy family/name, `B_K`, native `l*`, `|V|`, `M`,
  achieved budget, retained original indices, seed, cache lengths, runtime pins,
  mandatory mode/selection identity, and digest.

CPU correction disposition: **accepted after independent re-review.** The hook,
native/forced identity rules, forced-record evaluation transport, literal
edge/no-op matrix, M-RoPE identity record, and fail-closed visual-index API are
accepted. Native `PruningTrace` semantics remain unchanged. The seven-boundary
opt-in real-model parity execution is now authorized exactly once on the
sealed clean committed runtime with a fresh no-replace JSON artifact path and
the reviewed successor handoff/launcher. Job `62265662` completed and admitted
the seven-boundary gate; Task 3 is complete.

## Task 4 — Implement ranking and coverage policies test-first

- [x] Implement deterministic score top-M with stable score/index ordering for
  aggregate and literal scores. Name these score policies, not native CTP.
- [x] Implement global uniform random without replacement over the combined
  post-BTP+QTP visual population.
- [x] Implement page-stratified random matching aggregate-score top-M's exact
  per-page counts.
- [x] Implement normalized 4×4 grid assignment for every post-QTP token.
- [x] Implement grid-stratified allocation: uniformly sample nonempty cells when
  `M` is smaller than their count; otherwise assign one per nonempty cell, then
  use largest-remainder proportional allocation to residual capacity.
- [x] Implement coverage-matched identity shuffle with exact aggregate-score
  page×cell counts.
- [x] Seed all random policies by `(experiment version, qid, boundary, policy,
  repetition)` and restore original sequence order.
- [x] Test exact cardinality, uniqueness, enumerable-population uniformity,
  seed repeatability, multi-page behavior, per-page/cell quotas, capacity
  exhaustion, and edge budgets.
- [x] Emit page/grid occupancy, center, dispersion, and overlap measurements.

Acceptance: every ranking-only arm differs only in selected visual identities
or its declared allocation constraint; boundary, budget, and shared inputs are
identical.

CPU disposition: **accepted after fresh strict-TDD implementation and
independent review.** The accepted module and tests are
`src/docprune/ctp_controls.py` and `tests/test_ctp_controls.py`. Task 4 adds no
decoder integration or experiment result; Task 5 remains a separate step.

## Task 5 — Separate native-policy and ranking-only factories

- [x] Add corrected-runtime factories for BTP+QTP/no CTP, literal native
  threshold CTP, and aggregate-logit native threshold CTP.
- [x] Add separate factories named `literal-score-top-m` and
  `aggregate-score-top-m`.
- [x] Persist native threshold/count versus forced top-M identity in manifests.
- [x] When the aggregate-native threshold has no boundary-score tie, assert
  that `aggregate-score-top-m` selects exactly the native aggregate mask at the
  same `M`. When it has a tie, persist the full tied set and the deterministic
  tie-breaking difference instead of treating the masks as automatically
  identical.
- [x] Add developmental fixed-retention score/random policies at 55%, 65%, and
  80% of post-QTP tokens with deterministic rounding.
- [x] Add tests preventing a native threshold policy from being mislabeled or
  silently forced to another method's `M`.

Acceptance: native-policy tables test complete reconstructions; ranking-only
tables test score identity at matched budgets; they cannot be merged silently.

CPU disposition: **accepted after strict TDD and independent correction
review.** Native, ranking, random, and coverage identities are fail-closed in
factory, manifest, runtime selection, result JSONL, and resume validation.
Random seed context uses the exact canonical values persisted in the manifest;
geometry-aware controls fail closed until truthful post-QTP geometry is
supplied. Task 5 ran no model, retrieval, GPU, launcher, or experiment job.

## Task 6 — Run only the developmental random/coverage gate

- [x] Seal a fixed-page short smoke cohort from development data and expected
  page/feature hashes.
- [ ] Write a SOL handoff with clean commit, environment, inputs, output roots,
  launcher hashes, short resources, and fail-closed recovery authority.
- [ ] Run mask/cache/numerical-drift smoke cells across broad compatible GPUs;
  use one GPU family for canonical QA.
- [ ] Run development aggregate/literal score, global random, page-stratified,
  grid-stratified, and coverage-matched policies at native `M`.
- [ ] Run 10 random masks per question and compute mask Monte Carlo error. If
  above `0.25 F1`, run the additional masks required to reach 20 before any
  holdout sealing.
- [ ] Run the 55/65/80% sensitivity curve on development data only.
- [ ] Compute developmental variance, equivalence power, required holdout `N`,
  and final random repetition count.
- [ ] Seal the method holdout atomically without reading candidate outcomes.
- [ ] Stop for review. Do not submit confirmatory jobs in this task.

Acceptance: the primary analysis code is exercised on development data, random
uncertainty is controlled, and a power-backed method holdout is sealed.

Current next action (2026-08-27): commit the independently accepted Task 6
runtime in the isolated clean checkout, bind that commit and fixture v2/gate
v4 hashes in an exact SOL handoff, then submit the four one-QID portability
smokes. Do not submit the 64-QID L40S matrix until the four-way comparator
admits input, budget, geometry, cache, M-RoPE, and trace identity.

## Task 7 — Implement the explicit-visual-state removal curve

- [ ] Add all-visual physical deletion at `B_input`, `B_0`, `B_6`, `B_13`,
  `B_20`, `B_23`, and `B_26`; retain all nonvisual states.
- [ ] Add native `B_l*` as a separate diagnostic, not part of the fixed grid.
- [ ] Use BTP+QTP/no CTP as the paired reference.
- [ ] Implement max-statistic document-cluster bootstrap simultaneous one-sided
  95% lower bounds across fixed boundaries.
- [ ] Implement persistent-boundary logic: lower bound greater than `-1.0 F1`
  at the candidate and every later tested boundary; any later failure prevents
  persistence.
- [ ] Report non-monotonic curves and describe only bracketing among sampled
  boundaries.
- [ ] Add evidence-opportunity strata: document recall@4=1, no-CTP EM correct,
  no-CTP F1 positive, and `B_input` visually sensitive using answer change or
  `0.1` nat/token best-reference likelihood loss.

Acceptance: outputs are named an explicit-visual-state removal/dependence curve;
“information horizon” is unavailable without separately reproduced near-zero
standalone token information.

## Task 8 — Set up MinerU whole-region masks

- [ ] Run pinned MinerU on a tiny cached-page smoke set and save page-space
  regions, revision, configuration, and hashes.
- [ ] Define a region schema with document/page, ID/type, polygon/box, reading
  order, and token cost.
- [ ] Map every post-BTP+QTP Qwen visual token footprint to exactly one primary
  region or residual source using deterministic overlap/boundary rules.
- [ ] Partition large residual background into normalized 4×4 page cells before
  attribution.
- [ ] Visualize regions and corresponding token masks for smoke pages.
- [ ] Cache MinerU output and token mapping once; interventions may not rerun
  MinerU.
- [ ] Test 100% coverage, byte-identical replay, overlaps, empty regions, and
  residual partitioning.

Acceptance: regions are whole binary sources over actual deep visual tokens;
they are never substituted for tokens or treated as evidence labels.

## Task 9 — Implement the answer-conditioned regional oracle pilot

Historical preparation (2026-08-28): the exact physical-deletion and
surrogate contracts are accepted through `e9310cd`. L40S smoke job `62315446`
completed `0:0` and was admitted; A100 portability job `62315546` was canceled
while pending at elapsed zero after the L40S gate passed. Clean successor
`7616b29b4dc5ba33584a6e26371281be6886188f` added the original 96-mask runner,
dual-target completion-manifest-last publisher, terminal validator, and frozen
one-question CPU analysis. Exact historical authority remains in
[`DOCPRUNE_TASK9_REGIONAL_DEVELOPMENT_L40S_2026-08-28.md`](../../../sol/handoffs/DOCPRUNE_TASK9_REGIONAL_DEVELOPMENT_L40S_2026-08-28.md).

Original result (2026-08-31): job `62323129` terminally admitted all 96
physical interventions. At `M=2689`, accepted-answer LDS was `0.88783` and
held-out RMSE was `0.24044` versus constant `1.01629`, but minimum five-refit
selection Jaccard was `0.63636`. Generated-response LDS was `0.58798`, held-out
RMSE was `0.98257` versus constant `0.93324`, and minimum selection Jaccard was
`0.71111`. The then-active dual-target reliability gate failed. This historical
decision and its absence of a one-question interval remain unchanged.

Paired diagnostics (2026-08-31): B13 job `62423463` completed `0:0` in
`00:07:23`; corrected input job `62424211` completed `0:0` in `00:04:23`.
Each terminally admitted all 320 masks. Failed input preflight attempt
`62423876` created no output and loaded no model. Generated-response analysis
showed B13 LDS `0.71593` and RMSE `0.97905` versus constant `1.34080`; input
LDS `0.64886` and RMSE `0.88932` versus constant `1.19600`. Input deletion did
not improve stability or fidelity, so `B_13` remains primary.

The accepted-answer reconstruction on the same admitted raw rows is the pilot
progression evidence. At B13 it produced held-out LDS `0.91484`, RMSE `0.19368`
versus constant `0.99428`, and coefficient-refit Spearman mean/min
`0.66975`/`0.57412`; selected-set Jaccard mean/min was `0.70779`/`0.65`. The
user approved bypassing the old `0.8` identity gate because exact support
identity is not the deployment estimand. Direct budgeted-set outcomes and
budget-local fidelity replace it for this developmental pilot.

Completed foundations:

- [x] Generate and admit the original 64 Bernoulli-0.5 fit masks plus 32 global
  holdouts, then generate and admit 256 fit plus 64 global holdouts at both
  `B_13` and `B_input` on the same question.
- [x] Cache full boundary states and physically delete selected whole-region
  unions before later blocks while retaining original M-RoPE positions.
- [x] Compute accepted-answer and exact unpruned generated-response normalized
  full-sequence likelihood targets from the same raw intervention rows.
- [x] Fit the pinned ContextCite Lasso surrogate and report held-out
  LDS/Spearman, RMSE versus the fit-target-mean constant, coefficient stability,
  selection stability, and solver warnings without inventing a one-question
  confidence interval.
- [x] Implement exact whole-region knapsack: first maximize attainable cost
  `M′ <= M`, then maximize coefficient sum at `M′`; reverse attribution
  minimizes coefficient sum at the same `M′`.
- [x] Record the old dual-target/Jaccard rejection without deleting or
  reinterpreting it, and approve the new oracle-pilot claim boundary.

Pilot preparation and implementation:

- [x] Consolidate unique accepted Task 6–9 Git work on the active Task 9
  branch: Task 6 merge `689432e`, Task 7 merge `c5b212f`, and Task 8 merge
  `c38cd9c`.
- [x] Seal the preliminary random cohort from the authenticated 245-question
  BTP+QTP/no-CTP pool: exactly 24 canonical-EM-correct and 24 canonical-EM-
  wrong QIDs, sampled without replacement by a recorded seed. Persist the
  90/155 eligible-pool counts, source paths/hashes, selected QIDs, and natural-
  pool weights without reading attribution or arm outcomes. Canonical artifact:
  `/scratch/lmalveau/docprune/task9-preliminary-random48-v1/cohort.json`; file
  SHA-256 `123607a6a1226b4e3436f43cb82d45e64e8a3008e9ab6deefd7526efeabd0273`;
  internal cohort SHA-256
  `465fcf6e8e0adee6e79845db8cb1d6f1fbc97e01d02b7cf5c1e28c7c33c3c5f9`.
- [x] Implement the corrected preliminary comparison. For each question,
  native aggregate-threshold DocPrune independently selects its crossing layer
  `l*_q`, exact retained token count `M_q`, and native retained token IDs before
  any ContextCite fit or outcome is available. Freeze `l*_q` for that
  question. Evaluate unpruned, native DocPrune, gold-support ContextCite,
  gold-margin ContextCite where a distinct non-gold unpruned response exists,
  and one deterministic region-size-aware random comparator. ContextCite and
  random use the closest attainable whole-region cost at or below `M_q`; every
  pruned arm uses the same physical-deletion implementation and original
  M-RoPE positions. Keep FastV excluded.
- [x] Report preliminary correct/wrong strata separately; paired normalized
  token-F1 and EM differences; win/tie/loss; rescue; preservation; gold-
  likelihood and gold-margin changes; question-clustered uncertainty; and an
  optional 90/245 versus 155/245 reweighted descriptive result. Do not use its
  selector outcomes to choose the subsequent enriched cohort.
- [ ] Implement and test an outcome-blind cohort sealer for 48 unique questions:
  16 uniform-anchor, 16 traceable distractor-error, 8 high-ambiguity/correct,
  and 8 clean-control questions. It may read accepted and already-observed
  unpruned answers and region text, but must reject ContextCite and attention
  outcomes. Randomly sample within frozen pools and seal a balanced 16-question
  mask-count calibration subset.
- [ ] Define and audit deterministic same-type candidate extraction, wrong-
  answer/gold-region mappings, ambiguity score/tier, clean-control confidence
  threshold, primary-stratum precedence, insufficient-pool failure, and a
  frozen manual-annotation schema before cohort sealing.
- [ ] Extend the cached MinerU-to-post-QTP mapping path across all sealed pilot
  pages without retrieval or feature rebuilding. Require 100% assignment to a
  whole MinerU region or bounded residual cell and publish mapping manifests
  before scoring outcomes.
- [x] Generate per question 256 unique Bernoulli-0.5 fit masks, 32 unique global
  holdouts, and 32 unique native-budget-local holdouts centered on
  `M_q / |V_q|` within ±5% of the visual population. Seal vectors and hashes
  before model scoring.
- [ ] Add teacher-forced accepted-answer attention capture at each frozen
  `l*_q`. Produce
  privilege-matched whole-region scores by summing answer-token attention over
  member visual tokens and averaging references; keep mean-over-member-token
  and literal-score variants as named sensitivities.
- [x] Capture and reproduce native aggregate-threshold DocPrune's independently
  chosen layer, exact budget, and retained-token IDs. Do not add FastV,
  uniform-random pruning, or coverage-matched pruning arms. The completed
  1,213-question Task 6 random-versus-DocPrune holdout remains separate evidence
  and is never recomputed for Task 9.
- [ ] Convert ContextCite coefficients to signed raw inclusion-feature space.
  Fit the canonical 256-row surrogate plus five deterministic 80%-without-
  replacement refits; use their elementwise coefficient median for the primary
  robust selector and retain the canonical fit as a required sensitivity.
- [ ] At each question's independently selected native budget `M_q`, use the
  existing exact two-stage whole-region knapsack for every regional arm. Add
  reverse robust ContextCite and the audited gold-in/distractor-out diagnostic
  on the traceable stratum. Never broadcast coefficients or split regions.
- [ ] Directly generate answers for unpruned, native DocPrune, gold-attention,
  robust ContextCite, canonical ContextCite, reverse ContextCite, the five
  refit-selected sets, the matched regional random control where prescribed,
  and the traceable-only audited constraint at `M_q` using identical physical
  deletion and original M-RoPE.
- [ ] Report global and primary-budget-local LDS/Spearman and RMSE/constant,
  treating failures as warnings rather than reinstating the superseded
  identity gate. Report token-weighted overlap, signed distractor behavior,
  direct set regret, generated F1/EM, accepted-answer likelihood, retained
  tokens, runtime, and memory.
- [ ] Report selector headroom and its privilege/intervention decompositions by
  stratum. On traceable errors report rescue, gold-vs-wrong likelihood-margin
  change, and gold/distractor retain/remove states; on baseline-correct cases
  report harm. Cluster descriptive intervals by support-document component.
  Never report the enriched panel as a population estimate.
- [x] Before enriched-panel sealing, run the CPU mask-count ablation on all 48
  completed preliminary questions using five deterministic independent subsets
  at 64, 96, 128, and 192 of the existing 256 fit masks, with canonical 256 as
  reference. Reuse the same 32 global plus 32 native-budget-local holdouts;
  report fidelity, coefficient agreement, region/token-set agreement, and seal
  every unique reduced-mask selection for GPU evaluation.
- [ ] GPU-evaluate 192 masks first, reusing exact canonical selections and
  generating only the unique noncanonical 192 sets. Compare with the existing
  canonical-256 and native DocPrune outcomes. Explicitly report the
  four rescue cases, baseline-correct preservation, wrong-stratum mean F1,
  selected-context gold likelihood, and win/tie/loss. Adopt 128 or 192 only
  after the frozen functional rule passes; otherwise retain 256. Only if 192
  passes, evaluate 128 under the same rule. Do not adopt 64 or 96 based on
  compute cost alone.
- [x] Implement terminal manifest-last admission and a unified analysis JSON
  containing question rows, strata, arm/budget contrasts, model-readable metric
  definitions, artifact hashes, and claim limits.
- [x] Run bounded real-input validation, then launch from one clean committed
  runtime. Use
  first-available suitable CUDA GPUs with recorded GPU identity for four-
  question batch jobs (the final job may contain fewer; never mask-level
  shards), 24 GiB host RAM, no requeue, fresh
  roots, and no partial-outcome inspection. Do not compare absolute timing
  across GPU families. Do not submit the
  48-question pilot until the handoff binds a clean consolidated commit.
  Corrected implementation `2a66d79c105d28ba4ddcd53b9a3015b5db624b67`
  was admitted after validator fix `90f27d7ed8b99ad10f1a5fe405c131127456ae5d`.
  Replacement array `62464099` and targeted retry array `62466113` completed all
  48 unique questions. Canonical unified result:
  `/scratch/lmalveau/docprune/task9-preliminary-dynamic48-90f27d7-unified-v1/analysis.json`.

Acceptance: the pilot is sealed and reproducible, and every selected set is
evaluated through the actual deletion operator. The old identity-Jaccard
failure is reported but does not block. The result remains a privileged
answer-conditioned regional oracle diagnostic, never vanilla ContextCite, a
token oracle, a deployable selector, or a population/holdout claim.

## Task 10 — Conditional Wang standalone-contribution study

The 2026-08-27 approved scientific amendment resolves the Task 1 source-audit
blocker. A faithful released-code arm physically deletes all visual states or
retains one 2x2 window, ranks that window's first-gold-token probability,
broadcasts the score to its four members, and does not use the saved no-visual
baseline in ordering. A local zero-mask intervention is a separate optional
transfer/estimand diagnostic, not Wang implementation fidelity.

- [~] This task requires separate user approval after Tasks 1 and 3. It is not
  automatically activated by regional-attribution failure.
- [~] Count exact windows, tokens, and continuations; microbenchmark a small
  continuation batch; and estimate GPU time/storage before implementation or
  submission.
- [~] Limit the first study to one boundary and the smaller of eight questions
  or 40,000 window-specific continuations.
- [~] Port the pinned `cal_info`/`info_prune` physical-deletion semantics:
  retain one 2x2 visual-token window, score its first-gold-token probability,
  broadcast that score to all four members, and exclude the saved no-visual
  baseline from ordering.
- [~] Add normalized full-answer sequence validation on the bounded sample.
- [~] If the separate local zero-mask transfer/estimand diagnostic is used,
  compare it with faithful window-deletion ranking and local physical top-M
  deletion, preserve divergence, and never label it Wang implementation
  fidelity.
- [~] Label the faithful result “Wang released-code 2x2-window
  standalone-contribution ranking,” not ContextCite, oracle, or optimal
  selection.

Acceptance: after the separate post-Task-3 approval, a reviewed cost envelope,
faithful released-code window-deletion implementation, and any separately
labeled local transfer test exist before a wider Wang study can be proposed.

## Task 11 — Confirmatory short-job gate

- [ ] Write a fresh SOL handoff binding the sealed method holdout, final random
  repetition count, clean runtime, exact cached pages/features, canonical GPU
  family, short shards, artifact schema, and recovery rules.
- [ ] Rerun BTP+QTP/no CTP, literal native, and aggregate native controls under
  the corrected runtime on the identical method-holdout pages.
- [ ] Run aggregate-score top-M, literal-score top-M, and all four random/
  coverage policies at identical per-question `B_l*` and `M`.
- [ ] Fail closed on retrieval/index access, page/feature drift, budget drift,
  duplicate indices, missing rows, cohort leakage, or digest mismatch.
- [ ] Analyze the locked primary contrast and secondary Holm-adjusted coverage
  family without changing margins, seeds, clusters, or cohort membership.
- [ ] Report native-policy results separately from ranking-only results.
- [ ] Run the fixed-boundary dependence curve only after its no-op/deletion
  smoke gate passes.
- [ ] Do not run regional attribution on the method holdout unless its entire
  development admission gate passes and a separate handoff authorizes it.
- [ ] Do not start a 2,441-question expansion or new dataset.

Acceptance: every result is input-, layer-, budget-, runtime-, and cohort-valid;
the primary outcome is classified only as equivalent, superior, inferior, or
unresolved under the locked procedure.

## Task 9 H200 baseline-wrong confirmation — approved 2026-09-02

- [x] Retain 256 fitting masks after the frozen 192-mask comparison failed.
- [x] Seal 100 new baseline-wrong questions from the authenticated 1,213-question
  Task 6 holdout, excluding the preliminary 48 and preferring one question per
  support-document component.
- [x] Freeze exact cached top-4 pages, no retrieval, dynamic DocPrune layer and
  budget, whole-region physical deletion, and the same pilot comparison arms.
- [x] Remove all 64 extra surrogate holdout masks; fit support and margin from
  the same 256 interventions and make generated-answer outcomes primary.
- [x] Prepare path-parameterized cohort, fixed-input, fit-only analysis, batch,
  selected-arm, aggregation, and H200 handoff tooling.
- [x] On the source/SOL computer, seal all 100 retrieval-free fixed inputs and
  package the exact cohort, selected rows, 400 PNGs, 253 unique PDFs, 253 unique
  feature shards, and provenance metadata without the global retrieval index.
  The 1,014 raw files total 2,826,385,204 bytes; `MANIFEST.sha256` has SHA-256
  `d5eb76ff0e0382d487387e31d9a28cde37ae1de8020644a111ec659d2f1f5bb5`.
- [~] Transfer that sealed CPU-input bundle through the Mac bridge to
  `/mnt/data1/eunwooim/DocPrune/task9-h200-local-data/inputs/transferred/` and
  verify every checksum. Do not preprocess a partial transfer.
- [x] Prepare tracked H200 setup for two isolated environments: frozen
  DocPrune and pinned MinerU. A single environment is rejected because their
  validated Python/PyTorch/Transformers versions conflict. Environment setup
  may overlap the transfer after the read-only survey is recorded.
- [ ] On H200, authenticate and relocate the fixed inputs without changing
  source bytes; run pinned MinerU on the 400 sealed pages, capture frozen
  post-BTP/QTP geometry for 100 QIDs, build exactly 100 mappings, and pass
  model-free validation for ordinals 0–99. No retrieval or global index.
- [ ] After separate exact-GPU approval, run one experiment smoke and then
  questions 1–99 sequentially as resumable one-question units on one approved
  CoRAL GPU. Additional GPUs require separate explicit approval.
- [ ] Aggregate exactly 100 admitted results with support-component-clustered
  uncertainty and preserve a unified analysis JSON.

Current next action: while the Mac-to-H200 transfer finishes, pull this branch
and create the two isolated H200 environments after the recorded survey. Then
verify all 1,014 transferred files before relocation or preprocessing.

## Task 12 — Close the stage

- [ ] Append source inventories, cohort seals, power analysis, job IDs, GPU
  identities, artifact hashes, all results, failures, and decisions to
  [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md).
- [ ] Update [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md) only through an approved
  pre-outcome scientific amendment.
- [ ] Update repository navigation and active SOL authority.
- [ ] Preserve paper snapshots, environments, failed roots, and canonical
  outputs outside Git.
- [ ] State the exact local-reconstruction, cohort, model, budget, layer, and
  privileged-attribution claim boundaries.

Acceptance: another agent can reproduce the result and exact next action from
these documents plus the active SOL handoff.

## Current next action

Finish and verify the Mac-to-H200 transfer while the H200 agent prepares the
separate DocPrune and MinerU environments. After verification, H200 relocates
the sealed fixed inputs, runs pinned MinerU and geometry, builds/validates all
100 mappings, and requests approval for one exact smoke GPU. No method-holdout
run is currently authorized. Task 10 remains inactive and separately gated.
