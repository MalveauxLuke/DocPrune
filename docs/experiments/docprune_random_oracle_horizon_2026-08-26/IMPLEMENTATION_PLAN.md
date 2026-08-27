# DocPrune Random/Coverage/Attribution Implementation Plan

> **Execution rule:** Read this file, [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md),
> and [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) before acting. Complete one
> checked step at a time, update the log with evidence, and stop at every
> approval or SOL-handoff gate.

Status: Tasks 1–5 complete; Task 3 seven-boundary live all-kept parity is
admitted by job `62265662`; Task 5 CPU integration is independently accepted;
Task 6 CPU runtime and fixed-page gate accepted; clean SOL smoke handoff next
Baseline runtime: `dd5f000a909a718826541df816a4b65396764e1c`
Canonical revision: approved Wang semantics amendment, 2026-08-27; approved
review amendment, 2026-08-26

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

## Task 9 — Implement exploratory regional attribution

- [ ] Generate 64 deterministic Bernoulli-0.5 fit masks plus 32 independent
  held-out whole-region masks.
- [ ] Cache full `B_K` states and physically delete each selected union mask
  before continuing later blocks.
- [ ] Compute per-reference normalized teacher-forced full-answer
  log-likelihood and save the maximum accepted-reference target.
- [ ] Compute the normalized full-sequence likelihood of the unpruned model
  response as a separate contributive target.
- [ ] Fit the pinned ContextCite Lasso surrogate; log every interface change.
- [ ] Report held-out LDS and Spearman separately with bootstrap intervals,
  error versus constant, and five-refit coefficient/selection stability.
- [ ] Implement whole-region knapsack: maximize attainable cost `M′ <= M`, then
  maximize coefficient sum at `M′`; reverse attribution minimizes coefficient
  sum at the same `M′`.
- [ ] Compare aggregate-score, literal-score, and random at exact achieved `M′`.
  Never broadcast coefficients, split regions, or use token index tie-breaking.
- [ ] Directly validate held-out deployed whole-region physical masks.
- [ ] Apply the full admission gate before examining attribution QA outcomes:
  both correlation points at least `0.5`, both lower bounds above `0.2`, better
  than constant, selection Jaccard at least `0.8`, and deployed top masks above
  reverse masks on at least 80% of development questions with a positive mean
  paired difference whose 95% interval excludes zero.

Acceptance: either the exploratory regional procedure passes every gate or is
logged as rejected. It is never called a token oracle or deployable selector.

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

Tasks 1–5 are complete after final independent review. Task 3 CPU
implementation and live gate are complete: job `62265662` admitted the
seven-boundary all-kept parity gate under
`sol/handoffs/DOCPRUNE_QWEN_FORCED_BOUNDARY_PARITY_2026-08-27.md`. Task 4 CPU
controls and Task 5 corrected-runtime policy integration are independently
accepted. Stop here: Task 6 remains unstarted and requires a new continuation
decision before any retrieval/index access, feature work, holdout sealing,
model run, GPU, launcher, job, or developmental experiment. Geometry-aware
controls additionally require truthful post-QTP geometry transport. The final
method holdout remains intentionally unsealed and deferred to Task 6. Task 10
remains inactive and separately approval gated.
