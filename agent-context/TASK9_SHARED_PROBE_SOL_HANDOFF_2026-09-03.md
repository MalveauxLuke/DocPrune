# Task 9 shared-probe SOL implementation handoff

## Objective

Prepare and implement the new Task 9 shared-probe feasibility experiment
specified in
[`../docs/experiments/docprune_random_oracle_horizon_2026-08-26/TASK9_SHARED_PROBE_RESEARCH_REVIEW_2026-09-03.md`](../docs/experiments/docprune_random_oracle_horizon_2026-08-26/TASK9_SHARED_PROBE_RESEARCH_REVIEW_2026-09-03.md).
This is a new experiment track. Do not alter, reuse as training data, or block on
the locked 100-question baseline-wrong confirmation.

SOL owns cohort construction, implementation, CPU analysis, targeted tests,
packaging, and Git handoff. All new model/GPU data-generation jobs run on the
CoRAL H200 server after the corresponding code and sealed inputs are delivered.

## Read first

1. [`../AGENTS.md`](../AGENTS.md)
2. [`CURRENT_TASK.md`](CURRENT_TASK.md)
3. [`../docs/experiments/docprune_random_oracle_horizon_2026-08-26/EXPERIMENT_PLAN.md`](../docs/experiments/docprune_random_oracle_horizon_2026-08-26/EXPERIMENT_PLAN.md)
4. [`../docs/experiments/docprune_random_oracle_horizon_2026-08-26/IMPLEMENTATION_PLAN.md`](../docs/experiments/docprune_random_oracle_horizon_2026-08-26/IMPLEMENTATION_PLAN.md)
5. [`../docs/experiments/docprune_random_oracle_horizon_2026-08-26/EXPERIMENT_LOG.md`](../docs/experiments/docprune_random_oracle_horizon_2026-08-26/EXPERIMENT_LOG.md)
6. The shared-probe review linked above
7. [`../docs/experiments/docprune_random_oracle_horizon_2026-08-26/TASK9_SHARED_PROBE_LAYER_TRAJECTORY_AMENDMENT_2026-09-03.md`](../docs/experiments/docprune_random_oracle_horizon_2026-08-26/TASK9_SHARED_PROBE_LAYER_TRAJECTORY_AMENDMENT_2026-09-03.md)
8. [`../h200/task9-shared-probe/HANDOFF.md`](../h200/task9-shared-probe/HANDOFF.md)
9. The existing H200 setup and runtime notes under
   [`../h200/task9-baseline-wrong-100/`](../h200/task9-baseline-wrong-100/)

## Starting rule: synchronize before editing

Fetch and fast-forward `codex/task9-h200-baseline-wrong-100` before inspecting
or modifying code. The H200 agent has already pushed environment and MinerU
work to that branch. A sparse checkout is a local Git setting, not a separate
repository history: inspect the files committed by H200 rather than trying to
copy its working tree. Stop on uncommitted overlap instead of overwriting it.

## Frozen initial scope

- Target exactly 600 new questions if eligibility permits.
- Exclude every QID in the 48-question development pilot and the locked
  100-question confirmation cohort.
- Use one question per support-document component and keep documents disjoint
  across all splits. If 600 independent components are unavailable, stop and
  report the eligible counts; do not silently admit document overlap.
- Use only authenticated existing corpus rows, exact already-cached ordered
  top-4 pages, and persisted features. Never run retrieval or load/rebuild the
  global retrieval index.
- Default split: 360 balanced train questions (180 correct/180 wrong), 120
  balanced validation questions (60/60), 60 balanced primary-test questions
  (30/30), and a separate 60-question natural-prevalence secondary test. The
  natural-prevalence draw is random from the remaining eligible document pool.
  Freeze split membership before any new intervention outcomes are read.
- Preserve template/vendor metadata when available so a template-held-out
  analysis can be defined without reselection.
- Use the review's 32-mask training allocation per question: 8 singleton, 16
  native-budget-local, 4 less-aggressive, and 4 pair/small-set masks. Balance
  inclusion frequency and record deterministic seeds.
- Train directly from centered observed mask outcomes for separate gold and
  self heads. Do not use fitted ContextCite/LASSO coefficients as labels.
- Freeze the physical-deletion teacher boundary at `B13` for every question.
  Keep the feature read layer separate: the implementation may extract and
  compare pre-answer QK features from prespecified layers, but all labels must
  remain B13 deletion responses. Do not use each question's dynamic DocPrune
  layer as the teacher boundary.
- Initial representation work trains matched independent probes at every
  zero-based decoder read block `0..13`. Required feature families are
  metadata plus native DocPrune, pooled region hidden state, compact
  final-prompt-query-to-region QK summaries, and hidden plus QK. All features
  precede every accepted or generated answer token.
- Validation alone selects the best single-layer family. It is then compared
  with a small local-trajectory linear model using the current compact feature,
  its adjacent difference, and the mean over the latest four readable layers.
  Required matched controls distinguish ordered trajectory information from
  capacity or unordered multi-layer aggregation.
- Full-prefix low-rank layer aggregation is conditional on a frozen local-
  trajectory validation gate. Rank-4 question-region bilinear probing remains
  a conditional within-layer diagnostic. Do not begin with an RNN, transformer
  over layers, or raw cross-layer hidden-state differencing.
- Keep the frozen VLM, answer targets, deletion operator, positional identity,
  layer/budget definitions, and question-equal loss semantics explicit.

## Fast two-push execution

### Push 1 — unblock H200 data generation

Do this first and keep it small:

1. Implement or adapt a deterministic 600-question sealer using the existing
   authenticated Task 6/Task 9 records. Write cohort, split, exclusion,
   provenance, and checksum manifests. Fail closed on duplicates, overlap,
   missing cached page identities/features, or retrieval access.
2. Build the transfer inventory for the exact selected top-4 pages and required
   persisted artifacts. Large bytes remain outside Git.
3. Adapt only the minimum existing preprocessing/intervention scaffolding H200
   needs to begin resumable GPU teacher-data generation for the sealed cohort.
   Reuse the verified MinerU and geometry machinery; do not redesign it.
   Phase 1 need not select a read layer or train a probe; its B13 teacher
   outcomes are shared by every later read-layer model.
4. Add every required tracked path to the shared-probe sparse-checkout list.
5. Run targeted CPU validation only, commit, push, and provide the cohort bundle
   plus checksums to H200. Do not wait for the complete probe implementation.

The H200 agent may then begin authenticated MinerU/geometry/mapping and masked
teacher-data jobs while SOL continues Push 2. No new MinerU smoke is required;
the existing verified smoke is sufficient unless code or pinned semantics
change materially.

### Push 2 — complete runnable probe scaffolding

1. Use the existing 48×256 pilot for CPU target-structure/additivity analysis:
   repeated fit/holdout splits, mask-size-only, ridge, LASSO, low-rank pairwise,
   budget-local response prediction, and interaction summaries. Freeze the
   numerical Gate 0 decision before new-cohort outcomes are inspected.
2. Implement the matched independent read-block `0..13` feature/probe sweep for
   metadata/DocPrune, region hidden, compact pre-answer QK, and combined
   features without materializing a full quadratic attention matrix. Train
   direct centered gold/self mask-outcome heads with question-equal weighting,
   document-disjoint loading, checkpointing, and resumable evaluation.
3. Freeze validation-only best-snapshot selection, then implement the small
   local-trajectory model and current/mean/delta/parameter/order/history
   controls. Full-prefix aggregation remains conditional on the frozen local-
   trajectory gate; a later deletion-boundary sweep is not part of Phase 2.
4. Implement the required held-out singleton, random budget-matched, and
   policy-like mask evaluation plus actual routed QA evaluation. Keep the
   learned-policy outcomes sealed until training/model selection is frozen.
5. Add focused unit/fixture tests and CPU dry-run validators. Do not rerun the
   full historical suite.
6. Update the H200 handoff with exact commands, hashes, paths, expected counts,
   resume behavior, admission checks, and GPU parallelization limits. Update
   the sparse-checkout manifest so that a pull contains every tracked runtime
   file.
7. Commit and push the clean implementation, then transfer any remaining
   non-Git artifacts with a checksum manifest.

## Delegation and worktree discipline

Use subagents for genuinely independent work. Prefer `gpt-5.6-luna` at high
reasoning for bounded implementation/tests and xhigh for the mask/statistical
contract or cross-cutting review. Keep the primary agent responsible for
reading authority documents, integration, scientific consistency, and final
verification. Do not spawn agents merely to rescan the repository.

Worktrees are allowed for isolated parallel edits, but record `git worktree
list` before delegation and remove every temporary worktree after its work is
integrated or rejected. The final handoff must show a clean status and no new
orphaned worktrees. Never delete a pre-existing user worktree as cleanup.

## Required SOL deliverables

- Sealed cohort/split JSON and hashes, with eligibility/exclusion audit.
- Transfer manifest for all non-Git inputs; no dataset bytes in Git.
- H200-ready teacher-data runners and validators.
- Independent read-layer feature/probe sweep, controlled local-trajectory
  comparison, direct mask-outcome training/evaluation, and focused tests.
- Gate 0 CPU analysis artifact and frozen decision.
- Updated H200 handoff, sparse-checkout paths, exact commands, and expected
  artifacts.
- Experiment-log entry containing commits, hashes, failures, and decisions.

Do not launch SOL GPU jobs, H200 jobs, or inspect new-cohort outcomes from this
handoff. H200 execution begins under its own handoff and GPU policy.
