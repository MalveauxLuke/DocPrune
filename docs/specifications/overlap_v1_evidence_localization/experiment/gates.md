# Active POC Gates and Stop Conditions

## Authority gate

No SOL work starts until:

- the owner reviews the written specification;
- the unified POC implementation plan is approved;
- `docs/EXPERIMENT_WORKSPACES.md` records a real checkout, branch, scratch
  root, and recovery authority;
- local and remote commit SHAs match and the tracked worktree is clean;
- `sol/CURRENT_SOL_TASK.md` activates exactly one stage; and
- that task pins exact inputs/configs, commands, outputs, and recovery.

POC authorization does not bypass stage activation. HierDoc/A1 architecture
files do not authorize later answer-sufficiency work.

## Stage 00 candidate gate

Pass requires V1 hash verification, deterministic repeated builds, one frozen
`candidate_revision`, exact source-to-eligible/excluded accounting, no V1
writes, at least one accepted covering candidate for every eligible question,
oracle/exclusion reports by required slice, crop/member-union/reading-order
audit, and no gold/query-relative fields in reusable candidates.

If coverage is poor, stop and refreeze a corrected candidate definition under
a new revision. Historical 95% overall and 90% strict multi-box suggestions are
guidance, not fabricated measurements; actual adequacy receives control-plane
review before Stage 01.

## Stage 01 Qwen pairwise gate

Pass requires:

- immutable Qwen model/environment/prompt/processor lock;
- official score parity within `atol=5e-3, rtol=5e-3` in BF16;
- complete finite R0 train/validation scores;
- train-only negative mining with partial/alternative/answer-string/context
  guards;
- completed deterministic audit of at least 200 hard candidates;
- reported false-negative/uncertain rates;
- no validation/test/holdout optimizer rows;
- exact R1 initialization/input/score parity with R0;
- finite resumable training and validation-only checkpoint selection; and
- matched validation R0/R1 report with document-clustered uncertainty.

R1 need not beat R0 to continue; both remain controls. Inability to certify
verified negatives stops R1 and M0/M1 training. Internal test remains unopened.

## Stage 02 MiniVGent systems gate

M0/M1 data training cannot begin until all thirteen architecture checks pass:

1. official score-path parity;
2. all-hidden-state versus selective-tap parity for layers 14 and 28;
3. exact `image_grid_thw` token count;
4. row-major grid reconstruction;
5. rendered candidate-to-grid overlay audit;
6. full-page, border, thin, table, and disconnected-member ROI tests;
7. M0 off-diagonal isolation;
8. M1 candidate permutation equivariance;
9. zero backbone gradients and optimizer parameters;
10. finite nonzero gradients in every added module;
11. exact trainable parameter count;
12. added-weight-only checkpoint round trip; and
13. deterministic 16-question real overfit.

The real overfit requires at least 90% training-loss reduction, Recall@1 at
least 0.95 on those rows, finite added-module gradients, zero Qwen gradients,
and successful restore. Parity, orientation, pooled-gap, gradient, parameter,
checkpoint, or OOM failure preserves evidence and stops M0/M1.

No candidate cap is chosen before p50/p90/p95/max and resource profiling. A
necessary cap creates a new view hash and post-cap oracle report.

## Stage 03 screen gate

Use one registered seed and a deterministic source/document-balanced sample of
up to 10,000 eligible training questions. M0/M1 must start from byte-identical
added-module initialization and use identical rows, order, loss, optimizer,
schedule, and validation view.

Promote four-block M1 only if two-block M1 is stable, improves validation
Recall@1 or MRR over M0 without material regression in the other, and fits the
approved allocation. Otherwise confirm the two-block model. Before Stage 04,
freeze all architecture/config/model/input hashes, exact three seeds,
thresholds, optimizer/schedule, and checkpoint-selection rules.

## Stage 04 confirmatory gate

Internal test opens exactly once only after the confirmatory registration hash
is written. Any post-opening architecture, prompt, processor, loss, threshold,
hyperparameter, depth, or checkpoint-selection change invalidates the
confirmatory claim and must be a newly declared experiment.

M1 promotion requires:

- M1-minus-M0 Recall@1 at least +2.0 absolute points;
- positive direction in all three seeds;
- document-clustered 95% interval excluding zero;
- gain on answer-string-absent rows;
- M1 no more than 2.0 absolute Recall@1 points behind R1; and
- every representation/integrity check still passing.

Failure is a valid negative/inconclusive result; retain the strongest simpler
system.

## Stage 05 holdout gate

InfographicsVQA opens only after Stage 04 model choice is frozen. Require exact
Stage 04 model/checkpoint/config/prompt/processor/threshold hashes. Run once;
results cannot change any POC choice. A mismatch stops Stage 05.

## Program-wide stops

Stop when:

- any V1/candidate/view/model/environment/predecessor hash differs;
- a create-once output exists with different contents;
- a candidate revision changes after scoring;
- split/holdout leakage occurs;
- non-finite values are not fully accounted for;
- a model input contains answer/gold/audit/source leakage;
- predictions/questions/candidates are missing or duplicated;
- compute occurs on a login node;
- paths/recovery are unresolved;
- an operation falls outside the active stage; or
- an active POC task attempts HierDoc, A1, answer-sufficiency, or multi-hop
  work.

Preserve failure artifacts and stop rather than silently changing the model,
candidate set, prompt, loss, or metric.
