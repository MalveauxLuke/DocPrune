# H200 handoff: Task 9 shared causal-deletion probe

## Start condition

This handoff begins when SOL delivers all three of the following:

1. a sealed 600-question cohort/split manifest and authenticated external
   bundle;
2. a clean pushed commit containing the Phase 1 H200 teacher-data runners and
   validators; and
3. checksums and an explicit admission statement for that phase.

Until then, read and plan only. Do not run retrieval, preprocessing, MinerU,
geometry, model loading, or GPU work for this new track. Continue to treat the
existing 100-question confirmation as separate work with separate artifacts.

## Scientific objective

Test whether compact pre-answer frozen-state features predict the
gold-conditioned and self-conditioned post-boundary physical-deletion response
of semantic document regions on unseen documents, then determine whether the
predicted deletion risk supports corrective or safe pruning without test-time
interventions.

The new cohort is not the 48-question pilot and not the locked 100-question
confirmation. The 48 remains development data for Gate 0; the 100 remains
locked oracle-headroom confirmation data.

## Frozen initial model scope

- Train directly on centered mask outcomes, with separate gold and self heads.
- Every intervention label uses physical deletion at the fixed decoder
  boundary `B13`. The QK read layer may vary only according to the prespecified
  layer sweep; it does not change the B13 deletion target. Reject any runner
  that substitutes native dynamic DocPrune layers for this teacher boundary.
- Model 2, the pre-answer QK-linear probe, is the initial implementation.
- Model 3 rank-4 bilinear is the only prespecified escalation.
- Models 0 and 1 are deferred for this initial implementation. Do not add them
  opportunistically on H200.
- Preserve document-disjoint splits, exact cached top-4 page identities,
  question-equal weighting, the frozen VLM/deletion semantics, and the sealed
  32-mask schedule.

## Phase 1 — receive and generate teacher data

After pulling the admitted SOL commit, reapply
[`SPARSE_CHECKOUT_PATHS.txt`](SPARSE_CHECKOUT_PATHS.txt) and verify every
required path is present. Authenticate the external bundle before use; reject
partial, extra, missing, or mismatched files. Never retrieve or load the global
retrieval index.

Use the existing verified MinerU environment and smoke evidence. Do **not** run
another MinerU smoke merely because this cohort is larger. A new smoke is
required only if the SOL commit changes the pinned MinerU implementation,
model, command semantics, or output contract; stop and ask before doing so.

Run only the exact Phase 1 commands supplied by the updated SOL handoff:

1. resumable missing-page MinerU processing;
2. resumable post-BTP/QTP geometry capture;
3. deterministic region mapping and CPU validation;
4. the sealed 32-mask-per-question intervention schedule producing both gold
   and self outcomes from each masked computation; and
5. any separately sealed held-out singleton, budget-random, and policy-like
   response sets authorized for evaluation.

Maximize safe throughput using the parallelization contract provided by SOL,
but obey physical CoRAL GPU ownership. Inspect GPUs 4–7 immediately before
each launch, use only explicitly approved IDs/count, remain reachable, and do
not infer permission from apparent idleness. Record GPU identity, peak memory,
runtime, failures, retries, and completion-manifest-last publication. Retry
only failed resumable units.

Do not train a probe or inspect aggregate scientific outcomes during Phase 1
unless the pushed handoff explicitly admits that step. Report artifact counts
and integrity only so SOL can finish implementation without outcome leakage.

## Phase 2 — pull finalized implementation and execute

SOL will push a second clean commit and replace/extend this handoff with exact
feature extraction, training, validation, evaluation, and aggregation commands.
Before executing it:

- pull the commit and reapply sparse checkout;
- verify the declared code and artifact hashes;
- run targeted CPU validators;
- confirm the 48-question Gate 0 decision and model/split choices were frozen
  before new-cohort outcomes were inspected; and
- stop if exact commands, expected counts, resume behavior, or output roots are
  absent.

Then run the authorized H200 jobs as resumable units. Keep training/model
selection isolated from document-held-out test results. Evaluate new test
outcomes only after the chosen checkpoint and routing policy are sealed.

## Storage isolation

Use a new ignored root such as
`/mnt/data1/eunwooim/DocPrune/task9-shared-probe-local-data/`; never mix it with
`task9-h200-local-data` from the 100-question confirmation. Keep environments,
caches, models, and temporary files under `/mnt/data2/eunwooim/`. Never write
large data to Git, `/`, ARC-only storage, or cross-lab drives.

## Stop conditions

Stop rather than improvise if the cohort or split hashes differ, a QID overlaps
the 48 or locked 100, documents cross splits, cached page/feature identities are
missing, retrieval would be needed, an execution checkout is dirty, GPU
ownership is unclear, or the latest SOL handoff lacks an exact runnable command.
