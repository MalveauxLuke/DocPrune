# Regional attribution: implementation status

This file is a high-level progress map. It does not define scientific choices,
exact commands, resource requests, or acceptance thresholds. Those belong in
the [experiment plan](EXPERIMENT_PLAN.md) and scoped machine handoffs.

## Current state

| Task | State | Owner | Remaining work |
| --- | --- | --- | --- |
| Foundational controls and regional implementation | Complete | SOL/source | None unless current code changes |
| 48-question development pilot | Complete | SOL | Preserve results; do not rerun |
| 256-versus-192 mask ablation | Complete | SOL | Keep 256 masks |
| 100-question baseline-wrong confirmation | Active | H200 | Finish authenticated preprocessing, validate, run one experiment smoke, execute remaining questions, aggregate |
| 600-question shared-probe study | Active | SOL + H200 | Deliver Phase 1 bundle/code, generate teacher data, finalize Phase 2 probe code, train/evaluate, analyze |

## Execution order

### Track A — 100-question confirmation

1. Synchronize the preserved H200 execution branch and verify the transferred
   sealed inputs.
2. Complete only missing MinerU, geometry, mapping, and CPU-validation units.
3. Run one experiment smoke on one explicitly approved CoRAL GPU.
4. Run the remaining one-question units, retry failures only, then aggregate
   exactly 100 admitted results.

Exact commands and stop conditions: [H200 confirmation handoff](../../../h200/task9-baseline-wrong-100/HANDOFF.md).

### Track B — 600-question shared probe

1. SOL seals the document-disjoint cohort and pushes the minimum authenticated
   Phase 1 teacher-data implementation and bundle.
2. H200 generates resumable teacher data while SOL finishes Phase 2 feature,
   probe, validation, and aggregation code.
3. SOL pushes one finalized Phase 2 commit and handoff.
4. H200 extracts features, trains/selects on validation, freezes the choice,
   evaluates the held-out tests, and returns artifacts for CPU analysis.

Exact phase boundaries: [SOL handoff](../../../agent-context/TASK9_SHARED_PROBE_SOL_HANDOFF_2026-09-03.md)
and [H200 handoff](../../../h200/task9-shared-probe/HANDOFF.md).

## Immediate next action

Synchronize the preserved H200 execution branch, authenticate the transferred
inputs for both tracks, and continue each track from its scoped handoff. Do not
repeat completed development runs or MinerU smoke tests whose implementation
and contract have not changed.

## Maintenance rule

Keep this file short. Update only task state, ownership, remaining work, and the
immediate next action. Put scientific amendments in `EXPERIMENT_PLAN.md` and
append execution evidence to `experiment-log/`.
