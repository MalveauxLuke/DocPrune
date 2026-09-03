# H200 Task 9 shared-probe agent instructions

## Authority

Read, in order:

1. [`../../AGENTS.md`](../../AGENTS.md)
2. [`../../agent-context/CURRENT_TASK.md`](../../agent-context/CURRENT_TASK.md)
3. [`../../docs/experiments/docprune_random_oracle_horizon_2026-08-26/TASK9_SHARED_PROBE_RESEARCH_REVIEW_2026-09-03.md`](../../docs/experiments/docprune_random_oracle_horizon_2026-08-26/TASK9_SHARED_PROBE_RESEARCH_REVIEW_2026-09-03.md)
4. [`../../agent-context/TASK9_SHARED_PROBE_SOL_HANDOFF_2026-09-03.md`](../../agent-context/TASK9_SHARED_PROBE_SOL_HANDOFF_2026-09-03.md)
5. [`HANDOFF.md`](HANDOFF.md)
6. [`../task9-baseline-wrong-100/CORAL_POLICY.md`](../task9-baseline-wrong-100/CORAL_POLICY.md)

The CoRAL H200 policy is runtime ground truth. SOL scheduler instructions are
context only. This directory governs the new shared-probe track, not the locked
100-question confirmation.

## Scope boundary

Do not select or modify the cohort, retrieve pages, invent missing commands, or
begin from this Git-only handoff. Start only after receiving the sealed cohort,
checksum-verified external bundle, and the corresponding clean SOL commit.
During the two-push workflow, run only the phase explicitly admitted by the
latest pushed handoff.

All new model/GPU teacher-data and probe jobs run on H200. CPU validation and
aggregation may also run here when prescribed. Keep artifacts separate from
the 100-question confirmation and follow the established `/mnt/data1` project
and `/mnt/data2` environment/cache policy.
