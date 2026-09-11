# Current SOL task

## Role

SOL is the source-development and CPU-analysis environment for Task 9. New Task 9 GPU preprocessing and experiment runs are assigned to H200 unless a later approved handoff explicitly changes that boundary.

## Current state

- Canonical source: `/home/lmalveau/DocPrune` on consolidated `main`.
- Non-Git local data: `/home/lmalveau/docprune-data`.
- Large historical and current run artifacts: `/scratch/lmalveau/docprune`.
- The old Task 9 CPU-analysis and H200 worktrees were merged into `main` and removed.
- The H200 execution branch is preserved as `codex/task9-h200-baseline-wrong-100` without a second local worktree.
- No SOL GPU job is authorized by this file.

## Active responsibilities

1. Maintain and test the consolidated implementation.
2. Prepare immutable cohorts, manifests, transfer bundles, and checksums without retrieval.
3. Analyze returned H200 artifacts on CPU.
4. Update the canonical Task 9 implementation plan and experiment log.

## Active handoffs

- [600-question shared-probe SOL handoff](../agent-context/TASK9_SHARED_PROBE_SOL_HANDOFF_2026-09-03.md)
- [100-question H200 confirmation authority](../agent-context/TASK9_H200_BASELINE_WRONG100_HANDOFF_2026-09-02.md)
- [H200 baseline-wrong execution](../h200/task9-baseline-wrong-100/HANDOFF.md)
- [H200 shared-probe execution](../h200/task9-shared-probe/HANDOFF.md)

## Rules

- Reuse the sealed ordered top-four pages and persisted features.
- Never run fresh/global retrieval unless the user explicitly changes the experiment.
- Keep generated data and model caches outside Git.
- Use the experiment log for historical jobs and results; do not expand this file into another run ledger.

## Next action

Support the H200 agents with source fixes, CPU-only validation, and packaging as requested by the active handoffs. Stop before any unapproved SOL GPU submission.
