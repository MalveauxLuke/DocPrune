# Evidence-DINO-Units SOL Workspace

This directory routes the current Stage 0–4 dataset-construction task. The SOL
agent starts in `~/COLPALI_binary_classification`, then creates and works from
the separate sibling Git worktree `~/Evidence-DINO-Units`.

## Read first

1. root `AGENTS.md`
2. `../docs/SOL_INSTRUCTIONS.md`
3. `AGENTS.md`
4. `CURRENT_SOL_TASK.md`
5. `task_spec/evidence_dino_units_stages_0_4.md`
6. `task_spec/evidence_dino_units_training_plan_final.md` through Stage 4

## Active handoff

- `CURRENT_SOL_TASK.md`: concise current state and first action.
- `task_spec/evidence_dino_units_stages_0_4.md`: binding detailed execution
  contract.
- `task_spec/evidence_dino_units_training_plan_final.md`: byte-identical
  companion to the canonical full plan.
- `task_spec/evidence_dino_units_workspace_bootstrap.md`: exact sibling
  worktree creation and verification.
- `task_spec/evidence_dino_units_workspace_paths.txt`: home/scratch/Git policy
  by path.
- `task_spec/evidence_dino_units_sparse_checkout.txt`: focused SOL worktree
  surface.

No active SBATCH wrapper exists yet. The SOL agent creates focused Stage 1–4
wrappers only after the preceding contract/gate permits them.

## Archive

The unlaunched MMLongBench gold-page task, its contract, and its OCR wrappers
are preserved under:

```text
archive/current_tasks/
archive/task_specs/
archive/jobs/mmlongbench_prepared_unlaunched/
```

They are procedural history, not active instructions. Other completed and
superseded SOL jobs remain grouped under their existing archive domains.
