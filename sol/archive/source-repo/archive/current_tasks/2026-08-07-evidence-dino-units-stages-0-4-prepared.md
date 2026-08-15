# Archived SOL Task: Evidence-DINO-Units Stages 0–4

**Archived:** 2026-08-07

**Reason:** The owner changed `sol/CURRENT_SOL_TASK.md` to surface the prepared
BoundingDocs document-QA acquisition handoff. This preserves the prior SOL
state; it does not cancel or delete the Evidence-DINO-Units contract, branch,
or workspace design.

## State at archive time

- Prepared locally; not started on SOL.
- Active project: Evidence-DINO-Units dataset construction.
- Execution scope: Stages 0–4 only.
- Source/control checkout: `~/COLPALI_binary_classification`.
- Separate experiment workspace: `~/Evidence-DINO-Units`.
- Experiment branch:
  `codex/evidence-dino-units-dataset-stages-0-4`.
- Scratch root: `/scratch/$USER/evidence_dino_units`.
- Binding contract: `../../task_spec/evidence_dino_units_stages_0_4.md`.
- Full-plan companion:
  `../../task_spec/evidence_dino_units_training_plan_final.md`.
- No job ID or run ID existed.

## Previous next action

1. From `~/COLPALI_binary_classification`, fetch the pushed `main` and
   experiment branch.
2. Read root `AGENTS.md`, `docs/SOL_INSTRUCTIONS.md`, `sol/AGENTS.md`, and the
   binding contract.
3. Follow `sol/task_spec/evidence_dino_units_workspace_bootstrap.md` to create
   the sibling worktree `~/Evidence-DINO-Units`.
4. Verify the exact branch, remote commit, clean worktree, and scratch
   preflight before beginning Stage 0.
5. Execute the Stage 0 contract and gate before any environment, download, or
   dataset implementation.

## Previous hard boundary

Stop after Stage 4 and push only the experiment branch.

Do not execute Stage 5. Do not download model weights, run DeepSeek-OCR-2,
acquire MMLongBench, construct candidates, train models, or serve a website.

## Blockers at archive time

- The shared handoff and experiment branch had to be pushed and verified
  before SOL bootstrap.
- SOL had to confirm network access, immutable Hub revisions, and at least
  350 GiB of free scratch before Stage 3.
