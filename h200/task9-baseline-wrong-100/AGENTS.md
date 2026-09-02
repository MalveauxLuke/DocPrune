# H200 Task 9 agent instructions

## Authority

Read, in order:

1. [`../../AGENTS.md`](../../AGENTS.md)
2. [`../../agent-context/CURRENT_TASK.md`](../../agent-context/CURRENT_TASK.md)
3. [`../../docs/experiments/docprune_random_oracle_horizon_2026-08-26/EXPERIMENT_PLAN.md`](../../docs/experiments/docprune_random_oracle_horizon_2026-08-26/EXPERIMENT_PLAN.md)
4. [`../../docs/experiments/docprune_random_oracle_horizon_2026-08-26/IMPLEMENTATION_PLAN.md`](../../docs/experiments/docprune_random_oracle_horizon_2026-08-26/IMPLEMENTATION_PLAN.md)
5. [`../../docs/experiments/docprune_random_oracle_horizon_2026-08-26/EXPERIMENT_LOG.md`](../../docs/experiments/docprune_random_oracle_horizon_2026-08-26/EXPERIMENT_LOG.md)
6. [`HANDOFF.md`](HANDOFF.md) and [`CORAL_POLICY.md`](CORAL_POLICY.md)

The CoRAL H200 rules here are runtime ground truth. SOL/HTC files are retained
only for provenance and implementation history. Do not submit an SOL job or
copy SOL scheduler assumptions onto this server.

## First action: observation only

The sparse Git checkout at `/mnt/data1/eunwooim/DocPrune` is the sole permitted
setup action before the survey. After that checkout, and before environment
creation, downloads, transfer, compilation, model loading, or GPU work, run the
read-only commands in [`SURVEY_COMMANDS.sh`](SURVEY_COMMANDS.sh). Then record
the outputs and conclusions in [`ENVIRONMENT_SURVEY.md`](ENVIRONMENT_SURVEY.md).
Do not otherwise change the machine during the survey.

## Scope

The experiment and launch code are already implemented. Do not redesign the
cohort, mask count, arms, model, retrieval, or statistics. Small path or
environment compatibility fixes are allowed only after recording the survey;
document them before running the smoke.

- Use only the sealed 100 QIDs.
- Use only their exact cached ordered top-4 pages and persisted features.
- Never run retrieval or load the global retrieval index.
- Use 256 fit masks and zero holdout masks.
- Keep model, prompt, decoding, Lasso, mask distribution, dynamic DocPrune
  policy, and whole-region deletion frozen.
- Run one smoke question before production.
- Large/generated artifacts stay outside Git on CoRAL storage.

## Source/H200 work boundary

The source computer must seal the fixed inputs, reuse authenticated MinerU page
outputs, run MinerU only for missing unique pages, capture frozen post-BTP/QTP
geometry, build all 100 mappings, and validate the complete bundle before
transfer. The H200 agent must not recreate or modify those artifacts and must
not install MinerU or its model. It only receives and authenticates the
completed bundle, runs CPU-only transferred-input validation, then performs the
smoke, production, retry, and aggregation stages.

## Storage and GPU safety

Never write caches, temporary files, environments, checkpoints, or outputs to
`/`. Use `/mnt/data1/eunwooim` for the checkout/artifacts and
`/mnt/data2/eunwooim` for environments/caches unless the survey proves those
paths differ. Do not use ARC-only or cross-lab storage.

CoRAL GPUs are physical IDs 4–7. Check them before every launch. A run expected
to exceed 15 minutes requires a channel notice first. Do not use GPUs 0–3 in
normal operation. If temporarily borrowing another lab's idle GPU, remain
reachable and vacate within 15 minutes of a request.
