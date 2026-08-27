# Repository Navigation

## Active authority

| Path | Purpose |
|---|---|
| [`README.md`](../README.md) | Project goal and research question |
| [`AGENTS.md`](../AGENTS.md) | Minimal repository instructions |
| [`agent-context/INDEX.md`](../agent-context/INDEX.md) | Short context routing |
| [`agent-context/CURRENT_TASK.md`](../agent-context/CURRENT_TASK.md) | Current state and next action |
| [`sol/CURRENT_SOL_TASK.md`](../sol/CURRENT_SOL_TASK.md) | SOL execution stop/go authority |

## Research and provenance

| Path | Purpose |
|---|---|
| [`references/`](../references/README.md) | Retained papers and authored notes |
| [`agent-context/research/`](../agent-context/research/architecture_registry/README.md) | Inherited research registry |
| [`docs/specifications/`](specifications/README.md) | Inherited nonbinding specifications |
| [`docs/SOURCE_INVENTORY.md`](SOURCE_INVENTORY.md) | Retain/remove record and verification data |
| [`docs/superpowers/specs/2026-08-15-docprune-reproduction-design.md`](superpowers/specs/2026-08-15-docprune-reproduction-design.md) | Binding DocPrune reproduction design |
| [`docs/superpowers/specs/2026-08-17-docprune-sol-recovery-design.md`](superpowers/specs/2026-08-17-docprune-sol-recovery-design.md) | Approved staged SOL recovery design |
| [`docs/superpowers/plans/2026-08-17-docprune-sol-recovery.md`](superpowers/plans/2026-08-17-docprune-sol-recovery.md) | SOL recovery implementation plan |
| [`docs/reproduction/DOCPRUNE.md`](reproduction/DOCPRUNE.md) | Implementation, commands, trace schema, and validation state |
| [`docs/reproduction/RECONSTRUCTION_GAPS.md`](reproduction/RECONSTRUCTION_GAPS.md) | Paper omissions and explicit reconstruction choices |
| [`docs/reproduction/DISCREPANCY_AUDIT.md`](reproduction/DISCREPANCY_AUDIT.md) | Preliminary score, retrieval, corpus, and pruning-fidelity audit |
| [`docs/reproduction/CODE_PAPER_FIDELITY_AUDIT.md`](reproduction/CODE_PAPER_FIDELITY_AUDIT.md) | Executable source-by-source comparison with the DocPrune paper and supplement |
| [`docs/reproduction/PAPER_AMBIGUITY_AUDIT_2026-08-26.md`](reproduction/PAPER_AMBIGUITY_AUDIT_2026-08-26.md) | Fresh paper-first ambiguity audit after aggregate-logit CTP diagnostics |
| [`docs/reproduction/FAIR_CTP_BASELINE_2026-08-26.md`](reproduction/FAIR_CTP_BASELINE_2026-08-26.md) | Frozen controls and admission gate for fair CTP method comparisons |
| [`docs/reproduction/docprune_handoff_2026-08-24/`](reproduction/docprune_handoff_2026-08-24/README.md) | Compact LLM handoff: completed benchmark, findings, exact artifacts, and next actions |
| [`docs/experiments/docprune_random_oracle_horizon_2026-08-26/EXPERIMENT_PLAN.md`](experiments/docprune_random_oracle_horizon_2026-08-26/EXPERIMENT_PLAN.md) | Revised scientific contract for native/ranking CTP, global and coverage-controlled random, regional attribution, and visual-state dependence |
| [`docs/experiments/docprune_random_oracle_horizon_2026-08-26/IMPLEMENTATION_PLAN.md`](experiments/docprune_random_oracle_horizon_2026-08-26/IMPLEMENTATION_PLAN.md) | Ordered execution anchor, paper-code-first setup, gates, and current next action |
| [`docs/experiments/docprune_random_oracle_horizon_2026-08-26/EXPERIMENT_LOG.md`](experiments/docprune_random_oracle_horizon_2026-08-26/EXPERIMENT_LOG.md) | Canonical source, run, artifact, failure, result, and decision ledger |
| [`archive/source-repo-governance/`](../archive/source-repo-governance/) | Exact original governance and context |

## SOL and reusable examples

| Path | Purpose |
|---|---|
| [`docs/SOL_INSTRUCTIONS.md`](SOL_INSTRUCTIONS.md) | General SOL operating rules |
| [`sol/README.md`](../sol/README.md) | Active SOL entry point |
| [`sol/handoffs/DOCPRUNE_M3DOCVQA_BENCHMARK_HANDOFF.md`](../sol/handoffs/DOCPRUNE_M3DOCVQA_BENCHMARK_HANDOFF.md) | Active commit-pinned M3DocVQA benchmark handoff |
| [`sol/handoffs/DOCPRUNE_SOL_SMOKE_RECOVERY_HANDOFF.md`](../sol/handoffs/DOCPRUNE_SOL_SMOKE_RECOVERY_HANDOFF.md) | Historical corrected-runtime smoke recovery |
| [`sol/handoffs/DOCPRUNE_QWEN_FORCED_BOUNDARY_PARITY_DRAFT.md`](../sol/handoffs/DOCPRUNE_QWEN_FORCED_BOUNDARY_PARITY_DRAFT.md) | Draft-only Task 3 all-kept parity handoff; not execution authority |
| [`sol/handoffs/M3DOCVQA_DEV_ACQUISITION_PROBE_HANDOFF.md`](../sol/handoffs/M3DOCVQA_DEV_ACQUISITION_PROBE_HANDOFF.md) | Historical corpus acquisition and processor probe |
| [`examples/sbatch/`](../examples/sbatch/README.md) | Project-neutral SBATCH examples |
| [`environments/docprune-sol.yml`](../environments/docprune-sol.yml) | Candidate pinned SOL environment; GPU validation pending |
| [`sol/archive/source-repo/`](../sol/archive/source-repo/) | Historical source handoffs and wrappers |

Historical files are reference material. Only the active authority table can
activate work in this repository.
