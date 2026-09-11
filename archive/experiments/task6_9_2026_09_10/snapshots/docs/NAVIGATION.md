# Repository navigation

## Current work

| Document | Purpose |
| --- | --- |
| [Current task](../agent-context/CURRENT_TASK.md) | Current authority, boundaries, and next action |
| [Results](results/README.md) | Consolidated findings and artifact locations |
| [Regional attribution experiments](experiments/regional-attribution/INDEX.md) | Current scientific authority and progressive navigation |
| [SOL operations](../sol/README.md) | SOL/H200 division of work and execution entry points |

## Method and reproduction

| Document | Purpose |
| --- | --- |
| [Implementation guide](reproduction/DOCPRUNE.md) | Architecture, commands, and trace schema |
| [Reconstruction gaps](reproduction/RECONSTRUCTION_GAPS.md) | Paper omissions and explicit local choices |
| [Code/paper fidelity audit](reproduction/CODE_PAPER_FIDELITY_AUDIT.md) | Source-by-source paper comparison |
| [Discrepancy audit](reproduction/DISCREPANCY_AUDIT.md) | Retrieval, scoring, corpus, and pruning discrepancies |
| [Paper ambiguity audit](reproduction/PAPER_AMBIGUITY_AUDIT_2026-08-26.md) | CTP ambiguity review |
| [Fair CTP baseline](reproduction/FAIR_CTP_BASELINE_2026-08-26.md) | Frozen controls for method comparisons |
| [DocPrune paper record](../references/papers/docprune-cvpr-2026.md) | Paper, supplement, and source links |

## Regional attribution experiments

| Document | Purpose |
| --- | --- |
| [Folder index](experiments/regional-attribution/INDEX.md) | Choose only the document needed for the task |
| [Scientific plan](experiments/regional-attribution/EXPERIMENT_PLAN.md) | Canonical design organized by task |
| [Implementation status](experiments/regional-attribution/IMPLEMENTATION_PLAN.md) | High-level progress and next action |
| [Experiment logs](experiments/regional-attribution/experiment-log/INDEX.md) | Track-specific jobs, artifacts, failures, and results |
| [Mask-count analysis](experiments/regional-attribution/analysis/mask-count-and-nesting-2026-09-02.md) | 64–256 mask ablation and adaptive-diagnostic analysis |

## Operations

| Path | Purpose |
| --- | --- |
| [`sol/`](../sol/README.md) | SOL policy, state, and historical handoffs |
| [`h200/task9-baseline-wrong-100/`](../h200/task9-baseline-wrong-100/HANDOFF.md) | Locked 100-question H200 confirmation |
| [`h200/task9-shared-probe/`](../h200/task9-shared-probe/HANDOFF.md) | 600-question H200 teacher/probe execution |
| [`examples/sbatch/`](../examples/sbatch/README.md) | Scheduler launchers |
| [`environments/`](../environments/README.md) | Environment definitions |

## Historical material

Superseded plans and conversation summaries are indexed in [history](history/README.md). The exact tree before the 2026-09-05 documentation cleanup is preserved by Git tag `pre-docprune-organization-2026-09-05`.
