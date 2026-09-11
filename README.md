# DocPrune

Current research: learning corrective region selection for frozen document VLMs.
The canonical scientific design is [ExperimentPlan.md](docs/ExperimentPlan.md).
The active task is [SOL Colfeatures17](docs/experiments/corrective-selection/COLFEATURES17.md):
tests, fixed-page feature extraction, evidence gathering, and temporary Git transfer.
The later selector, teacher pipeline and new data splits remain unimplemented.

- [Current task](agent-context/CURRENT_TASK.md): status and next work.
- [Experiment workspace](docs/experiments/corrective-selection/README.md): design contracts and stages.
- [Readiness decisions](docs/experiments/corrective-selection/READINESS.md): choices required for implementation and execution.
- [Resolved retention inventory](legacy/RETENTION.md): all files kept by the owner’s decision.
- [Repository navigation](docs/NAVIGATION.md): baseline, evidence, and operations.

`src/docprune/` retains the tested reproduction and reusable evaluation/region code.
Task-numbered modules and Qwen2 integration now live in `src/docprune/_legacy/`,
with their public import names preserved. Their defaults do not define this study.
`configs/corrective-selection/` is the configuration entry point for future stages.

Superseded experiments, handoffs, and launchers are in
[the archive](archive/experiments/task6_9_2026_09_10/README.md). They are historical
evidence, not active execution instructions. Only the bounded task in `sol/CURRENT_SOL_TASK.md` is active for remote execution.

Retained older material is collected in the [legacy index](legacy/README.md).
