# Task 9 H200 baseline-wrong 100 handoff

The active operational handoff is
[`../h200/task9-baseline-wrong-100/HANDOFF.md`](../h200/task9-baseline-wrong-100/HANDOFF.md).

This file exists so agents entering through `agent-context/` do not follow the
older SOL next action. Read the root `AGENTS.md`, the current task, the three
canonical experiment documents, and then the H200-scoped `AGENTS.md`. After
the required sparse checkout, the H200 agent begins with a read-only machine
survey and records it before any environment setup. It should not redesign or
recode the experiment.

The source computer must first seal all fixed inputs, run MinerU, capture
geometry, build all 100 mappings, validate them, and package the authenticated
bundle. The H200 agent must not repeat those construction steps; after its
survey and environment setup it only receives and authenticates that bundle,
runs CPU-only transferred-input validation, then performs smoke, production,
retries if necessary, and aggregation.
