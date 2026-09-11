# Current task — SOL Col-style feature package for 17 cases

Updated 2026-09-11. Owner authorized preparation, commit/push, and SOL execution
of the bounded analysis/gathering/feature-extraction task.

Read [task specification](../docs/experiments/corrective-selection/COLFEATURES17.md)
and [binding SOL handoff](../sol/CURRENT_SOL_TASK.md). Interpret the owner's “17
pages” as the 17 existing cases with 68 fixed page instances. Complete input
packet is temporarily tracked in `transfer/colfeatures17/input/`.

SOL runs tests, verifies the packet, smoke-tests pinned ColQwen2.5, extracts query,
page and region features, gathers available referenced artifacts, and publishes
an exhaustive hashed package through this repository. The local receiver will
verify it before those transfer directories are removed from tracking.

No selector training, new answerer/mask-bank execution, global retrieval, new
cohort, or additional retriever run is active. Canonical
[ExperimentPlan.md](../docs/ExperimentPlan.md) remains unchanged. This bounded
task does not complete all Stage 0 experiments. The 18 correct comparison cases
are not included in the existing packet.

The prior cleanup and all 69 retention decisions are complete; see
[legacy index](../legacy/README.md). Existing legacy tests have 34 known failures;
new task integrity tests and SOL feature smoke/verification must pass. GPU
extraction has not run during local preparation.
