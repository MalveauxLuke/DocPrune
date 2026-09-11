# Retained legacy material

The owner chose to keep every file in the earlier 69-file inventory on 2026-09-11.
These areas preserve baseline implementation and evidence without presenting them
as the active experiment. [Current work](../docs/experiments/corrective-selection/README.md).

| Material | Location |
| --- | --- |
| Older task, segmentation and Qwen2 implementation | [Source](../src/docprune/_legacy/README.md), with existing import names preserved |
| Older task tests | [Legacy tests](../tests/legacy/README.md), still collected by pytest |
| Baseline PDF/evaluation helpers | [Helpers](examples/m3docvqa/) |
| Frozen baseline config | [Configuration](configs/docprune-m3docvqa.toml) |
| Frozen baseline environment | [Environment](environments/docprune-sol.yml) |
| Complete curated evidence packet | [Evidence](evidence/2026-09-06/README.md) |
| Resolved retention inventory | [All 69 entries](RETENTION.md) |
| Old experiment contracts and launchers | [Historical archive](../archive/experiments/task6_9_2026_09_10/README.md) |

The root `pyproject.toml` remains package metadata. The current scientific plan
remains at `docs/ExperimentPlan.md`. Retention is resolved; model/data/runtime
choices remain in the new experiment's readiness document.

Archive snapshots and source evidence retain historical content. Their original
commands are provenance, not current execution authority. See `RELOCATION.json`
for every moved path and pre-move digest.
