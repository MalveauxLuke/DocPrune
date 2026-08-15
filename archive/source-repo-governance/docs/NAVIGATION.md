# Repository Navigation

For short agent routing, start with
[`agent-context/INDEX.md`](../agent-context/INDEX.md).

## Central-command entry points

| Path | Purpose |
|---|---|
| `README.md` | Repository role, current authority, and top-level boundaries |
| `docs/ACTIVE_PROJECTS.md` | Active, paused-resumable, completed, and historical research status |
| `docs/EXPERIMENT_WORKSPACES.md` | External execution locations, destination gates, and recovery authority |
| `agent-context/INDEX.md` | Short task-specific routing for agents |
| `agent-context/research/architecture_registry/README.md` | Nonbinding paper, proposal, and controlled-experiment registry |
| `archive/README.md` | Archive policy and current type-first historical routing |
| `archive/projects/README.md` | Staged project-first archive migration entry point |

## Active overlap-first V1 evidence-localization authority

| Path | Purpose |
|---|---|
| `docs/specifications/overlap_v1_evidence_localization/README.md` | Active POC scope, later sufficiency boundary, authority, and staged activation |
| `docs/specifications/overlap_v1_evidence_localization/architectures/` | Active candidate/Qwen/MiniVGent contracts plus deferred HierDoc/A1 contracts |
| `docs/specifications/overlap_v1_evidence_localization/experiment/` | Program, artifacts, metrics, gates, and stage files |
| `docs/specifications/overlap_v1_evidence_localization/SOURCE_COVERAGE.md` | Preservation map from prior designs and plans |
| `agent-context/CURRENT_TASK.md` | Concise state and next actions |
| `agent-context/modules/overlap_v1_evidence_localization.md` | Durable subsystem summary and execution gate |
| `sol/CURRENT_SOL_TASK.md` | SOL remains stopped until a binding handoff exists |
| `src/candidates/` | Planned frozen segment and eligibility implementation surface |
| `src/training/` | Planned stock scoring, mining, fine-tuning, and metric surface |

## Workspace and storage

The control checkout stays on `main` and does not execute experiments. The
POC destination, branch, and scratch root are still required.
See
[`EXPERIMENT_WORKSPACES.md`](EXPERIMENT_WORKSPACES.md) for every registered
active or paused execution line.

## Current stage sequence

```text
freeze candidates + eligibility
  -> Qwen R0/R1 + audited negatives
  -> MiniVGent systems preflight
  -> one-seed R0/R1/M0/M1 screen
  -> three-seed confirmatory run
  -> optional locked holdout
```

HierDoc H0/H1 and A1 begin only in the later answer-sufficiency experiment.

## Paused work

The previous segmentation/routing current task is under
`archive/agent-context/current_tasks/`. Its subsystem modules are under
`archive/agent-context/modules/`.

The unlaunched MMLongBench gold-page SOL task, contract, and wrappers are under:

```text
sol/archive/current_tasks/
sol/archive/task_specs/
sol/archive/jobs/mmlongbench_prepared_unlaunched/
```

Paused specifications and retained MMLongBench inputs/results remain
temporarily in their existing paths while their workspace entries are marked
`destination-required`. Read them only when planning an approved migration or
resuming the named project.

## Generated and historical artifacts

- Runtime Visual-CoT data belongs on scratch.
- Git retains only code, small manifests/audits, reports, hashes, and
  provenance.
- Historical code/tests/reports remain under `archive/`.
- Historical SOL jobs and task specs remain under `sol/archive/`.
- `tmp/`, caches, logs, model weights, archives, extracted images, and viewer
  builds are not source.
