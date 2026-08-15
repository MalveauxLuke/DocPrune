# Archive

This directory preserves completed or superseded research code, tests, job
wrappers, viewers, plans, and reports. Archiving means the material remains
available for reproducibility and Git history but is not part of the active
runtime or default test suite.

## Rules

- Archive complete domain bundles: source, focused tests, environment/job
  wrappers, and the final handoff or report.
- Use `git mv` so file history remains visible.
- Do not silently copy active or paused-resumable code into the archive. Paused
  implementations move to verified external workspaces; completed historical
  implementations move through project-specific archive plans.
- Archived Python is not imported by active packages.
- Archived jobs are not presented as current submission commands.
- Default `rg`, IDE indexing, and agent context should skip this directory.
- Do not create `.bak-*` files. Git history and named archive snapshots replace
  ad hoc backups.

## Structure

```text
archive/
├── agent-context/        # inactive agent-routing modules
│   └── current_tasks/    # dated superseded CURRENT_TASK snapshots
├── apps/                 # superseded viewers and applications
├── code/
│   ├── colqwen/
│   ├── mmlongbench/
│   ├── sciegqa/
│   └── segment_evidence/
├── environments/        # retired environment definitions
├── plans/               # completed implementation, experiment, and repository plans
│   ├── evidence_dino/   # completed Evidence-DINO handoff plans
│   └── repository/      # completed repository and registry plans
├── projects/            # project-first indexes and verified migrated bundles
├── references/          # historical research references
├── reports/             # findings, task reports, and run summaries
├── specifications/      # completed or superseded contracts
└── tests/               # tests and fixtures coupled to archived code
```

[`projects/README.md`](projects/README.md) is the project-first archive entry
point. Existing type-first paths in this file remain authoritative until each
project receives a separate migration plan, verification gate, and commit.

Completed architecture-registry design and implementation records live under
`archive/specifications/completed/` and `archive/plans/repository/`. The
completed Evidence-DINO SOL handoff plan lives under
`archive/plans/evidence_dino/`; its current binding contracts remain outside
the archive.

The self-authored MinerU smoke fixture now lives beside the archived SciEGQA
tests at `archive/tests/fixtures/mineru_smoke/`.

The retired MMLongBench corpus preparation implementation and test are under
`archive/code/mmlongbench/` and `archive/tests/mmlongbench/`. Its short agent
module is under `archive/agent-context/modules/`, and its superseded SOL
contract is under `sol/archive/task_specs/`.

The hierarchical-segmentation task that was active before Evidence-DINO-Units
is preserved at
`archive/agent-context/current_tasks/2026-07-28-hierarchical-segmentation-routing.md`.
Its short subsystem modules are preserved under
`archive/agent-context/modules/`.

The completed overlap-first document-corpus task state is preserved at
`archive/agent-context/current_tasks/2026-08-11-overlap-first-document-corpus-v1-complete.md`;
the immutable package and current evidence-localization POC remain outside the
archive.

Cluster-specific historical material remains under `sol/archive/` because SOL
agents route there through `agent-context/SOL.md`:

```text
sol/archive/jobs/
├── colqwen/
├── mmlongbench_completed/
├── sciegqa/
├── segment_evidence/
└── vidore/
```

The prepared-but-unlaunched MMLongBench gold-page task is specifically routed
through `sol/archive/current_tasks/`,
`sol/archive/task_specs/mmlongbench_gold_pages_deepseek_ocr2.md`, and
`sol/archive/jobs/mmlongbench_prepared_unlaunched/`.
