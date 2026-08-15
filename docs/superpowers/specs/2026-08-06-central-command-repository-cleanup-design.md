# Central-Command Repository Cleanup Design

**Status:** Approved by the owner on 2026-08-06. Implementation requires the
staged plans and verification gates defined below.

**Date:** 2026-08-06

## Goal

Convert this checkout from a hybrid command center, experiment workspace, and
artifact store into the central command for COLPALI research. The repository
will retain research direction, canonical specifications, active-task routing,
experiment-location records, compact provenance, final findings, and organized
historical records. Model execution, datasets, generated viewers, run outputs,
and long-lived experiment implementations will live in named external
workspaces.

## Chosen approach

Perform an in-place, staged conversion of the current repository before making
any Git-history decision. This preserves all current work and lets each project
be classified and migrated with an auditable recovery path. A fresh-history
central repository or a history rewrite may be considered only after the
current tree satisfies the central-command contract; it is not part of the
ordinary cleanup implementation.

This approach is preferred over immediately creating a new repository because
the current checkout already contains the active Evidence-DINO authority,
uncommitted approved research-registry work, and links to multiple historical
projects. It is preferred over rewriting history first because tree
classification must be correct before destructive history surgery can be
safe.

## Repository role

The central-command repository owns:

- the root research overview and navigation;
- `agent-context/` routing, current-task state, and concise subsystem memory;
- canonical project specifications and decision records;
- the research architecture and proposal registry;
- an experiment-workspace registry containing locations, branches, owners,
  status, and recovery instructions;
- small manifests, checksums, provenance, and final scientific findings; and
- organized archives for completed, rejected, superseded, or abandoned work.

The central-command repository does not own:

- datasets, source-document collections, PDFs, or extracted image corpora;
- model weights, checkpoints, environments, or caches;
- generated viewers, page images, OCR payloads, or per-run outputs;
- Slurm logs or scratch material;
- long-lived implementation code for experiments executed elsewhere; or
- duplicate archives whose canonical copy is already verified.

A small self-authored test fixture may remain tracked only when a current
repository-contract test requires it and the fixture is explicitly allowlisted.
Historical fixtures belong with their archived project.

## Target information architecture

```text
README.md                         # central-command identity and current focus
agent-context/                    # concise agent routing and current authority
docs/
├── ACTIVE_PROJECTS.md            # active, paused, completed, and historical map
├── NAVIGATION.md                 # human and agent entry points
├── EXPERIMENT_WORKSPACES.md      # external execution locations and recovery state
├── decisions/                    # durable cross-project decisions
└── specifications/               # canonical active and paused research contracts
agent-context/research/
└── architecture_registry/        # paper reviews, proposals, experiment records
archive/
├── README.md                     # archive policy and project index
├── projects/
│   └── <project>/
│       ├── README.md             # status, conclusions, provenance, recovery
│       ├── code/                 # historical code retained for reproduction
│       ├── tests/                # tests coupled to historical code
│       ├── reports/              # final reports and selected run summaries
│       ├── viewers/              # viewer source only when scientifically necessary
│       └── artifacts.md          # external artifact locations and checksums
└── repository/                   # completed repository-wide plans and decisions
sol/
├── CURRENT_SOL_TASK.md           # current SOL routing only
├── task_spec/                    # current binding SOL contracts
└── archive/                      # historical jobs linked from project archive indexes
```

Archived SOL jobs may remain under `sol/archive/` because cluster routing uses
that boundary. Each historical project README must link to its SOL material so
the archive is project-discoverable even when the physical job files stay in
the SOL tree.

## Project lifecycle and status

Every project record must use exactly one status:

- `active`: currently authorized work;
- `paused-resumable`: intentionally stopped with a valid resume contract;
- `completed`: finished with a retained conclusion;
- `superseded`: replaced by a named design or project;
- `rejected`: evaluated and intentionally not pursued; or
- `abandoned`: incomplete and not currently worth resuming.

Active and paused projects receive entries in `docs/ACTIVE_PROJECTS.md` and
`docs/EXPERIMENT_WORKSPACES.md`. Completed, superseded, rejected, and abandoned
projects receive project indexes under `archive/projects/`. A paused project is
not archived merely because it is inactive; its implementation is first moved
to an external workspace, and its central specification remains lightweight
and resumable.

## External experiment contract

Before experiment code or artifacts leave the repository, create or verify an
experiment-workspace record containing:

- stable project name and status;
- absolute local and SOL paths where applicable;
- Git remote, branch, and commit when version controlled;
- artifact root and ownership;
- source repository commit from which material was migrated;
- checksums or immutable identifiers for scientific inputs and final outputs;
- the central specification and final-report links; and
- exact recovery or reproduction instructions.

No destination may be invented. If an approved project has no external
workspace, its record is marked `destination-required`, and physical removal
from the central repository pauses at that gate. The control-plane cleanup may
continue around it.

## Artifact migration and deletion safety

Tracked experiment artifacts are migrated in project-sized batches. For each
batch:

1. enumerate the exact source files and byte totals;
2. identify the canonical external destination;
3. copy or confirm the destination without deleting the source;
4. verify counts, byte totals, and SHA-256 checksums;
5. write the retained artifact manifest and recovery instructions;
6. remove the tracked source with Git-aware moves or deletions;
7. verify repository links and tests; and
8. commit that project batch independently.

Ignored local datasets, archives, temporary viewers, and recovery worktrees
follow the same inventory and verification rule. Deletion is allowed only when
the canonical copy is proven or the owner explicitly chooses disposal after
seeing the exact target list. Local payloads must never be moved into the Git
archive merely to make the working directory look clean.

## Archive migration

The current type-first archive is converted incrementally. Create the
project-level README and artifact manifest first, then move the project's code,
tests, reports, and viewer sources as one reviewable bundle. References are
updated in the same commit. Cluster jobs may remain physically under
`sol/archive/jobs/<project>/` but must be linked from the project README.

Initial historical projects are:

- ColQwen verifier and retrieval pilots;
- SciEGQA and MinerU parsing pilots;
- completed MMLongBench OCR and deep-parse runs;
- segment-evidence training work;
- Vidore retrieval runs; and
- completed repository and architecture-registry plans.

The MMLongBench segmentation lab, gold-page reranker, hierarchical evidence
routing, and segment-reranker corpus remain `paused-resumable` until their
external workspace destinations and exact retained specifications are
confirmed.

## Active Evidence-DINO boundary

Evidence-DINO-Units remains the sole active research project. Its binding
Stages 0–4 authority remains:

- `agent-context/CURRENT_TASK.md`;
- `docs/specifications/DETR_GroundingDINO_Datasetplan/evidence_dino_units_training_plan_final.md`;
- `sol/task_spec/evidence_dino_units_stages_0_4.md`; and
- `sol/task_spec/evidence_dino_units_workspace_bootstrap.md`.

The separate `~/Evidence-DINO-Units` workspace and
`codex/evidence-dino-units-dataset-stages-0-4` branch remain protected. Cleanup
must not authorize Stage 5, download weights, acquire MMLongBench, or execute
the experiment. Placeholder implementation directories used by the binding
workspace contract remain until that experiment has a permanent external
repository contract.

The separate `main-project` worktree is outside this cleanup and must not be
modified.

## Root documentation changes

`README.md`, `docs/NAVIGATION.md`, `docs/ACTIVE_PROJECTS.md`, and
`agent-context/INDEX.md` become the central-command entry points. Together they
must answer:

1. What research project is active?
2. What work is paused, and where is its resume contract?
3. Where is each experiment actually executed?
4. Where are historical conclusions and reproducibility records?
5. Which files are authoritative versus nonbinding research proposals?
6. What data and generated material are forbidden from Git?

Stale language that presents paused scripts, retained results, or viewers as an
active runtime is removed.

## Repository guardrails

Add focused repository-contract tests that:

- reject newly tracked model weights, archives, raw PDFs, generated page
  images, caches, and run logs outside explicit allowlists;
- reject individual tracked files over the configured central-command size cap;
- require every external experiment record to contain status, location,
  authority, and recovery fields;
- require every archived project to contain a project README;
- ensure default tests do not collect archived tests;
- validate the active Evidence-DINO Stage 0–4 boundary; and
- verify relative Markdown links.

Use standard-library tooling and existing test infrastructure. Do not add a
production dependency solely for repository hygiene.

The initial per-file cap is 5 MiB. Existing violations are recorded as
migration debt and removed project by project; the test blocks new violations
without making the repository unusable before migration finishes. The only
initial binary allowlist is the archived self-authored MinerU smoke fixture and
its checksum record.

## Branch and Git-history policy

Merged cleanup branches may be deleted only after verifying that their tips are
ancestors of the retained branch and that no worktree uses them. Remote branch
deletion is recorded separately from local deletion. The active Evidence-DINO
branch and `main-project` branch/worktree are retained.

Current-tree cleanup and Git-history cleanup are separate projects. After the
tree migration is complete, produce a measured comparison of:

1. retaining existing history;
2. creating a fresh central-command repository while preserving this one
   read-only; and
3. rewriting history with a documented force-push and clone-migration plan.

No history rewrite, force-push, or replacement remote is authorized by this
design alone.

## Implementation sequencing

1. Preserve and commit the already approved architecture-registry and obvious
   archival changes as reviewable units.
2. Rewrite central-command navigation and add the external-workspace registry.
3. Add repository guardrails in migration-compatible mode.
4. Create project-first archive indexes, then migrate completed historical
   bundles one project at a time.
5. Resolve destinations and externalize paused experiment implementations.
6. Externalize tracked generated artifacts with verified manifests.
7. Inventory and clean ignored local payloads through explicit recovery gates.
8. Remove verified merged branches.
9. Measure the final tree and present the separate Git-history decision.

## Implementation-plan decomposition

This design is a cleanup program, not one indivisible implementation task. It
must be executed through separate, independently reviewable plans:

1. **Control plane:** root documentation, experiment-workspace registry,
   migration-compatible guardrails, and preservation of the already approved
   registry/archive edits.
2. **Historical archive:** project-first indexes followed by one migration plan
   per completed historical project.
3. **Paused experiments:** one externalization plan per paused project after
   its destination is known.
4. **Tracked artifacts:** one verified artifact-migration plan per owning
   project.
5. **Local payloads:** a read-only inventory followed by explicit, recoverable
   relocation or deletion batches.
6. **Branches and history:** verified merged-branch cleanup, followed by a
   separate Git-history decision document after the tree is clean.

The first implementation plan covers only the control plane. Later plans
consume the workspace registry and guardrails it creates.

## Verification

Each implementation batch must run:

- focused tests for the files or contracts changed;
- the repository-structure and active Evidence-DINO contract tests;
- relative Markdown-link validation;
- shell syntax checks for any moved SOL wrappers;
- `git diff --check`; and
- the full default test suite before the batch is reported complete.

Artifact migrations additionally require matching file counts, byte totals, and
SHA-256 manifests before the source copy can be removed.

## Success criteria

The cleanup is complete when:

- the root clearly identifies this checkout as the central command;
- active authority, nonbinding proposals, paused work, and historical evidence
  are visibly distinct;
- every experiment implementation and heavy artifact has a named external
  location or an explicit unresolved-destination gate;
- no paused experiment implementation is collected by the default test suite;
- generated viewers, result trees, datasets, PDFs, archives, model weights,
  caches, and logs are absent from the tracked central-command surface except
  explicit fixtures;
- every historical project is recoverable through a project-level archive
  index and artifact manifest;
- current Evidence-DINO scope and the `main-project` worktree are unchanged;
- repository guardrails prevent the same sprawl from recurring; and
- the final tree-size and Git-history measurements are reported separately.

## Explicitly separate approvals

The owner's acceptance of the cleanup direction authorizes the reversible
central-command reorganization and verified project migrations. The following
remain separate execution gates because their exact targets or recovery
consequences cannot be inferred safely:

- choosing or creating an external destination for a project that has none;
- deleting ignored local datasets, archives, temporary builds, or recovery
  worktrees whose canonical copy has not been proven;
- deleting remote branches;
- rewriting Git history, force-pushing, or replacing the remote repository;
  and
- changing the active Evidence-DINO stage boundary.
