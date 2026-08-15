# Branch Lineage and Policy

Status date: 2026-08-11

## Active roles

### `main`

Central-command line:

- active/paused project status;
- canonical overlap-first V1 and evidence-localization POC specifications;
- agent-context routing;
- archived prior task state; and
- SOL bootstrap handoff.

Experiment execution does not occur in the `main` control checkout. External
locations and unresolved destination gates are registered in
`docs/EXPERIMENT_WORKSPACES.md`.

The project owner explicitly authorized pushing these shared changes to
`main`.

### Evidence-localization POC execution branch

No branch has been approved. The experiment is `destination-required` in
`docs/EXPERIMENT_WORKSPACES.md`. Do not implement or execute it on `main`, and
do not reuse an older experiment branch.

### `codex/evidence-dino-units-dataset-stages-0-4`

Retained paused SOL implementation/result line for the earlier Stages 0–4
handoff. It is checked out as `~/Evidence-DINO-Units`.

The branch remains recoverable historical project state. It does not authorize
the overlap-first V1 POC and must not be repurposed for it.

Do not merge experiment work into `main` without a separate owner review and
approval.

### `main-project`

Separate query-planner research with independent history. Do not merge, rebase,
prune, or otherwise modify it.

Its protected local execution location is recorded in
`docs/EXPERIMENT_WORKSPACES.md`.

## Historical lineage

Prior MMLongBench OCR, academic gold-page, and repository-consolidation
branches were sequential stages whose retained work is represented on `main`.
Their exact historical explanation remains available in Git history and the
repository consolidation plan.

## SOL branch gate

Before execution, prove:

1. the source checkout has fetched `origin`;
2. the experiment worktree is on the exact experiment branch;
3. local `HEAD` equals the fetched remote branch;
4. the branch contains the prepared `origin/main` handoff;
5. the worktree is clean; and
6. source/base/execution commits are recorded.

No force push is allowed. SOL pushes only the experiment branch.
