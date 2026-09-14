# DocPrune Repository Instructions

General working behavior is defined by the global `~/.codex/AGENTS.md`. This
file contains only durable, repository-wide DocPrune rules. Task-specific plans,
handoffs, and experiment logs apply only when the user explicitly directs the
agent to them.

## Scope

This repository concerns token compression for document QA.

## H200 operations

Before doing anything involving H200s, read
[`h200-operations/README.md`](h200-operations/README.md) and
[`h200-operations/CORAL_POLICY.md`](h200-operations/CORAL_POLICY.md).
For connections, transfers or shared-account sessions, also read the local,
Git-ignored `h200-operations/H200_SOL_MAC_RUNBOOK.md`. Read the selected task's
scoped instructions when applicable; archived handoffs do not authorize new
work or supply default model settings and GPU assignments.

## Working rules

- Preserve immutable sources and keep generated or large artifacts outside Git.
  A task may keep runtime inputs and outputs inside its working checkout only
  under an explicitly ignored project-local directory; never stage or commit
  those bytes.
- For diagnostic replays or controlled evaluations with sealed retrieval
  artifacts, use the cached ordered page identities and persisted page features
  by default. Never run fresh/global retrieval, load the global retrieval index,
  or rebuild retrieval artifacts unless the user explicitly asks for fresh
  retrieval. Verify this before submission; any accidental fresh-retrieval run
  is non-canonical and must not be scored or merged with fixed-retrieval controls.
- Keep navigation files current when adding durable research material.
- Customarily record meaningful findings, insights, decisions, and experimental
  results in `agent-context/findings/YYYY-MM-DD-<topic>.md`, one file per chat.
  Update it intermittently at useful checkpoints, at the agent's discretion—not
  after every insight or experiment. Link evidence and distinguish measurements
  from hypotheses; this complements any required experiment logs.
- Never add secrets, model weights, datasets, caches, or generated outputs.
- Owner-authorized exception: `h200/correction-depth/recipe/` may contain the
  reviewed 40-case assembly metadata (questions, answer contracts, page IDs,
  source ledger and hashes) for sparse Git transfer. PDF, feature, model and
  generated segmentation bytes remain outside Git.
- Owner-authorized exception: the curated presentation packet in
  `legacy/evidence/2026-09-06/` may include selected evidence images,
  measurement extracts and source snapshots for Git-based transfer. This does not
  authorize adding raw experiment directories or other generated outputs.

- Owner-authorized temporary exception (2026-09-11):
  `transfer/colfeatures17/input/` and `transfer/colfeatures17/result/` may track
  hashed, chunked archives of the selected 17-case images/evidence and extracted
  Col-style features. This task-specific exception overrides the general
  generated-artifact prohibition above. It excludes model weights, credentials,
  caches and unrelated datasets. Remove from tracking only after verified local
  receipt, following `docs/experiments/corrective-selection/COLFEATURES17.md`.

## Resource requests

- Default to the minimum hardware needed for the task: GPU count and memory,
  CPU cores, host RAM, and wall time. Use any compatible GPU rather than
  restricting the model unnecessarily. Respect the scheduler minimums; verify
  them live rather than treating an old number as universal. For an unmeasured
  smoke, start with a justified minimal request and record actual peaks. Increase
  resources only for a measured failure/requirement or an explicit owner request.
