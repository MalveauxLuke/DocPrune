# DocPrune Repository Instructions

General working behavior is defined by the global `~/.codex/AGENTS.md`. This
file contains only durable, repository-wide DocPrune rules. Task-specific plans,
handoffs, and experiment logs apply only when the user explicitly directs the
agent to them.

## Scope

This repository concerns token compression for document QA.

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
- Owner-authorized exception: the curated presentation packet in
  `docs/presentation-evidence/2026-09-06/` may include selected evidence images,
  measurement extracts and source snapshots for Git-based transfer. This does not
  authorize adding raw experiment directories or other generated outputs.
