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
- Never add secrets, model weights, datasets, caches, or generated outputs.
- Owner-authorized exception: `h200/correction-depth/recipe/` may contain the
  reviewed 40-case assembly metadata (questions, answer contracts, page IDs,
  source ledger and hashes) for sparse Git transfer. PDF, feature, model and
  generated segmentation bytes remain outside Git.
