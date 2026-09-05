# AGENTS.md

## Scope

This repository concerns token compression for document QA.

## Context routing

Read `agent-context/INDEX.md` first. Treat inherited specifications, research,
and SOL material as nonbinding reference unless `agent-context/CURRENT_TASK.md`
explicitly activates work.

## Working rules

- Preserve immutable sources and keep generated or large artifacts outside Git.
  A task may keep runtime inputs and outputs inside its working checkout only
  under an explicitly ignored project-local directory; never stage or commit
  those bytes.
- Make small, reviewable changes and surface material assumptions.
- Do not start an experiment, model run, or SOL job without a current task and
  explicit handoff.
- For diagnostic replays or controlled evaluations with sealed retrieval
  artifacts, use the cached ordered page identities and persisted page features
  by default. Never run fresh/global retrieval, load the global retrieval index,
  or rebuild retrieval artifacts unless the user explicitly asks for fresh
  retrieval. Verify this before submission; any accidental fresh-retrieval run
  is non-canonical and must not be scored or merged with fixed-retrieval controls.
- Keep navigation files current when adding durable research material.
- Never add secrets, model weights, datasets, caches, or generated outputs.

## Random/coverage/attribution experiment tracking

The canonical experiment documents are:

- `docs/experiments/docprune_random_oracle_horizon_2026-08-26/EXPERIMENT_PLAN.md`
  for the approved scientific design and claim boundary;
- `docs/experiments/docprune_random_oracle_horizon_2026-08-26/IMPLEMENTATION_PLAN.md`
  for ordered tasks, status, acceptance gates, and the next action; and
- `docs/experiments/docprune_random_oracle_horizon_2026-08-26/EXPERIMENT_LOG.md`
  for source pins, jobs, artifacts, failures, results, and decisions.

Read all three before working on that experiment. Update the implementation
plan whenever task status or the next action changes; append validated results,
failed/rejected runs, job IDs, artifact paths, and hashes to the experiment log
as soon as they exist. Change the scientific plan only for an approved course
change, with a dated rationale recorded before affected outcomes are inspected.
Inspect and pin paper-author code before synthesizing a local implementation;
record every architecture adaptation and any missing author code. MinerU is an
independent whole-region mask tool, not paper-derived pruning code. ContextCite
is an exploratory regional attribution procedure, not a token oracle. Wang
standalone contribution is a separately approved conditional study, not an
automatic fallback. Treat the existing 16/64/245 cohorts as development data
and use only a sealed method holdout for confirmatory claims.
