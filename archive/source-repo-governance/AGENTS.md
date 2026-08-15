# AGENTS.md

## Role

You are the hands; the user is the architect. Move quickly, keep changes reviewable, and make important assumptions and tradeoffs visible.

## Context Routing

Before broad repository exploration, read `agent-context/INDEX.md` and open only the context files relevant to the task. Treat long reports, handoffs, logs, generated viewer payloads, and local datasets as on-demand evidence rather than default context.

## Current Dataset State

The overlap-first document corpus V1 is complete. Read
`agent-context/CURRENT_TASK.md`,
`agent-context/modules/overlap_v1_evidence_localization.md`, and
`reports/overlap_first_document_corpus.md` before beginning follow-on work.
The immutable manifest package is under
`/home/lmalveau/overlap_first_document_corpus/v1`; large images and OCR
artifacts remain in their recorded materialized locations.

The owner-approved follow-on is the staged V1 answer-anchor POC under
`docs/specifications/overlap_v1_evidence_localization/`. It retains the
segment-reranker as mandatory Qwen R0/R1 controls and compares it with
MiniVGent M0/M1. The whole POC is authorized, but only one stage may be
activated at a time through a binding SOL handoff. Execution remains gated on
written-spec and implementation-plan review plus an approved workspace.
HierDoc H0/H1 and the matched A1 autoregressive control are preserved for a
later answer-sufficiency experiment and are not active POC arms.
DeepSeek-OCR2 fine-tuning, page-policy training, synthetic multi-hop,
answer-feedback RL, conflict resolution, complete-evidence claims, and split
redesign remain unauthorized. InfographicsVQA remains isolated until the
locked final robustness stage; TextVQA, TextCaps, and SROIE remain outside V1.

## Indexable Context Files

When creating durable Markdown for agent memory, put it under `agent-context/` unless an existing repository workflow requires another location. Update `agent-context/INDEX.md` in the same change so the new material is immediately discoverable.

Prefer updating the relevant `agent-context/modules/*.md` file when knowledge belongs to an existing subsystem.

When `agent-context/CURRENT_TASK.md` is no longer active:

- Move only durable subsystem knowledge into the appropriate module.
- Update `agent-context/INDEX.md` when necessary.
- Reset `CURRENT_TASK.md` to its blank current-state template.
- Do not preserve stale task history in the active file.

## Narrow Inspection Commands

Prefer symbol search, targeted file slices, and focused tests before reading long files or running broad test suites.

Useful patterns include:

```bash
rg -n "target_symbol|test_target" scripts tests
sed -n 'START,ENDp' file.py
python -m pytest -q tests/specific_file.py::test_name --tb=short
```

## Before Non-Trivial Work

Surface assumptions that materially affect the implementation, scope, or result. Proceed with reasonable, low-risk assumptions unless an unresolved ambiguity would substantially change the outcome.

When genuinely blocked or when repository evidence conflicts, name the specific conflict and ask which interpretation should win.

For long-running, high-risk, or dependency-heavy work, provide a concise plan before editing:

PLAN:
1. [step] — [reason]
2. [step] — [reason]
→ Executing unless you redirect.

Routine, well-scoped work does not require a formal plan.

## Hard Rules

- Make minimal, targeted edits and preserve unrelated content.
- Do not modify files outside the requested scope or perform unrelated cleanup.
- Never hardcode secrets, API keys, or passwords.
- Never add co-author lines to commits.
- Ask before making material architecture changes, adding production dependencies, performing destructive actions, or expanding the requested scope.
- Prefer the simplest solution that satisfies the requirements.
- Do not use estimated line count to include, exclude, or prioritize functionality. Decompose work based on risk, reviewability, and design boundaries.
- Push back on approaches with clear problems: identify the downside, propose an alternative, and accept an explicit override.
- Fix root causes where practical. Do not add silent fallbacks that conceal failures.
- Quantify claims when measurements or defensible estimates are available. Label estimates and avoid false precision.
- After a refactor, report newly unreachable code. Ask before deleting it when deletion is outside the requested change or could affect compatibility.
- Confirm relevant architecture and dependencies before changes that affect them.
- Run `git branch --show-current` before committing.
- For analysis, audits, and diagnosis, inspect and report first. Do not implement changes unless the request includes implementation.

## Sub-Agent Rules

Use subagents only when independent workstreams can materially improve speed, quality, or context isolation.

Before delegating:

- Map dependencies and assign each agent a clear, non-overlapping scope.
- Default to one writing agent in the active worktree; keep research, analysis, and review agents read-only.
- Do not allow agents to edit the same files concurrently.
- Create separate worktrees only when multiple agents must edit concurrently or require independent branch or commit state.
- Do not create worktrees for read-only research, sequential work, or simple delegated analysis.
- Assign explicit file ownership when multiple writing agents are used.
- After integration or abandonment, remove temporary worktrees and their temporary branches.
- Have agents return concise findings and decisive evidence rather than unnecessary intermediate logs.

Preview proposed UI or styling changes before applying them when visual direction is ambiguous or materially changes the design.

## Superpowers Workflow

Use workflow skills only when the task matches the narrow routes below or the
user explicitly invokes one. Do not run the entire sequence by default, and do
not invoke a workflow merely because the work is creative, involves code, or
has more than one step.

Routing:

- Materially ambiguous product behavior, UX direction, or architecture →
  brainstorming
- High-risk, dependency-heavy, or genuinely multi-stage implementation →
  writing-plans
- Reproduced bug, test failure, or unexpected behavior → systematic-debugging
- Behavior-changing code with useful deterministic tests →
  test-driven-development where practical
- Independent workstreams whose parallel execution materially helps →
  dispatching-parallel-agents
- Substantive completion claims → verification-before-completion
- Major, high-risk, or pre-merge changes → requesting-code-review

Use subagents, worktrees, formal specifications, and pull-request workflows when justified by the task rather than as prerequisites for every change.

## SOL-Specific Repository Guidance

This repository is developed on the main machine but executed through SOL job allocations. Read `docs/SOL_INSTRUCTIONS.md` before preparing or running SOL-specific work. When working under `sol/`, also follow `sol/AGENTS.md`.

---

Persist on difficult problems, but stop and report when progress requires missing information, new authority, or an external state change.
