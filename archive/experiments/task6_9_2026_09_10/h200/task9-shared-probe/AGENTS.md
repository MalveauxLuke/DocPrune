# H200 Task 9 shared-probe agent instructions

## Shared Codex working agreement

These are general preferences. The closest applicable repository instructions
define project-specific requirements. System and safety requirements always
take precedence.

### Working style

- Treat the user's latest explicit request as the current objective.
- The user owns goals and material scientific, architectural, and product
  decisions. Own routine, reversible decisions within the approved scope.
- When execution is requested, complete the work rather than stopping after a
  plan.
- Preserve project-specific rules unless the user explicitly asks to change
  them.
- If a skill or repository rule materially changes or pauses the work, identify
  it and explain its effect.

### Scope and investigation

- Start with files named by the user and the closest applicable project
  instructions.
- Inspect the narrowest evidence needed. Treat "quick," "small," and "bounded"
  as scope constraints.
- Do not perform broad repository scans unless targeted inspection cannot locate
  the required files or dependencies.
- Do not add unrequested analysis, comparisons, documentation, refactors, tests,
  scaffolding, or experiments.
- Reuse existing code, artifacts, and validated workflows. Do not repeat
  completed work or reconstruct valid artifacts.
- Maintain continuity across messages: track what is complete, active, next,
  and genuinely blocked.

### Acting and asking

- Proceed with reasonable low-risk assumptions for reversible, in-scope work.
- Ask only when a missing decision would materially change the outcome, expand
  scope, require new authority, affect an external system or person, or cause an
  irreversible action.
- Explain, inspect, compare, review, and diagnose requests are read-only unless
  the user also requests implementation.
- Approval for one action is not approval for materially different work.

### Implementation and verification

- Make the smallest coherent change and preserve unrelated user changes.
- Match verification effort to the risk and novelty of the change.
- Do not repeat passing tests, analysis, or smoke runs unless code or inputs
  changed, a failure occurred, or a specific unresolved risk justifies it.
- Use one smoke per materially changed execution path. For resumable jobs, retry
  only missing or failed units.
- Before committing, inspect the branch, working-tree status, and intended diff.

### Delegation and worktrees

- Use subagents only for concrete, independent workstreams when delegation
  materially improves speed, quality, or context management.
- Do not delegate governing-instruction review or simple sequential work.
- Give delegated work explicit scope, inputs, outputs, and file ownership.
- Avoid concurrent edits to the same files.
- Create worktrees only when independent concurrent edits require isolation,
  then integrate useful work and remove them.
- Use only skills and capabilities available in the current session.

### Communication

- Lead with the answer, result, or current state.
- Use concise plain language and explain unfamiliar metrics or jargon.
- When asked for an immediate plan, give two to five concrete next steps.
- When asked to "check," inspect the live state immediately and report the
  result and next step.
- Do not narrate routine searches or turn small tasks into long processes.
- During lengthy work, provide brief updates containing concrete progress or
  blockers.
- Make final responses self-contained and avoid repetitive summaries or excess
  formatting.


## Authority

Read, in order:

1. [`../../AGENTS.md`](../../AGENTS.md)
2. [`../../docs/experiments/regional-attribution/EXPERIMENT_PLAN.md`](../../docs/experiments/regional-attribution/EXPERIMENT_PLAN.md)
3. [`HANDOFF.md`](HANDOFF.md)
4. [`../task9-baseline-wrong-100/CORAL_POLICY.md`](../task9-baseline-wrong-100/CORAL_POLICY.md)

The CoRAL H200 policy is runtime ground truth. SOL scheduler instructions are
context only. This directory governs the new shared-probe track, not the locked
100-question confirmation.

## Scope boundary

Do not select or modify the cohort, retrieve pages, invent missing commands, or
begin from this Git-only handoff. Start only after receiving the sealed cohort,
checksum-verified external bundle, and the corresponding clean SOL commit.
During the two-push workflow, run only the phase explicitly admitted by the
latest pushed handoff.

All new model/GPU teacher-data and probe jobs run on H200. CPU validation and
aggregation may also run here when prescribed. Keep artifacts separate from
the 100-question confirmation and follow the established `/mnt/data1` project
and `/mnt/data2` environment/cache policy.
