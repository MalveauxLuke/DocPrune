# SOL Workflow

## Workspace state

`~/COLPALI_binary_classification` is the control checkout. The completed V1
package is `/home/lmalveau/overlap_first_document_corpus/v1`.

The overlap-first V1 answer-anchor POC does not yet have an approved
execution checkout, branch, or scratch root. Do not reuse the retained
`~/Evidence-DINO-Units` worktree or its scratch paths.

## Required first reads

1. root `AGENTS.md`
2. `docs/SOL_INSTRUCTIONS.md`
3. `sol/AGENTS.md`
4. `sol/CURRENT_SOL_TASK.md`
5. `agent-context/CURRENT_TASK.md`
6. `docs/specifications/overlap_v1_evidence_localization/README.md`

## Bootstrap gate

There is no POC bootstrap yet. Wait for an owner-approved destination and
a binding SOL handoff. Do not invent a layout, branch, scratch root, model
revision, or authority order.

## Approved future boundary

After a binding handoff exists, the allowed sequence is frozen semantic
candidates, answer-anchor eligibility, stock/tuned Qwen R0/R1, MiniVGent
systems verification, one-seed R0/R1/M0/M1 screening, confirmatory training,
and an optional matched held-out comparison.

Do not mutate V1, use InfographicsVQA for development, execute a stage not named
by the handoff, train HierDoc/A1 or DeepSeek-OCR2, generate multi-hop data, run
answer-feedback RL, resolve corpus conflicts, or redesign splits.

## State and archive

Keep `sol/CURRENT_SOL_TASK.md` concise and current. Archived task state,
contracts, and job wrappers are history, not executable instructions.

After submitting a job, confirm it starts. Monitor further only when explicitly
requested, using `modules/sol_job_monitoring.md`.
