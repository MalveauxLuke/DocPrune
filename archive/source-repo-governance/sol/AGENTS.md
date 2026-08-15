# SOL Agent Guidance

## Completed dataset and pending follow-on

The overlap-first document corpus V1 is complete. The owner approved a staged
answer-anchor POC comparing segment-reranker Qwen R0/R1 with MiniVGent M0/M1,
but there is no active SOL stage or binding handoff. HierDoc H0/H1 and A1 are
reserved for a later answer-sufficiency experiment.
Read `CURRENT_SOL_TASK.md` before any future execution.

Before any future submission:

1. read `CURRENT_SOL_TASK.md`;
2. fetch `origin`;
3. verify `main`, the local commit, and the remote commit;
4. require a clean tracked worktree; and
5. confirm the new task's approved scratch root.

## Completed artifact

The final manifest package is
`/home/lmalveau/overlap_first_document_corpus/v1`. Its split and exclusion
policy is summarized in `CURRENT_SOL_TASK.md`.

Keep `CURRENT_SOL_TASK.md` short: current state, exact commits, next action,
job/run IDs, relevant paths, and blockers. Put detailed requirements and
reports elsewhere.

## Compute and storage

- Login nodes are for Git, light inspection, editing, and submission only.
- Environment creation, downloads, hashing, extraction, image decoding,
  indexing, normalization, and split generation require allocations.
- Code and small Git-safe outputs live in `~/COLPALI_binary_classification`.
- Large data, caches, overlays, logs, and intermediate tables live under
  `/scratch/$USER/boundingdocs_document_qa`.
- Scratch is temporary; required manifests, audits, reports, and provenance
  must return to Git-safe paths.
- Never commit datasets, images, archive parts, models, environments, caches,
  machine-specific symlinks, or Slurm logs.

## Task boundary

The whole single-hop POC is approved, but only the stage named in
`CURRENT_SOL_TASK.md` is executable. A Stage 00 handoff may authorize only
frozen candidate/eligibility/oracle implementation and materialization. Later
POC stages require predecessor hashes and a new active handoff. No active POC
stage may execute HierDoc or A1. Until a handoff exists, do not submit jobs.
V1 mutation, InfographicsVQA-driven model selection,
DeepSeek-OCR2 fine-tuning, page-policy training, synthetic multi-hop,
answer-feedback RL, conflict adjudication, complete-evidence claims, and split
redesign remain outside scope.

## Archive and monitoring

Move superseded task state, contracts, wrappers, and reports into their named
`archive/` domains. Do not create `.bak` files.

After submitting a job, confirm it starts, then stop monitoring unless
continued monitoring was explicitly requested. When monitoring is requested,
read `../agent-context/modules/sol_job_monitoring.md`.
