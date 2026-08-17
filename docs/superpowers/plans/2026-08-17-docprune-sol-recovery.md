# DocPrune SOL Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the stale SOL authority with separate, executable handoffs for corrected-runtime smoke recovery and complete M3DocVQA acquisition plus processor probing.

**Architecture:** Keep the failed runtime and evidence immutable, activate only a smoke-recovery handoff pinned to `64ea70c`, and stage a second handoff that cannot run until smoke success is recorded. Reuse the hashed FlashAttention artifact and use a Slurm array for dataset acquisition.

**Tech Stack:** Git worktrees, Bash, Slurm, Mamba, Python 3.10, Playwright, M3DocRAG/M3DocVQA, Pytest, Ruff.

## Global Constraints

- Preserve runtime `99dbece9f7cd09abdfe35c1ba6b61020218e6f1e` and all prior evidence.
- Pin corrected runtime `64ea70c66a8f9e3dbce804d1fde265a4a2b8b09d`.
- Pin M3DocRAG `29e6ac2294d6b87075a1d45b8a8df175b214248a`.
- Preserve PyTorch 2.4.1, CUDA 12.1, Transformers 4.46.3, and FlashAttention 2.5.8.
- Do not authorize indexing, benchmarking, generation, evaluation, or training.

---

### Task 1: Replace stale active authority

**Files:**
- Modify: `agent-context/CURRENT_TASK.md`
- Modify: `sol/README.md`
- Modify: `sol/CURRENT_SOL_TASK.md`
- Modify: `sol/handoffs/DOCPRUNE_SOL_HANDOFF.md`
- Modify: `docs/NAVIGATION.md`

**Interfaces:**
- Consumes: the recorded failure at job `61567743` and corrected commit `64ea70c`
- Produces: one unambiguous active smoke-recovery authority and one explicitly inactive follow-up handoff

- [ ] Update the current-task state to record completed environment construction, the stopped smoke, and the verified correction.
- [ ] Point `CURRENT_SOL_TASK.md` and `sol/README.md` only at the smoke-recovery handoff.
- [ ] Mark the original handoff superseded without deleting its historical commands.
- [ ] Add the recovery design, plan, and handoffs to repository navigation.

### Task 2: Add the bounded smoke-recovery handoff

**Files:**
- Create: `sol/handoffs/DOCPRUNE_SOL_SMOKE_RECOVERY_HANDOFF.md`

**Interfaces:**
- Consumes: corrected runtime commit, existing environment, verified wheel, and pinned M3DocRAG commit
- Produces: one structural-smoke result and a complete stop report

- [ ] Specify clean replacement worktrees without mutating the failed runtime or dirty upstream checkout.
- [ ] Verify source and wheel hashes before reinstalling the exact artifact.
- [ ] Update the environment, install pinned source trees, and record a new freeze.
- [ ] Submit only `10_docprune_smoke.sbatch`, define pass/fail evidence, and stop.

### Task 3: Add the staged acquisition/probe workflow

**Files:**
- Create: `examples/sbatch/20_m3docvqa_download_array.sbatch`
- Modify: `examples/sbatch/README.md`
- Create: `sol/handoffs/M3DOCVQA_DEV_ACQUISITION_PROBE_HANDOFF.md`

**Interfaces:**
- Consumes: a recorded passing smoke, pinned M3DocRAG builder, and complete dev metadata
- Produces: validated dev PDFs, one deterministic page image, checksums, and `processor-contract.json`

- [ ] Add a fail-closed Slurm array wrapper around `--proc_id` and `--n_proc`.
- [ ] Specify metadata generation and the 2,441-question/3,368-PDF integrity gates.
- [ ] Specify recoverable quarantine and retry behavior for missing or corrupt PDFs.
- [ ] Specify deterministic first-page rendering, immutable model revisions, schema validation, reporting, and stop boundaries.

### Task 4: Verify and commit

**Files:**
- Verify all files above.

**Interfaces:**
- Consumes: completed authority and handoff documents
- Produces: a clean reviewable branch ready for integration

- [ ] Run `bash -n examples/sbatch/20_m3docvqa_download_array.sbatch`.
- [ ] Run the complete Pytest suite and Ruff.
- [ ] Check Markdown links, stale active-handoff references, placeholders, hashes, and Git whitespace.
- [ ] Confirm the failed runtime remains clean at `99dbece` and no SOL job or dataset operation was submitted.
- [ ] Commit the complete recovery authority change.
