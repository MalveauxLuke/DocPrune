# Task 9 H200 Baseline-Wrong 100 Implementation Plan

> **For Codex:** Use superpowers:executing-plans to implement this plan in the isolated `codex/task9-h200-baseline-wrong-100` worktree.

**Goal:** Deliver a sealed 100-question baseline-wrong cohort and a ready-to-run H200 confirmation path with 256 fitting masks and no extra mask holdouts.

**Architecture:** Extend the existing Task 9 pilot scaffolding narrowly. Build the cohort by joining the authenticated Task 6 holdout results/eligibility and excluding the sealed random-48 cohort. Generalize fixed-page materialization, add fit-only attribution analysis, and provide path-parameterized H200 launch scripts and handoff files.

**Tech stack:** Python, pytest, existing DocPrune/Qwen2-VL runtime, ContextCite Lasso environment, Bash launch templates.

### Task 1: Cohort and fixed-page inputs

- Add deterministic baseline-wrong/component-aware selection to `src/docprune/experiment_design.py`.
- Add a sealer CLI joining the Task 6 cached results, sealed eligibility, and random-48 exclusion.
- Generalize the existing fixed-page input builder for a 100-question single stratum without changing page identities.
- Add targeted cohort and fixture tests.

### Task 2: Fit-only 256-mask attribution

- Permit zero holdout masks in the generic regional mask design and admission path.
- Add fit-only dual-target analysis that creates gold-support, optional gold-margin, and matched random selections without LDS/RMSE.
- Make the existing regional runner retain the native dynamic DocPrune comparison metadata when a confirmation cohort is supplied with zero holdouts.
- Add targeted zero-holdout and analysis tests.

### Task 3: Confirmation driver and aggregation

- Add path-parameterized per-batch driver and fit-only analyzer using existing mapping and selected-arm generation.
- Add a 100-question aggregator with per-question outcomes, paired summaries, rescue rate, likelihood/margin changes, and support-component-clustered uncertainty.
- Add validate-only checks; do not run model jobs locally.

### Task 4: H200 package

- Add scoped `h200/task9-baseline-wrong-100/AGENTS.md`, experiment handoff, CoRAL policy, environment survey template, sparse-checkout manifest, environment manifest, and smoke/production launch templates.
- Make H200 authority explicit and SOL instructions context-only.
- Pin all caches/temp/output paths to `/mnt/data1` or `/mnt/data2`, never `/`.

### Task 5: Targeted verification and handoff

- Run only the new/affected Task 9 tests and CLI validate-only paths.
- Inspect the diff, record exact next H200 actions, and leave the worktree ready for user commit/transfer.
