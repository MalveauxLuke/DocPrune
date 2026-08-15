# Arm B Stage 2 Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Arm B a faithful, fail-closed ColQwen2 LoRA + calibrated MaxSim Stage 2 comparison against frozen B2, R1, and B4-ft using train/validation only.

**Architecture:** Keep the existing ColQwen2 loading and shared metric/calibration utilities, but add a dedicated Stage 2 Arm B runner. The runner will use the published adapter in place, exact render_v2 crop policy, length-normalized MaxSim, affine calibrator plus learnable log temperature, BCE plus configurable in-group softmax loss, deterministic query-group sampling, best-validation checkpointing, and the same score/metric artifact protocol used by R1/B4-ft.

**Tech Stack:** Python 3, PyTorch, colpali-engine, PIL, unittest, Slurm/SBATCH, existing stage0 dataset and metric helpers.

## Global Constraints

- Train and validation only; test is forbidden in code paths, wrappers, score files, and audits.
- Do not touch SciEGQA test or launch Stages 3/4/5.
- Use the published `vidore/colqwen2-v1.0` adapter in place; do not add or reinitialize a second adapter.
- Trainable set is exactly published LM q/k/v/o/gate/up/down LoRA tensors plus `custom_text_proj`, affine calibrator, and `log_tau`; vision remains frozen.
- Arm B document input uses render_v2 crops: all member boxes for text sections, image/table member boxes only for visual bundles; figure-title text is text-only.
- Enforce the 4,608 ColQwen document-token cap by common-factor crop downscaling and log affected sections/distributions before optimizer steps.
- Use bf16 model computation, fp32 calibrator/log temperature, AdamW, LoRA lr `5e-5`, calibrator/log temperature lr `1e-3`, weight decay `0.01` only on LoRA, 1.0 gradient clipping, four epochs, deterministic eight-query-group steps, 2.5% warmup, linear decay, validation PR-AUC checkpoint selection, and validation temperature/threshold fitting.
- Pilot λ values are `0.0`, `0.5`, and `1.0`; the primary Arm B pilot is λ=`0.5`.
- Record every Slurm job/check in `sol/sol_logs.md`.

---

### Task 1: Add Arm B contract tests before implementation

**Files:**
- Modify: `tests/test_segment_evidence_sol_wrappers.py`
- Modify: `tests/test_segment_evidence_stage0.py`

**Interfaces:**
- Tests will target `validate_stage2_armb_pilot`, `stage2_armb_schedule`, `armb_group_loss_components`, `configure_colqwen_trainable`, and the Arm B crop/token helpers.

- [ ] **Step 1: Write failing tests** for λ validation, 2.5% warmup, exact group-loss behavior including degenerate groups, exact crop policy, and trainable-set rejection of non-LoRA tensors.
- [ ] **Step 2: Run the focused tests** and confirm they fail because the new interfaces do not yet exist or do not yet enforce the contract.
- [ ] **Step 3: Keep the test fixtures CPU-only** with fake tensors/models and temporary images; do not load models or access the test split.

### Task 2: Correct Arm B rendering, token cap, and trainable-set assertions

**Files:**
- Modify: `scripts/run_segment_evidence_stage.py`
- Modify: `tests/test_segment_evidence_sol_wrappers.py`
- Modify: `tests/test_segment_evidence_stage0.py`

**Interfaces:**
- Add a render-v2 Arm B crop selector that takes a section and returns ordered crop paths according to section kind.
- Add a document-embedding helper that measures `n_d`, applies one common downscale factor when needed, reprocesses crops, and returns embeddings plus `{n_d_before, n_d_after, downscaled, factor}`.
- Strengthen `configure_colqwen_trainable` to require the exact published adapter name pattern and reject all forbidden/non-published trainables.

- [ ] **Step 1: Implement the minimal helpers** to make Task 1 tests pass, preserving the existing B2/ArmB shared-forward certification interface.
- [ ] **Step 2: Add fail-closed assertions** for crop metadata lengths, unsupported section kinds, empty document embeddings, and post-cap `n_d > 4608`.
- [ ] **Step 3: Run the focused tests**, then the complete lightweight suite.

### Task 3: Implement the Stage 2 Arm B runner and wrapper

**Files:**
- Modify: `scripts/run_segment_evidence_stage.py`
- Create: `sol/run_segment_evidence_stage2_arm_b.sbatch`
- Modify: `tests/test_segment_evidence_sol_wrappers.py`

**Interfaces:**
- `validate_stage2_armb_pilot(name)` accepts `ArmB`, `ArmB-lambda0`, `ArmB-lambda1` and returns `{pilot_name, lambda_group, peak_lr}`.
- `stage2_armb_schedule(...)` returns the exact schedule with `warmup_steps=ceil(max_steps*0.025)`.
- `armb_group_loss_components(scores, group_pairs, log_tau)` returns BCE sum, valid group-softmax sum, and valid-group count.
- CLI mode is `stage2-arm-b-pilot`; wrapper passes `--splits train validation` and never test.

- [ ] **Step 1: Add the failing wrapper/CLI tests** for all three pilot names, train/validation-only guardrails, A100 resources, and absence of test scope.
- [ ] **Step 2: Implement the runner** using the existing sampler and scoring utilities, with all negatives in section-id order, per-step pair normalization, λ-weighted group loss, optimizer groups, diagnostics, checkpoint restore, final scores, calibration, thresholds, orientation, strata, and manifests.
- [ ] **Step 3: Add preflight logging** of unique-section crop/token metadata before the first optimizer step and write trainable names plus adapter hash/report.
- [ ] **Step 4: Run wrapper tests and `py_compile`; do not submit yet if any contract test fails.

### Task 4: Re-certify nested B2/ArmB behavior and verify launch readiness

**Files:**
- Modify: `sol/CURRENT_SOL_TASK.md`
- Modify: `sol/sol_logs.md`

- [ ] **Step 1: Record the read-only B4-ft status check** for job `58884205`.
- [ ] **Step 2: Run the local full lightweight suite and a CPU dry-run of schedule/loss/guardrail helpers.**
- [ ] **Step 3: Run the allowed Stage 0.5 validation-only certification** after rendering changes; require ArmB-step0 vs B2 exact equivalence and no test artifacts.
- [ ] **Step 4: If certification passes, rerun frozen Stage 1 B2 on train/validation against the corrected crop policy, aggregate its refreshed score files, and record that the old B2 report is superseded.**
- [ ] **Step 5: After refreshed B2 artifacts exist, submit only the Arm B Stage 2 pilots with the approved wrapper and log job IDs/resources/output roots.**

### Task 5: Audit completed Arm B scores independently

**Files:**
- Modify: `sol/CURRENT_SOL_TASK.md`
- Modify: `sol/sol_logs.md`

- [ ] **Step 1: Check each Arm B job with `squeue` and `sacct`, recording status and elapsed time.**
- [ ] **Step 2: Verify each pilot has exactly 19,474 train and 4,155 validation rows, unique expected pair IDs, matching labels/splits, and zero test IDs.**
- [ ] **Step 3: Recompute validation calibration threshold and all metrics from score JSONL only, then cross-check PR-AUC/ROC-AUC with sklearn.**
- [ ] **Step 4: Compare Arm B λ variants with B2/B2b/R1/R1-lr1e4/B4-ft overall and by validation strata, and write the evidence-backed readiness assessment without claiming multimodal specificity unless the comparison supports it.
