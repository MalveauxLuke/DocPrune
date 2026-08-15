# Current SOL Task

## Purpose

This file is the short handoff for the COLQWEN/COLPALI segment-evidence experiment. Treat the detailed specifications as binding:

1. `sol/task_spec/current_design_architecture_plan.md`
2. `sol/task_spec/segment_evidence_experiment_plan.md`

Do not redesign the experiment from this file. Use it to understand current status, output paths, blockers, and the next executable steps.

## Guardrails

- Do not touch SciEGQA test until the final Stage 5 test pass.
- Do not launch Stage 3 ablations, Stage 4 confirmation, or final test evaluation yet.
- Stage 2 pilots are allowed. Full final fine-tuning/confirmation is not.
- Use official Qwen3-VL-Reranker template for B6/R1; never substitute the Arm A prompt.
- Use last-non-padding decision-position gather from `attention_mask`; never use `logits[:, -1, :]` under padding.
- Do not implement answer leakage checks as "answer string absent from input." Answers may appear in evidence. Ensure the answer field is never consumed by input builders, and keep the poison-string sentinel check.
- Fail closed on unclear schema keys. Do not guess page image path, SHA, document ID, caption/member text, abstracted-query fields, or visual-token accounting.
- Do not infer document IDs from `page_id`.
- Current canonical split is page-grouped. Do not silently replace it with a document-grouped split.
- Log all SOL/Slurm jobs in `sol/sol_logs.md` with job ID, purpose, resources, duration/status, and output path.

## Repo And Data

- Repo: `/home/lmalveau/COLPALI_binary_classification`
- Branch: `COLQWEN_binary_classification`
- Dataset: `/scratch/lmalveau/sciegqa_train_4k/evidence_option4_10431/combined_evidence/20260707T210746Z`
- Output root: `/scratch/lmalveau/segment_evidence_classifier`
- HF cache root: `/scratch/lmalveau/hf_cache`
- Environment: `/home/lmalveau/mamba-envs/segment-evidence`

## Completed Stages

## Implementation Coverage

The codebase has broad scaffolding, but not every remaining Stage 2 pilot has a finished launchable runner.

| Component | Current Coverage | Status |
|---|---|---|
| Dataset/schema/splits/render cache | CPU scaffold, real dataset constants, page-split checks, poison sentinel, render artifacts | Complete for current stages |
| Stage 0 wrappers | Env/template smoke, render/token accounting, frozen-score smoke, Arm A/R/B 20-step microtrains | Complete |
| Stage 0.5 certification | B2/B2b/B4/B6/R1-step0/ArmB-step0 certification, scorer equivalence checks, external ViDoRe diagnostic | Complete |
| Stage 1 frozen baselines | B0, B0k, B1, B2, B2b, B3, B4-zs, B5, B6 train/validation scoring and aggregate report | Complete |
| Stage 2 Arm R | Real `stage2-r1-pilot` runner and `sol/run_segment_evidence_stage2_r1.sbatch`; `R1` and `R1-lr1e4` completed | Complete for R1 pilots |
| B4 text reranker | B4-zs frozen scoring exists and is in Stage 1; official-style prompt checks exist; B4-ft Stage 2 runner/wrapper implemented; B4-ft job `58884205` completed and score/metric audit passed | Complete for B4-ft pilot |
| Arm B / ColQwen2 | Stage 0 20-step microtrain exists; B2 frozen baseline exists; ArmB-step0 vs B2 equivalence was certified before the render-policy repair; new Stage 2 runner/wrapper and contract tests are implemented locally | Stage 0.5 re-certification failed closed on token cap; Arm B Stage 2 pilots blocked |
| Stage 2 comparison report | Inputs exist for Stage 1, R1 pilots, and audited B4-ft | Still needed after Arm B pilots |

### Stage 0 Local Scaffolding

Complete.

- Implemented CPU-only scaffold in `scripts/segment_evidence_stage0.py`.
- Implemented SOL runner/wrappers in `scripts/run_segment_evidence_stage.py` and `sol/*.sbatch`.
- Lightweight tests passed repeatedly; latest full lightweight suite after Stage 2 support: `mamba run -n colqwen25 python -m unittest tests.test_segment_evidence_stage0 tests.test_segment_evidence_sol_wrappers` -> 45 tests OK.
- Real dataset constants: train/validation/test pairs = 19,474 / 4,155 / 4,103; degenerate groups = 196 / 42 / 36; schedule = 606 steps/epoch, 2,424 max steps, 73 warmup steps.

### Stage 0-SOL

Complete.

- Final B6/R1 Stage 0 job: `58727082`, COMPLETED in 00:26:21 on `sg008`.
- Output: `/scratch/lmalveau/segment_evidence_classifier/stage0/20260708T200747Z_r1`
- B6 frozen 64-group smoke: PR-AUC 0.857617, Rank@1 0.838710.
- R1 20-step microtrain completed and verified:
  - Model: `Qwen/Qwen3-VL-Reranker-2B`
  - Revision: `4bd860ac4f15ad1897a214615cccc700f8f71818`
  - Yes/no IDs: 9693 / 2152
  - Official template SHA: `47c758cb74d7f1e20e22483949a5ba4c8c1f4515126ad173da1c63211f472aa7`
  - Arm R LoRA targets: 168
  - Trainable params: 31,195,136
  - No forbidden trainables found (`lm_head`, embeddings, vision, merger, `o_proj`).

### Stage 0.5 Pipeline Certification

Complete.

- Final certification job with Arm A diagnostic: `58740160`, COMPLETED in 00:09:12 on `sg029`.
- Output: `/scratch/lmalveau/segment_evidence_classifier/certification/20260708T234801Z_arm_a_diag`
- Hard certification passed:
  - B0 base-rate invariant.
  - B2/B2b Platt slopes positive.
  - R1-step0 vs B6 exact equivalence: max_abs_diff 0.0 over 258 pairs.
  - ArmB-step0 vs B2 exact equivalence: max_abs_diff 0.0 over 258 pairs.
  - Score orientation checks passed.
  - External ViDoRe diagnostic paths passed.
- Arm A diagnostic only:
  - Frozen ArmA-step0 PR-AUC 0.790862, Rank@1 0.854839 on the same 258-pair certification slice.
  - B6/R1-step0 PR-AUC 0.851434, Rank@1 0.838710 on that slice.

### Stage 1 Frozen Baselines

Complete.

- Output: `/scratch/lmalveau/segment_evidence_classifier/stage1/20260709T002335Z/stage1_baselines`
- Report: `reports/baselines.md`
- Artifacts:
  - `metrics.json`
  - `validation_stratified_metrics.json`
  - `calibration_summary.json`
  - `compute_summary.json`
  - `orientation_report.json`
  - `score_files_manifest.json`
- Score files are complete for every baseline: 19,474 train rows and 4,155 validation rows.
- Orientation passed for B1, B2, B2b, B3, B4-zs, B5, and B6.
- Platt slopes positive for B2, B2b, and B3.

Validation headline metrics:

| Baseline | Val PR-AUC | Val Rank@1 | Val MRR | Val ECE |
|---|---:|---:|---:|---:|
| B0 | 0.249819 | 0.275100 | 0.539250 | 0.019647 |
| B0k | 0.284894 | 0.215863 | 0.500954 | 0.120956 |
| B1 | 0.701831 | 0.782129 | 0.877644 | 0.060773 |
| B2 | 0.277541 | 0.334337 | 0.570659 | 0.128715 |
| B2b | 0.286501 | 0.322289 | 0.563370 | 0.128453 |
| B3 | 0.672459 | 0.751004 | 0.854350 | 0.038584 |
| B4-zs | 0.731293 | 0.810241 | 0.886245 | 0.039099 |
| B5 | 0.856484 | 0.877510 | 0.935241 | 0.021017 |
| B6 | 0.830414 | 0.869478 | 0.927293 | 0.204803 |

### Stage 2 Arm R Pilots

R1 and R1-lr1e4 are complete. Remaining Stage 2 pilots are not complete.

First 8h attempt:

- Jobs `58779077` (`R1`) and `58779129` (`R1-lr1e4`) timed out after 8h, not from a Python exception.
- Partial logs showed both had already reached the peak performance region.
- Output preserved under `/scratch/lmalveau/segment_evidence_classifier/stage2/20260709T100500Z/stage2_pilots/`.

Completed 16h reruns:

- Output root: `/scratch/lmalveau/segment_evidence_classifier/stage2/20260709T190803Z/stage2_pilots`

| Pilot | Job | Status | Best Step | Val PR-AUC | Val Rank@1 | Train PR-AUC | Runtime |
|---|---|---|---:|---:|---:|---:|---|
| R1 | `58809033` | COMPLETED on `sg014` | 1750 | 0.939316 | 0.924699 | 0.989452 | 11:46:35 |
| R1-lr1e4 | `58809092` | COMPLETED on `sg006` | 1250 | 0.939595 | 0.937751 | 0.988821 | 10:06:44 |

Interpretation:

- Frozen B6/step0 validation PR-AUC was 0.830414, so the best R1 pilot improved by about +10.9 PR-AUC points.
- R1-lr1e4 is the current Arm R pilot winner by validation PR-AUC and Rank@1, though PR-AUC is nearly tied with R1.
- Both pilots show a big early gain followed by plateau / mild overfit:
  - R1 peaked at step 1750, then drifted down through step 2424.
  - R1-lr1e4 peaked at step 1250, then drifted down by step 2000.
- The 16h reruns were necessary for complete reports and final score files, but they did not reveal late improvement beyond the 8h peak region.

Integrity audit after completed R1 pilots:

- Both pilots wrote exact train/validation score counts:
  - train: 19,474 rows
  - validation: 4,155 rows
- No duplicate pair IDs.
- No missing or extra expected pair IDs.
- No label mismatches.
- No split mismatches.
- Zero test IDs.

Independent metric audit after completed R1 pilots:

- Completed 2026-07-10 with a standalone score-file script that reads only `scores/{pilot}/{train,validation}.jsonl`, recomputes the validation max-F1 threshold, and recomputes all metrics without importing the project metric helpers.
- The recomputed thresholds matched `calibration_summary.json` exactly:
  - R1: threshold 0.5362752781934714, F1 0.8709369024856597.
  - R1-lr1e4: threshold 0.4817518366254629, F1 0.8742228598756575.
- Recomputed metrics matched `metrics.json` to floating-point noise:
  - R1 max absolute diff: train 1.23e-15, validation 3.27e-16.
  - R1-lr1e4 max absolute diff: train 4.43e-15, validation 4.45e-16.
- A separate sklearn check in `/home/lmalveau/mamba-envs/colqwen25` matched recorded PR-AUC/ROC-AUC to <= 3.4e-16 on train and validation.
- No test score files were read or produced.

## Current Next Steps

The immediate next work is Stage 2 continuation. The R1 metric audit is complete; do not rerun it unless score files change.

1. Completed: cheap independent metric audit for completed R1/R1-lr1e4.

2. Completed: `B4-ft`.
   - Implemented `stage2-b4-ft-pilot` in `scripts/run_segment_evidence_stage.py`.
   - Added wrapper `sol/run_segment_evidence_stage2_b4_ft.sbatch`.
   - Verification passed:
     - `/home/lmalveau/mamba-envs/colqwen25/bin/python -m py_compile scripts/run_segment_evidence_stage.py tests/test_segment_evidence_sol_wrappers.py`
     - `/home/lmalveau/mamba-envs/colqwen25/bin/python -m unittest tests.test_segment_evidence_stage0 tests.test_segment_evidence_sol_wrappers` -> 48 tests OK.
   - Slurm job `58884205` completed after 09:26:06 on `sg239`.
   - Output: `/scratch/lmalveau/segment_evidence_classifier/stage2/20260711T033627Z/stage2_pilots/B4-ft`.
   - Best step: 1750.
   - Validation metrics: PR-AUC 0.9149634181190884, Rank@1 0.9046184738955824, MRR 0.949380856760375, ECE 0.023859919004813396.
   - Train metrics: PR-AUC 0.9928337410774363, Rank@1 0.9730951356005165, MRR 0.9853781030276945, ECE 0.03032077087257275.
   - Score files exist for train and validation with expected row counts: 19,474 / 4,155.
   - Text-only `Qwen/Qwen3-Reranker-0.6B` LoRA fine-tune.
   - Use official B4 text reranker prompt/template from the model-card pattern already used for B4-zs.
   - Same train/validation only discipline.
   - Purpose: determine whether OCR/text-only training matches R1; this is the key multimodal-specificity check.
   - Score-file integrity audit passed on 2026-07-13:
     - No duplicate pair IDs.
     - No missing or extra expected IDs.
     - No label, query/page/section identity, or split mismatches.
     - Zero test IDs.
   - Independent metric audit passed on 2026-07-13:
     - Recomputed validation threshold matched exactly: 0.5407034707906192.
     - Recomputed F1 matched exactly: 0.8332546055739253.
     - Recomputed train and validation metrics matched `metrics.json` exactly in the standalone implementation (max absolute diff 0.0).
     - Separate sklearn AP/ROC check matched recorded PR-AUC/ROC-AUC to <= 4.5e-16.

3. Blocked: Arm B Stage 2 implementation and certification.
   - Added `stage2-arm-b-pilot` and `sol/run_segment_evidence_stage2_arm_b.sbatch`.
   - Primary `ArmB` uses λ=0.5; optional `ArmB-lambda0` and `ArmB-lambda1` are supported.
   - Runner is train/validation only, uses the published adapter in place, logs the exact trainable set, preflights document-token counts, applies the 4,608 cap, and emits R1-style artifacts.
   - Validation-only Stage 0.5 re-certification job `58884590` failed after 00:03:35 on `sg237`.
   - Output target: `/scratch/lmalveau/segment_evidence_classifier/certification/20260710T_arm_b_renderfix`.
   - Failure: `Stage0Blocker`: `B2/ArmB certification section_id candidate_8c0d3b4d2c546b9587a2 document token cap failed after downscale: n_d=4651, cap=4608`.
   - Do not submit Arm B training until this certification failure is root-caused, fixed, and the re-certification passes.
   - If certification passes, rescore frozen B2 on train/validation with the corrected Arm B crop policy, then submit ArmB and λ pilots.
   - Purpose: determine whether training rescues the retrieval-style path relative to the corrected nested B2.

4. Produce a Stage 2 comparison report:
   - Include B6, B2/B2b, B4-zs, R1, R1-lr1e4, B4-ft, and Arm B pilots.
   - Overall validation PR-AUC, Rank@1, MRR, ECE.
   - Stratified validation metrics, especially text vs visual bundles.
   - Explicitly state whether R1 still wins after B4-ft and Arm B comparisons.

5. Only after Stage 2 comparison is clean:
   - Decide whether Stage 3 ablations are justified.
   - Do not touch test.

## Open Blockers / Caveats

- Arm B re-certification failed closed in job `58884590` because one certification candidate still exceeded the 4,608 document-token cap after downscale (`n_d=4651`). Arm B Stage 2 training remains blocked.
- Arm B Stage 2 pilots are not yet run; the local runner now supports `ArmB` (λ=0.5), `ArmB-lambda0`, and `ArmB-lambda1`.
- Arm B rendering was corrected to exclude figure-title pixels from visual bundles, enforce the 4,608 document-token cap, use all same-page negatives in section-id order, and include BCE + λ group-softmax loss. Existing full Stage 1 B2 scores are therefore stale until B2 is rescored with the corrected crop policy.
- Local Arm B verification: `py_compile` passed and the lightweight suite passed with 54 tests.
- Exact B4-ft implementation should be verified against the official Qwen3-Reranker-0.6B prompt/template behavior; fail closed if uncertain.
- Real Qwen3-VL visual-token accounting remains a fail-closed/stubbed hook beyond current smoke checks.
- Stage 2 R1 results strongly support task-adaptation signal, but do not by themselves prove multimodal specificity. B4-ft and Arm B are needed before making that claim.
- Test-set evaluation remains blocked until Stage 5.

## Key Files

- Runner: `scripts/run_segment_evidence_stage.py`
- Stage 0 utilities/tests: `scripts/segment_evidence_stage0.py`, `tests/test_segment_evidence_stage0.py`
- SOL wrapper tests: `tests/test_segment_evidence_sol_wrappers.py`
- Stage 0 wrapper: `sol/run_segment_evidence_stage0.sbatch`
- Stage 0.5 wrapper: `sol/run_segment_evidence_certification.sbatch`
- Stage 1 wrapper: `sol/run_segment_evidence_stage1_baselines.sbatch`
- Stage 2 R1 wrapper: `sol/run_segment_evidence_stage2_r1.sbatch`
- SOL job log: `sol/sol_logs.md`

## Detailed Job History

Detailed historical job IDs, failures, fixes, resources, and durations are recorded in `sol/sol_logs.md`. Keep that file as the append-only Slurm log; keep this file short.
