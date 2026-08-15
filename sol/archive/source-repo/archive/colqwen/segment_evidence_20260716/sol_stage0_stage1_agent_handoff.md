# Agent Handoff: COLQWEN Segment-Evidence Stage 0-SOL and Stage 1 Baselines

## 0. Purpose

Move the segment-evidence project from **local Stage 0 scaffolding** to **real SOL Stage 0 smoke tests**, then to **Stage 1 frozen baselines**.

The goal right now is not to produce headline results. The goal is to prove that every pipeline path is correct before spending serious compute.

This handoff is intentionally strict. Do not improvise around model templates, score extraction, dataset fields, split usage, or run order.
/Users/god/Downloads/sol_stage0_stage1_agent_handoff.md

---

## 1. Current state

Local Stage 0 scaffolding has been implemented only.

Reported completed work:

- Added `scripts/segment_evidence_stage0.py`
  - schema validation
  - split checks
  - constants round trip
  - render cache
  - deterministic sampler
  - attention-mask decision-logit gather
  - poison sentinel answer-field check
  - B0 / B0k / B3
  - metrics, calibration, bootstrap utilities
  - JSONL logging and manifest helpers
  - compute summary
  - opt-in fail-closed model hooks
- Added `tests/test_segment_evidence_stage0.py`
  - 20 CPU-only unit tests
  - no model downloads
  - no GPU requirements
- Updated `sol/CURRENT_SOL_TASK.md`
  - Stage 0 status
  - remaining blockers only

Verification already reported:

```text
mamba run -n colqwen25 python -m unittest tests.test_segment_evidence_stage0
-> 20 tests OK
```

Real scratch schema/constants pass already reported:

```text
train / validation / test pairs: 19,474 / 4,155 / 4,103
degenerate groups: 196 / 42 / 36
schedule: 606 steps/epoch / 2,424 max_steps / 73 warmup_steps
```

No SOL jobs, full fine-tuning, test-set evaluation, model downloads, or expensive ablations have been launched yet.

Remaining blockers:

```text
pinned HF revisions
qwen3_vl environment load check
exact B4 template confirmation
real Qwen3-VL visual-token accounting
```

Important path note:

The user describes the repo as `COLQWEN_binary_classification`, but existing paths show `/home/lmalveau/COLPALI_binary_classification`. Verify the real checkout before making changes. If the mismatch matters, stop and report it. Do not guess.

---

## 2. Dataset

Use the combined Option 4 dataset:

```text
/scratch/$USER/sciegqa_train_4k/evidence_option4_10431/combined_evidence/20260707T210746Z/
```

Dataset size:

```text
retained pages: 4,463
retained queries: 6,918
quarantined pages: 2,167
quarantined queries: 3,513
```

Fresh split:

```text
train queries: 4,842
validation queries: 1,038
test queries: 1,038
```

Reported split integrity:

```text
no page leakage
no query leakage
no pair leakage
```

Do not use quarantined pages or queries.

Do not touch test during Stage 0-SOL, Stage 1, or Stage 2 pilot selection.

---

## 3. Non-negotiable guardrails

Do not launch any of the following until Stage 0-SOL passes:

```text
full fine-tuning
final A1 training
test-set evaluation
expensive ablations
multi-seed confirmation
abstracted-query final test
Stage 3 ablation grid
```

Every unresolved ambiguity must fail closed. Do not infer:

```text
schema key mappings
repo path
model template
B4 prompt format
B6 prompt format
yes/no token IDs
score direction
score extraction position
visual-token accounting method
ColQwen MaxSim normalization
whether validation or test should be used
```

The current stage is validation-first. Treat short SOL jobs as debugging and pipeline validation jobs, not as final results.

---

## 4. Required new baseline: B6

Add B6 immediately.

```text
B6 = frozen Qwen/Qwen3-VL-Reranker-2B zero-shot
```

Purpose:

```text
Test whether the official Qwen3-VL multimodal reranker already solves this task without project-specific fine-tuning.
```

Rules:

- Use `Qwen/Qwen3-VL-Reranker-2B`.
- Pin and log the exact HF revision SHA.
- Use the model's own official model-card template and usage code.
- Do not approximate B6 using the Arm A prompt unless the official usage explicitly matches.
- Assert no `<think>` block unless the official B6 template unexpectedly contains one; if it does, stop and inspect.
- Assert lowercase `"yes"` and `"no"` are single tokens under the B6 tokenizer.
- Gather decision logits at the last non-padding real token. Never use `logits[:, -1, :]` under padding.
- Score with `z_yes - z_no` after validating orientation.
- Use the same `render_v2` payloads as Arm A.
- Inference only.
- Fit temperature and thresholds on validation only.
- Never fit anything on test.

Interpretation:

```text
If B6 ~= Arm A:
  The honest finding is probably: use the off-the-shelf reranker; task-specific LoRA adds little.

If Arm A >> B6:
  Task-specific fine-tuning is justified.

If B6 > A1:
  Do not spend Stage 3 compute on Arm A ablations yet. Run E-A9: initialize from Qwen3-VL-Reranker-2B instead of Qwen3-VL-2B-Instruct.
```

---

## 5. Stage 0-SOL job sequence

Run these before Stage 1.

```text
S0-env-pin
S0-render-token-accounting
S0-frozen-score-64
S0-A1-microtrain-20
S0-Barm-microtrain-20
```

Stop if any of these fail.

---

## 6. Job 0A: environment and model-load pinning

This is the first SOL job. It should not score the dataset.

Load and pin these models:

```text
Qwen/Qwen3-VL-2B-Instruct
Qwen/Qwen3-VL-Reranker-2B
Qwen/Qwen3-Reranker-0.6B
vidore/colqwen2-v1.0
vidore/colqwen2.5-v0.2
```

Required outputs:

```text
hf_revisions.json
pip_freeze.txt
model_load_report.json
tokenizer_report.json
one rendered dummy prompt per model family
selected attention backend: FA2 if it passes, otherwise SDPA
```

Required checks:

```text
transformers supports qwen3_vl
all model revisions are pinned
"yes" is one token for Arm A, B4, and B6
"no" is one token for Arm A, B4, and B6
Arm A / Qwen3-VL-Instruct prompt has no <think> block
B4 text reranker prompt has exactly one <think></think>
B6 uses its own official reranker template
B6 is not silently forced through the Arm A template
decision logits are gathered from the last non-padding real token
logits[:, -1, :] is not used for padded batches
```

Failure policy:

If any tokenizer/template/model-load assertion fails, stop and report the exact failing assertion. Do not patch around it silently.

---

## 7. Job 0B: render and token-accounting smoke

Validate `render_v2` with the real processors.

Run on either:

```text
first 64 validation query-groups by query_id
```

or a fixed stratified subset:

```text
32 text groups + 32 visual-bundle groups
```

Do not touch test.

Required outputs:

```text
render cache manifest
per-pair text token counts
per-pair Qwen3-VL visual-token counts
per-pair ColQwen document-token counts
downscale flags
p50 / p95 / max token counts by section kind
SHA-256 verification report for page images
serialized input samples for sentinel checking
```

Pass criteria:

```text
no missing page images
no SHA mismatch
no malformed visual bundles
no missing caption resolution path
no runaway token counts
no prompt truncation after chat templating
no answer sentinel in serialized model inputs
```

Visual/doc token accounting matters scientifically. Log it carefully.

---

## 8. Job 0C: 64-group frozen scoring smoke

Run frozen scorers only on the same fixed 64 validation groups.

Required scorers:

```text
B1: frozen Qwen3-VL-2B-Instruct
B4-zs: frozen Qwen3-Reranker-0.6B text reranker
B6: frozen Qwen3-VL-Reranker-2B
B2-mini: frozen ColQwen2 MaxSim
B2b-mini: frozen ColQwen2.5 MaxSim, if it loads cleanly
B3: BM25
```

Required outputs:

```text
scores_smoke/{model}/validation_64.jsonl
raw decision logits or raw MaxSim scores
score histograms
group metrics
pair metrics
duplicate deterministic rerun diff
```

Pass criteria:

```text
all methods score exactly the same pair_ids
every non-degenerate group has exactly one positive
no NaN scores
no Inf scores
scores are not constant
rerunning gives byte-identical outputs or score-identical outputs after rounding
B0/B0k/B3 floors look sane
B1/B4/B6 are not accidentally using yes/no backwards
```

For B1, B4, and B6, explicitly compute both directions:

```text
z_yes - z_no
z_no - z_yes
```

The correct direction should do better if the model has any zero-shot signal. If the reversed direction is better, do not silently flip the sign. Inspect the template, token IDs, and decision-position gather.

---

## 9. Job 0D: 20-step microtrain for Arm A and Arm B

This is still Stage 0. It is not real training.

Run:

```text
Arm A: 20 optimizer steps on 64 train groups
Arm B: 20 optimizer steps on the same 64 train groups
```

Evaluate every 10 steps on the fixed 64 validation groups.

Required outputs:

```text
train_log.jsonl
eval_log.jsonl
diagnostics.jsonl
trainable parameter names
LoRA update norms
gradient norms
memory and throughput
sampler_trace.jsonl
```

Pass criteria:

```text
loss decreases or at least moves in the right direction on the tiny train subset
LoRA trainable set is exactly the intended set
gradients are nonzero
gradients do not explode
positive/negative decision margins improve after 20 steps
memory is feasible
step time is feasible
```

Arm A trainable-module assertion:

```text
Allowed trainable modules:
  intended LM LoRA modules only

Forbidden trainable modules:
  lm_head
  embed_tokens
  vision tower
  merger
  any accidental extra module
```

Discover the language-model module prefix at runtime. Do not hardcode the prefix if `named_modules()` shows a different structure.

---

## 10. Arm A calibration smoke

Arm A calibration smoke must check four things:

```text
1. Decision-position correctness
2. Logit-orientation correctness
3. Temperature fitting correctness
4. Threshold freezing correctness
```

Required output fields:

```text
raw_val_nll
temp_scaled_val_nll
raw_ece
temp_scaled_ece
T_star
t_F1
t_HR
val_precision_at_t_F1
val_recall_at_t_F1
val_precision_at_t_HR
val_recall_at_t_HR
rank_metrics_unchanged_after_temperature = true
```

Temperature scaling should not change Rank@1 or MRR. If it does, something is wrong because temperature scaling is monotonic.

---

## 11. ColQwen / Arm B sanity checks

For ColQwen, “working” does not mean “good score.” It means the scoring path is correct.

Must verify:

```text
MaxSim implementation matches intended ColQwen scoring path
score is query-length normalized as maxsim / N_q
frozen B2 is exactly Arm B with zero training steps
E-B4 train-affine-only reproduces B2 within numerical tolerance
ColQwen2 and ColQwen2.5 score files use the same pair list and group structure
```

First ColQwen validation output must include:

```text
raw_maxsim_distribution_by_label
length_normalized_m_distribution_by_label
platt_a
platt_b
val_pr_auc_raw_m
val_pr_auc_platt_prob
val_rank1_raw_m
val_mrr_raw_m
n_doc_tokens_p50_p95_max
n_query_tokens_p50_p95_max
```

For B2 group-ranking metrics, use raw `m`, not calibrated probability. Still log whether Platt slope `a` is positive. If `a` is negative, treat it as a sign-orientation alarm.

---

## 12. Stage 1: full frozen baselines

Only run Stage 1 after Stage 0-SOL passes.

Do not touch test.

Stage 1 must produce:

```text
reports/baselines.md
```

Required baselines:

| ID | Baseline | Purpose |
|---|---|---|
| B0 | random / prior floor | metric plumbing floor |
| B0k | section-kind prior | detects kind/base-rate shortcut |
| B1 | zero-shot Qwen3-VL-2B-Instruct | value before LoRA |
| B2 | frozen ColQwen2 + Platt | primary “MaxSim is enough” null |
| B2b | frozen ColQwen2.5 + Platt | strongest retrieval-similarity null |
| B3 | BM25 | lexical-overlap floor |
| B4-zs | frozen Qwen3-Reranker-0.6B text reranker | text-only reranker sanity |
| B5 | frozen VLM hidden-state linear probe | representation-vs-adaptation diagnostic |
| B6 | frozen Qwen3-VL-Reranker-2B | off-the-shelf multimodal reranker null |

Use train only where fitting is defined:

```text
B5 trains logistic regression on train features.
B2/B2b/B3/B4/B6 fit calibration/thresholds on validation.
```

Do not use test for fitting, thresholding, calibration, reporting, debugging, or selection.

---

## 13. Stage 2 pilots after Stage 1 only

Only after Stage 1 reports are clean, Stage 2 pilots may begin.

Stage 2 pilot set:

```text
A1
A1-lr5e5
A1-lr2e4
Barm-lambda0
Barm-lambda05
Barm-lambda1
B4-ft
```

Comparison logic:

```text
A1 vs B2/B2b:
  Does task-specific VLM cross-encoding beat retrieval similarity?

A1 vs B6:
  Does task-specific fine-tuning beat the official multimodal reranker?

A1 vs B4-ft:
  Do pixels/multimodal input help beyond OCR text?

Arm B vs B2:
  Does adapting ColQwen help beyond frozen MaxSim?
```

Do not start Stage 2 until Stage 0-SOL and Stage 1 are clean.

---

## 14. Stage 2 decision tree

### Case 1: B6 weak, A1 strong

Continue toward Stage 3. This supports the current thesis:

```text
off-the-shelf multimodal reranker does not solve the task;
task-specific LoRA helps.
```

### Case 2: B6 strong, A1 only slightly better

Reframe the contribution:

```text
Qwen3-VL-Reranker already transfers well;
task-specific LoRA gives small adaptation gains.
```

This is still useful, but it is more dataset/evaluation/calibration than new modeling.

### Case 3: B6 beats A1

Do not spend Stage 3 compute on Arm A ablations yet.

Run:

```text
E-A9: initialize from Qwen3-VL-Reranker-2B instead of Qwen3-VL-2B-Instruct
```

### Case 4: B4-ft matches A1 and B6

The multimodal claim is weak.

Likely interpretation:

```text
OCR text carries most of the signal.
Pixels are not adding much, or the visual stratum is too small/noisy.
```

Reframe as OCR-text evidence localization with visual-bundle analysis as a secondary diagnostic.

---

## 15. Things to avoid

Do not launch full A1 before B6/B4/B2 frozen baselines are clean.

If A1 fails before baselines are clean, you will not know whether the cause is:

```text
bad training
bad render_v2
bad template
wrong yes/no token IDs
wrong decision-position gather
bad calibration
bad ColQwen comparison
weak dataset signal
```

If A1 succeeds before baselines are clean, you will not know whether it beats the real off-the-shelf reranker or only weak/buggy baselines.

Do not run B6 with a hand-approximated Arm A prompt. B6 must use the official template.

Do not silently flip yes/no sign orientation.

Do not use `logits[:, -1, :]` for padded batches.

Do not touch test.

---

## 16. Expected final report from this agent pass

At the end of this pass, report:

```text
repo path used
commit hash
conda/mamba environment
HF revisions pinned
whether qwen3_vl loaded successfully
attention backend selected
Stage 0-SOL jobs run
Stage 0-SOL outputs written
Stage 0-SOL pass/fail status
B6 implementation status
Stage 1 baselines run
validation-only metrics summary
reports/baselines.md path
any failed assertions
whether Stage 2 pilots are now allowed
```

Keep the report concise but exact.

---

## 17. Minimal execution order

```text
1. Verify repo path/name.
2. Verify dataset path and audits.
3. Fill constants.json from audits.
4. Build/check environment.
5. Pin model revisions.
6. Run tokenizer/template/model-load assertions.
7. Run render/token-accounting smoke.
8. Run 64-group frozen scoring smoke.
9. Run 20-step Arm A and Arm B microtrains.
10. Fix any Stage 0-SOL failures.
11. Add/verify B6 as required baseline.
12. Run Stage 1 frozen baselines on train/validation only.
13. Produce reports/baselines.md.
14. Stop and report. Do not proceed into full training unless explicitly asked after Stage 1 passes.
```
