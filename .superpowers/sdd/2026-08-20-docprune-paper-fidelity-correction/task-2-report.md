# Task 2 Report: FlashAttention CTP and paper-aligned stage timing

Commit: `a811263` (`fix: align Qwen backend and efficiency timing`)

## Changes

- `_load_qwen` now passes `attn_implementation="flash_attention_2"` and `torch_dtype=torch.bfloat16`; the loaded Qwen vision configuration is explicitly bfloat16.
- CTP projects Q from `hidden[:, -1:, :]`, projects full K, applies only the final rotary row to Q, and avoids allocating a square causal mask on the FlashAttention path. Eager attention retains the explicit causal mask and fixture behavior.
- `GenerationResult`, `AnswerOutput`, and `SampleTiming` carry encoder and decoder seconds. Encoder timing covers synchronized vision execution; decoder timing covers synchronized language-model prefill, first logits, and greedy decode. Stock generation uses a vision-module hook; CUDA fails closed if that hook observes no vision execution. CPU fakes use a deterministic fallback for test-only stage fields.
- Orchestration measures retrieval, page loading, complete QA wall time, and total sample wall time. Whole-QA peak allocated GPU memory remains the memory measurement boundary.
- Metrics now report encoder/decoder/page-load/QA/retrieval/total timing, encoder and decoder samples per second, arithmetic-mean per-sample paper drop rates, and separately labeled token-weighted drop rates.
- Production run identity includes GPU/compute capability, Python/CUDA/PyTorch/Transformers versions, bfloat16/float32 precision, FlashAttention backend, allocator definition, synchronized timer boundaries, and warmup count/sample identity. Production validation checks the complete identity and rejects missing, nonfinite, or nonpositive stage fields. Disabled profiler fields remain absent.
- Canonical evaluation no longer has a schema-4-only artifact check; non-schema-5 nested index manifests are routed through the schema-5 loader.

Files changed:

- `src/docprune/qwen2vl/decoder.py`
- `src/docprune/qwen2vl/model.py`
- `src/docprune/answerers.py`
- `src/docprune/m3docrag.py`
- `src/docprune/metrics.py`
- `src/docprune/m3docvqa_factory.py`
- `src/docprune/evaluation.py`
- `tests/qwen2vl/test_decoder.py`
- `tests/qwen2vl/test_model.py`
- `tests/test_answerers.py`
- `tests/test_evaluation.py`
- `tests/test_m3docrag.py`
- `tests/test_m3docvqa_factory.py`
- `tests/test_metrics.py`

## TDD RED evidence

The mandated pre-change focused command was:

```text
env PYTHONPATH=/home/lmalveau/DocPrune-benchmark/src /home/lmalveau/mamba-envs/docprune-sol/bin/python -m pytest tests/qwen2vl/test_decoder.py tests/qwen2vl/test_model.py tests/test_answerers.py tests/test_metrics.py tests/test_evaluation.py tests/test_m3docvqa_factory.py -v
```

Before adding the new assertions, the existing suite was the baseline:

```text
=========================== short test summary info ============================
SKIPPED [1] tests/qwen2vl/test_model.py:149: set DOCPRUNE_REAL_MODEL_TEST=1 to run the cached real-model probe
======================== 79 passed, 1 skipped in 5.99s =========================
```

After adding the intended RED tests and before production edits, the literal focused RED command was:

```text
env PYTHONPATH=/home/lmalveau/DocPrune-benchmark/src /home/lmalveau/mamba-envs/docprune-sol/bin/python -m pytest tests/qwen2vl/test_decoder.py::test_flash_ctp_recomputes_final_query_without_square_causal_mask tests/test_metrics.py::test_paper_drop_rates_are_arithmetic_mean_of_per_sample_ratios tests/test_m3docrag.py::test_adapter_preserves_official_retrieval_order_and_trace tests/test_m3docvqa_factory.py::test_model_loaders_explicitly_place_production_models_on_cuda -v
```

The literal failure summary was:

```text
tests/qwen2vl/test_decoder.py::test_flash_ctp_recomputes_final_query_without_square_causal_mask FAILED [ 25%]
tests/test_metrics.py::test_paper_drop_rates_are_arithmetic_mean_of_per_sample_ratios FAILED [ 50%]
tests/test_m3docrag.py::test_adapter_preserves_official_retrieval_order_and_trace FAILED [ 75%]
tests/test_m3docvqa_factory.py::test_model_loaders_explicitly_place_production_models_on_cuda FAILED [100%]
========================= 4 failed in 4.90s =========================
```

Reasons were respectively: FlashAttention still called `_causal_mask`; timing dataclasses did not accept stage fields; `AnswerOutput` did not carry stage fields; and the Qwen loader did not pass `attn_implementation`.

## GREEN and verification evidence

Focused implementation suite:

```text
env PYTHONPATH=/home/lmalveau/DocPrune-benchmark/src /home/lmalveau/mamba-envs/docprune-sol/bin/python -m pytest tests/qwen2vl/test_decoder.py tests/qwen2vl/test_model.py tests/test_answerers.py tests/test_metrics.py tests/test_evaluation.py tests/test_m3docvqa_factory.py tests/test_m3docrag.py -q
93 passed, 1 skipped in 4.51s
```

Full CPU suite:

```text
env PYTHONPATH=/home/lmalveau/DocPrune-benchmark/src /home/lmalveau/mamba-envs/docprune-sol/bin/python -m pytest -q
328 passed, 1 skipped in 10.18s
```

Final post-commit checks:

```text
env PYTHONPATH=/home/lmalveau/DocPrune-benchmark/src /home/lmalveau/mamba-envs/docprune-sol/bin/python -m ruff check src tests
All checks passed!

git diff --check
exit 0 (no output)
```

## Cached real-model probe

Command:

```text
env PYTHONPATH=/home/lmalveau/DocPrune-benchmark/src DOCPRUNE_REAL_MODEL_TEST=1 /home/lmalveau/mamba-envs/docprune-sol/bin/python -m pytest tests/qwen2vl/test_model.py::test_real_model_all_kept_matches_stock_first_step_logits_without_download -q
```

Result:

```text
SKIPPED [1] tests/qwen2vl/test_model.py:153: real-model probe requires CUDA; CPU tests never load the 7B checkpoint
1 skipped in 2.64s
```

This is an environmental skip, not a silent omission: `torch.cuda.is_available()` is false in the current environment, so the probe cannot load the cached 7B model or execute the CUDA/FlashAttention comparison. No model download was attempted.

## Self-review

- Verified the all-kept tiny-model generation equivalence tests remain green, including first-step logits and generated suffixes.
- Verified FlashAttention CTP does not call the square-mask allocator and eager CTP cache behavior remains green.
- Verified disabled profiler fields remain omitted from serialized timing and aggregate output.
- Verified measurement identity is strict on required sections, backend, precision, allocator, timer-boundary values, compute capability shape, and warmup/sample digest.
- Verified schema-4 checks in canonical evaluation are no longer stale and all nested non-5 manifests fail through the canonical loader.
- Verified no unrelated worktree changes were included in the commit.

## Concerns

- CUDA and the cached Qwen checkpoint are unavailable in this environment, so real-model numerical equivalence and production FlashAttention kernel execution remain unobserved here; the CPU suite and deterministic fake coverage pass.
- The benchmark’s absolute throughput/memory values remain reconstruction measurements on the configured hardware, as required by the design.

## Fix Round 1

Round 1 addressed all open Critical/Important/spec findings from review.

### Changes and files

- Corrected CTP rotary application in `src/docprune/qwen2vl/decoder.py`: the final query row receives only the final rotary row, while the full key sequence receives all rotary rows. Added a literal attention-equivalence regression in `tests/qwen2vl/test_decoder.py`.
- Updated `src/docprune/evaluation.py` production index validation to require schema 5 before invoking the canonical `_load_index_manifest` loader. Added schema-5 acceptance/schema-4 rejection coverage in `tests/test_evaluation.py`.
- Made production measurement identity require a non-null, exact two-integer compute capability, with missing, short, string, and boolean cases covered in `tests/test_evaluation.py`.
- Made production timing validation require finite, strictly positive retrieval, page-load, QA, total-sample, encoder, and decoder fields. `src/docprune/m3docvqa_factory.py` exposes strict validation for production callers; canonical evaluation and aggregate reproduction enforce the same rule. `tests/test_m3docvqa_factory.py`, `tests/test_metrics.py`, and `tests/test_evaluation.py` cover missing, zero, and nonfinite values.
- Aligned stock and sparse timing boundaries in `src/docprune/answerers.py` and `src/docprune/qwen2vl/model.py`: synchronized hooks observe only visual encoder modules and language-model prefill/greedy-decode modules (including the LM head), while preparation/compaction remains in QA and sample wall time. CUDA paths fail closed when the stage hooks observe no execution; CPU fakes retain deterministic fallback timing. `tests/test_answerers.py` covers observable stock stage execution.
- Added canonical hardware/result classification to `src/docprune/metrics.py`, measurement identity, summaries, and production validation. A100/non-A6000 hardware is classified as `reconstruction_measurement`; actual RTX A6000 hardware is classified as `paper_hardware_parity`. Exact classification presence and values are covered in `tests/test_metrics.py` and `tests/test_evaluation.py`.
- Removed the remaining stale schema-4 conditional in the secondary evaluation artifact check and kept disabled TFLOPs absent.

Modified implementation/test files:

```text
src/docprune/answerers.py
src/docprune/evaluation.py
src/docprune/m3docvqa_factory.py
src/docprune/metrics.py
src/docprune/qwen2vl/decoder.py
src/docprune/qwen2vl/model.py
tests/qwen2vl/test_decoder.py
tests/test_answerers.py
tests/test_evaluation.py
tests/test_m3docvqa_factory.py
tests/test_metrics.py
```

### TDD RED evidence

The intended round-1 RED command was run before the production fixes:

```text
env PYTHONPATH=/home/lmalveau/DocPrune-benchmark/src /home/lmalveau/mamba-envs/docprune-sol/bin/python -m pytest tests/qwen2vl/test_decoder.py::test_ctp_attention_matches_literal_final_query_full_key_rotary_fixture tests/test_evaluation.py::test_production_measurement_identity_requires_compute_capability tests/test_evaluation.py::test_production_index_validation_accepts_schema_five_and_rejects_schema_four tests/test_metrics.py::test_production_summary_rejects_missing_or_zero_stage_timings -v
```

Literal RED result before production edits:

```text
test_ctp_attention_matches_literal_final_query_full_key_rotary_fixture FAILED
  numerical mismatch: 24/24 values mismatched; greatest absolute difference 0.002968...
test_production_measurement_identity_requires_compute_capability FAILED
  expected an invalid production report, but the report was valid
test_production_index_validation_accepts_schema_five_and_rejects_schema_four FAILED
  schema-5 input was rejected with "index manifest schema_version must equal 4"
test_production_summary_rejects_missing_or_zero_stage_timings FAILED
  summarize_jsonl() did not yet accept the strict require_positive argument
============================== 4 failed in ... ==============================
```

These failures demonstrated the rotary-row mismatch, missing compute-capability fail-closed behavior, stale schema-4 production gate, and absent strict aggregate timing validation. Additional RED assertions covered the six required stage fields, boundary timing hooks, and hardware classification before their corresponding production changes.

### GREEN and verification

Required focused suite:

```text
env PYTHONPATH=/home/lmalveau/DocPrune-benchmark/src /home/lmalveau/mamba-envs/docprune-sol/bin/python -m pytest tests/qwen2vl/test_decoder.py tests/qwen2vl/test_model.py tests/test_answerers.py tests/test_metrics.py tests/test_evaluation.py tests/test_m3docvqa_factory.py tests/test_m3docrag.py -q
109 passed, 1 skipped in 4.46s
```

Full CPU suite:

```text
env PYTHONPATH=/home/lmalveau/DocPrune-benchmark/src /home/lmalveau/mamba-envs/docprune-sol/bin/python -m pytest -q
344 passed, 1 skipped in 7.91s
```

Ruff and diff checks:

```text
env PYTHONPATH=/home/lmalveau/DocPrune-benchmark/src /home/lmalveau/mamba-envs/docprune-sol/bin/python -m ruff check src tests
All checks passed!

git diff --check
exit 0 (no output)
```

### Cached real-model probe

```text
env PYTHONPATH=/home/lmalveau/DocPrune-benchmark/src DOCPRUNE_REAL_MODEL_TEST=1 /home/lmalveau/mamba-envs/docprune-sol/bin/python -m pytest tests/qwen2vl/test_model.py::test_real_model_all_kept_matches_stock_first_step_logits_without_download -q
```

Result:

```text
SKIPPED [1] tests/qwen2vl/test_model.py:153: real-model probe requires CUDA; CPU tests never load the 7B checkpoint
1 skipped in 1.60s
```

This is an environmental skip: `torch.cuda.is_available()` is false, so the cached CUDA/FlashAttention probe cannot execute. No model download was attempted and the skip is explicit in the test output.

### Self-review and concerns

- The corrected CTP path now has a literal numerical equivalence regression in addition to shape/mask assertions; tiny eager all-kept generation and first-step-logit equivalence remain green.
- Production validation now fails closed on schema, compute capability, hardware classification, timing presence, timing finiteness, and strict positivity, while fixture relaxation remains explicit and isolated.
- Stock and sparse stage timers use synchronized module boundaries and preserve complete QA/sample wall timing; GPU hook absence fails closed rather than silently labeling preparation as model execution.
- Classification is derived from the observed hardware string; A100 is explicitly a reconstruction measurement and is never labeled RTX A6000 parity.
- CUDA kernel execution and cached real-model numerical equivalence remain unobserved because this environment has no CUDA device; deterministic CPU fakes and the full CPU suite pass.

Round-1 implementation/test commit: `e9316b7`.

## Fix Round 2

Round 2 closes the remaining strict timing, boundary, synchronization, and aggregate-classification findings.

### Changes

- Production record validation now checks the raw JSON timing mapping before `SampleTiming` construction. Strict paths require explicitly present, finite, strictly positive retrieval, page-load, QA, total-sample, encoder, and decoder fields; a derived `SampleTiming.total_seconds` fallback cannot satisfy production presence.
- The evaluate CLI now invokes production record validation for resume and newly written rows and emits summaries through strict timing validation with an explicitly passed canonical result classification. Independent evaluation validation also performs the raw production check before aggregate reproduction.
- Sparse DocPrune timing now uses one synchronized encoder boundary around the complete compact vision execution (including `rot_pos_emb`) and one decoder boundary beginning before prompt `embed_tokens`, covering prefill, LM head, and every greedy decode call. Sequence/mask compaction remains outside the decoder boundary.
- Stock module hooks now record CUDA events without synchronizing per module; all event samples resolve with one post-generation synchronization. CPU fakes retain perf-counter intervals. Missing expected CUDA stage observations remain fail-closed.
- `StageMetrics.to_dict()` and strict `summarize_jsonl` now carry and validate exact canonical result classification. Strict standalone summaries reject absent classification; classification is no longer opportunistically injected from an adjacent manifest.

Round-2 implementation/test files:

```text
src/docprune/answerers.py
src/docprune/cli.py
src/docprune/evaluation.py
src/docprune/m3docvqa_factory.py
src/docprune/metrics.py
src/docprune/qwen2vl/model.py
tests/qwen2vl/test_model.py
tests/test_answerers.py
tests/test_cli.py
tests/test_evaluation.py
tests/test_m3docvqa_factory.py
tests/test_metrics.py
```

### TDD RED

The intended RED command was:

```text
env PYTHONPATH=/home/lmalveau/DocPrune-benchmark/src /home/lmalveau/mamba-envs/docprune-sol/bin/python -m pytest tests/qwen2vl/test_model.py::test_module_timer_hooks_use_cuda_events_without_per_module_synchronization tests/test_metrics.py::test_strict_summary_rejects_derived_total_and_missing_classification tests/test_metrics.py::test_strict_aggregate_carries_exact_result_classification tests/test_m3docvqa_factory.py::test_production_resume_validation_requires_raw_total_sample_seconds -v
```

Literal initial RED output:

```text
ImportError: cannot import name '_resolve_module_timer_samples' from 'docprune.qwen2vl.model'
=========================== short test summary info ============================
ERROR tests/qwen2vl/test_model.py
============================== 1 error in ... ================================
```

After the initial timer interface was added, the remaining RED assertions exposed the expected intermediate gaps: strict summary reported a generic stage-boundary error instead of naming missing `total_sample_seconds`, and the newly introduced production resume test exposed an accidentally misplaced parameterized timing assertion. Those were corrected before the GREEN run. The event test also proved that the final implementation performs zero per-module CUDA synchronizations and one resolution synchronization.

### GREEN and verification

Required focused suite:

```text
env PYTHONPATH=/home/lmalveau/DocPrune-benchmark/src /home/lmalveau/mamba-envs/docprune-sol/bin/python -m pytest tests/qwen2vl/test_model.py tests/test_answerers.py tests/test_metrics.py tests/test_evaluation.py tests/test_cli.py tests/test_m3docvqa_factory.py tests/test_m3docrag.py -q
139 passed, 1 skipped in 4.66s
```

Full CPU suite:

```text
env PYTHONPATH=/home/lmalveau/DocPrune-benchmark/src /home/lmalveau/mamba-envs/docprune-sol/bin/python -m pytest -q
348 passed, 1 skipped in 7.78s
```

Ruff and diff checks:

```text
env PYTHONPATH=/home/lmalveau/DocPrune-benchmark/src /home/lmalveau/mamba-envs/docprune-sol/bin/python -m ruff check src tests
All checks passed!

git diff --check
exit 0 (no output)
```

### Cached real-model probe

```text
env PYTHONPATH=/home/lmalveau/DocPrune-benchmark/src DOCPRUNE_REAL_MODEL_TEST=1 /home/lmalveau/mamba-envs/docprune-sol/bin/python -m pytest tests/qwen2vl/test_model.py::test_real_model_all_kept_matches_stock_first_step_logits_without_download -q
```

Result:

```text
SKIPPED [1] tests/qwen2vl/test_model.py:191: real-model probe requires CUDA; CPU tests never load the 7B checkpoint
1 skipped in 1.65s
```

This is an explicit environmental skip because `torch.cuda.is_available()` is false; no model download was attempted.

### Self-review and concerns

- Rechecked round-1 rotary numerical equivalence, schema-5 canonical loading, exact compute capability validation, and classification validation through the full CPU suite; no regressions occurred.
- Sparse timers now have exactly one stage timer each and include the requested visual rotary-position computation and prompt embedding/greedy decode boundaries.
- Stock CUDA hooks no longer call synchronized timers in every module hook; event intervals accumulate repeated calls and resolve once after generation. CPU deterministic fakes remain supported.
- Strict aggregate output requires a canonical classification passed from validated measurement identity; fixture/non-strict summaries remain available without that production requirement.
- CUDA/FlashAttention execution and cached real-model equivalence remain unobserved in this CPU-only environment.

Round-2 implementation/test commit: `cc10336`.
