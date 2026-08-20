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
