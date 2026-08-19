# Task 6: Official quality metrics and independent result validation

## Implementation

- Added `evaluate_m3docvqa` with the pinned M3DocVQA list EM/F1 formulas,
  number normalization, exact maximum answer-bag alignment, modality slices,
  explicit single/multi-hop classification, every question-type slice, and
  document recall at 1/2/4/5/10. The hop/type slices are computed
  independently and do not inherit the stale upstream dictionary assignment
  bug.
- Added `M3DocVQAMetrics` and deterministic quality-plus-efficiency summary
  serialization. Quality summaries record `observed_retrieval_depth` so
  recall at 5/10 is not misrepresented when an evaluation cell records only
  top-1, top-2, or top-4 pages.
- Added `validate_benchmark_run` and the `validate-run` CLI command. Validation
  checks exact result count, duplicate/missing/unexpected qids, source order
  and source question/answer content, source-question SHA-256, run/index
  manifest canonical digests, nested index identity, full pinned index artifact
  validation when an index manifest is available, page rows, finite scores,
  monotonic traces, measurement fields, explicit warmup exclusion, profiler
  gating, and summary reproduction.
- Extended answer/timing records with peak allocated GPU bytes, explicit
  warmup state, profiler-gated definition/FLOPs, and a documented measurement
  definition. CUDA peak counters are synchronized/reset at the start of the
  complete QA path and synchronized/read after generation. The runner performs
  one unrecorded warmup and only then marks measured rows as warmup-excluded;
  resumed processes warm up before their first pending row as well.

## Hardening round

- Production model loaders now require CUDA, explicitly move both Qwen and
  ColPali models to CUDA in bfloat16, and fail before model imports when no GPU
  is available.
- Production validation now requires a complete evaluate manifest, exact
  measurement declaration, raw run-config and index source bytes plus hashes,
  validated CorpusIdentity/PDF and archive inputs, schema-4 index artifacts,
  and cross-checked runtime/resource/corpus/processor/pruning/source-order
  identity. Fixture relaxation requires an explicit complete fixture identity
  and never bypasses the default 2,441-question validation.
- Selected runs validate qids as an ordered subsequence of the full source;
  production records require a positive GPU peak, mode-aware traces, finite
  timings, and a callable warmup before pending rows. `word2number` is now a
  declared required dependency and answer evaluation fails closed if absent.

## Hardening round 2

- Fixture validation is now opt-in through the `allow_fixture=True` API
  argument and is unavailable to the CLI or the default 2,441-question
  contract. Fixture corpus identities are still reconstructed and validated;
  self-declared fixture flags cannot relax production validation.
- Raw run-config mode and page count are checked before identity validation,
  trace counts are strictly positive, and every result must explicitly carry
  a boolean `profiler_enabled` state matching the run declaration.

## TDD evidence

The required focused command initially failed during collection because the
new evaluator module did not exist:

```text
PYTHONPATH=src .../python -m pytest \
  tests/test_evaluation.py tests/test_metrics.py tests/test_cli.py -q
ModuleNotFoundError: No module named 'docprune.evaluation'
```

The implementation then added literal metric, validator, CLI, measurement,
source-integrity, profiler, and warmup tests.

## Verification

```text
PYTHONPATH=src .../python -m pytest -q
252 passed, 1 skipped (opt-in cached real-model probe)

PYTHONPATH=src .../python -m pytest \
  tests/test_evaluation.py tests/test_metrics.py tests/test_cli.py -q
59 passed

.../ruff check [all Task-6 changed source/tests]
All checks passed!

.../ruff format --check [all Task-6 changed source/tests]
5 files already formatted

git diff --check
clean
```

The sealed two-record validator fixture reproduces both the quality summary
(`overall` EM/F1 and observed retrieval depth) and the efficiency summary.
No model weights, datasets, indexes, caches, or raw profiles were added to
Git.
