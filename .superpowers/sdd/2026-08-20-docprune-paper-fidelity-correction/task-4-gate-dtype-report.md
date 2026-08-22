# Task 4 gate dtype regression report

## Status

`DONE_WITH_CONCERNS`: the control-side gate dtype regression is fixed and
verified. No runtime source, Slurm job, or scratch attempt was changed.

## Root cause

The independent stock ColPali equivalence call in
`examples/sbatch/12_docprune_m3docvqa_gate.sbatch` passed the processor's
FP32 `pixel_values` directly to a CUDA BF16 ColPali vision tower. The indexing
path was unaffected because `encode_colpali_page()` already prepares pixels
from the vision parameter's device and dtype.

## Change

- Added `examples/m3docvqa/gate_dtype.py` with
  `prepare_stock_colpali_batch()`.
- The helper obtains the exact first vision-tower parameter, fails closed when
  the vision tower has no parameters or its parameters are non-floating, and
  casts only `pixel_values` to that parameter's device and dtype.
- Integer/token tensors are copied without dtype conversion.
- The gate now calls the helper immediately before the unmodified stock
  ColPali forward used for persisted complete-sequence equivalence. Runtime
  `src/` and all gate assertions, pins, root checks, and six-cell contract
  checks remain unchanged.
- Added behavioral tests using a small fake BF16 vision module and real
  tensors, including absent-parameter and non-floating-parameter fail-closed
  cases.

## TDD evidence

RED, before adding the helper:

```text
/home/lmalveau/mamba-envs/docprune-sol/bin/pytest -q tests/test_m3docvqa_gate_dtype.py
ERROR collecting ... ModuleNotFoundError: No module named 'gate_dtype'
```

GREEN after the minimal helper and gate call:

```text
/home/lmalveau/mamba-envs/docprune-sol/bin/pytest -q tests/test_m3docvqa_gate_dtype.py
3 passed in 1.41s
```

## Verification

```text
PYTHONPATH=src /home/lmalveau/mamba-envs/docprune-sol/bin/pytest -q tests/test_m3docvqa_launchers.py
45 passed in 5.57s

PYTHONPATH=src /home/lmalveau/mamba-envs/docprune-sol/bin/pytest -q
407 passed, 1 skipped in 11.99s

/home/lmalveau/mamba-envs/docprune-sol/bin/ruff check src tests examples/m3docvqa
All checks passed!

bash -n examples/sbatch/12_docprune_m3docvqa_gate.sbatch
exit 0

git diff --check
exit 0
```

The one skipped test is the pre-existing opt-in cached real-model CUDA probe.

## Concerns

- Real CUDA/BF16 model execution was not run locally; the corrected gate must
  still be exercised by the next authorized diagnostic/benchmark run.
- No Slurm jobs were submitted or canceled, and no scratch attempt was
  created.
