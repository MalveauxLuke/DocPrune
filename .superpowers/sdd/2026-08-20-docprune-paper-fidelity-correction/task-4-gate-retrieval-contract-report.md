# Task 4 gate retrieval contract report

## Status

`DONE_WITH_CONCERNS`: the control-side gate now consumes the active
`RetrievalOutput.page_features` contract, preserves ordered fail-closed page /
feature alignment, and leaves runtime `src/` and the pinned runtime SHA
untouched. No Slurm job was submitted or canceled, and no scratch attempt was
created.

## Root cause

The semantic gate aligned retrieved page identities with feature identities by
iterating `observed.features`. The active runtime contract defines
`RetrievalOutput.page_features`; therefore the gate stopped with
`AttributeError` before semantic QA.

## Changes

- Corrected the gate heredoc in `examples/sbatch/12_docprune_m3docvqa_gate.sbatch`
  to iterate `observed.page_features`.
- Added a behavioral regression in `tests/test_m3docvqa_launchers.py` that
  extracts and executes the gate's feature-row block against real
  `RetrievalOutput` and `RetrievedPageFeatures` instances with real torch
  tensors, then runs `align_retrieved_page_context` and verifies the original
  retrieved order.
- Did not add a runtime `features` compatibility alias or change retrieval,
  feature tensors, answerer interfaces, resource assertions, dtype handling,
  or `PYTHONPATH` handling.

## TDD evidence

RED, before the gate correction:

```text
PYTHONPATH=src:examples/m3docvqa /home/lmalveau/mamba-envs/docprune-sol/bin/python -m pytest -q tests/test_m3docvqa_launchers.py -k ordered_page_features_contract
1 failed, 46 deselected
AttributeError: 'RetrievalOutput' object has no attribute 'features'
```

GREEN after the one-line gate correction:

```text
PYTHONPATH=src:examples/m3docvqa /home/lmalveau/mamba-envs/docprune-sol/bin/python -m pytest -q tests/test_m3docvqa_launchers.py -k ordered_page_features_contract
1 passed, 46 deselected
```

## Verification

```text
PYTHONPATH=src:examples/m3docvqa /home/lmalveau/mamba-envs/docprune-sol/bin/python -m pytest -q tests/test_m3docvqa_launchers.py tests/test_benchmark_seal.py tests/test_m3docvqa_gate_dtype.py
61 passed

PYTHONPATH=src:examples/m3docvqa /home/lmalveau/mamba-envs/docprune-sol/bin/python -m pytest -q
409 passed, 1 skipped
```

The skipped test is the existing opt-in cached real-model probe, which
requires `DOCPRUNE_REAL_MODEL_TEST=1` and CUDA.

```text
/home/lmalveau/mamba-envs/docprune-sol/bin/ruff check src tests examples
All checks passed!

bash -n examples/sbatch/12_docprune_m3docvqa_gate.sbatch
exit 0

git diff --check
exit 0
```

## Concerns

- Real CUDA/FlashAttention-2 gate execution was not run in this CPU-only
  environment; the next authorized gate run must exercise the corrected
  wrapper against the pinned runtime.
- The focused test executes the checked-in gate extraction block behaviorally;
  it does not submit or invoke the full model-consuming semantic gate.
