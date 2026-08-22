# Task 4 runtime 15301ea control reseal report

## Status

`DONE_WITH_CONCERNS`: the control/docs/tests reseal is implemented around
runtime `15301ea557288a4f67fc3c85228bf5e148014d17`. Runtime `src/` is
unchanged. No runtime checkout, scratch root, control record, configuration,
Slurm job, or scheduler submission was created.

## Active authority

- Runtime checkout: `/home/lmalveau/DocPrune-runtime-15301ea`
- Fresh attempt root: `/scratch/lmalveau/docprune/benchmark-15301ea/attempt-1`
- Semantic gate: HTC/public QoS, `gpu:l40:1`, 8 CPUs, 128G, 1 hour, no
  GPU-model constraint.
- Six index jobs: HTC/public QoS, `gpu:l40:1`, 8 CPUs, 128G, 4 hours, no
  GPU-model constraint.
- Six-cell evaluation array: HTC/public QoS, exact `gpu:a100:1` with
  `a100_80`, 8 CPUs, 128G, 4 hours, concurrency `%6`.
- Comparator: HTC CPU-only, 8 CPUs, 128G, 1 hour.

Only exact A100-80 evaluation rows are eligible for timing, throughput, or
memory claims. The gate/index L40 jobs are non-measurement.

## Preserved history

The active handoff and current-authority docs retain the immutable old-runtime
production graph (`61968793`, `61968794`–`61968797`, `61968799`–`61968800`,
`61968821`, `61968823`), canceled scheduling probes (`61969352`, `61969614`),
L40 failures (`61970394`, `61972695`, `61973090`, `61974092`), and the canceled
A100-40GB hedge (`61974173`) with their exact outcomes. No historical root is
reused or promotable.

## TDD evidence

RED after changing the launcher tests but before changing controls:

```text
7 failed, 42 passed
```

The failures were the expected stale A100 resources, old runtime/root pins,
and old scheduler graph. GREEN after the minimal wrapper/docs/test changes:

```text
env PYTHONPATH=src /home/lmalveau/mamba-envs/docprune-sol/bin/python -m pytest -q tests/test_m3docvqa_launchers.py
49 passed in 4.30s
```

The focused suite behaviorally executes the fake-sbatch graph, checks exact
`afterok` dependencies and immutable exports, and guards mutation back to
A100 gate/index requests or stale runtime/root pins.

## Verification

```text
env PYTHONPATH=/home/lmalveau/DocPrune-benchmark/src /home/lmalveau/mamba-envs/docprune-sol/bin/python -m pytest -q
412 passed, 1 skipped in 10.60s

/home/lmalveau/mamba-envs/docprune-sol/bin/ruff check src tests examples/m3docvqa
All checks passed!

for wrapper in examples/sbatch/{11,12,13,14,15}_docprune_m3docvqa*.sbatch; do bash -n "$wrapper"; done
exit 0

git diff --check
exit 0

env PYTHONPATH=src /home/lmalveau/mamba-envs/docprune-sol/bin/pytest -q tests/test_cli.py -k dry_run
2 passed, 32 deselected

env PYTHONPATH=src /home/lmalveau/mamba-envs/docprune-sol/bin/docprune-m3docvqa inspect --config configs/docprune-m3docvqa.toml --pages 1
exit 0
```

The skipped test is the pre-existing opt-in cached real-model CUDA probe.

## Concerns

- Real CUDA/FlashAttention-2 execution and scheduler-start evidence were not
  run locally; the next authorized run must exercise the pinned runtime and
  exact hardware contract.
- The live scheduler evidence requires the evaluation array to use HTC/public
  QoS and four hours. If any evaluation cell exceeds four hours, recovery must
  use a new reviewed chunk/resume design rather than mutating this seal.
