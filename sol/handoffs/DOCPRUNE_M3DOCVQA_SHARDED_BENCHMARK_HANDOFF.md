# Sharded M3DocVQA Benchmark Handoff

## Authority and scope

The user approved replacing the failed monolithic six-cell evaluation with durable short HTC
shards. No historical job or artifact may be canceled, deleted, or modified. The first work is
exactly four representative 64-question top-4 shards in each mode (256 paired questions total).
Those eight runs are final top-4 shard inputs and must not be recomputed when the matrix continues.

## Exact pins

```text
runtime commit: 4e2473bdbbc2e4eca0e92c30d4a0633044501ccf
runtime checkout: /home/lmalveau/DocPrune-runtime-4e2473b
control checkout: /home/lmalveau/DocPrune-control-4e2473b
control seal: /scratch/lmalveau/docprune/benchmark-4e2473b/attempt-1/control.json
attempt root: /scratch/lmalveau/docprune/benchmark-4e2473b/attempt-1
environment: /home/lmalveau/mamba-envs/docprune-sol
PDF tools: /home/lmalveau/mamba-envs/m3docvqa-acquisition
corpus: /scratch/lmalveau/docprune/datasets/m3docvqa
HF cache: /scratch/lmalveau/hf_cache
M3DocRAG checkout: /home/lmalveau/src/m3docrag-benchmark-29e6ac2
M3DocRAG commit: 29e6ac2294d6b87075a1d45b8a8df175b214248a
factory: docprune.m3docvqa_factory:create_evaluation_workload
source promoted inputs: /scratch/lmalveau/docprune/benchmark-15301ea/attempt-2
```

The full control commit and tree are the exact values sealed in `control.json`. The control and
runtime checkouts must be clean, detached at those identities, and match the seal before execution.

## Plan and resources

`shard-plan/plan.json` binds the authoritative question-file SHA, all 2,441 source-order QIDs,
39 disjoint shards (38×64 plus 9), and a proportional exact-`metadata.type` 256-question checkpoint
in shards 0–3. Every evaluation task uses HTC/public, one constrained A100 80GB, 8 CPUs, 128 GB RAM,
and 1:15 walltime. Runtime GPU name and memory are checked before model load.

## Authorized checkpoint submission

Submit `examples/sbatch/16_docprune_m3docvqa_eval_shards.sbatch` twice with `--array=0-3%4`, once
for `MODE=all-kept,PAGES=4` and once for `MODE=docprune,PAGES=4`. Export every exact pin above plus
`CONTROL_COMMIT` from the seal, `EXPECTED_COMMIT` equal to the runtime commit,
`EXPECTED_ATTEMPT_ROOT` equal to the attempt root, and the pinned factory. Submit
`17_docprune_m3docvqa_checkpoint.sbatch` with `afterok` on both arrays.

The checkpoint job merges existing shard bytes into two 256-question runs, validates them, and
publishes `checkpoints/top4-256/checkpoint.json` with paired F1 uncertainty and encoder/decoder
speedups. It performs no evaluation.

## Authorized continuation after checkpoint inspection

If the checkpoint trend is credible, submit the remaining top-4 work as `--array=4-38`, and submit
top-1 and top-2 for both modes as `--array=0-38`. A completed valid shard exits before model load,
so retries cannot recompute it. After all six cell arrays succeed, submit
`18_docprune_m3docvqa_merge_array.sbatch`, then submit the existing signed six-cell comparator
`15_docprune_m3docvqa_compare.sbatch` with `afterok` dependencies.

## Recovery and outputs

- Never use a monolithic 2,441- or 256-question evaluation job.
- Never request 24 hours; shard jobs are fixed at 1:15.
- Do not cancel unrelated or historical jobs, including smoke job `62004161`.
- A timed-out exact shard may be resubmitted with the same plan/task ID; runtime resume validates
  the unchanged selection and prefix.
- Do not overwrite an existing plan, checkpoint, canonical merge, or comparison.
- Shards: `eval-shards/{mode}/top{pages}/shard-NNNN/run`.
- Checkpoint: `checkpoints/top4-256/`.
- Canonical cells: `eval/{mode}/top{pages}/run`.
- Final comparison: `comparison/`.

