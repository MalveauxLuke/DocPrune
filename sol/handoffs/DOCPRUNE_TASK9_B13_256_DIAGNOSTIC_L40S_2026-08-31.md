# Task 9 B13 256-Mask Diagnostic L40S Handoff

Date: 2026-08-31
Status: approved and prepared; no outcomes inspected.

## Frozen diagnostic

- QID: `e1e6ed53f9ad11813845088f4cf2f6b1`
- intervention: the existing 95-region physical deletion at `B_13`
- masks: seeds `0..255` fit; seeds `256..319` held out
- decision target: exact unpruned generated response only
- analysis scale: ContextCite sequence-probability logit divided by generated
  non-EOS token count, reconstructed from sealed raw mean sequence
  log-likelihood and sealed generated token IDs
- solver: pinned released ContextCite StandardScaler plus Lasso
- no new question, retrieval, global index, feature rebuild, shard, array,
  automatic retry, or method holdout

The accepted-reference target remains in the raw dual-target package only for
backward-compatible terminal validation and is excluded from this decision.

Runtime: `/home/lmalveau/DocPrune-task9-b13-256-runtime-39aa5f5` at
`39aa5f5142cd1ef87291e72f30bcfb23d813195f`.

Launcher: `examples/sbatch/40_docprune_task9_b13_256_diagnostic.sbatch`, SHA-256
`2c58d5ffaa92bda9e94c054bfd6ea53326178f0d50f8f6b70009688c1260b8c1`.

Fresh root:
`/scratch/lmalveau/docprune/task9-b13-256-39aa5f5-v1`; terminal output is its
`output` directory.

Resources: HTC/public, one exact NVIDIA L40S, 8 CPUs, 24 GiB host RAM, 10
minutes, no requeue.

The real-input `--validate-only` path authenticated all 320 consecutive seeds,
the fixed inputs, and the mapping without loading the model or creating output.

Submit exactly once:

```bash
sbatch \
  --export=ALL,RUNTIME_DIR=/home/lmalveau/DocPrune-task9-b13-256-runtime-39aa5f5,RUNTIME_COMMIT=39aa5f5142cd1ef87291e72f30bcfb23d813195f,JOB_ROOT=/scratch/lmalveau/docprune/task9-b13-256-39aa5f5-v1 \
  /home/lmalveau/DocPrune-task9-b13-256-runtime-39aa5f5/examples/sbatch/40_docprune_task9_b13_256_diagnostic.sbatch
```

Do not inspect partial output. Require terminal admission before analysis. The
approved input-level 256+64 diagnostic receives a separate runtime and handoff.

Submitted job `62423463` completed `0:0` in `00:07:23` on `scg027`. The
embedded validator returned `admitted-task9-regional-development` for all 320
masks. Completion-manifest file SHA-256 is
`db46773f64c45fdbcbc9109af6f92458a96378ec177e6a3c61c978b61d0dec86`.
