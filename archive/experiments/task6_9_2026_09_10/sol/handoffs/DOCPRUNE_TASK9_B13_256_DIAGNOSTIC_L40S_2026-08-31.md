# Task 9 B13 256-Mask Diagnostic L40S Handoff

Date: 2026-08-31
Status: completed, terminally admitted, and analyzed. Do not resubmit.

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

## Completed analyses and disposition

The canonical generated-response analysis is
`/scratch/lmalveau/docprune/task9-paired-256-diagnostics-c49abb5-v2/analysis.json`
(file SHA-256
`cfe9b9456b0040aa9bcee1ce7f332d67401f203a8a53646402a905f009798a18`,
internal SHA-256
`4d5f27cfd8a069ed8fff976a24b1529780b6f0a9430c5b53949bd9ddf3e40605`).
B13 generated-response LDS was `0.7159341`; RMSE was `0.979053` versus
constant `1.340796`; selection Jaccard mean/minimum was
`0.622318`/`0.546667`.

The accepted-answer reconstruction is
`/scratch/lmalveau/docprune/task9-paired-256-accepted-answer-0d40fad-v1/analysis.json`
(file SHA-256
`5c67c85a0223bbd8ee84d8787a40789ac2f3cf19a4bde8dd706fcde1ad8e8d98`,
internal SHA-256
`2e695fbb4413392bf8dae633294174881458c4d2fcfe3ced67a3996cdd3e65fc`).
B13 accepted-answer LDS was `0.9148352`; RMSE was `0.193681` versus constant
`0.994279`; coefficient-refit Spearman mean/minimum was
`0.669753`/`0.574119`; selection Jaccard mean/minimum was
`0.707792`/`0.650000`; and no solver warning occurred.

The user subsequently superseded the old `0.8` exact-set identity gate and
approved B13 as the boundary for a controlled 48-question answer-conditioned
oracle pilot. That new pilot requires a separate preparation handoff, clean
runtime, smoke, and submission authority. This completed diagnostic handoff
authorizes no rerun, new question, or pilot submission.
