# Task 9 Input-Level 256-Mask Diagnostic L40S Handoff

Date: 2026-08-31
Status: approved and prepared; no outcomes inspected.

This paired diagnostic uses the same question, 95 post-BTP/QTP regions, mask
seeds, target, solver, fixed pages, and hardware contract as the B13 diagnostic.
Its only experimental change is `B_input`: selected visual states are physically
deleted at decoder input before decoder block 0 rather than after block 13.
This is decoder-input ablation, not removal of pixels before the vision encoder.

- fit seeds: `0..255`
- held-out seeds: `256..319`
- decision target: exact unpruned generated-response ContextCite sequence-
  probability logit divided by generated non-EOS token count
- no accepted-reference decision, new question, retrieval, global index,
  feature rebuild, shard, array, or automatic retry

Runtime: `/home/lmalveau/DocPrune-task9-input-256-runtime-f1d163c` at
`f1d163c8ef3922311f35db4deabd8420621c4e58`.

Launcher: `examples/sbatch/41_docprune_task9_input_256_diagnostic.sbatch`,
SHA-256 `783f2043a89162710a9f8b124d86d5e2150ecd9f855506aeb96ec99a5a96aa33`.

Fresh root: `/scratch/lmalveau/docprune/task9-input-256-f1d163c-v1`.

Resources: HTC/public, one exact NVIDIA L40S, 8 CPUs, 24 GiB host RAM, 10
minutes, no requeue.

The real-input no-model validation authenticated `B_input`, all 320 seeds, the
fixed 3,586-token mapping, and created no output. Existing forced-boundary tests
cover the B_input cache and M-RoPE topology; terminal admission remains required
before outcome analysis.

Submit exactly once:

```bash
sbatch \
  --export=ALL,RUNTIME_DIR=/home/lmalveau/DocPrune-task9-input-256-runtime-f1d163c,RUNTIME_COMMIT=f1d163c8ef3922311f35db4deabd8420621c4e58,JOB_ROOT=/scratch/lmalveau/docprune/task9-input-256-f1d163c-v1 \
  /home/lmalveau/DocPrune-task9-input-256-runtime-f1d163c/examples/sbatch/41_docprune_task9_input_256_diagnostic.sbatch
```

Do not inspect partial output. Require terminal admission before analysis.
