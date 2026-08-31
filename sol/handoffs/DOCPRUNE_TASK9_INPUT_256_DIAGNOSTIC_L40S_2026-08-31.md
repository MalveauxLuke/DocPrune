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

Runtime: `/home/lmalveau/DocPrune-task9-input-256-runtime-8e956be` at
`8e956be8259e4d72ba1aa7b534649c2f411ef16c`.

Launcher: `examples/sbatch/41_docprune_task9_input_256_diagnostic.sbatch`,
SHA-256 `ff8d6c2c38b5a09d9006ad9cf9b1c8376ceaa4be4d81a7dbe9f7ee934d99b81e`.

Fresh root: `/scratch/lmalveau/docprune/task9-input-256-8e956be-v2`.

Resources: HTC/public, one exact NVIDIA L40S, 8 CPUs, 24 GiB host RAM, 10
minutes, no requeue.

The real-input no-model validation authenticated `B_input`, all 320 seeds, the
fixed 3,586-token mapping, and created no output. Existing forced-boundary tests
cover the B_input cache and M-RoPE topology; terminal admission remains required
before outcome analysis.

Submit exactly once:

```bash
sbatch \
  --export=ALL,RUNTIME_DIR=/home/lmalveau/DocPrune-task9-input-256-runtime-8e956be,RUNTIME_COMMIT=8e956be8259e4d72ba1aa7b534649c2f411ef16c,JOB_ROOT=/scratch/lmalveau/docprune/task9-input-256-8e956be-v2 \
  /home/lmalveau/DocPrune-task9-input-256-runtime-8e956be/examples/sbatch/41_docprune_task9_input_256_diagnostic.sbatch
```

Do not inspect partial output. Require terminal admission before analysis.

Historical attempt `62423876` used runtime `f1d163c8` and failed at the clean
dependency-pin preflight in six seconds because the launcher omitted one
character from the M3DocRAG commit. It created no output and loaded no model.
Commit `8e956be` adds a regression assertion and corrects only that pin.

Corrected successor job `62424211` was submitted and initially recorded as
`PENDING (Priority)` with one L40S, 8 CPUs, 24 GiB, 10 minutes, and no requeue.
