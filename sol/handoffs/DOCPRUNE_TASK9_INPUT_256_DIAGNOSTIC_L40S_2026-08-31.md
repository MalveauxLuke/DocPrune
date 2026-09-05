# Task 9 Input-Level 256-Mask Diagnostic L40S Handoff

Date: 2026-08-31
Status: corrected successor completed, terminally admitted, and analyzed. Do
not resubmit either input attempt.

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

## Completion, analyses, and disposition

Corrected successor `62424211` completed `0:0` in `00:04:23` on `scg027` and
terminally admitted all 320 masks. Completion-manifest file SHA-256 is
`8c2051e559064fcbc47ae317e123aaccb2cb685a3930c39c61b6ea9a6efe752e`.

The canonical generated-response analysis is
`/scratch/lmalveau/docprune/task9-paired-256-diagnostics-c49abb5-v2/analysis.json`
(file SHA-256
`cfe9b9456b0040aa9bcee1ce7f332d67401f203a8a53646402a905f009798a18`,
internal SHA-256
`4d5f27cfd8a069ed8fff976a24b1529780b6f0a9430c5b53949bd9ddf3e40605`).
Input generated-response LDS was `0.6488553`; RMSE was `0.889320` versus
constant `1.196005`; selection Jaccard mean/minimum was
`0.610084`/`0.530864`.

The accepted-answer reconstruction is
`/scratch/lmalveau/docprune/task9-paired-256-accepted-answer-0d40fad-v1/analysis.json`
(file SHA-256
`5c67c85a0223bbd8ee84d8787a40789ac2f3cf19a4bde8dd706fcde1ad8e8d98`,
internal SHA-256
`2e695fbb4413392bf8dae633294174881458c4d2fcfe3ced67a3996cdd3e65fc`).
Input accepted-answer LDS was `0.8995421`; RMSE was `0.219347` versus constant
`1.009503`; coefficient-refit Spearman mean/minimum was
`0.652819`/`0.589037`; selection Jaccard mean/minimum was
`0.683227`/`0.626506`; and no solver warning occurred.

Moving the same region intervention to decoder input did not improve
predictive or selection stability, so `B_input` is closed as a diagnostic and
`B_13` remains the controlled-pilot boundary. This handoff authorizes no retry,
new question, or pilot submission.
