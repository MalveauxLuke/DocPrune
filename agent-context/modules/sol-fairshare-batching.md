# SOL throughput, fairshare and queue-time decisions

Owner priority (2026-09-17): minimize fairshare cost while avoiding excessive queue waits. Neither fastest elapsed time nor largest batch is the sole objective. Use actual evidence and keep experiments bounded.

## Accounting evidence

ASU documents CHE as weighted allocated resources times actual job runtime: https://docs.rc.asu.edu/fairshare/ . Slurm describes partition TRESBillingWeights: https://slurm.schedmd.com/tres.html . Live `scontrol show partition htc` and each job's `AllocTRES.billing` take precedence over older web examples. The live A30 weight was25 while the documentation example says20. Include generic and typed GPU billing as actually reported; do not infer the total from the GPU name alone.

Smoke63519851: billing35, elapsed227s, hence about2.207CHE. It requested genericGPU1, CPU2,24000M and received A10080. MaxRSS14272868KiB (~13.61GiB). Checkpoint loading32s; one short-mask cached G/S call .322s. Total runtime3:47 includes unmeasured imports/preparation and checks; do not attribute the entire residual to loading.

## Decision procedure

1. Size CPU and host memory from measured RSS/CPU requirements and verified scheduler acceptance, not GPU VRAM. GPU memory is a separate constraint. Reduce cautiously; owner prefers resumable OOM/timeouts to systematic over-allocation.
2. Compare `sbatch --test-only` estimates for a few plausible short time limits. These are transient estimates, not reservations. Use `squeue --start` after actual submission. Queue waiting does not reserve a running GPU; distinguish wait from charged runtime.
3. Prefer one loaded worker for a small bank when setup dominates. Shard only if the expected queue/runtime benefit justifies repeated setup and extra concurrent allocation. Model: CHE = sum(billing_rate * actual_runtime_hours); elapsed completion = queue + critical-path runtime. No universal optimum without these measurements and a wait-time preference.
4. Within the allocated worker, group similar token lengths, try a bounded set of batches and select measured lower seconds per completed mask. Larger VRAM permits larger batches but does not establish a speedup. Keep masks, page resolution, target strings and positions unchanged.
5. Persist every mask, source pin, calibration, fallback reason and job receipt. Check numerical agreement separately from throughput. Smaller near-tie score differences are not reliable training labels merely because an engineering tolerance passed.
6. Compare realized CHE per completed question/mask and end-to-end turnaround after each run. Stop tuning when its overhead exceeds plausible savings. Keep any unproven cross-GPU optimum labeled as a hypothesis.

## Current bounded implementation

`experiments/omp10/batching.py`: independent full-prefix scoring, padded masks of one question, original multimodal positions, aligned DeepStack streams. Trials1/2/4 use the two longest masks, stop on first slowdown (<5% improvement), >.05 G/S difference, predicted memory pressure or OOM. Uses live free memory; no GPU-name table. Explicit fallback to the last valid size, including1; production OOM halves the batch. Calibration repeated per new question is a bounded validation expense, not extra OMP supervision. Cap4 is a tuning-cost choice for this220-mask pilot, not a claim that larger batches are inefficient.

`run.py --smoke-then-run` keeps the loaded model across the two-case smoke and production. It requires both smoke cases to pass first, and saves separate immutable output under `omp10-20260917-batch-v1`. Failed original shared-KV smoke remains preserved under `omp10-20260917-v1`.

Queue comparison at12:04 Arizona:5/10/15min genericGPU requests all predicted immediate start onsg001. Ten minutes is a short resumable initial envelope; no queue evidence favored5min. Submission and final accounting receipts must be added once available.

Live minimum confirmed: `sbatch --test-only --mem=16000M` was rejected with “GPU jobs require at least24000MB of memory per node”. Keep24000M for GPU jobs; do not treat the partition default24576 as the exact enforced minimum.

Combined job63521070 submitted with code8d62af18833f86a3afda08e914e6b6e4e14fa60d, GPU1/CPU2/24000M/10min, generic compatible GPU. Exact launch receipt on scratch under `stage/omp10-20260917-batch-v1/launch.json`. Initial pending reasonPriority; test-only immediate estimates are not guaranteed starts. CPU preparation allocation63520988 explicitly released after submission. Runtime results pending; do not claim measured batching speedup yet.

Recovery lesson: job63521070 completed both smoke cases and6/10 banks, then hit duplicate semantic IDs because two admitted positions shared a pixel-cache key. CPU preflight should validate page-occurrence identity across the entire cohort before GPU use, not only the smoke subset. Distinguish cache deduplication from independent positions in a reader prompt. The recovery preserves132 completed calls with hashed parent provenance and performs only88 remaining calls; no repeated smoke when all scoring code and smoke layouts are unchanged. Actual first job5:04 at billing35 ~=2.956CHE; failed original smoke2.207CHE is a separate development cost. Batching trials observed so far selected1; no speedup claimed.
