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

Continuation63522358 submitted at commit5f8bbdf1a714608eb2d4406bd5e32435691b28e4: genericGPU1,CPU2,24000M,5min,htc/public, outputs batch-v2. Test-only estimates changed sharply: HTC5min13:49, HTC3min14:18, ordinarypublic and general/private several days later, A10040 slice also days later. Public/private was rejected as an invalid partition/QoS combination; the documented preemptible route is general/private. No jobs were allocated by those tests. Crucially the actual generic HTC continuation started promptly onsg004 despite the pessimistic estimate. Do not churn/re-submit jobs solely on forecasts; compare a few reasonable options and inspect the actual job. Preparation allocation63522281 explicitly released.

Thread heartbeat `monitor-ten-question-sol-run` checks every5min until completion/blocker. It must preserve existing132 scores, verify total220 and10 OMP fits, and record final accounting before pausing. At this checkpoint the continuation is running; results are not yet complete.

## Verified completion — 2026-09-17

Continuation63522358 COMPLETED on sg004 (A100-SXM4-80GB), elapsed3:18, AllocTRES billing35, MaxRSS13182740KiB (~12.57GiB), charged-equivalent estimate1.925CHE. Both production-bearing jobs together used8:22 and4.881CHE; original failed smoke adds2.207CHE (development total~7.088CHE, excludes small CPU preparations). Requested GPU1/CPU2/24000M/5min. No extra GPU job remains necessary.

Final receipt `omp10-20260917-batch-v2/runs/shard-0-of-1.json` is complete, contract00e5ab72b4701c2031384e207e6abe1308d82e3df14e52529391c5fe28a0d7f4, with ten distinct questions each22 measurements. Six hashed reuse receipts preserve132 measurements and their OMP fits from batch-v1; four new OMP files cover88 new measurements. Verified receipt and file listing through browser shell. Visual interpretation of nominated regions is still pending.

All four continuation questions selected batch1 after batch2 numerical rejection. Shakespeare calibration: B1 1.1015s/mask, B2 1.6340s/mask, max G/S difference0.20165. These rejected batch2 scores did not enter the production bank. This is evidence against unconditionally treating batching differences as harmless, not a diagnosis of their cause. Keep full-prefix reference scores; the original0.03261 shared-KV deviation was not adopted as the production path.

Practical conclusion for this small pilot: generic compatible GPU, one loaded worker, short resumable wall time, measured batch selection, enforced minimum host RAM, and preservation of completed scores. Queue estimates proved pessimistic; no universal cross-GPU or queue/fairshare optimum has been demonstrated. Pause the monitoring heartbeat after this completion check.

## Batching slowdown: read-only code audit, 2026-09-17

No additional GPU run or production-code change in this audit. Inspected experiments/omp10/batching.py, src/docprune/stage2/answerer.py, src/docprune/stage2/qwen.py, run.py and pinned Transformers4.57.3 upstream masking_utils.py/Qwen3-VL source.

Verified: B1 dispatches to independent likelihood() calls; B2 assembles a left-padded dense batch. Both use full-prefix G then S, no KV reuse, frozen BF16 SDPA. Both project only answer-prediction positions through lm_head. Vision features are cached before mask calls, so repeated image encoding is not the batching explanation. B2 assembles prefixes once for both targets, while B1 assembles again for each target: prefix reuse itself favors B2, not a reason it must be slower.

Leading hypothesis: unequal lengths introduce zeros in the attention mask. Pinned Transformers _ignore_causal_mask_sdpa permits the causal fast path for all-ones masks but not padded masks; its SDPA mask builder then builds an explicit 4D mask. This proves a conditional mask-construction difference, NOT the selected CUDA kernel or its measured cost. Source: https://raw.githubusercontent.com/huggingface/transformers/v4.57.3/src/transformers/masking_utils.py . Qwen3-VL passes the supplied mask through create_causal_mask. Padding also adds dense work; actual overhead needs saved length pairs, not guesses. Do not remove padding masks to regain speed, as that changes valid attention.

Other verified limitations: calibration times one trial per size in fixed B1->B2 order, without shape-specific warm-ups or repeats; it synchronizes correctly around elapsed time. Stops at first slowdown/parity failure, so does not establish global batch optimum. Packing allocates tensors, copies rows and concatenates DeepStack streams; host synchronizations occur during validation and score extraction. Their relative cost is unmeasured. GPU peak counters reset per question AND trial, so final stats are not whole-job peaks. Tiny CPU float32 parity tests do not establish BF16 CUDA equivalence.

Proposed bounded diagnostic, before any optimization: one loaded model, one existing question/two frozen masks; compare (A) two sequential B1 calls, (B) B2 duplicate long mask with no padding, (C) B2 actual unequal masks, (D) B1 short mask explicitly padded to the long length through the same padded scorer. D needs a diagnostic-only route because production currently delegates len1. Preserve all real multimodal positions, attention exclusion and G/S targets. Warm up each shape once; use three interleaved timed repeats, synchronized end-to-end timing and per-phase CUDA events. Capture one short CPU/CUDA profiler trace per condition outside timing repeats, identifying attention kernels, packing/copies, decoder and answer head. Record actual lengths/padding ratio, per-target/token score errors, full-process and per-condition memory, GPU model and library versions. No new dataset questions, no full-bank rerun, no batch4/8 sweep initially. GPU execution is needed to confirm kernel dispatch, numerical drift and actual speed; CPU inspection alone cannot do that. If padding is implicated, evaluate length buckets or a separately validated padding-aware backend; if packing dominates, reuse invariant preparation; if only initial-shape overhead dominates, repair calibration methodology. Preserve existing production scores regardless.
