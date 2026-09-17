# Ten-error Qwen3 masking and OMP audit

Owner approved steps 1–3 on 2026-09-17: review/freeze ten genuine errors (3 text, 3 table, 4 image), two-case masked-scoring smoke, then 22 independent masks per question and OMP/visual inspection. No three-check/seven-follow-up expansion, selector training or new retrieval is authorized here.

## Current executable stage: freeze reviewed metadata, then two-case smoke

Checkout: /home/lmalveau/DocPrune. Existing baseline revision: 04b0d49e24b754a0fca55cd590051d8ab98c5930. Read-only baseline/catalog inspection uses this recorded baseline revision; new masking code must be tested and pinned separately before GPU submission.
Inputs: /scratch/lmalveau/docprune-m3doc600/20260915-v2/admitted471-v2/stage. Canonical baseline: baseline-qwen3-8b-admitted-v2; contract ad208c8f709e475507835f2f87054830c6d5f3dfaca962c5ffd805fbfd40f364. Reuse frozen pool, presentation order and existing images/layouts.
Environment for metadata-only preparation: system Python3 in a CPU allocation. Browser shell: check hostname and SLURM_JOB_ID first. Request salloc -p lightwork -q public -c 1 --mem=1G -t 00:10:00 for brief metadata preparation/transfer; no inference or sustained compute there. Outputs: stage/omp10-20260917-v1/review (new directory only). Command scope: read assessment/catalog, emit deterministic candidate-review metadata; no source mutation or deletion. If allocation expires, renew only the necessary small allocation and resume existing preparation. Preserve unrelated jobs.

## GPU stage: smoke gated

Use existing /scratch/lmalveau/docprune-initial300/20260915-v1/envs/qwen3-baseline and Qwen3-VL-8B-Instruct revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b. No download. Preserve BF16 SDPA, canonical prompt, processor limits and original self answer. Map stable regions to original reader tokens; never substitute visual blackout for token deletion.

Runner: experiments/omp10/run.py; deterministic mask/fit implementation: discovery.py. Five focused tests cover independent masks, signed sparse recovery, quiet data, invalid banks and shared-prefix target isolation; three existing bridge tests also passed. The reviewed ten comprise 3 TextQ, 3 TableQ, 4 ImageQ, five original-four and five supplemented. Frozen IDs/evidence/answer-file hashes are written by experiments/omp10/freeze.py from the transferred review-decisions.json; no mask outcome informed selection. The two smoke cases are the largest original visual-token context and the largest context of a different modality.

Exact executable pin: require DP_OMP_COMMIT equal to the tested/pushed Git revision, recorded with the freeze SHA and sbatch receipt in stage/omp10-20260917-v1/launch.json before submission. The launcher refuses a different HEAD or modified runtime source. This receipt binds the concrete hash without embedding a self-referential commit hash in source.

In the CPU allocation, pull the tested commit and run:
python3 experiments/omp10/freeze.py --root /scratch/lmalveau/docprune-m3doc600/20260915-v2/admitted471-v2/stage --decisions /scratch/lmalveau/docprune-m3doc600/20260915-v2/admitted471-v2/stage/omp10-20260917-v1/review/review-decisions.json

Smoke command: export DP_OMP_COMMIT=<tested full SHA>; sbatch --exclude=sg048,sg049 --output=<scratch>/omp10-20260917-v1/logs/smoke-%j.log sol/omp10/run.sbatch. Check with sbatch --test-only first. Generic compatible GPU1, CPU2, 24000M host RAM and 10min, batch1; exclude only measured-insufficient20GB MIG. Current partition defaults 24576MB/GPU; verify acceptance of the prior measured24000M request. Preserve all admitted pages and original tokens. No full-corpus retrieval/download/install.

Smoke tests physical deletion in every DeepStack stream, native all-keep parity, repeated G/S and shared-prefill versus independent-continuation scoring. Score-parity absolute tolerance .02, repeat tolerance1e-5, native logits atol.02/rtol.005 plus same top1. These are engineering gates, not significance thresholds. Production requires the matching passed smoke receipt. G scores the complete gold answer (joint JSON list for list questions), S the original generated continuation excluding terminal EOS. Bernoulli(.5) region bits yield half the tokens in expectation, not a per-mask token quota.

Production: DP_OMP_SMOKE=0 DP_OMP_SHARDS=<measured count> sbatch --array=0-<count-minus-one>%<concurrency> --time=<measured> --exclude=sg048,sg049 --output=<scratch>/omp10-20260917-v1/logs/run-%A_%a.log sol/omp10/run.sbatch. Size sharding/time from the completed smoke, amortizing one model load and one vision encoding per question; do not reserve maximum concurrency without a measured reason. Record concrete resources in launch.json. Outputs: contract.json, question banks, original-context anchors, 22 per-question measurements, OMP hypotheses, smoke and run receipts. Every mask is durable/resumable; stop on contract failure, preserve results and retry only missing work. No automatic context reduction or mask repair on OOM. Increase resources only after measured failure. No further adaptive calls or training.
