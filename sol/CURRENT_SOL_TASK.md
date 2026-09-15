# Active — 17-question, three-arm acquisition on SOL

Owner authorized 2026-09-14: sharded scoring on SOL and current-execution
baselines instead of historical likelihood equality. Binding handoff:
[adaptive-acquisition/SCORING.md](adaptive-acquisition/SCORING.md).

Checkout /home/lmalveau/DocPrune, main at submitted ACQUISITION_CODE_COMMIT.
The launcher enforces the exact commit and a clean tracked checkout.
Environment /home/lmalveau/mamba-envs/docprune-acquisition-sol/bin/python.
Inputs /scratch/lmalveau/docprune-adaptive-acquisition/20260914-smoke01/inputs.
Model and caches remain in that existing scratch root; source environment.sh.
Outputs /scratch/lmalveau/docprune-adaptive-acquisition/20260914-smoke01/score-ARRAY_ID/shard-N.
Command: sbatch --export=ALL,ACQUISITION_CODE_COMMIT=<tested-commit> sol/adaptive-acquisition/score.sbatch.
Array 1-17%4: Q01-Q17, all three arms per shard, 32 observations per arm.
Each shard: one compatible whole GPU >=23000 MiB, 2 CPUs, 24000 MiB RAM,
20 minutes, htc/public. No new lightwork allocation or model download needed.
Preserve partial outputs. Retry only failed/incomplete shards at the same code,
runtime and baseline identity. No changed tolerances or silent hardware resume.
Scope: likelihood acquisition and per-question summaries, not selector training
or a claim that decoded-candidate assessment is complete.


## Failure reconciliation authorized 2026-09-14

Array 63243809 completed 14 questions; Q03/Q15 stopped on historical generation
identity and Q17 stopped before loading the reader. Preserve all original outputs.
Retry only array tasks 3,15,17 in fresh output directories at the newly tested
commit. The scheduler check now explicitly selects ARRAY_ID_TASK_ID and rejects
ambiguous/mismatched records. Generation text, tokens, EOS and cached parity
are printed before the existing generation gate. Q03/Q15 may stop again; those
replays recover evidence missing from the first logs, not permission to silently
change the S target or ignore real answer changes. Q17 executes full scoring if
its normal question guard succeeds. Resources remain one compatible GPU,
24000 MiB RAM, two CPUs, 20 minutes per shard; up to three simultaneous retries.
The production code pin remains unchanged for the 14 completed questions.
