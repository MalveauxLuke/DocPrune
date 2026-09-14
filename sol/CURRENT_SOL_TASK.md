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
