# Active — one-question adaptive acquisition smoke on SOL

Owner authorized 2026-09-14. Binding handoff:
[adaptive-acquisition/HANDOFF.md](adaptive-acquisition/HANDOFF.md).

Checkout /home/lmalveau/DocPrune, main. The submitted ACQUISITION_CODE_COMMIT
pins the exact tested commit; smoke.sbatch verifies it and a clean tracked tree.
Environment: /home/lmalveau/mamba-envs/docprune-acquisition-sol/bin/python.
Inputs: /scratch/lmalveau/docprune-adaptive-acquisition/20260914-smoke01/inputs.
Scratch/output root: /scratch/lmalveau/docprune-adaptive-acquisition/20260914-smoke01.
Resources/command: one H100, 2 CPUs, 48 GiB, 20 min, htc/public;
sbatch --export=ALL,ACQUISITION_CODE_COMMIT=<verified-commit> sol/adaptive-acquisition/smoke.sbatch.
Reuse browser lightwork for setup; stage only the sealed input package and exact
model snapshot. Scope ends after Q12 smoke and receipt review. No full scoring.
Preserve failed outputs; only narrow contract-preserving fixes/retries permitted.

The prior Colfeatures17 task is complete. Its historical handoff remains in
Git history and docs/experiments/corrective-selection/COLFEATURES17.md.
