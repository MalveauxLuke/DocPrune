#!/bin/bash
#SBATCH --job-name=docprune-acq-smoke
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/lmalveau/docprune-adaptive-acquisition/20260914-smoke01/smoke-%j.log
set -euo pipefail
cd /home/lmalveau/DocPrune
source sol/adaptive-acquisition/environment.sh
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PWD/src"
export LD_LIBRARY_PATH="/home/lmalveau/mamba-envs/docprune-sol/lib:/home/lmalveau/mamba-envs/docprune-sol/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
test -n "${ACQUISITION_CODE_COMMIT:-}"
test "$(git rev-parse HEAD)" = "$ACQUISITION_CODE_COMMIT"
test -z "$(git status --porcelain --untracked-files=no)"
"$ACQUISITION_SOL_PYTHON" -B scripts/stage0_acquisition.py doctor --package "$ACQUISITION_SOL_ROOT/inputs" --snapshot "$ACQUISITION_SOL_SNAPSHOT"
"$ACQUISITION_SOL_PYTHON" -B scripts/stage0_acquisition.py smoke --platform sol --execute-gpu --cases Q12 --package "$ACQUISITION_SOL_ROOT/inputs" --snapshot "$ACQUISITION_SOL_SNAPSHOT" --output "$ACQUISITION_SOL_ROOT/smoke-$SLURM_JOB_ID"
