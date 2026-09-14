#!/bin/bash
#SBATCH --job-name=docprune-parity-q12
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100_40|a100_80|l40|l40s|h100"
#SBATCH --cpus-per-task=2
#SBATCH --mem=24000M
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/lmalveau/docprune-adaptive-acquisition/20260914-smoke01/parity-%j.log
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
"$ACQUISITION_SOL_PYTHON" -B scripts/diagnose_acquisition_parity.py --execute-gpu --package "$ACQUISITION_SOL_ROOT/inputs" --snapshot "$ACQUISITION_SOL_SNAPSHOT" --output "$ACQUISITION_SOL_ROOT/parity-$SLURM_JOB_ID"
