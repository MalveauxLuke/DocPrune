#!/bin/bash
#SBATCH --job-name=docprune-acq-score
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --array=1-17%4
#SBATCH --gres=gpu:1
#SBATCH --constraint="a30|a100_40|a100_80|l40|l40s|h100"
#SBATCH --cpus-per-task=2
#SBATCH --mem=24000M
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/lmalveau/docprune-adaptive-acquisition/20260914-smoke01/score-%A_%a.log
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
test "$SLURM_ARRAY_TASK_ID" -ge 1
test "$SLURM_ARRAY_TASK_ID" -le 17
printf -v ACQUISITION_CASE 'Q%02d' "$SLURM_ARRAY_TASK_ID"
ACQUISITION_SCORE_RUN="${ACQUISITION_SCORE_RUN:-$ACQUISITION_SOL_ROOT/score-$SLURM_ARRAY_JOB_ID}"
"$ACQUISITION_SOL_PYTHON" -B scripts/stage0_acquisition.py score --platform sol --execute-gpu --cases "$ACQUISITION_CASE" --package "$ACQUISITION_SOL_ROOT/inputs" --snapshot "$ACQUISITION_SOL_SNAPSHOT" --output "$ACQUISITION_SCORE_RUN/shard-$SLURM_ARRAY_TASK_ID"
