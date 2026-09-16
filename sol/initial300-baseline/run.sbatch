#!/usr/bin/env bash
#SBATCH --job-name=dp300-q3-baseline
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24000M
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/lmalveau/docprune-initial300/20260915-v1/logs/%x-%A-%a.out
#SBATCH --error=/scratch/lmalveau/docprune-initial300/20260915-v1/logs/%x-%A-%a.err
set -euo pipefail
cd /home/lmalveau/DocPrune
source sol/initial300-baseline/environment.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
if [ "${DP_BASELINE_SMOKE:-0}" = 1 ]; then
  "$DP_ENV/bin/python" -u experiments/initial300/baseline.py --root "$DP_ROOT" --snapshot "$DP_SNAPSHOT" --smoke
else
  : "${DP_BASELINE_SHARDS:?Supply shard count}"
  "$DP_ENV/bin/python" -u experiments/initial300/baseline.py --root "$DP_ROOT" --snapshot "$DP_SNAPSHOT" --shard "$SLURM_ARRAY_TASK_ID" --shards "$DP_BASELINE_SHARDS"
fi
