#!/usr/bin/env bash
#SBATCH --job-name=dp471-q3-baseline
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24000M
#SBATCH --time=00:10:00
set -euo pipefail
cd /home/lmalveau/DocPrune
source sol/initial300-baseline/environment.sh
export DP_ROOT=/scratch/lmalveau/docprune-m3doc600/20260915-v2/admitted471-v2/stage
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
git diff --exit-code HEAD -- experiments/m3doc600 sol/m3doc471-baseline
# Hold one writer per smoke or production shard; duplicate submissions exit early.
mkdir -p "$DP_ROOT/baseline-locks"
exec 9>"$DP_ROOT/baseline-locks/${DP_BASELINE_SMOKE:-0}-${SLURM_ARRAY_TASK_ID:-smoke}.lock"
flock -n 9 || { echo 'Another identical baseline worker owns this shard'; exit 1; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
if [ "${DP_BASELINE_SMOKE:-0}" = 1 ]; then
 exec "$DP_ENV/bin/python" -u experiments/m3doc600/baseline.py --root "$DP_ROOT" --snapshot "$DP_SNAPSHOT" --smoke
else
 : "${DP_BASELINE_SHARDS:?Supply shard count}"
 exec "$DP_ENV/bin/python" -u experiments/m3doc600/baseline.py --root "$DP_ROOT" --snapshot "$DP_SNAPSHOT" --shard "$SLURM_ARRAY_TASK_ID" --shards "$DP_BASELINE_SHARDS"
fi
