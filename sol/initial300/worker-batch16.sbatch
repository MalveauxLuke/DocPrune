#!/usr/bin/env bash
#SBATCH --job-name=dp300-worker
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24000M
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/lmalveau/docprune-initial300/20260915-v1/logs/%x-%A-%a.out
#SBATCH --error=/scratch/lmalveau/docprune-initial300/20260915-v1/logs/%x-%A-%a.err
set -euo pipefail
cd /home/lmalveau/DocPrune
source sol/initial300/environment.sh
: "${DP300_ENGINE:?Choose colqwen or mineru}"
: "${DP300_SHARDS:?Specify the exact shard count}"
: "${DP300_SMOKE_RECEIPT:?Provide inspected smoke admission receipt}"
"$DP300_MINER_PY" "$DP300_CODE/admit.py" --root "$DP300_ROOT" --check "$DP300_SMOKE_RECEIPT"
(cd "$DP300_CODE" && sha256sum -c CODE_SHA256SUMS)
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
test "$DP300_ENGINE" = mineru
exec /usr/bin/time -v "$DP300_MINER_PY" "$DP300_CODE/extract_layout_batch16.py" --root "$DP300_ROOT" --model "$DP300_MINER_MODEL" --batch 16 --shard "$SLURM_ARRAY_TASK_ID" --shards 8
