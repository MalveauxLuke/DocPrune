#!/usr/bin/env bash
#SBATCH --job-name=dp600-production
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24000M
#SBATCH --time=00:25:00

# Submission example (the log directory must already exist):
#   export DP600_LOG_ROOT=/scratch/lmalveau/docprune-m3doc600/20260915-v2/logs/production
#   export DP600_COMMIT="$(git rev-parse HEAD)"
#   export DP600_STAGE=/scratch/lmalveau/docprune-m3doc600/20260915-v2/admitted471-v1/stage
#   export DP600_ENGINE=colqwen
#   export DP600_SHARDS=8
#   mkdir -p "$DP600_LOG_ROOT"
#   sbatch --array="0-$((DP600_SHARDS - 1))" \
#     --output="$DP600_LOG_ROOT/%x-%A_%a.out" \
#     --error="$DP600_LOG_ROOT/%x-%A_%a.err" \
#     --export=ALL \
#     sol/m3doc600/production.sbatch
# Use a separately named log root and the same array/shard settings for mineru.

set -euo pipefail

cd /home/lmalveau/DocPrune
source sol/m3doc600/environment.sh

: "${DP600_STAGE:?Production stage root containing catalog.json is required}"
: "${DP600_ENGINE:?Engine must be colqwen or mineru}"
: "${DP600_SHARDS:?Positive shard count is required}"
: "${SLURM_ARRAY_TASK_ID:?Submit as a zero-based Slurm array}"
: "${SLURM_ARRAY_TASK_COUNT:?Slurm array metadata is required}"
: "${SLURM_ARRAY_TASK_MIN:?Slurm array metadata is required}"
: "${SLURM_ARRAY_TASK_MAX:?Slurm array metadata is required}"
: "${SLURM_ARRAY_TASK_STEP:?Slurm array metadata is required}"

if [[ ! "$DP600_SHARDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "DP600_SHARDS must be a positive integer: $DP600_SHARDS" >&2
  exit 2
fi
if (( SLURM_ARRAY_TASK_MIN != 0 ||
      SLURM_ARRAY_TASK_MAX != DP600_SHARDS - 1 ||
      SLURM_ARRAY_TASK_STEP != 1 ||
      SLURM_ARRAY_TASK_COUNT != DP600_SHARDS ||
      SLURM_ARRAY_TASK_ID < 0 ||
      SLURM_ARRAY_TASK_ID >= DP600_SHARDS )); then
  echo "Array must be exactly 0-$((DP600_SHARDS - 1)) for DP600_SHARDS=$DP600_SHARDS" >&2
  exit 2
fi
if [[ "$DP600_STAGE" != /* || "$DP600_STAGE" != "$DP600_ROOT"/* ]]; then
  echo "DP600_STAGE must be an absolute child of DP600_ROOT ($DP600_ROOT): $DP600_STAGE" >&2
  exit 2
fi
if [[ ! -f "$DP600_STAGE/catalog.json" ]]; then
  echo "Missing production catalog: $DP600_STAGE/catalog.json" >&2
  exit 2
fi

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

case "$DP600_ENGINE" in
  colqwen)
    exec /usr/bin/time -v "$DP600_COL_PY" experiments/m3doc600/colqwen.py \
      --root "$DP600_STAGE" \
      --base "$DP600_BASE" \
      --adapter "$DP600_ADAPTER" \
      --batch 4 \
      --shard "$SLURM_ARRAY_TASK_ID" \
      --shards "$DP600_SHARDS"
    ;;
  mineru)
    exec /usr/bin/time -v "$DP600_MINER_PY" experiments/m3doc600/extract_layout.py \
      --root "$DP600_STAGE" \
      --model "$DP600_MINER_MODEL" \
      --batch 16 \
      --shard "$SLURM_ARRAY_TASK_ID" \
      --shards "$DP600_SHARDS"
    ;;
  *)
    echo "DP600_ENGINE must be colqwen or mineru: $DP600_ENGINE" >&2
    exit 2
    ;;
esac
