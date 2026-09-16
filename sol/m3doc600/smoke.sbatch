#!/usr/bin/env bash
#SBATCH --job-name=dp600-smoke
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24000M
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/lmalveau/docprune-m3doc600/20260915-v1/logs/%x-%j.out
#SBATCH --error=/scratch/lmalveau/docprune-m3doc600/20260915-v1/logs/%x-%j.err
set -euo pipefail
cd /home/lmalveau/DocPrune
source sol/m3doc600/environment.sh
: "${DP600_ENGINE:?colqwen or mineru}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
if [[ "$DP600_ENGINE" == colqwen ]]; then
 for batch in 4 8 16; do
  /usr/bin/time -v "$DP600_COL_PY" experiments/m3doc600/colqwen.py --root "$DP600_ROOT/smoke/colqwen-b$batch" --base "$DP600_BASE" --adapter "$DP600_ADAPTER" --batch "$batch"
 done
elif [[ "$DP600_ENGINE" == mineru ]]; then
 for batch in 8 16; do
  /usr/bin/time -v "$DP600_MINER_PY" experiments/m3doc600/extract_layout.py --root "$DP600_ROOT/smoke/mineru-b$batch" --model "$DP600_MINER_MODEL" --batch "$batch"
 done
else exit 2; fi
