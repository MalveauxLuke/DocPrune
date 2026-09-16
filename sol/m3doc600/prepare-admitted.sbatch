#!/usr/bin/env bash
#SBATCH --job-name=dp471-prepare
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
set -euo pipefail
cd /home/lmalveau/DocPrune
source sol/m3doc600/environment.sh
root="$DP600_ROOT/admitted471-v2"
exec "$DP600_MINER_PY" experiments/m3doc600/prepare_admitted.py \
  --pool "$root/pool.json" \
  --corpus /scratch/lmalveau/docprune/datasets/m3docvqa \
  --out "$root/stage" \
  --reuse-catalog "$DP600_ROOT/retrieved-stage/catalog.json"
