#!/usr/bin/env bash
#SBATCH --job-name=dp471-combine
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
set -euo pipefail
cd /home/lmalveau/DocPrune
source sol/m3doc600/environment.sh
exec "$DP600_COL_PY" experiments/m3doc600/combine.py \
  --root "$DP600_ROOT/admitted471-v2/stage"
