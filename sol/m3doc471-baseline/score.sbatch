#!/usr/bin/env bash
#SBATCH --job-name=dp471-q3-score
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:03:00
set -euo pipefail
cd /home/lmalveau/DocPrune
source sol/initial300-baseline/environment.sh
exec "$DP_ENV/bin/python" experiments/m3doc600/score_baseline.py --root /scratch/lmalveau/docprune-m3doc600/20260915-v2/admitted471-v2/stage
