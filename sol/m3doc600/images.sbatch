#!/usr/bin/env bash
#SBATCH --job-name=dp600-images
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/lmalveau/docprune-m3doc600/20260915-v2/logs/%x-%j.out
#SBATCH --error=/scratch/lmalveau/docprune-m3doc600/20260915-v2/logs/%x-%j.err
set -euo pipefail
cd /home/lmalveau/DocPrune
source sol/m3doc600/environment.sh
exec "$DP600_MINER_PY" experiments/m3doc600/fetch_images.py --root "$DP600_ROOT"
