#!/usr/bin/env bash
#SBATCH --job-name=dp600-image-match
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/lmalveau/docprune-m3doc600/20260915-v2/logs/%x-%j.out
#SBATCH --error=/scratch/lmalveau/docprune-m3doc600/20260915-v2/logs/%x-%j.err
set -euo pipefail
cd /home/lmalveau/DocPrune
source sol/m3doc600/environment.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec /usr/bin/time -v "$DP600_MINER_PY" experiments/m3doc600/match_images.py --root "$DP600_ROOT" --corpus /scratch/lmalveau/docprune/datasets/m3docvqa
