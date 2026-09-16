#!/usr/bin/env bash
#SBATCH --job-name=dp600-render
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/lmalveau/docprune-m3doc600/20260915-v1/logs/%x-%j.out
#SBATCH --error=/scratch/lmalveau/docprune-m3doc600/20260915-v1/logs/%x-%j.err
set -euo pipefail
cd /home/lmalveau/DocPrune
source sol/m3doc600/environment.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec /usr/bin/time -v "$DP600_MINER_PY" experiments/m3doc600/prepare_smoke.py --pool "$DP600_ROOT/cohort/pool.json" --corpus /scratch/lmalveau/docprune/datasets/m3docvqa --out "$DP600_ROOT/retrieved-stage" --all-retrieved
