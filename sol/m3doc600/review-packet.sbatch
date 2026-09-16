#!/usr/bin/env bash
#SBATCH --job-name=dp600-review
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/lmalveau/docprune-m3doc600/20260915-v2/logs/%x-%j.out
#SBATCH --error=/scratch/lmalveau/docprune-m3doc600/20260915-v2/logs/%x-%j.err
set -euo pipefail
cd /home/lmalveau/DocPrune
source sol/m3doc600/environment.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
args=()
if [[ "${DP600_PACKET_SCOPE:-calibration}" == all ]]; then args+=(--all); fi
exec /usr/bin/python3 experiments/m3doc600/build_review_packets.py --root "$DP600_ROOT" --corpus /scratch/lmalveau/docprune/datasets/m3docvqa "${args[@]}"
