#!/usr/bin/env bash
#SBATCH --job-name=dp300-pages
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/lmalveau/docprune-initial300/20260915-v1/logs/prepare-%j.out
#SBATCH --error=/scratch/lmalveau/docprune-initial300/20260915-v1/logs/prepare-%j.err
set -euo pipefail
cd /home/lmalveau/DocPrune
source sol/initial300/environment.sh
(cd "$DP300_CODE" && sha256sum -c CODE_SHA256SUMS)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
(cd "$DP300_ROOT/input" && sha256sum -c TRANSFER_SHA256SUMS > "$DP300_ROOT/logs/input-checksums-${SLURM_JOB_ID}.txt")
(cd "$DP300_CODE" && "$DP300_MINER_PY" -m unittest -q test_common)
/usr/bin/time -v "$DP300_MINER_PY" "$DP300_CODE/prepare.py" --root "$DP300_ROOT"
