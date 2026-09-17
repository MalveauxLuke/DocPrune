#!/usr/bin/env bash
#SBATCH --job-name=dp-varlen-prof
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24000M
#SBATCH --time=00:05:00
set -euo pipefail
: "${SLURM_JOB_ID:?Allocation required}" "${DP_OMP_COMMIT:?Exact tested revision required}"
case "$(hostname -s)" in *login*) exit 2;; esac
cd /home/lmalveau/DocPrune
test "$(git rev-parse HEAD)" = "$DP_OMP_COMMIT"
git diff --exit-code HEAD -- experiments/omp10 experiments/m3doc600 src/docprune/stage2 sol/omp10
export DP_BASELINE_COMMIT="$DP_OMP_COMMIT"
source sol/initial300-baseline/environment.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DP_OMP_ROOT=/scratch/lmalveau/docprune-m3doc600/20260915-v2/admitted471-v2/stage
exec "$DP_ENV/bin/python" -u experiments/omp10/profile_padding.py --root "$DP_OMP_ROOT" --snapshot "$DP_SNAPSHOT" --out "$DP_OMP_ROOT/varlen-profile-$SLURM_JOB_ID" --varlen
