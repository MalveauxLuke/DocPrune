#!/usr/bin/env bash
#SBATCH --job-name=dp-omp10-batch
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24000M
#SBATCH --time=00:10:00
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
DP_OMP_REVIEW="$DP_OMP_ROOT/omp10-20260917-v1/review/frozen.json"
mkdir -p "$DP_OMP_ROOT/omp10-20260917-batch-v1/locks"
exec 9>"$DP_OMP_ROOT/omp10-20260917-batch-v1/locks/${DP_OMP_SMOKE:-1}-${SLURM_ARRAY_TASK_ID:-0}.lock"
flock -n 9 || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv
if [ "${DP_OMP_COMBINED:-0}" = 1 ]; then
 exec "$DP_ENV/bin/python" -u experiments/omp10/run.py --root "$DP_OMP_ROOT" --review "$DP_OMP_REVIEW" --snapshot "$DP_SNAPSHOT" --smoke-then-run
elif [ "${DP_OMP_SMOKE:-1}" = 1 ]; then
 exec "$DP_ENV/bin/python" -u experiments/omp10/run.py --root "$DP_OMP_ROOT" --review "$DP_OMP_REVIEW" --snapshot "$DP_SNAPSHOT" --smoke
else
 : "${DP_OMP_SHARDS:?Set measured shard count}"
 exec "$DP_ENV/bin/python" -u experiments/omp10/run.py --root "$DP_OMP_ROOT" --review "$DP_OMP_REVIEW" --snapshot "$DP_SNAPSHOT" --shard "${SLURM_ARRAY_TASK_ID:?}" --shards "$DP_OMP_SHARDS"
fi
