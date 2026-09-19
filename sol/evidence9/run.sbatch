#!/usr/bin/env bash
#SBATCH --job-name=dp-evidence9
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24000M
#SBATCH --time=00:10:00
#SBATCH --exclude=sg048,sg049,sg050
set -euo pipefail
: "${SLURM_JOB_ID:?}" "${DP_BASELINE_COMMIT:?}"
case "$(hostname -s)" in *login*) exit 2;; esac
cd /home/lmalveau/DocPrune
test "$(git rev-parse HEAD)" = "$DP_BASELINE_COMMIT"
git diff --exit-code HEAD -- experiments/evidence9 src/docprune/stage2/answerer.py src/docprune/stage2/qwen.py experiments/selector_smoke/run.py
source sol/initial300-baseline/environment.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DP_STAGE=/scratch/lmalveau/docprune-m3doc600/20260915-v2/admitted471-v2/stage
DP_OUT="$DP_STAGE/evidence9-20260919-v1"
mkdir -p "$DP_OUT"
exec 9>"$DP_OUT/run.lock"
flock -n 9 || exit 1
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total --format=csv -l 2 >"$DP_OUT/device-observations-$SLURM_JOB_ID.csv" &
DP_MONITOR=$!
trap 'kill "$DP_MONITOR" 2>/dev/null || true' EXIT
"$DP_ENV/bin/python" -u experiments/evidence9/run.py --root "$DP_STAGE" --output "$DP_OUT" --reader "$DP_SNAPSHOT"
