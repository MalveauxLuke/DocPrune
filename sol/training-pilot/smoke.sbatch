#!/usr/bin/env bash
#SBATCH --job-name=dp-api-pilot-smoke
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24000M
#SBATCH --time=00:10:00
set -euo pipefail
: "${SLURM_JOB_ID:?}" "${DP_PILOT_COMMIT:?}"
case "$(hostname -s)" in *login*) exit 2;; esac
cd /home/lmalveau/DocPrune
test "$(git rev-parse HEAD)" = "$DP_PILOT_COMMIT"
git diff --exit-code HEAD -- experiments/training_pilot src/docprune/stage2 src/docprune/answer_adjudication.py sol/training-pilot
export DP_BASELINE_COMMIT="$DP_PILOT_COMMIT"
source sol/initial300-baseline/environment.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="/scratch/lmalveau/docprune-m3doc600/20260915-v2/admitted471-v2/selector-smoke-env/peft-0.18.0${PYTHONPATH:+:$PYTHONPATH}"
DP_STAGE=/scratch/lmalveau/docprune-m3doc600/20260915-v2/admitted471-v2/stage
DP_OUT="$DP_STAGE/training-pilot-smoke-v1"
DP_SELECTOR=/scratch/lmalveau/hf_cache/hub/models--Qwen--Qwen3-VL-Reranker-2B/snapshots/4bd860ac4f15ad1897a214615cccc700f8f71818
mkdir -p "$DP_OUT"
exec 9>"$DP_OUT/run.lock"
flock -n 9 || exit 1
DP_GPU_UUID=$("$DP_ENV/bin/python" -c 'import torch; u=str(torch.cuda.get_device_properties(0).uuid); print(u if u.startswith("GPU-") else "GPU-"+u)')
nvidia-smi -i "$DP_GPU_UUID" --query-gpu=timestamp,uuid,name,utilization.gpu,memory.used,memory.total --format=csv -l 1 >"$DP_OUT/utilization-$SLURM_JOB_ID.csv" &
DP_MONITOR=$!
trap 'kill "$DP_MONITOR" 2>/dev/null || true' EXIT
DP_PHASES=(teacher train evaluate)
for DP_PHASE in "${DP_PHASES[@]}"; do
  "$DP_ENV/bin/python" -u experiments/training_pilot/run.py --phase "$DP_PHASE" --root "$DP_STAGE" --output "$DP_OUT" --reader "$DP_SNAPSHOT" --selector "$DP_SELECTOR"
done
