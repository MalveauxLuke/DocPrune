#!/usr/bin/env bash
#SBATCH --job-name=dp-taskprompt-smoke
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24000M
#SBATCH --time=00:05:00
set -euo pipefail
: "${SLURM_JOB_ID:?}" "${DP_PILOT_COMMIT:?}" "${DP_BANK_AUDIT:?}"
case "$(hostname -s)" in *login*) exit 2;; esac
cd /home/lmalveau/DocPrune
test "$(git rev-parse HEAD)" = "$DP_PILOT_COMMIT"
git diff --exit-code HEAD -- experiments/training_pilot src/docprune/stage2 sol/training-pilot
export DP_BASELINE_COMMIT="$DP_PILOT_COMMIT"
source sol/initial300-baseline/environment.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="/scratch/lmalveau/docprune-m3doc600/20260915-v2/admitted471-v2/selector-smoke-env/peft-0.18.0${PYTHONPATH:+:$PYTHONPATH}"
DP_STAGE=/scratch/lmalveau/docprune-m3doc600/20260915-v2/admitted471-v2/stage
DP_OUT="$DP_STAGE/training-pilot-quality-v1-task-prompt-preflight-seed0"
DP_SELECTOR=/scratch/lmalveau/hf_cache/hub/models--Qwen--Qwen3-VL-Reranker-2B/snapshots/4bd860ac4f15ad1897a214615cccc700f8f71818
mkdir -p "$DP_OUT"
exec 9>"$DP_OUT/job.lock"
flock -n 9 || exit 1
"$DP_ENV/bin/python" -u experiments/training_pilot/train64.py \
  --root "$DP_STAGE" \
  --output "$DP_OUT" \
  --audit "$DP_BANK_AUDIT" \
  --selector "$DP_SELECTOR" \
  --seed 0 \
  --prompt-condition evidence-v1 \
  --branches frozen-only \
  --preflight-only
