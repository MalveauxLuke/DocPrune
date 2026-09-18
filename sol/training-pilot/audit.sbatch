#!/usr/bin/env bash
#SBATCH --job-name=dp-pilot-bank-audit
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --cpus-per-task=2
#SBATCH --mem=6000M
#SBATCH --time=00:08:00
set -euo pipefail
: "${SLURM_JOB_ID:?}" "${DP_PILOT_COMMIT:?}"
case "$(hostname -s)" in *login*) exit 2;; esac
cd /home/lmalveau/DocPrune
test "$(git rev-parse HEAD)" = "$DP_PILOT_COMMIT"
git diff --exit-code HEAD -- experiments/training_pilot src/docprune/stage2 sol/training-pilot
export DP_BASELINE_COMMIT="$DP_PILOT_COMMIT"
source sol/initial300-baseline/environment.sh
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="/scratch/lmalveau/docprune-m3doc600/20260915-v2/admitted471-v2/selector-smoke-env/peft-0.18.0${PYTHONPATH:+:$PYTHONPATH}"
DP_STAGE=/scratch/lmalveau/docprune-m3doc600/20260915-v2/admitted471-v2/stage
"$DP_ENV/bin/python" -u experiments/training_pilot/audit_banks.py --root "$DP_STAGE" --collection "$DP_STAGE/training-pilot-quality-v1" --output "$DP_STAGE/training-pilot-v1/bank-audit-$SLURM_JOB_ID.json"
