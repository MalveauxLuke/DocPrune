#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Compute allocation required}"
case "$(hostname -s)" in *login*) exit 2;; esac
cd /home/lmalveau/DocPrune
: "${DP_BASELINE_COMMIT:?Exact tested revision required}"
test "$(git rev-parse HEAD)" = "$DP_BASELINE_COMMIT"
git diff --exit-code HEAD -- experiments/initial300/baseline.py experiments/initial300/common.py sol/initial300-baseline
export DP_ROOT=/scratch/lmalveau/docprune-initial300/20260915-v1
export DP_ENV="$DP_ROOT/envs/qwen3-baseline"
export HF_HOME="$DP_ROOT/cache/qwen3-hf"
export XDG_CACHE_HOME="$DP_ROOT/cache/xdg" PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TMPDIR="$DP_ROOT/tmp/$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
export DP_REVISION=0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
export DP_SNAPSHOT="$HF_HOME/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/$DP_REVISION"
