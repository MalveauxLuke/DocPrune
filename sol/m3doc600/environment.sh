#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Compute allocation required}"
: "${DP600_COMMIT:?Exact tested code revision required}"
case "$(hostname -s)" in *login*) exit 2;; esac
test "$(git rev-parse HEAD)" = "$DP600_COMMIT"
git diff --exit-code HEAD -- experiments/m3doc600 sol/m3doc600 scripts/extract_colfeatures17.py
export DP600_ROOT=/scratch/lmalveau/docprune-m3doc600/20260915-v1
export DP600_COL_PY=/home/lmalveau/.conda/envs/docprune-colfeatures17/bin/python
export DP600_MINER_PY=/scratch/lmalveau/docprune/tool-envs/random-coverage-attribution-v2/mineru/bin/python
export DP600_BASE=/scratch/lmalveau/docprune-colfeatures17/20260911-run01/hf-cache/hub/models--vidore--colqwen2.5-base/snapshots/92908120384b7a2110c5beda3ab29cbdb2c08e49
export DP600_ADAPTER=/scratch/lmalveau/docprune-colfeatures17/20260911-run01/hf-cache/hub/models--vidore--colqwen2.5-v0.2/snapshots/dcbe8d9cede518bce830488364ba0e40c873645b
export DP600_MINER_MODEL=/scratch/lmalveau/docprune/tool-envs/random-coverage-attribution-v2/hf-cache/models--opendatalab--MinerU2.5-Pro-2604-1.2B/snapshots/d3f5e08d073c21466bbabe21c71bb1e9c2e595da
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export HF_HOME="$DP600_ROOT/cache/huggingface" XDG_CACHE_HOME="$DP600_ROOT/cache/xdg" TORCH_HOME="$DP600_ROOT/cache/torch"
export TMPDIR="$DP600_ROOT/tmp/$SLURM_JOB_ID" TRITON_CACHE_DIR="$DP600_ROOT/cache/triton/$SLURM_JOB_ID"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$DP600_ROOT/logs"
