#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?A SOL compute allocation is required}"
case "$(hostname -s)" in *login*) echo 'Refusing computation on a login node' >&2; exit 2;; esac
 : "${DP300_COMMIT:?Supply the tested Git commit}"
test "$(git rev-parse HEAD)" = "$DP300_COMMIT"
git diff --exit-code HEAD -- experiments/initial300 sol/initial300 scripts/extract_colfeatures17.py scripts/colfeatures_package.py
export DP300_ROOT=/scratch/lmalveau/docprune-initial300/20260915-v1
export DP300_CODE=/home/lmalveau/DocPrune/experiments/initial300
export DP300_COL_PY=/home/lmalveau/.conda/envs/docprune-colfeatures17/bin/python
export DP300_MINER_PY=/scratch/lmalveau/docprune/tool-envs/random-coverage-attribution-v2/mineru/bin/python
export DP300_BASE=/scratch/lmalveau/docprune-colfeatures17/20260911-run01/hf-cache/hub/models--vidore--colqwen2.5-base/snapshots/92908120384b7a2110c5beda3ab29cbdb2c08e49
export DP300_ADAPTER=/scratch/lmalveau/docprune-colfeatures17/20260911-run01/hf-cache/hub/models--vidore--colqwen2.5-v0.2/snapshots/dcbe8d9cede518bce830488364ba0e40c873645b
export DP300_MINER_MODEL=/scratch/lmalveau/docprune/tool-envs/random-coverage-attribution-v2/hf-cache/models--opendatalab--MinerU2.5-Pro-2604-1.2B/snapshots/d3f5e08d073c21466bbabe21c71bb1e9c2e595da
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export HF_HOME="$DP300_ROOT/cache/huggingface" XDG_CACHE_HOME="$DP300_ROOT/cache/xdg" TORCH_HOME="$DP300_ROOT/cache/torch"
export TMPDIR="$DP300_ROOT/tmp/${SLURM_JOB_ID}" TRITON_CACHE_DIR="$DP300_ROOT/cache/triton/${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$DP300_ROOT/logs"
