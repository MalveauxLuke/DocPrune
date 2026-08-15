#!/usr/bin/env bash
#SBATCH --job-name=seg-evid-env
#SBATCH --partition=lightwork
#SBATCH --qos=public
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/setup_segment_evidence_env_%j.out
#SBATCH --error=logs/setup_segment_evidence_env_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/COLPALI_binary_classification}"
ENV_PREFIX="${ENV_PREFIX:-/home/lmalveau/mamba-envs/segment-evidence}"
ENV_FILE="${ENV_FILE:-$PROJECT_DIR/sol/segment_evidence_environment.yml}"

mkdir -p "$PROJECT_DIR/logs" "$(dirname "$ENV_PREFIX")"
cd "$PROJECT_DIR"

module load mamba/latest
export PYTHONNOUSERSITE=1

echo "PROJECT_DIR=$PROJECT_DIR"
echo "ENV_PREFIX=$ENV_PREFIX"
echo "ENV_FILE=$ENV_FILE"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none}"

if [[ -d "$ENV_PREFIX" ]]; then
  echo "Environment already exists: $ENV_PREFIX"
  mamba env update -p "$ENV_PREFIX" --file "$ENV_FILE"
else
  mamba env create -p "$ENV_PREFIX" --file "$ENV_FILE"
fi

source activate "$ENV_PREFIX"
python - <<'PY'
import sys
print("python", sys.executable)
import torch
import transformers
import peft
import sklearn
import scipy
import rank_bm25
import colpali_engine
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("peft", peft.__version__)
print("sklearn", sklearn.__version__)
print("scipy", scipy.__version__)
print("rank_bm25", getattr(rank_bm25, "__version__", "unknown"))
print("colpali_engine", getattr(colpali_engine, "__version__", "unknown"))
PY
