#!/usr/bin/env bash
#SBATCH --job-name=vidore-arxivqa-custom-cq2
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/vidore_arxivqa_custom_colqwen2_%j.out
#SBATCH --error=logs/vidore_arxivqa_custom_colqwen2_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/COLPALI_binary_classification}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/$USER/vidore_arxivqa_custom_colqwen2_runs/$(date -u +%Y%m%dT%H%M%SZ)}"
ENV_NAME="${ENV_NAME:-/home/lmalveau/mamba-envs/colqwen25}"
MODEL_ARCH="${MODEL_ARCH:-colqwen2}"
MODEL_NAME="${MODEL_NAME:-vidore/colqwen2-v1.0}"
DATASET_NAME="${DATASET_NAME:-vidore/arxivqa_test_subsampled}"

mkdir -p "$PROJECT_DIR/logs" "$OUTPUT_DIR"
cd "$PROJECT_DIR"

module load mamba/latest
source activate "$ENV_NAME"
ENV_PREFIX="${CONDA_PREFIX:-$ENV_NAME}"

export ENV_PREFIX
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:$ENV_PREFIX/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONNOUSERSITE=1
PYTHON_BIN="$ENV_PREFIX/bin/python"
export HF_HOME="${HF_HOME:-/scratch/$USER/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

echo "ENV_NAME=$ENV_NAME"
echo "ENV_PREFIX=$ENV_PREFIX"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "MODEL_ARCH=$MODEL_ARCH"
echo "MODEL_NAME=$MODEL_NAME"
echo "DATASET_NAME=$DATASET_NAME"
"$PYTHON_BIN" - <<'PY'
import sys
print(sys.executable)
import torch
import datasets
from colpali_engine.models import ColQwen2, ColQwen2Processor
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("datasets", datasets.__version__)
print("colqwen2 imports ok")
PY

"$PYTHON_BIN" scripts/run_vidore_arxivqa_colqwen.py \
  --dataset-name "$DATASET_NAME" \
  --split "${SPLIT:-test}" \
  --model-name "$MODEL_NAME" \
  --model-arch "$MODEL_ARCH" \
  --top-k "${TOP_K:-5}" \
  --image-batch-size "${IMAGE_BATCH_SIZE:-2}" \
  --query-batch-size "${QUERY_BATCH_SIZE:-16}" \
  --score-batch-size "${SCORE_BATCH_SIZE:-128}" \
  --device "${DEVICE:-cuda:0}" \
  --output-dir "$OUTPUT_DIR"
