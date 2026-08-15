#!/usr/bin/env bash
#SBATCH --job-name=vidore-arxivqa-official
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/vidore_arxivqa_official_%j.out
#SBATCH --error=logs/vidore_arxivqa_official_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/COLPALI_binary_classification}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/$USER/vidore_arxivqa_official_runs/$(date -u +%Y%m%dT%H%M%SZ)}"
ENV_NAME="${ENV_NAME:-/home/lmalveau/mamba-envs/colqwen25}"
MODEL_CLASS="${MODEL_CLASS:-colqwen2}"
MODEL_NAME="${MODEL_NAME:-vidore/colqwen2-v1.0}"
DATASET_NAME="${DATASET_NAME:-vidore/arxivqa_test_subsampled}"
SPLIT="${SPLIT:-test}"

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
echo "MODEL_CLASS=$MODEL_CLASS"
echo "MODEL_NAME=$MODEL_NAME"
echo "DATASET_NAME=$DATASET_NAME"
"$PYTHON_BIN" - <<'PY'
import sys
print(sys.executable)
import torch
import datasets
import vidore_benchmark
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("datasets", datasets.__version__)
print("vidore_benchmark", getattr(vidore_benchmark, "__version__", "unknown"))
print("colqwen imports ok")
PY

if "$PYTHON_BIN" -m vidore_benchmark.cli evaluate-retriever --help > "$OUTPUT_DIR/evaluate_retriever_help.txt" 2>&1; then
  VIDORE_CMD=("$PYTHON_BIN" -m vidore_benchmark.cli)
elif command -v vidore-benchmark >/dev/null 2>&1 && vidore-benchmark evaluate-retriever --help > "$OUTPUT_DIR/evaluate_retriever_help.txt" 2>&1; then
  VIDORE_CMD=(vidore-benchmark)
else
  echo "Unable to run the vidore-benchmark evaluator CLI." >&2
  echo "Captured help output:" >&2
  cat "$OUTPUT_DIR/evaluate_retriever_help.txt" >&2 || true
  exit 1
fi

grep -i "col" "$OUTPUT_DIR/evaluate_retriever_help.txt" || true

pushd "$OUTPUT_DIR" >/dev/null
"${VIDORE_CMD[@]}" evaluate-retriever \
  --model-class "$MODEL_CLASS" \
  --model-name "$MODEL_NAME" \
  --dataset-name "$DATASET_NAME" \
  --dataset-format qa \
  --split "$SPLIT" \
  --batch-query "${BATCH_QUERY:-16}" \
  --batch-passage "${BATCH_PASSAGE:-2}" \
  --batch-score "${BATCH_SCORE:-128}"
popd >/dev/null

find "$OUTPUT_DIR" -maxdepth 3 -type f | sort
