#!/usr/bin/env bash
#SBATCH --job-name=colqwen-doc-top5
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/colqwen_doc_top5_%j.out
#SBATCH --error=logs/colqwen_doc_top5_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/COLPALI_binary_classification}"
SUBSET_DIR="${SUBSET_DIR:-/scratch/$USER/mmdocir_colqwen_diverse_longdoc_pilot}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/$USER/mmdocir_colqwen_docscoped_runs/$(date -u +%Y%m%dT%H%M%SZ)}"
ENV_NAME="${ENV_NAME:-/home/lmalveau/mamba-envs/colqwen25}"
MODEL_NAME="${MODEL_NAME:-vidore/colqwen2.5-v0.1}"

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
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

echo "ENV_NAME=$ENV_NAME"
echo "ENV_PREFIX=$ENV_PREFIX"
echo "PYTHON_BIN=$PYTHON_BIN"
"$PYTHON_BIN" -c 'import sys; print(sys.executable); import torch; print(torch.__version__)'

"$PYTHON_BIN" scripts/run_colqwen_topk.py \
  --subset-dir "$SUBSET_DIR" \
  --model-name "$MODEL_NAME" \
  --scope document \
  --top-k 5 \
  --page-batch-size "${PAGE_BATCH_SIZE:-2}" \
  --query-batch-size "${QUERY_BATCH_SIZE:-16}" \
  --score-batch-size "${SCORE_BATCH_SIZE:-128}" \
  --device "${DEVICE:-cuda:0}" \
  --page-embeddings "$OUTPUT_DIR/page_embeddings.pt" \
  --query-embeddings "$OUTPUT_DIR/query_embeddings.pt" \
  --output "$OUTPUT_DIR/doc_top5_results.jsonl"
