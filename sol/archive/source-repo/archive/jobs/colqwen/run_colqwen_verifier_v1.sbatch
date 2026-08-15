#!/usr/bin/env bash
#SBATCH --job-name=colqwen-verifier-v1
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/colqwen_verifier_v1_%j.out
#SBATCH --error=logs/colqwen_verifier_v1_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/COLPALI_binary_classification}"
DATASET_DIR="${DATASET_DIR:-/scratch/$USER/mmdocir_colqwen2_verifier_training_dataset_v1}"
RUN_DIR="${RUN_DIR:-/scratch/$USER/mmdocir_colqwen2_docscoped_runs/20260625T035758Z}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/$USER/mmdocir_colqwen2_verifier_runs/v1/$(date -u +%Y%m%dT%H%M%SZ)}"
ENV_NAME="${ENV_NAME:-/home/lmalveau/mamba-envs/colqwen25}"
SEEDS="${SEEDS:-13 17 23}"
SHUFFLE_SEED="${SHUFFLE_SEED:-101}"
CONTENT_ONLY="${CONTENT_ONLY:-0}"
NO_INPUT_LAYERNORM="${NO_INPUT_LAYERNORM:-0}"

TRAIN_EXTRA_ARGS=()
if [[ "$CONTENT_ONLY" == "1" ]]; then
  TRAIN_EXTRA_ARGS+=(--content-only)
fi
if [[ "$NO_INPUT_LAYERNORM" == "1" ]]; then
  TRAIN_EXTRA_ARGS+=(--no-input-layernorm)
fi

mkdir -p "$PROJECT_DIR/logs" "$OUTPUT_DIR"
cd "$PROJECT_DIR"

module load mamba/latest
source activate "$ENV_NAME"
ENV_PREFIX="${CONDA_PREFIX:-$ENV_NAME}"

export ENV_PREFIX
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:$ENV_PREFIX/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONNOUSERSITE=1
PYTHON_BIN="$ENV_PREFIX/bin/python"

echo "ENV_NAME=$ENV_NAME"
echo "ENV_PREFIX=$ENV_PREFIX"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "DATASET_DIR=$DATASET_DIR"
echo "RUN_DIR=$RUN_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "SEEDS=$SEEDS"
echo "CONTENT_ONLY=$CONTENT_ONLY"
echo "NO_INPUT_LAYERNORM=$NO_INPUT_LAYERNORM"
"$PYTHON_BIN" - <<'PY'
import sys
print(sys.executable)
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
PY

"$PYTHON_BIN" archive/code/colqwen/train_colqwen_verifier.py \
  --dataset-dir "$DATASET_DIR" \
  --run-dir "$RUN_DIR" \
  --output-dir "$OUTPUT_DIR/main" \
  --seeds $SEEDS \
  --epochs "${EPOCHS:-12}" \
  --patience "${PATIENCE:-3}" \
  --lr "${LR:-0.0002}" \
  --group-batch-size "${GROUP_BATCH_SIZE:-8}" \
  --dropout "${DROPOUT:-0.2}" \
  --device "${DEVICE:-cuda:0}" \
  "${TRAIN_EXTRA_ARGS[@]}"

"$PYTHON_BIN" archive/code/colqwen/evaluate_colqwen_verifier.py "$OUTPUT_DIR/main/all_seed_metrics.json" --output-dir "$OUTPUT_DIR/main"

if [[ "${RUN_SHUFFLE_CONTROL:-1}" == "1" ]]; then
  "$PYTHON_BIN" archive/code/colqwen/train_colqwen_verifier.py \
    --dataset-dir "$DATASET_DIR" \
    --run-dir "$RUN_DIR" \
    --output-dir "$OUTPUT_DIR/label_shuffle_control" \
    --seeds "$SHUFFLE_SEED" \
    --epochs "${SHUFFLE_EPOCHS:-8}" \
    --patience "${PATIENCE:-3}" \
    --lr "${LR:-0.0002}" \
    --group-batch-size "${GROUP_BATCH_SIZE:-8}" \
    --dropout "${DROPOUT:-0.2}" \
    --device "${DEVICE:-cuda:0}" \
    --shuffle-labels \
    "${TRAIN_EXTRA_ARGS[@]}"

  "$PYTHON_BIN" archive/code/colqwen/evaluate_colqwen_verifier.py "$OUTPUT_DIR/label_shuffle_control/all_seed_metrics.json" --output-dir "$OUTPUT_DIR/label_shuffle_control"
fi

echo "Verifier outputs: $OUTPUT_DIR"
