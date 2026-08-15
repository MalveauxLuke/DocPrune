#!/usr/bin/env bash
#SBATCH --job-name=mmlb-ocr2
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --array=0-31%8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/lmalveau/mmlongbench_ocr2_unstructured_313/logs/ocr2_%A_%a.out
#SBATCH --error=/scratch/lmalveau/mmlongbench_ocr2_unstructured_313/logs/ocr2_%A_%a.err

set -euo pipefail
export PYTHONNOUSERSITE=1

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${PAGES_MANIFEST:?PAGES_MANIFEST is required}"
: "${ATTEMPT_INDEX:?ATTEMPT_INDEX is required}"
if [[ "$ATTEMPT_INDEX" != "1" && "$ATTEMPT_INDEX" != "2" ]]; then
  echo "ATTEMPT_INDEX must be 1 or 2" >&2
  exit 2
fi

PROJECT_DIR="${PROJECT_DIR:-$HOME/COLPALI_binary_classification}"
ROOT="${ROOT:-/scratch/$USER/mmlongbench_ocr2_unstructured_313}"
SHARD_COUNT="${SHARD_COUNT:-32}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
SHARD_NAME="$(printf 'shard_%03d' "$SHARD_INDEX")"
OUTPUT_BASE="$RUN_ROOT/deepseek_ocr2_attempts/attempt_$ATTEMPT_INDEX"
OUTPUT_ROOT="$OUTPUT_BASE/$SHARD_NAME"
ENV_NAME="/home/$USER/mamba-envs/deepseek-ocr2-sol"
PYTHON_BIN="$ENV_NAME/bin/python"
HF_HOME="${HF_HOME:-/scratch/$USER/hf_cache}"
MODEL_REVISION="aaa02f3811945a91062062994c5c4a3f4c0af2b0"

export HF_HOME
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH="$ENV_NAME/lib:$ENV_NAME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p "$ROOT/logs" "$HF_HOME" "$OUTPUT_ROOT"
cd "$PROJECT_DIR"

echo "git_commit=$(git rev-parse HEAD)"
echo "run_root=$RUN_ROOT"
echo "pages_manifest=$PAGES_MANIFEST"
echo "pages_manifest_sha256=$(sha256sum "$PAGES_MANIFEST" | awk '{print $1}')"
echo "attempt_index=$ATTEMPT_INDEX shard_index=$SHARD_INDEX shard_count=$SHARD_COUNT"
echo "model_revision=$MODEL_REVISION"
nvidia-smi

"$PYTHON_BIN" scripts/document_parsing/run_deepseek_ocr.py \
  --run-root "$RUN_ROOT" \
  --ocr-version 2 \
  --pages-manifest "$PAGES_MANIFEST" \
  --output-root "$OUTPUT_ROOT" \
  --shard-index "$SHARD_INDEX" \
  --shard-count "$SHARD_COUNT" \
  --attempt-index "$ATTEMPT_INDEX"
