#!/usr/bin/env bash
#SBATCH --job-name=sciegqa-4k-ocr2
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --array=0-31%8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/lmalveau/sciegqa_train_4k/logs/evidence_ocr2_%A_%a.out
#SBATCH --error=/scratch/lmalveau/sciegqa_train_4k/logs/evidence_ocr2_%A_%a.err

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
ROOT="${ROOT:-/scratch/$USER/sciegqa_train_4k}"
SHARD_COUNT="${SHARD_COUNT:-32}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
OUTPUT_BASE="${OUTPUT_BASE:-$RUN_ROOT/deepseek_ocr2_attempts}"
ENV_NAME="/home/$USER/mamba-envs/deepseek-ocr2-sol"
PYTHON_BIN="$ENV_NAME/bin/python"
HF_HOME="${HF_HOME:-/scratch/$USER/hf_cache}"
MODEL_NAME="deepseek-ai/DeepSeek-OCR-2"
MODEL_REVISION="aaa02f3811945a91062062994c5c4a3f4c0af2b0"
SHARD_NAME="$(printf 'shard_%03d' "$SHARD_INDEX")"
OUTPUT_ROOT="$OUTPUT_BASE/attempt_$ATTEMPT_INDEX/$SHARD_NAME"

export HF_HOME
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH="$ENV_NAME/lib:$ENV_NAME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p "$ROOT/logs" "$HF_HOME" "$OUTPUT_ROOT"
cd "$PROJECT_DIR"

echo "project=$PROJECT_DIR"
echo "git_commit=$(git rev-parse HEAD)"
echo "run_root=$RUN_ROOT"
echo "pages_manifest=$PAGES_MANIFEST"
echo "pages_manifest_sha256=$(sha256sum "$PAGES_MANIFEST" | awk '{print $1}')"
echo "selection_queries_sha256=$(sha256sum "$RUN_ROOT/selection/queries.jsonl" | awk '{print $1}')"
echo "attempt_index=$ATTEMPT_INDEX shard_index=$SHARD_INDEX shard_count=$SHARD_COUNT"
echo "output_base=$OUTPUT_BASE"
echo "output_root=$OUTPUT_ROOT"
nvidia-smi
"$PYTHON_BIN" - <<PY
import sys, torch, transformers
print("python", sys.executable)
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("gpu", torch.cuda.get_device_name(0))
print("bf16", torch.cuda.is_bf16_supported())
if not torch.cuda.is_bf16_supported():
    raise SystemExit("BF16 is required")
print("model", "$MODEL_NAME")
print("revision", "$MODEL_REVISION")
PY

"$PYTHON_BIN" "$PROJECT_DIR/scripts/document_parsing/run_deepseek_ocr.py" \
  --run-root "$RUN_ROOT" \
  --ocr-version 2 \
  --pages-manifest "$PAGES_MANIFEST" \
  --output-root "$OUTPUT_ROOT" \
  --shard-index "$SHARD_INDEX" \
  --shard-count "$SHARD_COUNT" \
  --attempt-index "$ATTEMPT_INDEX"
