#!/usr/bin/env bash
#SBATCH --job-name=deepseek-ocr
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/deepseek_ocr_%j.out
#SBATCH --error=logs/deepseek_ocr_%j.err

set -euo pipefail
export PYTHONNOUSERSITE=1
if [[ "${OCR_VERSION:-}" != "1" && "${OCR_VERSION:-}" != "2" ]]; then
  echo "OCR_VERSION must be 1 or 2" >&2
  exit 2
fi
: "${RUN_ROOT:?RUN_ROOT is required}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/COLPALI_binary_classification}"
ROOT="${ROOT:-/scratch/$USER/document_understanding}"
HF_HOME="${HF_HOME:-/scratch/$USER/hf_cache}"
if [[ "$OCR_VERSION" == "1" ]]; then
  ENV_NAME="/home/$USER/mamba-envs/deepseek-ocr1-sol"
  MODEL_NAME="deepseek-ai/DeepSeek-OCR"
  MODEL_REVISION="9f30c71f441d010e5429c532364a86705536c53a"
else
  ENV_NAME="/home/$USER/mamba-envs/deepseek-ocr2-sol"
  MODEL_NAME="deepseek-ai/DeepSeek-OCR-2"
  MODEL_REVISION="aaa02f3811945a91062062994c5c4a3f4c0af2b0"
fi
PYTHON_BIN="$ENV_NAME/bin/python"
export HF_HOME HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" TRANSFORMERS_CACHE="$HF_HOME/transformers"
export CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH="$ENV_NAME/lib:$ENV_NAME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p "$ROOT/logs" "$HF_HOME" "$RUN_ROOT"
nvidia-smi
"$PYTHON_BIN" - <<PY
import sys, torch, transformers
print("python", sys.executable)
print("torch", torch.__version__, "transformers", transformers.__version__)
print("gpu", torch.cuda.get_device_name(0), "bf16", torch.cuda.is_bf16_supported())
if not torch.cuda.is_bf16_supported(): raise SystemExit("BF16 is required")
print("model", "$MODEL_NAME", "revision", "$MODEL_REVISION")
PY
EXTRA_ARGS=()
if [[ "${SMOKE:-0}" == "1" ]]; then EXTRA_ARGS+=(--smoke); fi
"$PYTHON_BIN" "$PROJECT_DIR/scripts/document_parsing/run_deepseek_ocr.py" \
  --run-root "$RUN_ROOT" --ocr-version "$OCR_VERSION" "${EXTRA_ARGS[@]}"
