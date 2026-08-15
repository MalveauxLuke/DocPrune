#!/usr/bin/env bash
#SBATCH --job-name=deepseek-ocr-setup
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/lmalveau/sciegqa_train_4k/logs/deepseek_setup_%j.out
#SBATCH --error=/scratch/lmalveau/sciegqa_train_4k/logs/deepseek_setup_%j.err

set -euo pipefail
export PYTHONNOUSERSITE=1

if [[ "${OCR_VERSION:-}" != "1" && "${OCR_VERSION:-}" != "2" ]]; then
  echo "OCR_VERSION must be 1 or 2" >&2
  exit 2
fi

PROJECT_DIR="${PROJECT_DIR:-$HOME/COLPALI_binary_classification}"
ROOT="${ROOT:-/scratch/$USER/sciegqa_train_4k}"
HF_HOME="${HF_HOME:-/scratch/$USER/hf_cache}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-$ROOT/pip_cache}"
TMPDIR="${TMPDIR:-$ROOT/tmp/$SLURM_JOB_ID}"
FLASH_WHEEL_NAME="flash_attn-2.7.3+cu11torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
FLASH_WHEEL_DIR="$ROOT/deepseek_setup/wheels/$SLURM_JOB_ID"
FLASH_WHEEL_PATH="$FLASH_WHEEL_DIR/$FLASH_WHEEL_NAME"
FLASH_WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/$FLASH_WHEEL_NAME"
if [[ "$OCR_VERSION" == "1" ]]; then
  ENV_NAME="/home/$USER/mamba-envs/deepseek-ocr1-sol"
  ENV_FILE="$PROJECT_DIR/environments/deepseek-ocr1-sol.yml"
else
  ENV_NAME="/home/$USER/mamba-envs/deepseek-ocr2-sol"
  ENV_FILE="$PROJECT_DIR/environments/deepseek-ocr2-sol.yml"
fi
export HF_HOME PIP_CACHE_DIR TMPDIR
mkdir -p "$ROOT/logs" "$ROOT/deepseek_setup" "$HF_HOME" "$PIP_CACHE_DIR" "$TMPDIR" "$FLASH_WHEEL_DIR"

nvidia-smi
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader
module load mamba/latest
if [[ -e "$ENV_NAME" ]]; then
  mamba env update --prefix "$ENV_NAME" --file "$ENV_FILE"
else
  mamba env create --prefix "$ENV_NAME" --file "$ENV_FILE"
fi
PYTHON_BIN="$ENV_NAME/bin/python"
export LD_LIBRARY_PATH="$ENV_NAME/lib:$ENV_NAME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

"$PYTHON_BIN" -m pip install \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu118
"$PYTHON_BIN" -m pip install psutil
curl --fail --location --continue-at - --output "$FLASH_WHEEL_PATH" "$FLASH_WHEEL_URL"
"$PYTHON_BIN" -m pip install "$FLASH_WHEEL_PATH"
"$PYTHON_BIN" -m pip install flash-attn==2.7.3 --no-build-isolation
"$PYTHON_BIN" -m pip uninstall --yes psutil
if "$PYTHON_BIN" -m pip show psutil >/dev/null 2>&1; then
  echo "temporary psutil build dependency remains installed" >&2
  exit 1
fi
"$PYTHON_BIN" - <<'PY'
import sys
import torch
import transformers
import tokenizers
import flash_attn
import fitz
import img2pdf
import einops
import easydict
import addict
import PIL
import numpy
import huggingface_hub
print("python", sys.executable)
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("tokenizers", tokenizers.__version__)
print("flash_attn", flash_attn.__version__)
print("cuda", torch.cuda.is_available())
print("bf16", torch.cuda.is_bf16_supported())
if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
    raise SystemExit("allocated GPU must support CUDA BF16")
if flash_attn.__version__ != "2.7.3":
    raise SystemExit(f"unexpected flash-attn version: {flash_attn.__version__}")
PY
"$PYTHON_BIN" -m pip freeze > "$ROOT/deepseek_setup/ocr${OCR_VERSION}_environment_${SLURM_JOB_ID}.txt"
