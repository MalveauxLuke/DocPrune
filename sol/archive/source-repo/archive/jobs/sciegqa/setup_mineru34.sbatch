#!/usr/bin/env bash
#SBATCH --job-name=mineru34-setup
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/lmalveau/sciegqa_train_4k/logs/mineru_setup_%j.out
#SBATCH --error=/scratch/lmalveau/sciegqa_train_4k/logs/mineru_setup_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/COLPALI_binary_classification}"
ENV_NAME="${ENV_NAME:-/home/$USER/mamba-envs/mineru34-sol}"
HF_HOME="${HF_HOME:-/scratch/$USER/hf_cache}"
ROOT="${ROOT:-/scratch/$USER/sciegqa_train_4k}"
export PYTHONNOUSERSITE=1 HF_HOME

mkdir -p "$ROOT/logs" "$ROOT/mineru_smoke" "$HF_HOME"
echo "Inspecting allocated GPU/driver before dependency installation"
nvidia-smi
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader

module load mamba/latest
if [[ -e "$ENV_NAME" ]]; then
  mamba env update --prefix "$ENV_NAME" --file "$PROJECT_DIR/environments/mineru34-sol.yml"
else
  mamba env create --prefix "$ENV_NAME" --file "$PROJECT_DIR/environments/mineru34-sol.yml"
fi
source activate "$ENV_NAME"
ENV_PREFIX="${CONDA_PREFIX:-$ENV_NAME}"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:$ENV_PREFIX/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
PYTHON_BIN="$ENV_PREFIX/bin/python"

which python
"$PYTHON_BIN" - <<'PY'
import sys
from importlib.metadata import version
import torch
print("sys.executable", sys.executable)
print("mineru", version("mineru"))
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("Torch cannot access the allocated CUDA GPU")
print("gpu", torch.cuda.get_device_name(0))
print("free_vram", torch.cuda.mem_get_info(0))
PY
"$PYTHON_BIN" -m pip freeze > "$ROOT/mineru_smoke/setup_environment_freeze_${SLURM_JOB_ID}.txt"
