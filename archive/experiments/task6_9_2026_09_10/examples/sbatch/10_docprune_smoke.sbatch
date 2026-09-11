#!/usr/bin/env bash
#SBATCH --job-name=docprune-smoke
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

: "${PROJECT_DIR:?set PROJECT_DIR to the exact handoff checkout}"
: "${ENV_DIR:?set ENV_DIR to the built docprune-sol environment}"
: "${EXPECTED_COMMIT:?set EXPECTED_COMMIT from the approved handoff}"

RUN_ROOT="${RUN_ROOT:-/scratch/$USER/docprune/smoke/$SLURM_JOB_ID}"
mkdir -p "$RUN_ROOT"
cd "$PROJECT_DIR"

ACTUAL_COMMIT="$(git rev-parse HEAD)"
test "$ACTUAL_COMMIT" = "$EXPECTED_COMMIT"

export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-/scratch/$USER/hf_cache}"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export LD_LIBRARY_PATH="$ENV_DIR/lib:$ENV_DIR/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME"

export PATH="$ENV_DIR/bin:$PATH"
python -V | tee "$RUN_ROOT/python.txt"
nvidia-smi | tee "$RUN_ROOT/nvidia-smi.txt"
python - <<'PY' | tee "$RUN_ROOT/imports.txt"
import flash_attn
import torch
import transformers
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("flash_attn", flash_attn.__version__)
assert torch.cuda.is_available()
PY
python -m pytest -q | tee "$RUN_ROOT/pytest.txt"
ruff check . | tee "$RUN_ROOT/ruff.txt"
docprune-m3docvqa inspect \
  --config configs/docprune-m3docvqa.toml --pages 4 \
  | tee "$RUN_ROOT/inspect-top4.json"
git status --short | tee "$RUN_ROOT/git-status.txt"
test ! -s "$RUN_ROOT/git-status.txt"
