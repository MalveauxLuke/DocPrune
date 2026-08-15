#!/usr/bin/env bash
#SBATCH --job-name=mineru34-smoke
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/lmalveau/sciegqa_train_4k/logs/mineru_smoke_%j.out
#SBATCH --error=/scratch/lmalveau/sciegqa_train_4k/logs/mineru_smoke_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/COLPALI_binary_classification}"
ENV_NAME="${ENV_NAME:-/home/$USER/mamba-envs/mineru34-sol}"
ROOT="${ROOT:-/scratch/$USER/sciegqa_train_4k}"
HF_HOME="${HF_HOME:-/scratch/$USER/hf_cache}"
INPUT_PATH="${INPUT_PATH:-$PROJECT_DIR/archive/tests/fixtures/mineru_smoke/mineru_smoke.pdf}"
INPUT_KIND="${INPUT_KIND:-pdf}"
REQUIRED_TYPES="${REQUIRED_TYPES:-text,table,chart,chart_caption}"
RUN_LABEL="${RUN_LABEL:-synthetic}"
OUTPUT_DIR="$ROOT/mineru_smoke/$RUN_LABEL/${SLURM_JOB_ID}"
API_PORT=$((18000 + SLURM_JOB_ID % 20000))
API_URL="http://127.0.0.1:$API_PORT"

mkdir -p "$OUTPUT_DIR" "$HF_HOME"
module load mamba/latest
source activate "$ENV_NAME"
ENV_PREFIX="${CONDA_PREFIX:-$ENV_NAME}"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:$ENV_PREFIX/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONNOUSERSITE=1 HF_HOME CUDA_VISIBLE_DEVICES=0
export MINERU_MODEL_SOURCE=huggingface
export MINERU_API_MAX_CONCURRENT_REQUESTS=1
export MINERU_PROCESSING_WINDOW_SIZE=1
export MINERU_PDF_RENDER_THREADS=2
PYTHON_BIN="$ENV_PREFIX/bin/python"
MINERU_BIN="$ENV_PREFIX/bin/mineru"
MINERU_API_BIN="$ENV_PREFIX/bin/mineru-api"

nvidia-smi
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
    raise SystemExit("Torch cannot access CUDA")
print("gpu", torch.cuda.get_device_name(0))
print("free_vram", torch.cuda.mem_get_info(0))
PY

API_PID=""
cleanup() {
  if [[ -n "$API_PID" ]] && kill -0 "$API_PID" 2>/dev/null; then
    kill "$API_PID"
    wait "$API_PID" || true
  fi
}
trap cleanup EXIT

(
  cd "$OUTPUT_DIR"
  exec "$MINERU_API_BIN" --host 127.0.0.1 --port "$API_PORT" --enable-vlm-preload true
) >"$OUTPUT_DIR/api.out" 2>"$OUTPUT_DIR/api.err" &
API_PID=$!

healthy=0
for _ in $(seq 1 180); do
  if curl --fail --silent "$API_URL/health" > "$OUTPUT_DIR/health.json"; then
    healthy=1
    break
  fi
  if ! kill -0 "$API_PID" 2>/dev/null; then
    echo "MinerU API exited before becoming healthy" >&2
    exit 1
  fi
  sleep 2
done
if [[ "$healthy" != 1 ]]; then
  echo "MinerU API health timeout" >&2
  exit 1
fi

"$PYTHON_BIN" "$PROJECT_DIR/archive/code/sciegqa/run_mineru_smoke.py" \
  --input "$INPUT_PATH" \
  --input-kind "$INPUT_KIND" \
  --output-dir "$OUTPUT_DIR/result" \
  --mineru-bin "$MINERU_BIN" \
  --api-url "$API_URL" \
  --required-types "$REQUIRED_TYPES"
