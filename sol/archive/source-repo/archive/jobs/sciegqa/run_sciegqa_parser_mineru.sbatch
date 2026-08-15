#!/usr/bin/env bash
#SBATCH --job-name=sciegqa-mineru32
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/lmalveau/sciegqa_train_4k/logs/parser_mineru_%j.out
#SBATCH --error=/scratch/lmalveau/sciegqa_train_4k/logs/parser_mineru_%j.err

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$HOME/COLPALI_binary_classification}"
ROOT="${ROOT:-/scratch/$USER/sciegqa_train_4k}"
: "${RUN_ROOT:?RUN_ROOT is required}"
ENV_NAME="/home/$USER/mamba-envs/mineru34-sol"
PYTHON_BIN="/home/$USER/mamba-envs/mineru34-sol/bin/python"
MINERU_BIN="$ENV_NAME/bin/mineru"
MINERU_API_BIN="$ENV_NAME/bin/mineru-api"
HF_HOME="${HF_HOME:-/scratch/$USER/hf_cache}"
API_PORT=$((18000 + SLURM_JOB_ID % 20000))
API_URL="http://127.0.0.1:$API_PORT"
API_LOG_DIR="$RUN_ROOT/${SMOKE:+smoke/}mineru/api-${SLURM_JOB_ID}"
mkdir -p "$ROOT/logs" "$API_LOG_DIR" "$HF_HOME"
export PYTHONNOUSERSITE=1 HF_HOME CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH="$ENV_NAME/lib:$ENV_NAME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MINERU_MODEL_SOURCE=huggingface MINERU_API_MAX_CONCURRENT_REQUESTS=1
export MINERU_PROCESSING_WINDOW_SIZE=1 MINERU_PDF_RENDER_THREADS=2

nvidia-smi
GPU_IDENTITY=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
export GPU_IDENTITY
"$PYTHON_BIN" -c 'import sys, torch; from importlib.metadata import version; print(sys.executable, version("mineru"), torch.__version__, torch.cuda.is_available())'

API_PID=""
cleanup() {
  if [[ -n "$API_PID" ]] && kill -0 "$API_PID" 2>/dev/null; then
    kill "$API_PID"
    wait "$API_PID" || true
  fi
}
trap cleanup EXIT
(cd "$API_LOG_DIR" && exec "$MINERU_API_BIN" --host 127.0.0.1 --port "$API_PORT" --enable-vlm-preload true) >"$API_LOG_DIR/api.out" 2>"$API_LOG_DIR/api.err" &
API_PID=$!
healthy=0
for _ in $(seq 1 180); do
  if curl --fail --silent "$API_URL/health" > "$API_LOG_DIR/health.json"; then healthy=1; break; fi
  if ! kill -0 "$API_PID" 2>/dev/null; then echo "MinerU API exited" >&2; exit 1; fi
  sleep 2
done
[[ "$healthy" == 1 ]] || { echo "MinerU API health timeout" >&2; exit 1; }

EXTRA_ARGS=()
if [[ "${SMOKE:-0}" == "1" ]]; then EXTRA_ARGS+=(--smoke); fi
"$PYTHON_BIN" "$PROJECT_DIR/scripts/run_sciegqa_parser_mineru.py" \
  --run-root "$RUN_ROOT" --mineru-bin "$MINERU_BIN" --api-url "$API_URL" "${EXTRA_ARGS[@]}"
