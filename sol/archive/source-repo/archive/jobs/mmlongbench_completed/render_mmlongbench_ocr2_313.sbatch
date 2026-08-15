#!/usr/bin/env bash
#SBATCH --job-name=mmlb-ocr2-render
#SBATCH --partition=lightwork
#SBATCH --qos=public
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/lmalveau/mmlongbench_ocr2_unstructured_313/logs/render_%j.out
#SBATCH --error=/scratch/lmalveau/mmlongbench_ocr2_unstructured_313/logs/render_%j.err

set -euo pipefail
export PYTHONNOUSERSITE=1

: "${RUN_ROOT:?RUN_ROOT is required}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/COLPALI_binary_classification}"
PILOT_ROOT="${PILOT_ROOT:-$PROJECT_DIR/pilot_data/mmlongbench_ocr2_unstructured_313}"
ROOT="${ROOT:-/scratch/$USER/mmlongbench_ocr2_unstructured_313}"
PYTHON_BIN="/home/$USER/mamba-envs/deepseek-ocr2-sol/bin/python"

mkdir -p "$ROOT/logs" "$RUN_ROOT"
cd "$PROJECT_DIR"

echo "project=$PROJECT_DIR"
echo "git_commit=$(git rev-parse HEAD)"
echo "pilot_root=$PILOT_ROOT"
echo "page_plan_sha256=$(sha256sum "$PILOT_ROOT/page_plan.jsonl" | awk '{print $1}')"

"$PYTHON_BIN" scripts/mmlongbench/prepare_pilot.py render \
  --pilot-root "$PILOT_ROOT" \
  --run-root "$RUN_ROOT" \
  --matrix-scale 2.0
