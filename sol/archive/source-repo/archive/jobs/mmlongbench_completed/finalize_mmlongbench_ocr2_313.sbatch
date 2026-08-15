#!/usr/bin/env bash
#SBATCH --job-name=mmlb-ocr2-finalize
#SBATCH --partition=lightwork
#SBATCH --qos=public
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/lmalveau/mmlongbench_ocr2_unstructured_313/logs/finalize_%j.out
#SBATCH --error=/scratch/lmalveau/mmlongbench_ocr2_unstructured_313/logs/finalize_%j.err

set -euo pipefail
export PYTHONNOUSERSITE=1

: "${MODE:?MODE is required: build-retry, merge-package, or merge-package-incomplete}"
: "${RUN_ROOT:?RUN_ROOT is required}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/COLPALI_binary_classification}"
PILOT_ROOT="${PILOT_ROOT:-$PROJECT_DIR/pilot_data/mmlongbench_ocr2_unstructured_313}"
PYTHON_BIN="/home/$USER/mamba-envs/deepseek-ocr2-sol/bin/python"
RUN_ID="${RUN_ID:-$(basename "$RUN_ROOT")}"
RESULT_ROOT="$PROJECT_DIR/sol_results/mmlongbench_ocr2_unstructured_313/$RUN_ID"

cd "$PROJECT_DIR"
echo "git_commit=$(git rev-parse HEAD)"
echo "mode=$MODE run_root=$RUN_ROOT run_id=$RUN_ID"

case "$MODE" in
  build-retry)
    "$PYTHON_BIN" scripts/build_sciegqa_evidence_poc.py build-retry \
      --run-root "$RUN_ROOT"
    echo "retry_manifest=$RUN_ROOT/selection/retry_pages.jsonl"
    ;;
  merge-package)
    mkdir -p "$RUN_ROOT/deepseek_ocr2/raw"
    "$PYTHON_BIN" scripts/build_sciegqa_evidence_poc.py merge \
      --run-root "$RUN_ROOT"
    "$PYTHON_BIN" scripts/mmlongbench/prepare_pilot.py package-results \
      --run-root "$RUN_ROOT" \
      --page-plan "$PILOT_ROOT/page_plan.jsonl" \
      --output-root "$RESULT_ROOT"
    ;;
  merge-package-incomplete)
    mkdir -p "$RUN_ROOT/deepseek_ocr2/raw"
    "$PYTHON_BIN" scripts/build_sciegqa_evidence_poc.py merge \
      --run-root "$RUN_ROOT" \
      --allow-unretried-terminal
    "$PYTHON_BIN" scripts/mmlongbench/prepare_pilot.py package-results \
      --run-root "$RUN_ROOT" \
      --page-plan "$PILOT_ROOT/page_plan.jsonl" \
      --output-root "$RESULT_ROOT" \
      --allow-incomplete
    ;;
  *)
    echo "MODE must be build-retry, merge-package, or merge-package-incomplete" >&2
    exit 2
    ;;
esac
