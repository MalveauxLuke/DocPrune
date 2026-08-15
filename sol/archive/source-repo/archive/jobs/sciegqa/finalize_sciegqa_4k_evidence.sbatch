#!/usr/bin/env bash
#SBATCH --job-name=sciegqa-evidence-finalize
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/lmalveau/sciegqa_train_4k/logs/evidence_finalize_%j.out
#SBATCH --error=/scratch/lmalveau/sciegqa_train_4k/logs/evidence_finalize_%j.err

set -euo pipefail
export PYTHONNOUSERSITE=1
: "${RUN_ROOT:?RUN_ROOT is required}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/COLPALI_binary_classification}"
ENV_NAME="/home/$USER/mamba-envs/deepseek-ocr2-sol"
PYTHON_BIN="$ENV_NAME/bin/python"
export LD_LIBRARY_PATH="$ENV_NAME/lib:$ENV_NAME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$PROJECT_DIR"
echo "project=$PROJECT_DIR"
echo "git_commit=$(git rev-parse HEAD)"
echo "run_root=$RUN_ROOT"
echo "pages_sha256=$(sha256sum "$RUN_ROOT/selection/pages.jsonl" | awk '{print $1}')"
echo "queries_sha256=$(sha256sum "$RUN_ROOT/selection/queries.jsonl" | awk '{print $1}')"
"$PYTHON_BIN" scripts/build_sciegqa_evidence_poc.py merge --run-root "$RUN_ROOT"
"$PYTHON_BIN" scripts/build_sciegqa_evidence_poc.py finalize --run-root "$RUN_ROOT"
