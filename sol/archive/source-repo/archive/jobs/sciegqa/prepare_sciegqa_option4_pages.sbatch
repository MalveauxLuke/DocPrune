#!/usr/bin/env bash
#SBATCH --job-name=sciegqa-opt4-pages
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/lmalveau/sciegqa_train_4k/logs/option4_pages_%j.out
#SBATCH --error=/scratch/lmalveau/sciegqa_train_4k/logs/option4_pages_%j.err

set -euo pipefail
export PYTHONNOUSERSITE=1

PROJECT_DIR="${PROJECT_DIR:-$HOME/COLPALI_binary_classification}"
ROOT="${ROOT:-/scratch/$USER/sciegqa_train_4k}"
OPTION4_ROOT="${OPTION4_ROOT:-$ROOT/evidence_option4_10431}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$OPTION4_ROOT/ocr_input}"
PYTHON_BIN="${PYTHON_BIN:-/home/$USER/mamba-envs/mineru34-sol/bin/python}"
EXPECTED_ARCHIVE_SHA256="${EXPECTED_ARCHIVE_SHA256:-14c0fe4ab007d86a6ade60093c03cd6f0b5e25d5affba3f5b8e410cf9178f8c9}"

cd "$PROJECT_DIR"
echo "project=$PROJECT_DIR"
echo "git_commit=$(git rev-parse HEAD)"
echo "output_root=$OUTPUT_ROOT"
echo "option4_queries_sha256=$(sha256sum "$OPTION4_ROOT/selection/queries.jsonl" | awk '{print $1}')"
echo "new_pages_sha256=$(sha256sum "$ROOT/evidence_expansion_3700/selection/pages.jsonl" | awk '{print $1}')"

"$PYTHON_BIN" scripts/build_sciegqa_evidence_poc.py prepare-option4-pages \
  --archive "$ROOT/raw/images.tar" \
  --option4-selection-dir "$OPTION4_ROOT/selection" \
  --expansion-selection-dir "$ROOT/evidence_expansion_3700/selection" \
  --output-root "$OUTPUT_ROOT" \
  --expected-archive-sha256 "$EXPECTED_ARCHIVE_SHA256"
