#!/usr/bin/env bash
#SBATCH --job-name=sciegqa-extract
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/lmalveau/sciegqa_train_4k/logs/extract_%j.out
#SBATCH --error=/scratch/lmalveau/sciegqa_train_4k/logs/extract_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/COLPALI_binary_classification}"
ROOT="${ROOT:-/scratch/$USER/sciegqa_train_4k}"
PYTHON_BIN="${PYTHON_BIN:-/home/$USER/mamba-envs/colqwen25/bin/python}"
EXPECTED_ARCHIVE_SHA256="${EXPECTED_ARCHIVE_SHA256:-14c0fe4ab007d86a6ade60093c03cd6f0b5e25d5affba3f5b8e410cf9178f8c9}"
export PYTHONNOUSERSITE=1

cd "$PROJECT_DIR"
"$PYTHON_BIN" scripts/extract_sciegqa_4k_pages.py \
  --archive "$ROOT/raw/images.tar" \
  --jsonl "$ROOT/raw/SciEGQA-Train.jsonl" \
  --selection-dir "$ROOT/selection" \
  --pages-dir "$ROOT/pages" \
  --expected-archive-sha256 "$EXPECTED_ARCHIVE_SHA256"
