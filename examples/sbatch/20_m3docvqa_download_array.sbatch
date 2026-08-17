#!/usr/bin/env bash
#SBATCH --job-name=m3docvqa-dev-pdfs
#SBATCH --partition=lightwork
#SBATCH --qos=public
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

: "${BUILDER_DIR:?set BUILDER_DIR to the pinned M3DocRAG worktree}"
: "${ACQ_ENV_DIR:?set ACQ_ENV_DIR to the acquisition environment}"
: "${DATA_ROOT:?set DATA_ROOT to the M3DocVQA dev root}"
: "${N_PROC:?set N_PROC to the exact array width}"
: "${ATTEMPT:?set ATTEMPT to an immutable attempt label}"
: "${SLURM_ARRAY_TASK_ID:?submit this wrapper as a Slurm array}"

test "$SLURM_ARRAY_TASK_ID" -ge 0
test "$SLURM_ARRAY_TASK_ID" -lt "$N_PROC"
test -f "$DATA_ROOT/id_url_mapping.jsonl"
test -f "$DATA_ROOT/dev_doc_ids.json"

export PYTHONNOUSERSITE=1
export PATH="$ACQ_ENV_DIR/bin:$PATH"
export PLAYWRIGHT_BROWSERS_PATH="$DATA_ROOT/cache/ms-playwright"
export XDG_CACHE_HOME="$DATA_ROOT/cache/xdg"

mkdir -p \
  "$DATA_ROOT/pdfs_dev" \
  "$DATA_ROOT/download_logs/$ATTEMPT" \
  "$PLAYWRIGHT_BROWSERS_PATH" \
  "$XDG_CACHE_HOME"

cd "$BUILDER_DIR/m3docvqa"
export PYTHONPATH="$PWD/src"

python main.py download_pdfs \
  --metadata_path="$DATA_ROOT/id_url_mapping.jsonl" \
  --pdf_dir="$DATA_ROOT/pdfs_dev" \
  --result_log_dir="$DATA_ROOT/download_logs/$ATTEMPT" \
  --per_split_doc_ids="$DATA_ROOT/dev_doc_ids.json" \
  --first_n=-1 \
  --proc_id="$SLURM_ARRAY_TASK_ID" \
  --n_proc="$N_PROC" \
  --check_downloaded=True
