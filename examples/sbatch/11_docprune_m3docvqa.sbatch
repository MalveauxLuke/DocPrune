#!/usr/bin/env bash
#SBATCH --job-name=docprune-m3docvqa
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

: "${PROJECT_DIR:?set PROJECT_DIR to the exact handoff checkout}"
: "${ENV_DIR:?set ENV_DIR to the validated docprune-sol environment}"
: "${EXPECTED_COMMIT:?set EXPECTED_COMMIT from the approved handoff}"
: "${DOCPRUNE_FACTORY:?set module:function after the processor-contract gate passes}"
: "${QA_REVISION:?set immutable Qwen/Qwen2-VL-7B-Instruct revision}"
: "${RETRIEVER_REVISION:?set immutable vidore/colpali-v1 revision}"
: "${DATASET_REVISION:?set immutable m3docrag/m3docvqa revision}"

PAGES="${PAGES:-1}"
case "$PAGES" in 1|2|4) ;; *) echo "PAGES must be 1, 2, or 4" >&2; exit 2 ;; esac

RUN_ROOT="${RUN_ROOT:-/scratch/$USER/docprune/m3docvqa/top${PAGES}/$SLURM_JOB_ID}"
mkdir -p "$RUN_ROOT"
cd "$PROJECT_DIR"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"

export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-/scratch/$USER/hf_cache}"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export LD_LIBRARY_PATH="$ENV_DIR/lib:$ENV_DIR/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TOKENIZERS_PARALLELISM=false
export QA_REVISION RETRIEVER_REVISION DATASET_REVISION
mkdir -p "$HF_HOME"
export PATH="$ENV_DIR/bin:$PATH"

docprune-m3docvqa evaluate \
  --config configs/docprune-m3docvqa.toml \
  --pages "$PAGES" \
  --output "$RUN_ROOT/run" \
  --factory "$DOCPRUNE_FACTORY"

docprune-m3docvqa summarize \
  --results "$RUN_ROOT/run/results.jsonl" \
  | tee "$RUN_ROOT/summary-verified.json"
