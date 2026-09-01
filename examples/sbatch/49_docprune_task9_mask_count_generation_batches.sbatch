#!/usr/bin/env bash
# Four-question GPU batches for the Task 9 192-mask selected-response check.
#SBATCH --job-name=docprune-t9-mask192-gen
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --gres=gpu:1
#SBATCH --constraint="[a30|a100_40|a100_80|l40s|l40|h100|h100_80|h200]"
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --array=0-11%12
#SBATCH --no-requeue
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

: "${RUNTIME_DIR:?set the clean reviewed Task 9 runtime}"
: "${RUNTIME_COMMIT:?set its exact commit}"
: "${JOB_ROOT_BASE:?set a fresh existing Task 9 generation-root parent}"
: "${SLURM_ARRAY_TASK_ID:?submit this launcher as an array}"

readonly PYTHON=/home/lmalveau/mamba-envs/docprune-sol/bin/python
readonly ENV_DIR=/home/lmalveau/mamba-envs/docprune-sol
readonly PDFTOOLS_DIR=/home/lmalveau/mamba-envs/m3docvqa-acquisition
readonly M3DOCRAG_DIR=/home/lmalveau/src/m3docrag-task6-clean-20260828
readonly M3DOCRAG_COMMIT=29e6ac2294d6b87075a1d45b8a8df175b214248a
readonly JOB_ROOT="$JOB_ROOT_BASE/batch-$SLURM_ARRAY_TASK_ID"

test -d "$JOB_ROOT_BASE"
test ! -e "$JOB_ROOT"
test "$(git -C "$RUNTIME_DIR" rev-parse HEAD)" = "$RUNTIME_COMMIT"
test -z "$(git -C "$RUNTIME_DIR" status --porcelain --untracked-files=all)"
test "$(git -C "$M3DOCRAG_DIR" rev-parse HEAD)" = "$M3DOCRAG_COMMIT"
test -z "$(git -C "$M3DOCRAG_DIR" status --porcelain --untracked-files=all)"
mkdir "$JOB_ROOT"

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PATH="$PDFTOOLS_DIR/bin:$ENV_DIR/bin:$PATH"
export PYTHONPATH="$RUNTIME_DIR/src"
export LD_LIBRARY_PATH="$ENV_DIR/lib:$ENV_DIR/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HOME=/scratch/lmalveau/hf_cache
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_CACHE="$HUGGINGFACE_HUB_CACHE"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export DOCPRUNE_M3DOCRAG_ROOT="$M3DOCRAG_DIR"

"$PYTHON" "$RUNTIME_DIR/examples/run_task9_mask_count_generation_batch.py" \
  --batch-index "$SLURM_ARRAY_TASK_ID" \
  --job-root "$JOB_ROOT" \
  --runtime-dir "$RUNTIME_DIR" \
  --runtime-commit "$RUNTIME_COMMIT"
