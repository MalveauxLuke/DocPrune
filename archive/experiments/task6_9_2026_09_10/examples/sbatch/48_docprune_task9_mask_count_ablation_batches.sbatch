#!/usr/bin/env bash
# Four-question CPU batches for the reuse-only Task 9 mask-count ablation.
#SBATCH --job-name=docprune-t9-mask-ablation
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --array=0-11%12
#SBATCH --no-requeue
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

: "${RUNTIME_DIR:?set the clean reviewed Task 9 runtime}"
: "${RUNTIME_COMMIT:?set its exact commit}"
: "${JOB_ROOT_BASE:?set a fresh existing Task 9 ablation-root parent}"
: "${SLURM_ARRAY_TASK_ID:?submit this launcher as an array}"

readonly PYTHON=/scratch/lmalveau/docprune/tool-envs/random-coverage-attribution-v2/contextcite/bin/python
readonly SOL_SITE_PACKAGES=/home/lmalveau/mamba-envs/docprune-sol/lib/python3.10/site-packages
readonly JOB_ROOT="$JOB_ROOT_BASE/batch-$SLURM_ARRAY_TASK_ID"

test -d "$JOB_ROOT_BASE"
test ! -e "$JOB_ROOT"
test "$(git -C "$RUNTIME_DIR" rev-parse HEAD)" = "$RUNTIME_COMMIT"
test -z "$(git -C "$RUNTIME_DIR" status --porcelain --untracked-files=all)"
mkdir "$JOB_ROOT"

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$RUNTIME_DIR/src:$SOL_SITE_PACKAGES"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

"$PYTHON" "$RUNTIME_DIR/examples/run_task9_mask_count_ablation_batch.py" \
  --batch-index "$SLURM_ARRAY_TASK_ID" \
  --job-root "$JOB_ROOT" \
  --runtime-dir "$RUNTIME_DIR" \
  --runtime-commit "$RUNTIME_COMMIT"
