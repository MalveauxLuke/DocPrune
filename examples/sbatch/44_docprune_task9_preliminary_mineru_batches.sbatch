#!/usr/bin/env bash
# Four remaining Task 9 MinerU questions per recoverable array task.
#SBATCH --job-name=docprune-t9-mineru-b4
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --array=0-11%6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --no-requeue
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

: "${RUNTIME_DIR:?set the clean reviewed runtime checkout}"
: "${RUNTIME_COMMIT:?set the exact reviewed runtime commit}"
: "${BATCH_ROOT:?set a fresh existing batch root}"

readonly PYTHON=/home/lmalveau/mamba-envs/docprune-sol/bin/python
mapfile -t QUESTION_INDICES < <(
  PYTHONPATH="$RUNTIME_DIR/src" "$PYTHON" -c \
    'import sys; from docprune.experiment_design import task9_preprocessing_batch_indices; print(*task9_preprocessing_batch_indices(int(sys.argv[1])), sep="\n")' \
    "$SLURM_ARRAY_TASK_ID"
)
for QUESTION_INDEX in "${QUESTION_INDICES[@]}"; do
  export QUESTION_INDEX
  bash "$RUNTIME_DIR/examples/sbatch/42_docprune_task9_preliminary_mineru.sbatch"
done
