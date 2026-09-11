#!/usr/bin/env bash
# Seal document-disjoint splits for the existing Task 9 random-600 cohort.
#SBATCH --job-name=docprune-t9-split600
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/lmalveau/docprune/task9-split600-%j.out
#SBATCH --error=/scratch/lmalveau/docprune/task9-split600-%j.err

set -euo pipefail

readonly RUNTIME_DIR=/home/lmalveau/DocPrune-task9-h200-baseline-wrong-100
readonly PYTHON=/home/lmalveau/mamba-envs/docprune-sol/bin/python
readonly COHORT=/scratch/lmalveau/docprune/task9-shared-probe-random600-v1/cohort.json
readonly COHORT_FILE_SHA256=6c56c3a9b84818888ccf9a55601abfef0529be206a496874f05b588d979365a8
readonly SOURCE=/scratch/lmalveau/docprune/datasets/m3docvqa/multimodalqa/MMQA_dev.jsonl
readonly SOURCE_SHA256=31192a64bfc4ffc23123c1e6657a5b57dbb9515e9a37ecbbc8578cf894dd0e3b
readonly OUTPUT=/scratch/lmalveau/docprune/task9-shared-probe-random600-v1/splits.json
readonly SEED=docprune-task9-shared-probe-random600-splits-v1

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$RUNTIME_DIR/src"

"$PYTHON" "$RUNTIME_DIR/examples/seal_task9_random600_splits.py" \
  --cohort "$COHORT" \
  --cohort-file-sha256 "$COHORT_FILE_SHA256" \
  --source "$SOURCE" \
  --source-sha256 "$SOURCE_SHA256" \
  --seed "$SEED" \
  --output "$OUTPUT"

sha256sum "$OUTPUT"
