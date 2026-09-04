#!/usr/bin/env bash
# Materialize retrieval-free fixed inputs for the Task 9 random-600 cohort.
#SBATCH --job-name=docprune-t9-input600
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/lmalveau/docprune/task9-input600-%j.out
#SBATCH --error=/scratch/lmalveau/docprune/task9-input600-%j.err

set -euo pipefail

readonly RUNTIME_DIR=/home/lmalveau/DocPrune-task9-h200-baseline-wrong-100
readonly PYTHON=/home/lmalveau/mamba-envs/docprune-sol/bin/python
readonly ENV_DIR=/home/lmalveau/mamba-envs/docprune-sol
readonly PDFTOOLS_DIR=/home/lmalveau/mamba-envs/m3docvqa-acquisition
readonly COHORT=/scratch/lmalveau/docprune/task9-shared-probe-random600-v1/cohort.json
readonly COHORT_SHA256=6c56c3a9b84818888ccf9a55601abfef0529be206a496874f05b588d979365a8
readonly SPLITS=/scratch/lmalveau/docprune/task9-shared-probe-random600-v1/splits.json
readonly SPLITS_SHA256=2c1b99b51a087618f1b04818161098004b04c76e84f7e33c6085e88588385087
readonly SOURCE=/scratch/lmalveau/docprune/datasets/m3docvqa/multimodalqa/MMQA_dev.jsonl
readonly SOURCE_SHA256=31192a64bfc4ffc23123c1e6657a5b57dbb9515e9a37ecbbc8578cf894dd0e3b
readonly RESULTS=/scratch/lmalveau/docprune/benchmark-4e2473b/attempt-2/eval-quality/docprune/top4/run/results.jsonl
readonly RESULTS_SHA256=fb148d6ce33f92ef788aa0c65ada4a95f9ed81f575e9fe3e3097b54f37d1b217
readonly FEATURE_MANIFEST=/scratch/lmalveau/docprune/benchmark-4e2473b/attempt-2/indexes/docprune/top4/docprune/manifest.json
readonly PDF_DIR=/scratch/lmalveau/docprune/datasets/m3docvqa/pdfs_dev
readonly OUTPUT=/scratch/lmalveau/docprune/task9-shared-probe-random600-inputs-v1
readonly RUNTIME_COMMIT="$(git -C "$RUNTIME_DIR" rev-parse HEAD)"

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PATH="$PDFTOOLS_DIR/bin:$ENV_DIR/bin:$PATH"
export PYTHONPATH="$RUNTIME_DIR/src"
export LD_LIBRARY_PATH="$ENV_DIR/lib:$ENV_DIR/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

"$PYTHON" "$RUNTIME_DIR/examples/seal_task9_random600_inputs.py" \
  --cohort "$COHORT" --cohort-file-sha256 "$COHORT_SHA256" \
  --splits "$SPLITS" --splits-file-sha256 "$SPLITS_SHA256" \
  --source "$SOURCE" --source-sha256 "$SOURCE_SHA256" \
  --results "$RESULTS" --results-sha256 "$RESULTS_SHA256" \
  --feature-manifest "$FEATURE_MANIFEST" --pdf-dir "$PDF_DIR" \
  --output-root "$OUTPUT" --runtime-commit "$RUNTIME_COMMIT"
