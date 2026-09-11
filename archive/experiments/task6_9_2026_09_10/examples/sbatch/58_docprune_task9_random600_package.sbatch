#!/usr/bin/env bash
# Package the Task 9 random-600 fixed inputs for checksum-verified H200 transfer.
#SBATCH --job-name=docprune-t9-pack600
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/lmalveau/docprune/task9-pack600-%j.out
#SBATCH --error=/scratch/lmalveau/docprune/task9-pack600-%j.err

set -euo pipefail

readonly RUNTIME_DIR=/home/lmalveau/DocPrune-task9-h200-baseline-wrong-100
readonly PYTHON=/home/lmalveau/mamba-envs/docprune-sol/bin/python
readonly FIXED_ROOT=/scratch/lmalveau/docprune/task9-shared-probe-random600-inputs-v1
readonly SPLITS=/scratch/lmalveau/docprune/task9-shared-probe-random600-v1/splits.json
readonly SPLITS_SHA256=2c1b99b51a087618f1b04818161098004b04c76e84f7e33c6085e88588385087
readonly OUTPUT=/scratch/lmalveau/docprune/task9-shared-probe-random600-transfer-v1

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$RUNTIME_DIR/src"

"$PYTHON" "$RUNTIME_DIR/examples/package_task9_random600_h200_inputs.py" \
  --fixed-root "$FIXED_ROOT" \
  --splits "$SPLITS" --splits-file-sha256 "$SPLITS_SHA256" \
  --output-root "$OUTPUT"

cd "$OUTPUT"
sha256sum -c MANIFEST.sha256.sha256
sha256sum -c MANIFEST.sha256
