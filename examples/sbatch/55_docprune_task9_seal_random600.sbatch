#!/usr/bin/env bash
# Seal the Task 9 300-correct/300-wrong cohort from existing cached results.
#SBATCH --job-name=docprune-t9-seal600
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/lmalveau/docprune/task9-seal600-%j.out
#SBATCH --error=/scratch/lmalveau/docprune/task9-seal600-%j.err

set -euo pipefail

readonly RUNTIME_DIR=/home/lmalveau/DocPrune-task9-h200-baseline-wrong-100
readonly PYTHON=/home/lmalveau/mamba-envs/docprune-sol/bin/python
readonly RESULTS=/scratch/lmalveau/docprune/benchmark-4e2473b/attempt-2/eval-quality/docprune/top4/run/results.jsonl
readonly RESULTS_SHA256=fb148d6ce33f92ef788aa0c65ada4a95f9ed81f575e9fe3e3097b54f37d1b217
readonly OUTPUT=/scratch/lmalveau/docprune/task9-shared-probe-random600-v1/cohort.json
readonly SEED=docprune-task9-shared-probe-random600-v1

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$RUNTIME_DIR/src"

"$PYTHON" "$RUNTIME_DIR/examples/seal_task9_random600.py" \
  --results "$RESULTS" \
  --results-sha256 "$RESULTS_SHA256" \
  --seed "$SEED" \
  --output "$OUTPUT"

"$PYTHON" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["per_stratum"] == 300; assert sum(len(v) for v in p["selected_qids"].values()) == 600; assert len(set(p["selected_qids"]["baseline_correct"])) == 300; assert len(set(p["selected_qids"]["baseline_wrong"])) == 300; assert set(p["selected_qids"]["baseline_correct"]).isdisjoint(p["selected_qids"]["baseline_wrong"]); print(json.dumps({"validated": True, "question_count": 600, "cohort_sha256": p["cohort_sha256"]}, sort_keys=True))' "$OUTPUT"

sha256sum "$OUTPUT"
