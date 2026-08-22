#!/usr/bin/env bash
# Merge all 39 durable shards for each of the six canonical benchmark cells.
#SBATCH --job-name=docprune-merge-cells
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --array=0-5%6
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

: "${PROJECT_DIR:?set PROJECT_DIR to the exact control checkout}"
: "${CONTROL_COMMIT:?set CONTROL_COMMIT to this control commit}"
: "${CONTROL_RECORD:?set CONTROL_RECORD to the sealed attempt control.json}"
: "${RUNTIME_DIR:?set RUNTIME_DIR to the detached runtime checkout}"
: "${ENV_DIR:?set ENV_DIR to the validated environment}"
: "${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the runtime commit}"
: "${ATTEMPT_ROOT:?set ATTEMPT_ROOT to the benchmark attempt root}"
: "${EXPECTED_ATTEMPT_ROOT:?set EXPECTED_ATTEMPT_ROOT to the exact attempt root}"
: "${SLURM_ARRAY_TASK_ID:?submit this wrapper as a Slurm array}"

test "$(git -C "$PROJECT_DIR" rev-parse HEAD)" = "$CONTROL_COMMIT"
test -z "$(git -C "$PROJECT_DIR" status --porcelain --untracked-files=all)"
test "$(git -C "$RUNTIME_DIR" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$RUNTIME_DIR" status --porcelain --untracked-files=all)"
CONTROL_TREE="$(git -C "$PROJECT_DIR" rev-parse "${CONTROL_COMMIT}^{tree}")"
"$ENV_DIR/bin/python" - "$CONTROL_RECORD" "$CONTROL_COMMIT" "$CONTROL_TREE" "$PROJECT_DIR" <<'PY'
import json
import sys
from pathlib import Path

record = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if record != {
    "schema_version": 1,
    "control_commit": sys.argv[2],
    "tree_sha": sys.argv[3],
    "project_dir": str(Path(sys.argv[4]).resolve()),
}:
    raise SystemExit("sealed control record does not match the reviewed checkout")
PY
PYTHONPATH="$RUNTIME_DIR/src" "$ENV_DIR/bin/python" - "$ATTEMPT_ROOT" "$EXPECTED_ATTEMPT_ROOT" <<'PY'
import sys
from docprune.benchmark_seal import validate_attempt_root

validate_attempt_root(sys.argv[1], sys.argv[2])
PY

case "$SLURM_ARRAY_TASK_ID" in
  0) MODE=all-kept; PAGES=1 ;;
  1) MODE=all-kept; PAGES=2 ;;
  2) MODE=all-kept; PAGES=4 ;;
  3) MODE=docprune; PAGES=1 ;;
  4) MODE=docprune; PAGES=2 ;;
  5) MODE=docprune; PAGES=4 ;;
  *) echo "invalid merge array task" >&2; exit 2 ;;
esac

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$RUNTIME_DIR/src"
PLAN="$ATTEMPT_ROOT/shard-plan/plan.json"
test -f "$PLAN"
"$ENV_DIR/bin/python" "$PROJECT_DIR/examples/m3docvqa/merge_eval_shards.py" \
  --plan "$PLAN" \
  --shard-root "$ATTEMPT_ROOT/eval-shards/$MODE/top$PAGES" \
  --output "$ATTEMPT_ROOT/eval/$MODE/top$PAGES/run" \
  --expected-questions 2441
