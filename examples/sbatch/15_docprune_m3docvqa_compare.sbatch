#!/usr/bin/env bash
# Independently validate all six completed cells, then publish the signed
# six-cell JSON and Markdown comparison. Submit afterok on the eval array.
#SBATCH --job-name=docprune-m3docvqa-compare
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

: "${PROJECT_DIR:?set PROJECT_DIR to the exact control checkout}"
: "${CONTROL_COMMIT:?set CONTROL_COMMIT to this handoff commit}"
: "${CONTROL_RECORD:?set CONTROL_RECORD to the sealed attempt control.json}"
: "${RUNTIME_DIR:?set RUNTIME_DIR to the detached runtime checkout}"
: "${ENV_DIR:?set ENV_DIR to the validated docprune-sol environment}"
: "${PDFTOOLS_DIR:?set PDFTOOLS_DIR to the pinned Poppler environment}"
: "${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the immutable runtime commit}"
: "${M3DOCRAG_DIR:?set M3DOCRAG_DIR to the pinned M3DocRAG checkout}"
: "${M3DOCRAG_COMMIT:?set M3DOCRAG_COMMIT to the pinned upstream commit}"
: "${ATTEMPT_ROOT:?set ATTEMPT_ROOT to the benchmark attempt root}"
: "${EXPECTED_ATTEMPT_ROOT:?set EXPECTED_ATTEMPT_ROOT to the exact fresh benchmark attempt root}"
: "${CORPUS_ROOT:?set CORPUS_ROOT to the pinned M3DocVQA corpus}"
: "${DOCPRUNE_FACTORY:?set DOCPRUNE_FACTORY to the reviewed factory}"
: "${COMPARISON_JSON:?set COMPARISON_JSON to a new signed JSON report path}"
: "${COMPARISON_MARKDOWN:?set COMPARISON_MARKDOWN to a new Markdown report path}"

PYTHONPATH="$RUNTIME_DIR/src" "$ENV_DIR/bin/python" - "$ATTEMPT_ROOT" "$EXPECTED_ATTEMPT_ROOT" <<'PY'
import sys
from docprune.benchmark_seal import validate_attempt_root

validate_attempt_root(sys.argv[1], sys.argv[2])
PY
test "$COMPARISON_JSON" = "$ATTEMPT_ROOT/comparison/six-cell.json"
test "$COMPARISON_MARKDOWN" = "$ATTEMPT_ROOT/comparison/six-cell.md"

test "$(git -C "$PROJECT_DIR" rev-parse HEAD)" = "$CONTROL_COMMIT"
test -z "$(git -C "$PROJECT_DIR" status --porcelain --untracked-files=all)"
test -f "$CONTROL_RECORD"
CONTROL_TREE="$(git -C "$PROJECT_DIR" rev-parse "${CONTROL_COMMIT}^{tree}")"
"$ENV_DIR/bin/python" - "$CONTROL_RECORD" "$CONTROL_COMMIT" "$CONTROL_TREE" "$PROJECT_DIR" <<'PY'
import json
import sys
from pathlib import Path

record = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if record.get("schema_version") != 1:
    raise SystemExit("sealed control record schema mismatch")
if record.get("control_commit") != sys.argv[2] or record.get("tree_sha") != sys.argv[3]:
    raise SystemExit("sealed control record does not match the reviewed checkout")
if Path(str(record.get("project_dir", ""))).resolve() != Path(sys.argv[4]).resolve():
    raise SystemExit("sealed control record project path mismatch")
PY

source "$PROJECT_DIR/examples/m3docvqa/pdf_tools_preflight.sh"
test "$(git -C "$RUNTIME_DIR" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$RUNTIME_DIR" status --porcelain --untracked-files=all)"
test "$(git -C "$M3DOCRAG_DIR" rev-parse HEAD)" = "$M3DOCRAG_COMMIT"
test -z "$(git -C "$M3DOCRAG_DIR" status --porcelain --untracked-files=all)"
test ! -e "$COMPARISON_JSON"
test ! -e "$COMPARISON_MARKDOWN"

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PATH="$PDFTOOLS_DIR/bin:$ENV_DIR/bin:$PATH"
export PYTHONPATH="$RUNTIME_DIR/src"
export DOCPRUNE_M3DOCRAG_ROOT="$M3DOCRAG_DIR"
cd "$RUNTIME_DIR"

for mode in all-kept docprune; do
  for pages in 1 2 4; do
    run="$ATTEMPT_ROOT/eval/$mode/top$pages/run"
    test -d "$run"
    validation="$run/independent-validation.json"
    test ! -e "$validation"
    "$ENV_DIR/bin/docprune-m3docvqa" validate-run \
      --run "$run" --expected-questions 2441 > "$validation"
    "$ENV_DIR/bin/python" - "$validation" <<'PY'
import json
import sys

payload = json.loads(open(sys.argv[1], encoding="utf-8").read())
if payload.get("valid") is not True:
    raise SystemExit("independent single-run validation failed")
PY
  done
done

"$ENV_DIR/bin/docprune-m3docvqa" compare-runs \
  --corpus-root "$CORPUS_ROOT" \
  --all-kept-top1 "$ATTEMPT_ROOT/eval/all-kept/top1/run" \
  --all-kept-top2 "$ATTEMPT_ROOT/eval/all-kept/top2/run" \
  --all-kept-top4 "$ATTEMPT_ROOT/eval/all-kept/top4/run" \
  --docprune-top1 "$ATTEMPT_ROOT/eval/docprune/top1/run" \
  --docprune-top2 "$ATTEMPT_ROOT/eval/docprune/top2/run" \
  --docprune-top4 "$ATTEMPT_ROOT/eval/docprune/top4/run" \
  --json-output "$COMPARISON_JSON" \
  --markdown-output "$COMPARISON_MARKDOWN"
test -s "$COMPARISON_JSON"
test -s "$COMPARISON_MARKDOWN"
