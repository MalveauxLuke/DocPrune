#!/usr/bin/env bash
# Six immutable cells: all-kept/docprune crossed with top-1/top-2/top-4.
# Submit with afterok dependencies on all six matching index jobs and the
# semantic gate. The wrapper fails closed if the mapping is changed.
#SBATCH --job-name=docprune-m3docvqa-eval
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --array=0-5%6
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

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
: "${RUN_CONFIG_ROOT:?set RUN_CONFIG_ROOT to the six complete immutable run-config JSON files}"
: "${ALL_KEPT_INDEX_ROOT:?set ALL_KEPT_INDEX_ROOT to the validated all-kept index family}"
: "${DOCPRUNE_INDEX_ROOT:?set DOCPRUNE_INDEX_ROOT to the validated docprune index family}"
: "${ATTEMPT_ROOT:?set ATTEMPT_ROOT to the benchmark attempt root}"
: "${DOCPRUNE_FACTORY:?set DOCPRUNE_FACTORY to the reviewed factory}"
: "${SLURM_ARRAY_TASK_ID:?submit this wrapper as a Slurm array}"

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
export PATH="$PDFTOOLS_DIR/bin:$ENV_DIR/bin:$PATH"
pdfinfo_sha256="$(sha256sum "$PDFTOOLS_DIR/bin/pdfinfo" | awk '{print $1}')"
pdftoppm_sha256="$(sha256sum "$PDFTOOLS_DIR/bin/pdftoppm" | awk '{print $1}')"
test "$pdfinfo_sha256" = "5d0e1caa04f15391324c9e5f1d65753d8b925761eedc710c70e02f49b5080aae"
test "$pdftoppm_sha256" = "1102bc3f4a12f3d3d207ac8e39f463d0fb3c403511fb4c817e776e5251b53f33"

test "$SLURM_ARRAY_TASK_ID" -ge 0
test "$SLURM_ARRAY_TASK_ID" -le 5
case "$SLURM_ARRAY_TASK_ID" in
  0) MODE=all-kept; PAGES=1; INDEX_ROOT="$ALL_KEPT_INDEX_ROOT/top1" ;;
  1) MODE=all-kept; PAGES=2; INDEX_ROOT="$ALL_KEPT_INDEX_ROOT/top2" ;;
  2) MODE=all-kept; PAGES=4; INDEX_ROOT="$ALL_KEPT_INDEX_ROOT/top4" ;;
  3) MODE=docprune; PAGES=1; INDEX_ROOT="$DOCPRUNE_INDEX_ROOT/top1" ;;
  4) MODE=docprune; PAGES=2; INDEX_ROOT="$DOCPRUNE_INDEX_ROOT/top2" ;;
  5) MODE=docprune; PAGES=4; INDEX_ROOT="$DOCPRUNE_INDEX_ROOT/top4" ;;
  *) echo "invalid array index" >&2; exit 2 ;;
esac
export MODE PAGES
export RUN_CONFIG="$RUN_CONFIG_ROOT/${MODE}-top${PAGES}.json"
export INDEX_MANIFEST="$INDEX_ROOT/$MODE/manifest.json"
export CELL_ROOT="$ATTEMPT_ROOT/eval/${MODE}/top${PAGES}"
test -f "$RUN_CONFIG"
test -f "$INDEX_MANIFEST"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec "$SCRIPT_DIR/11_docprune_m3docvqa.sbatch"
