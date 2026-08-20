#!/usr/bin/env bash
# One immutable evaluation cell. The array wrapper supplies the selectors.
#SBATCH --job-name=docprune-m3docvqa-cell
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
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
: "${RUN_CONFIG:?set RUN_CONFIG to the complete immutable run-config JSON}"
: "${INDEX_MANIFEST:?set INDEX_MANIFEST to the matching mode/page index manifest}"
: "${MODE:?set MODE to all-kept or docprune}"
: "${PAGES:?set PAGES to 1, 2, or 4}"
: "${CELL_ROOT:?set CELL_ROOT to a new attempt-specific cell directory}"
: "${DOCPRUNE_FACTORY:?set DOCPRUNE_FACTORY to the reviewed factory}"

source "$PROJECT_DIR/examples/m3docvqa/pdf_tools_preflight.sh"
pdfinfo_sha256="$(sha256sum "$PDFTOOLS_DIR/bin/pdfinfo" | awk '{print $1}')"
pdftoppm_sha256="$(sha256sum "$PDFTOOLS_DIR/bin/pdftoppm" | awk '{print $1}')"
test "$pdfinfo_sha256" = "5d0e1caa04f15391324c9e5f1d65753d8b925761eedc710c70e02f49b5080aae"
test "$pdftoppm_sha256" = "1102bc3f4a12f3d3d207ac8e39f463d0fb3c403511fb4c817e776e5251b53f33"

case "$MODE" in all-kept|docprune) ;; *) echo "invalid MODE: $MODE" >&2; exit 2 ;; esac
case "$PAGES" in 1|2|4) ;; *) echo "PAGES must be 1, 2, or 4" >&2; exit 2 ;; esac
test -d "$PROJECT_DIR"
test "$(git -C "$PROJECT_DIR" rev-parse HEAD)" = "$CONTROL_COMMIT"
test -z "$(git -C "$PROJECT_DIR" status --porcelain --untracked-files=all)"
test -f "$CONTROL_RECORD"
CONTROL_TREE="$(git -C "$PROJECT_DIR" rev-parse "${CONTROL_COMMIT}^{tree}")"
python - "$CONTROL_RECORD" "$CONTROL_COMMIT" "$CONTROL_TREE" "$PROJECT_DIR" <<'PY'
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
test -d "$RUNTIME_DIR"
test -f "$RUN_CONFIG"
test -f "$INDEX_MANIFEST"
test -d "$M3DOCRAG_DIR"
test "$(git -C "$RUNTIME_DIR" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$RUNTIME_DIR" status --porcelain --untracked-files=all)"
test "$(git -C "$M3DOCRAG_DIR" rev-parse HEAD)" = "$M3DOCRAG_COMMIT"
test -z "$(git -C "$M3DOCRAG_DIR" status --porcelain --untracked-files=all)"

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PATH="$PDFTOOLS_DIR/bin:$ENV_DIR/bin:$PATH"
export PYTHONPATH="$RUNTIME_DIR/src"
export LD_LIBRARY_PATH="$ENV_DIR/lib:$ENV_DIR/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HOME="${HF_HOME:-/scratch/$USER/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TOKENIZERS_PARALLELISM=false
export DOCPRUNE_M3DOCRAG_ROOT="$M3DOCRAG_DIR"
mkdir -p "$CELL_ROOT"
cd "$RUNTIME_DIR"

RESUME_ARGS=()
if [[ -d "$CELL_ROOT/run" ]]; then
  RESUME_ARGS+=(--resume)
fi

docprune-m3docvqa evaluate \
  --config configs/docprune-m3docvqa.toml \
  --pages "$PAGES" \
  --mode "$MODE" \
  --run-config "$RUN_CONFIG" \
  --index-manifest "$INDEX_MANIFEST" \
  --output "$CELL_ROOT/run" \
  --factory "$DOCPRUNE_FACTORY" \
  "${RESUME_ARGS[@]}"

docprune-m3docvqa validate-run \
  --run "$CELL_ROOT/run" \
  --expected-questions 2441 \
  | tee "$CELL_ROOT/validation.json"

python - "$CELL_ROOT/validation.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    report = json.load(stream)
if report.get("valid") is not True:
    raise SystemExit("independent run validation failed")
PY

python - "$CELL_ROOT/run/summary.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    summary = json.load(stream)
if not isinstance(summary, dict):
    raise SystemExit("summary is not a JSON object")
efficiency = summary.get("efficiency")
if not isinstance(efficiency, dict) or efficiency["samples"] != 2441:
    raise SystemExit("summary efficiency does not contain exactly 2441 samples")
quality = summary.get("quality")
if not isinstance(quality, dict):
    raise SystemExit("summary is missing combined quality metrics")
if "count" in quality and quality["count"] != 2441:
    raise SystemExit("summary quality count is not exactly 2441")
PY
