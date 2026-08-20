#!/usr/bin/env bash
# Build one page-specific, mode-scoped manifest-bound index. Submit this
# wrapper once for every mode/page pair, with afterok on the semantic gate.
#SBATCH --job-name=docprune-m3docvqa-index
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
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
: "${MODE:?set MODE to all-kept or docprune}"
: "${PAGES:?set PAGES to 1, 2, or 4}"
: "${INDEX_ROOT:?set INDEX_ROOT to a new attempt-specific index directory}"
: "${DOCPRUNE_FACTORY:?set DOCPRUNE_FACTORY to the reviewed factory}"
: "${RESUME:=0}"

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
pdfinfo_sha256="$(sha256sum "$PDFTOOLS_DIR/bin/pdfinfo" | awk '{print $1}')"
pdftoppm_sha256="$(sha256sum "$PDFTOOLS_DIR/bin/pdftoppm" | awk '{print $1}')"
test "$pdfinfo_sha256" = "5d0e1caa04f15391324c9e5f1d65753d8b925761eedc710c70e02f49b5080aae"
test "$pdftoppm_sha256" = "1102bc3f4a12f3d3d207ac8e39f463d0fb3c403511fb4c817e776e5251b53f33"

case "$MODE" in all-kept|docprune) ;; *) echo "invalid MODE: $MODE" >&2; exit 2 ;; esac
case "$PAGES" in 1|2|4) ;; *) echo "PAGES must be 1, 2, or 4" >&2; exit 2 ;; esac
case "$RESUME" in 0|1) ;; *) echo "RESUME must be 0 or 1" >&2; exit 2 ;; esac
if [[ "$RESUME" == 0 ]]; then
  test ! -e "$INDEX_ROOT"
else
  test -d "$INDEX_ROOT"
fi
test -f "$RUN_CONFIG"
test -d "$RUNTIME_DIR"
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
mkdir -p "$(dirname "$INDEX_ROOT")"
cd "$RUNTIME_DIR"

EMBED_ARGS=(
  --config configs/docprune-m3docvqa.toml
  --pages "$PAGES"
  --mode "$MODE"
  --run-config "$RUN_CONFIG"
  --output "$INDEX_ROOT"
  --factory "$DOCPRUNE_FACTORY"
)
if [[ "$RESUME" == 1 ]]; then
  EMBED_ARGS+=(--resume)
fi
"$ENV_DIR/bin/docprune-m3docvqa" embed "${EMBED_ARGS[@]}"

MANIFEST="$INDEX_ROOT/$MODE/manifest.json"
test -f "$MANIFEST"
"$ENV_DIR/bin/docprune-m3docvqa" inspect --config configs/docprune-m3docvqa.toml \
  --pages "$PAGES" > "$INDEX_ROOT/inspection.json"

"$ENV_DIR/bin/python" - "$MANIFEST" "$MODE" "$PAGES" <<'PY'
import sys
from pathlib import Path
from docprune.m3docvqa_factory import _load_index_manifest

mode = sys.argv[2]
pages = int(sys.argv[3])
manifest_path = Path(sys.argv[1]).resolve()
manifest = _load_index_manifest(manifest_path)
if manifest.mode != mode or manifest.page_count != pages:
    raise SystemExit("index manifest mode/page mismatch")
if manifest_path.parent != manifest.artifact_root:
    raise SystemExit("index manifest artifact root mismatch")
print({"status": "passed", "manifest": str(manifest_path)})
PY
