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
: "${RUNTIME_DIR:?set RUNTIME_DIR to the detached runtime checkout}"
: "${ENV_DIR:?set ENV_DIR to the validated docprune-sol environment}"
: "${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the immutable runtime commit}"
: "${M3DOCRAG_DIR:?set M3DOCRAG_DIR to the pinned M3DocRAG checkout}"
: "${M3DOCRAG_COMMIT:?set M3DOCRAG_COMMIT to the pinned upstream commit}"
: "${RUN_CONFIG:?set RUN_CONFIG to the complete immutable run-config JSON}"
: "${MODE:?set MODE to all-kept or docprune}"
: "${PAGES:?set PAGES to 1, 2, or 4}"
: "${INDEX_ROOT:?set INDEX_ROOT to a new attempt-specific index directory}"
: "${DOCPRUNE_FACTORY:?set DOCPRUNE_FACTORY to the reviewed factory}"

case "$MODE" in all-kept|docprune) ;; *) echo "invalid MODE: $MODE" >&2; exit 2 ;; esac
case "$PAGES" in 1|2|4) ;; *) echo "PAGES must be 1, 2, or 4" >&2; exit 2 ;; esac
test ! -e "$INDEX_ROOT"
test "$(git -C "$PROJECT_DIR" rev-parse HEAD)" = "$CONTROL_COMMIT"
test -f "$RUN_CONFIG"
test -d "$RUNTIME_DIR"
test -d "$M3DOCRAG_DIR"
test "$(git -C "$RUNTIME_DIR" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$RUNTIME_DIR" status --porcelain)"
test "$(git -C "$M3DOCRAG_DIR" rev-parse HEAD)" = "$M3DOCRAG_COMMIT"
test -z "$(git -C "$M3DOCRAG_DIR" status --porcelain)"

export PYTHONNOUSERSITE=1
export PATH="$ENV_DIR/bin:$PATH"
export PYTHONPATH="$RUNTIME_DIR/src"
export HF_HOME="${HF_HOME:-/scratch/$USER/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TOKENIZERS_PARALLELISM=false
export DOCPRUNE_M3DOCRAG_ROOT="$M3DOCRAG_DIR"
mkdir -p "$(dirname "$INDEX_ROOT")"
cd "$RUNTIME_DIR"

docprune-m3docvqa embed \
  --config configs/docprune-m3docvqa.toml \
  --pages "$PAGES" \
  --mode "$MODE" \
  --run-config "$RUN_CONFIG" \
  --output "$INDEX_ROOT" \
  --factory "$DOCPRUNE_FACTORY"

MANIFEST="$INDEX_ROOT/$MODE/manifest.json"
test -f "$MANIFEST"
docprune-m3docvqa inspect --config configs/docprune-m3docvqa.toml \
  --pages "$PAGES" > "$INDEX_ROOT/inspection.json"

python - "$MANIFEST" "$MODE" "$PAGES" <<'PY'
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
mode = sys.argv[2]
pages = int(sys.argv[3])
payload = json.loads(manifest.read_text(encoding="utf-8"))
if payload.get("schema_version") != 4 or payload.get("mode") != mode or payload.get("page_count") != pages:
    raise SystemExit("index manifest mode/page/schema mismatch")
for key in ("manifest_sha256", "index_sha256", "embeddings_sha256", "token2pageuid_sha256", "completion_ledger_sha256"):
    value = payload.get(key)
    if not isinstance(value, str) or len(value) != 64:
        raise SystemExit(f"index manifest lacks {key}")
for key in ("index_path", "embeddings_path", "token2pageuid_path", "completion_ledger_path"):
    path = Path(str(payload[key]))
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"index artifact missing: {path}")
print(json.dumps({"status": "passed", "manifest": str(manifest.resolve())}, sort_keys=True))
PY
