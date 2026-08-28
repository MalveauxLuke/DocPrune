#!/usr/bin/env bash
# One-QID Task 6 developmental matrix shards on canonical L40S only.
#SBATCH --job-name=docprune-task6-l40s
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --array=0-63%16
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

: "${RUNTIME_DIR:?set the clean reviewed runtime checkout}"
: "${RUNTIME_COMMIT:?set the exact reviewed runtime commit}"
: "${MATRIX_ROOT:?set the existing native or fixed matrix root}"
: "${MATRIX_KIND:?set native, native-extension, fixed, or visual-state-fixed-grid}"
: "${SLURM_ARRAY_TASK_ID:?submit as the sealed 0-63 array}"

case "$MATRIX_KIND" in native|native-extension|fixed|visual-state-fixed-grid) ;; *) exit 2 ;; esac
case "$SLURM_ARRAY_TASK_ID" in ''|*[!0-9]*) exit 2 ;; esac
test "$SLURM_ARRAY_TASK_ID" -le 63

readonly FIXTURE=/scratch/lmalveau/docprune/task6-fixed-page-gate-v1/fixture-stage64-top4-v2.json
readonly FIXTURE_SHA256=32b3ddd6a1f608db509f002f9541dbc92317b59ac769f0b41fcb20bcc536652b
readonly GATE=/scratch/lmalveau/docprune/task6-fixed-page-gate-v1/gate-manifest-v4.json
readonly GATE_SHA256=4d44e297081b152c2e492ec68be58445c7ac17c8d106ceb2ddb1f6331fd57761
readonly ATTEMPT_ROOT=/scratch/lmalveau/docprune/benchmark-4e2473b/attempt-2
readonly RUN_CONFIG="$ATTEMPT_ROOT/run-configs/docprune-top4-task6-clean-m3-v1.json"
readonly RUN_CONFIG_SHA256=2233621303ccdf531267bb5bd2a7670775e54f90d04dede4fb19f264e8cad502
readonly INDEX_MANIFEST="$ATTEMPT_ROOT/indexes/docprune/top4/docprune/manifest.json"
readonly INDEX_MANIFEST_SHA256=ffa5979b3bf157adefcc132b0438af295ddafb2243377db4cdb8fcb8eaafe5da
readonly CONFIG="$RUNTIME_DIR/configs/docprune-m3docvqa.toml"
readonly CONFIG_SHA256=82463d2ef3296a199521f3f637b256341aad7938eb199cb55c6249debfea44aa
readonly ENV_DIR=/home/lmalveau/mamba-envs/docprune-sol
readonly PYTHON=/home/lmalveau/mamba-envs/docprune-sol/bin/python
readonly PDFTOOLS_DIR=/home/lmalveau/mamba-envs/m3docvqa-acquisition
readonly M3DOCRAG_DIR=/home/lmalveau/src/m3docrag-task6-clean-20260828
readonly M3DOCRAG_COMMIT=29e6ac2294d6b87075a1d45b8a8df175b214248a
SHARD_NAME="$(printf 'shard-%04d' "$SLURM_ARRAY_TASK_ID")"
readonly SHARD_NAME
readonly OUTPUT_DIR="$MATRIX_ROOT/$SHARD_NAME"

if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Task 6 shard output already exists: $OUTPUT_DIR" >&2
  exit 2
fi

test -d "$MATRIX_ROOT"
test -x "$PYTHON"
test "$(git -C "$RUNTIME_DIR" rev-parse HEAD)" = "$RUNTIME_COMMIT"
test -z "$(git -C "$RUNTIME_DIR" status --porcelain --untracked-files=all)"
test "$(git -C "$M3DOCRAG_DIR" rev-parse HEAD)" = "$M3DOCRAG_COMMIT"
test -z "$(git -C "$M3DOCRAG_DIR" status --porcelain --untracked-files=all)"
test "$(sha256sum "$FIXTURE" | cut -d' ' -f1)" = "$FIXTURE_SHA256"
test "$(sha256sum "$GATE" | cut -d' ' -f1)" = "$GATE_SHA256"
test "$(sha256sum "$RUN_CONFIG" | cut -d' ' -f1)" = "$RUN_CONFIG_SHA256"
test "$(sha256sum "$INDEX_MANIFEST" | cut -d' ' -f1)" = "$INDEX_MANIFEST_SHA256"
test "$(sha256sum "$CONFIG" | cut -d' ' -f1)" = "$CONFIG_SHA256"

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PATH="$PDFTOOLS_DIR/bin:$ENV_DIR/bin:$PATH"
export PYTHONPATH="$RUNTIME_DIR/src"
export LD_LIBRARY_PATH="$ENV_DIR/lib:$ENV_DIR/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HOME=/scratch/lmalveau/hf_cache
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_CACHE="$HUGGINGFACE_HUB_CACHE"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export DOCPRUNE_M3DOCRAG_ROOT="$M3DOCRAG_DIR"

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
case "$GPU_NAME" in *L40S*) ;; *) echo "canonical Task 6 shard requires L40S" >&2; exit 2 ;; esac

"$PYTHON" "$RUNTIME_DIR/examples/run_task6_matrix.py" \
  --config "$CONFIG" \
  --run-config "$RUN_CONFIG" \
  --index-manifest "$INDEX_MANIFEST" \
  --fixture "$FIXTURE" \
  --fixture-sha256 "$FIXTURE_SHA256" \
  --gate-manifest "$GATE" \
  --gate-manifest-sha256 "$GATE_SHA256" \
  --runtime-dir "$RUNTIME_DIR" \
  --runtime-commit "$RUNTIME_COMMIT" \
  --shard "$SLURM_ARRAY_TASK_ID" \
  --kind "$MATRIX_KIND" \
  --output "$OUTPUT_DIR"

EXPECTED=45
if [[ "$MATRIX_KIND" == fixed ]]; then EXPECTED=42; fi
if [[ "$MATRIX_KIND" == native-extension ]]; then EXPECTED=40; fi
if [[ "$MATRIX_KIND" == visual-state-fixed-grid ]]; then EXPECTED=8; fi
"$PYTHON" - "$OUTPUT_DIR/results.jsonl" "$EXPECTED" "$MATRIX_KIND" <<'PY'
import json
import sys
from pathlib import Path

rows = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines() if line]
expected = int(sys.argv[2])
kind = sys.argv[3]
if len(rows) != expected or [row["matrix_cell"] for row in rows] != list(range(expected)):
    raise SystemExit("Task 6 matrix shard is incomplete or out of order")
if any(row["matrix_kind"] != kind for row in rows):
    raise SystemExit("Task 6 matrix-kind drift")
if any(row["fixed_page_provenance"] is not True or row["global_index_loaded"] is not False for row in rows):
    raise SystemExit("Task 6 no-search evidence drift")
PY
