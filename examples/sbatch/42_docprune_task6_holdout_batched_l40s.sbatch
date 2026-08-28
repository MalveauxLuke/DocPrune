#!/usr/bin/env bash
# Twelve-QID batches for unfinished Task 6 primary holdout shards 36-1212.
#SBATCH --job-name=docprune-task6-batch
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --array=0-98%12
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

: "${RUNTIME_DIR:?set the clean reviewed runtime checkout}"
: "${RUNTIME_COMMIT:?set the exact reviewed runtime commit}"
: "${BATCH_ROOT:?set the fresh batched output root}"
: "${FIXTURE:?set the sealed 1213-QID fixed-page fixture}"
: "${FIXTURE_SHA256:?set its exact SHA-256}"
: "${GATE:?set the sealed holdout gate manifest}"
: "${GATE_SHA256:?set its exact SHA-256}"
: "${HOLDOUT_MANIFEST:?set the sealed holdout manifest}"
: "${HOLDOUT_MANIFEST_SHA256:?set its exact file SHA-256}"
: "${SLURM_ARRAY_TASK_ID:?submit as the sealed 0-98 array}"

case "$SLURM_ARRAY_TASK_ID" in ''|*[!0-9]*) exit 2 ;; esac
test "$SLURM_ARRAY_TASK_ID" -le 98
START_SHARD=$((36 + SLURM_ARRAY_TASK_ID * 12))
END_SHARD=$((START_SHARD + 11))
if (( END_SHARD > 1212 )); then END_SHARD=1212; fi
readonly START_SHARD END_SHARD

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

test -d "$BATCH_ROOT"
test -x "$PYTHON"
test -x "$RUNTIME_DIR/examples/probe_task7_l40s_gpu.sh"
test "$(git -C "$RUNTIME_DIR" rev-parse HEAD)" = "$RUNTIME_COMMIT"
test -z "$(git -C "$RUNTIME_DIR" status --porcelain --untracked-files=all)"
test "$(git -C "$M3DOCRAG_DIR" rev-parse HEAD)" = "$M3DOCRAG_COMMIT"
test -z "$(git -C "$M3DOCRAG_DIR" status --porcelain --untracked-files=all)"
test "$(sha256sum "$FIXTURE" | cut -d' ' -f1)" = "$FIXTURE_SHA256"
test "$(sha256sum "$GATE" | cut -d' ' -f1)" = "$GATE_SHA256"
test "$(sha256sum "$HOLDOUT_MANIFEST" | cut -d' ' -f1)" = "$HOLDOUT_MANIFEST_SHA256"
test "$(sha256sum "$RUN_CONFIG" | cut -d' ' -f1)" = "$RUN_CONFIG_SHA256"
test "$(sha256sum "$INDEX_MANIFEST" | cut -d' ' -f1)" = "$INDEX_MANIFEST_SHA256"
test "$(sha256sum "$CONFIG" | cut -d' ' -f1)" = "$CONFIG_SHA256"

for (( SHARD=START_SHARD; SHARD<=END_SHARD; SHARD++ )); do
  SHARD_NAME="$(printf 'shard-%04d' "$SHARD")"
  OUTPUT_DIR="$BATCH_ROOT/$SHARD_NAME"
  if [[ -e "$OUTPUT_DIR" ]]; then
    echo "Task 6 batched shard output already exists: $OUTPUT_DIR" >&2
    exit 2
  fi
done

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

"$RUNTIME_DIR/examples/probe_task7_l40s_gpu.sh" "$PYTHON" >/dev/null

for (( SHARD=START_SHARD; SHARD<=END_SHARD; SHARD++ )); do
  SHARD_NAME="$(printf 'shard-%04d' "$SHARD")"
  OUTPUT_DIR="$BATCH_ROOT/$SHARD_NAME"
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
    --shard "$SHARD" \
    --kind holdout-primary \
    --output "$OUTPUT_DIR"

  "$PYTHON" - "$OUTPUT_DIR/results.jsonl" "$SHARD" <<'PY'
import json
import sys
from pathlib import Path

rows = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines() if line]
shard = int(sys.argv[2])
if len(rows) != 21 or [row["matrix_cell"] for row in rows] != list(range(21)):
    raise SystemExit(f"Task 6 batched shard {shard} is incomplete or out of order")
if any(row["matrix_kind"] != "holdout-primary" for row in rows):
    raise SystemExit("Task 6 batched matrix-kind drift")
if any(
    row["fixed_page_provenance"] is not True or row["global_index_loaded"] is not False
    for row in rows
):
    raise SystemExit("Task 6 batched no-search evidence drift")
if rows[0]["policy_selection"]["policy"]["name"] != "aggregate-score-top-m":
    raise SystemExit("Task 6 batched aggregate comparator drift")
if [row["repetition"] for row in rows[1:]] != list(range(20)):
    raise SystemExit("Task 6 batched random-mask schedule drift")
PY
done
