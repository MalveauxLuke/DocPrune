#!/usr/bin/env bash
# Incremental stage-localization diagnostic: evaluate only questions 65-245.
# Plan shards 0-3 are the already-completed 64-question diagnostic.
#SBATCH --job-name=docprune-stage245
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --array=4-15%12
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100_40
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:45:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

: "${DIAG_ROOT:?set DIAG_ROOT}"
: "${STAGE:?set STAGE to btp-only or btp-qtp}"
: "${FACTORY:?set the reviewed diagnostic factory}"
: "${RUNTIME_DIR:?set the detached diagnostic runtime}"
: "${RUNTIME_COMMIT:?set the diagnostic runtime commit}"
: "${SLURM_ARRAY_TASK_ID:?run as the reviewed incremental array}"

case "$STAGE:$FACTORY" in
  btp-only:docprune.m3docvqa_factory\:build_btp_only_workload) ;;
  btp-qtp:docprune.m3docvqa_factory\:build_btp_qtp_workload) ;;
  *) echo "stage/factory mismatch" >&2; exit 2 ;;
esac

ATTEMPT_ROOT=/scratch/lmalveau/docprune/benchmark-4e2473b/attempt-2
ENV_DIR=/home/lmalveau/mamba-envs/docprune-sol
PDFTOOLS_DIR=/home/lmalveau/mamba-envs/m3docvqa-acquisition
M3DOCRAG_DIR=/home/lmalveau/src/m3docrag-benchmark-29e6ac2
M3DOCRAG_COMMIT=29e6ac2294d6b87075a1d45b8a8df175b214248a
RUN_CONFIG="$ATTEMPT_ROOT/run-configs/docprune-top4.json"
INDEX_MANIFEST="$ATTEMPT_ROOT/indexes/docprune/top4/docprune/manifest.json"
PLAN="$DIAG_ROOT/shard-plan/plan.json"
SHARD_ID="$(printf '%04d' "$SLURM_ARRAY_TASK_ID")"
QID_FILE="$DIAG_ROOT/shard-plan/shard-$SHARD_ID.qids"
CELL_ROOT="$DIAG_ROOT/eval-shards/$STAGE/shard-$SHARD_ID"

test "$SLURM_ARRAY_TASK_ID" -ge 4
test "$SLURM_ARRAY_TASK_ID" -le 15
test "$(git -C "$RUNTIME_DIR" rev-parse HEAD)" = "$RUNTIME_COMMIT"
test -z "$(git -C "$RUNTIME_DIR" status --porcelain --untracked-files=all)"
test "$(git -C "$M3DOCRAG_DIR" rev-parse HEAD)" = "$M3DOCRAG_COMMIT"
test -z "$(git -C "$M3DOCRAG_DIR" status --porcelain --untracked-files=all)"
test -f "$DIAG_ROOT/control.json"
test -f "$DIAG_ROOT/reference.json"
test -f "$PLAN"
test -f "$QID_FILE"

EXPECTED_QUESTIONS="$(
  PYTHONPATH="$RUNTIME_DIR/src" "$ENV_DIR/bin/python" - "$PLAN" "$QID_FILE" <<'PY'
import sys
from pathlib import Path

from docprune.evaluation_shards import load_shard_plan

plan = load_shard_plan(Path(sys.argv[1]))
shard_id = int(Path(sys.argv[2]).stem.removeprefix("shard-"))
expected = tuple(plan["shards"][shard_id]["question_ids"])
observed = tuple(Path(sys.argv[2]).read_text(encoding="utf-8").splitlines())
if observed != expected or not 1 <= len(expected) <= 16:
    raise SystemExit("diagnostic QIDs do not match the sealed incremental shard")
print(len(expected))
PY
)"

source "$RUNTIME_DIR/examples/m3docvqa/pdf_tools_preflight.sh"
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
export TOKENIZERS_PARALLELISM=false
export DOCPRUNE_M3DOCRAG_ROOT="$M3DOCRAG_DIR"

validate_retrieval_reference() {
  "$ENV_DIR/bin/python" - "$CELL_ROOT/run/results.jsonl" "$DIAG_ROOT/reference.json" <<'PY'
import json
import sys
from pathlib import Path

reference = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))["rows"]
for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    row = json.loads(line)
    qid = row["question_id"]
    if qid not in reference:
        raise SystemExit(f"question absent from sealed reference: {qid}")
    if row["retrieved_pages"] != reference[qid]["retrieved_pages"]:
        raise SystemExit(f"retrieved-page drift for {qid}")
PY
}

mkdir -p "$CELL_ROOT"
cd "$RUNTIME_DIR"

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
GPU_MEMORY_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)"
case "$GPU_NAME" in
  *A100*) ;;
  *) echo "incremental diagnostic requires an A100-40GB: $GPU_NAME" >&2; exit 2 ;;
esac
test "$GPU_MEMORY_MIB" -ge 39000
test "$GPU_MEMORY_MIB" -lt 50000

mapfile -t SAMPLE_IDS < "$QID_FILE"
test "${#SAMPLE_IDS[@]}" -eq "$EXPECTED_QUESTIONS"

if [[ -d "$CELL_ROOT/run" ]] && PYTHONPATH="$RUNTIME_DIR/src" "$ENV_DIR/bin/python" - "$CELL_ROOT/run" "$EXPECTED_QUESTIONS" <<'PY'
import sys
from pathlib import Path

from docprune.evaluation import validate_benchmark_run

valid = validate_benchmark_run(Path(sys.argv[1]), expected_questions=int(sys.argv[2])).valid
raise SystemExit(0 if valid else 1)
PY
then
  validate_retrieval_reference
  echo "validated diagnostic shard already complete"
  exit 0
fi

RESUME_ARGS=()
if [[ -d "$CELL_ROOT/run" ]]; then
  RESUME_ARGS+=(--resume)
fi

"$ENV_DIR/bin/docprune-m3docvqa" evaluate \
  --config configs/docprune-m3docvqa.toml \
  --pages 4 \
  --mode docprune \
  --run-config "$RUN_CONFIG" \
  --index-manifest "$INDEX_MANIFEST" \
  --output "$CELL_ROOT/run" \
  --factory "$FACTORY" \
  --sample-ids "${SAMPLE_IDS[@]}" \
  "${RESUME_ARGS[@]}"

"$ENV_DIR/bin/docprune-m3docvqa" validate-run \
  --run "$CELL_ROOT/run" \
  --expected-questions "$EXPECTED_QUESTIONS" \
  > "$CELL_ROOT/validation.json"

validate_retrieval_reference
