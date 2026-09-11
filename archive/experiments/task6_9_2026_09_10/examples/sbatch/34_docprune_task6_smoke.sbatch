#!/usr/bin/env bash
# One sealed nine-cell Task 6 portability smoke; GPU constraint is supplied at submission.
#SBATCH --job-name=docprune-task6-smoke
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

: "${RUNTIME_DIR:?set the clean reviewed runtime checkout}"
: "${RUNTIME_COMMIT:?set the exact reviewed runtime commit}"
: "${JOB_ROOT:?set a fresh existing job root}"
: "${GPU_EXPECTED:?set a30, a100_40, h100, or l40s}"
: "${PYTHON:=/home/lmalveau/mamba-envs/docprune-sol/bin/python}"

readonly FIXTURE=/scratch/lmalveau/docprune/task6-fixed-page-gate-v1/fixture-stage64-top4-v2.json
readonly FIXTURE_SHA256=32b3ddd6a1f608db509f002f9541dbc92317b59ac769f0b41fcb20bcc536652b
readonly GATE=/scratch/lmalveau/docprune/task6-fixed-page-gate-v1/gate-manifest-v4.json
readonly GATE_SHA256=4d44e297081b152c2e492ec68be58445c7ac17c8d106ceb2ddb1f6331fd57761
readonly ATTEMPT_ROOT=/scratch/lmalveau/docprune/benchmark-4e2473b/attempt-2
readonly RUN_CONFIG="$ATTEMPT_ROOT/run-configs/docprune-top4.json"
readonly INDEX_MANIFEST="$ATTEMPT_ROOT/indexes/docprune/top4/docprune/manifest.json"
readonly CONFIG="$RUNTIME_DIR/configs/docprune-m3docvqa.toml"
readonly OUTPUT_DIR="$JOB_ROOT/run"
readonly ENV_DIR=/home/lmalveau/mamba-envs/docprune-sol
readonly PDFTOOLS_DIR=/home/lmalveau/mamba-envs/m3docvqa-acquisition
readonly M3DOCRAG_DIR=/home/lmalveau/src/m3docrag-benchmark-29e6ac2
readonly M3DOCRAG_COMMIT=29e6ac2294d6b87075a1d45b8a8df175b214248a

case "$GPU_EXPECTED" in a30|a100_40|h100|l40s) ;; *) exit 2 ;; esac
test -d "$JOB_ROOT"
test ! -e "$OUTPUT_DIR"
test -x "$PYTHON"
test "$(git -C "$RUNTIME_DIR" rev-parse HEAD)" = "$RUNTIME_COMMIT"
test -z "$(git -C "$RUNTIME_DIR" status --porcelain --untracked-files=all)"
test "$(git -C "$M3DOCRAG_DIR" rev-parse HEAD)" = "$M3DOCRAG_COMMIT"
test -z "$(git -C "$M3DOCRAG_DIR" status --porcelain --untracked-files=all)"
test "$(sha256sum "$FIXTURE" | cut -d' ' -f1)" = "$FIXTURE_SHA256"
test "$(sha256sum "$GATE" | cut -d' ' -f1)" = "$GATE_SHA256"

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
GPU_MEMORY_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)"
case "$GPU_EXPECTED:$GPU_NAME" in
  a30:*A30*|a100_40:*A100*|h100:*H100*|l40s:*L40S*) ;;
  *) echo "GPU identity mismatch: expected $GPU_EXPECTED, received $GPU_NAME" >&2; exit 2 ;;
esac
case "$GPU_EXPECTED" in
  a30) test "$GPU_MEMORY_MIB" -ge 23000 ;;
  a100_40) test "$GPU_MEMORY_MIB" -ge 39000; test "$GPU_MEMORY_MIB" -lt 50000 ;;
  *) test "$GPU_MEMORY_MIB" -ge 39000 ;;
esac
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader > "$JOB_ROOT/gpu.txt"
git -C "$RUNTIME_DIR" show -s --format=fuller HEAD > "$JOB_ROOT/runtime-commit.txt"

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
  --shard 0 \
  --kind smoke \
  --output "$OUTPUT_DIR"

"$PYTHON" - "$OUTPUT_DIR/results.jsonl" "$FIXTURE" "$GPU_EXPECTED" <<'PY'
import json
import sys
from pathlib import Path

results = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines() if line]
fixture = json.loads(Path(sys.argv[2]).read_text())
expected_pages = fixture["questions"][0]["pages"]
expected_ids = [(row["doc_id"], row["page_index"], row["score"]) for row in expected_pages]
if len(results) != 9 or [row["matrix_cell"] for row in results] != list(range(9)):
    raise SystemExit("Task 6 smoke did not produce the exact nine-cell matrix")
for row in results:
    observed = [(page["doc_id"], page["page_index"], page["score"]) for page in row["retrieved_pages"]]
    if observed != expected_ids:
        raise SystemExit("Task 6 smoke fixed-page identity drifted")
    if row["fixed_page_provenance"] is not True or row["global_index_loaded"] is not False:
        raise SystemExit("Task 6 smoke no-search evidence is invalid")
    selection = row["policy_selection"]
    if not selection["prefill_cache_lengths"] or len(selection["retained_mrope_position_shape"]) != 3:
        raise SystemExit("Task 6 smoke lacks cache/M-RoPE evidence")
    if len(selection["retained_mrope_position_sha256"]) != 64:
        raise SystemExit("Task 6 smoke M-RoPE digest is invalid")
print(json.dumps({"gpu_role": sys.argv[3], "result_count": len(results)}, sort_keys=True))
PY

sha256sum \
  "$FIXTURE" \
  "$GATE" \
  "$OUTPUT_DIR/run_manifest.json" \
  "$OUTPUT_DIR/results.jsonl" \
  "$JOB_ROOT/gpu.txt" \
  "$JOB_ROOT/runtime-commit.txt" \
  > "$JOB_ROOT/artifacts.sha256"
