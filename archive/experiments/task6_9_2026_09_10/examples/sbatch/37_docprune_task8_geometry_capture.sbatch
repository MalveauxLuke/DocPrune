#!/usr/bin/env bash
# One-QID fixed-page post-BTP+QTP geometry capture; no search or Qwen weights.
#SBATCH --job-name=docprune-task8-geometry
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:15:00
#SBATCH --no-requeue
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

: "${RUNTIME_DIR:?set the clean reviewed runtime checkout}"
: "${RUNTIME_COMMIT:?set the exact reviewed runtime commit}"
: "${JOB_ROOT:?set a fresh existing job root}"

readonly PYTHON=/home/lmalveau/mamba-envs/docprune-sol/bin/python
readonly ENV_DIR=/home/lmalveau/mamba-envs/docprune-sol
readonly FIXTURE=/scratch/lmalveau/docprune/task6-fixed-page-gate-v1/fixture-stage64-top4-v2.json
readonly FIXTURE_SHA256=32b3ddd6a1f608db509f002f9541dbc92317b59ac769f0b41fcb20bcc536652b
readonly SMOKE_INPUT_MANIFEST=/scratch/lmalveau/docprune/task8-mineru-smoke-inputs-d2429ba-v1/smoke-input-manifest.json
readonly SMOKE_INPUT_MANIFEST_SHA256=f3c3c3eb00e8f457b66e25a98dbd444878285ddaabf7d364be90ff7036b26eab
readonly REFERENCE_RESULTS=/scratch/lmalveau/docprune/task6-native-l40s-2d2bea2-v1/shard-0000/results.jsonl
readonly REFERENCE_RESULTS_SHA256=50945fc78034d1ce54cc522e13a64d08c599e5be14076f43b7a39e43254ac3f0
readonly QID=e1e6ed53f9ad11813845088f4cf2f6b1
readonly EXPECTED_GEOMETRY_COUNT=3586
readonly EXPECTED_GEOMETRY_SHA256=47b32cf6156dcee41da7eab1686219c760dd73803a135534546c1a50519086e8
readonly OUTPUT="$JOB_ROOT/geometry.json"

test -x "$PYTHON"
test -d "$JOB_ROOT"
test ! -L "$JOB_ROOT"
test ! -e "$OUTPUT"
test "$(git -C "$RUNTIME_DIR" rev-parse HEAD)" = "$RUNTIME_COMMIT"
test -z "$(git -C "$RUNTIME_DIR" status --porcelain --untracked-files=all)"
test "$(sha256sum "$FIXTURE" | cut -d' ' -f1)" = "$FIXTURE_SHA256"
test "$(sha256sum "$SMOKE_INPUT_MANIFEST" | cut -d' ' -f1)" = "$SMOKE_INPUT_MANIFEST_SHA256"
test "$(sha256sum "$REFERENCE_RESULTS" | cut -d' ' -f1)" = "$REFERENCE_RESULTS_SHA256"

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PATH="$ENV_DIR/bin:$PATH"
export PYTHONPATH="$RUNTIME_DIR/src"
export LD_LIBRARY_PATH="$ENV_DIR/lib:$ENV_DIR/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HOME=/scratch/lmalveau/hf_cache
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_CACHE="$HUGGINGFACE_HUB_CACHE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
GPU_MEMORY_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
case "$GPU_NAME" in *L40S*) ;; *) echo "Task 8 geometry capture requires L40S" >&2; exit 2 ;; esac
test "$GPU_MEMORY_MIB" -ge 39000
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader > "$JOB_ROOT/gpu.txt"

"$PYTHON" "$RUNTIME_DIR/examples/run_task8_geometry_capture.py" \
  --fixture "$FIXTURE" \
  --fixture-sha256 "$FIXTURE_SHA256" \
  --smoke-input-manifest "$SMOKE_INPUT_MANIFEST" \
  --smoke-input-manifest-sha256 "$SMOKE_INPUT_MANIFEST_SHA256" \
  --reference-results "$REFERENCE_RESULTS" \
  --reference-results-sha256 "$REFERENCE_RESULTS_SHA256" \
  --qid "$QID" \
  --expected-geometry-count "$EXPECTED_GEOMETRY_COUNT" \
  --expected-geometry-sha256 "$EXPECTED_GEOMETRY_SHA256" \
  --output "$OUTPUT" \
  --runtime-commit "$RUNTIME_COMMIT"

"$PYTHON" - "$OUTPUT" "$EXPECTED_GEOMETRY_COUNT" "$EXPECTED_GEOMETRY_SHA256" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
if payload["status"] != "complete":
    raise SystemExit("Task 8 geometry capture is incomplete")
if payload["geometry_count"] != int(sys.argv[2]) or payload["geometry_sha256"] != sys.argv[3]:
    raise SystemExit("Task 8 geometry identity drifted")
if payload["global_index_loaded"] is not False or payload["retrieval_search_run"] is not False:
    raise SystemExit("Task 8 fixed-page no-search evidence drifted")
if payload["qwen_generation_model_loaded"] is not False:
    raise SystemExit("Task 8 geometry capture loaded forbidden Qwen generation weights")
if "L40S" not in payload["gpu"]["device_name"]:
    raise SystemExit("Task 8 geometry capture did not run on canonical L40S")
PY

sha256sum "$OUTPUT" "$JOB_ROOT/gpu.txt" > "$JOB_ROOT/artifacts.sha256"
