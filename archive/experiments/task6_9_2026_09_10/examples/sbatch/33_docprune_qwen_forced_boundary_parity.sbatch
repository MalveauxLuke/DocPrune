#!/usr/bin/env bash
# Runnable successor for the reviewed Task 3 seven-boundary all-kept parity gate.
# Submission still requires a current sealed SOL handoff naming this launcher.
#SBATCH --job-name=docprune-qwen-forced-boundary-parity
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

set -euo pipefail

: "${RUNTIME_DIR:?set RUNTIME_DIR to the clean committed checkout}"
: "${RUNTIME_COMMIT:?set RUNTIME_COMMIT to its exact reviewed 40-character commit}"
: "${OUTPUT_DIR:?set OUTPUT_DIR to a fresh absolute artifact directory}"
: "${FORCED_BOUNDARY_ARTIFACT:?set FORCED_BOUNDARY_ARTIFACT to a fresh path inside OUTPUT_DIR}"
: "${MODEL_PATH:?set MODEL_PATH to the cached pinned Qwen snapshot}"
: "${PROBE_PAGE:?set PROBE_PAGE to the fixed cached processor-probe PNG}"
: "${PYTHON:=/home/lmalveau/mamba-envs/docprune-sol/bin/python}"

readonly QWEN_REVISION=eed13092ef92e448dd6875b2a00151bd3f7db0ac

[[ "$RUNTIME_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "$OUTPUT_DIR" == /* ]]
test -d "$RUNTIME_DIR"
test -d "$MODEL_PATH"
test -f "$PROBE_PAGE"
test -x "$PYTHON"
test "$(git -C "$RUNTIME_DIR" rev-parse HEAD)" = "$RUNTIME_COMMIT"
test -z "$(git -C "$RUNTIME_DIR" status --porcelain --untracked-files=all)"
test ! -e "$OUTPUT_DIR"
mkdir "$OUTPUT_DIR"
OUTPUT_DIR="$(realpath -e "$OUTPUT_DIR")"
[[ "$FORCED_BOUNDARY_ARTIFACT" == "$OUTPUT_DIR/"* ]]
test ! -e "$FORCED_BOUNDARY_ARTIFACT"
test "$(dirname "$FORCED_BOUNDARY_ARTIFACT")" = "$OUTPUT_DIR"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
  > "$OUTPUT_DIR/gpu.txt"
git -C "$RUNTIME_DIR" show -s --format=fuller HEAD > "$OUTPUT_DIR/runtime-commit.txt"
sha256sum "$PROBE_PAGE" > "$OUTPUT_DIR/probe-page.sha256"

set +e
env \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  HF_DATASETS_OFFLINE=1 \
  PYTHONPATH="$RUNTIME_DIR/src" \
  DOCPRUNE_REAL_MODEL_TEST=1 \
  DOCPRUNE_QWEN_MODEL_PATH="$MODEL_PATH" \
  DOCPRUNE_QWEN_PROBE_PAGE="$PROBE_PAGE" \
  DOCPRUNE_QWEN_REVISION="$QWEN_REVISION" \
  DOCPRUNE_FORCED_BOUNDARY_ARTIFACT="$FORCED_BOUNDARY_ARTIFACT" \
  "$PYTHON" -m pytest -q \
    "$RUNTIME_DIR/tests/qwen2vl/test_model.py::test_real_model_forced_all_kept_boundaries_match_stock_without_download" \
    --junitxml "$OUTPUT_DIR/junit.xml" \
  2>&1 | tee "$OUTPUT_DIR/pytest.log"
pytest_status=${PIPESTATUS[0]}
set -e

test -f "$FORCED_BOUNDARY_ARTIFACT"
test -f "$OUTPUT_DIR/junit.xml"
sha256sum \
  "$OUTPUT_DIR/gpu.txt" \
  "$OUTPUT_DIR/runtime-commit.txt" \
  "$OUTPUT_DIR/probe-page.sha256" \
  "$FORCED_BOUNDARY_ARTIFACT" \
  "$OUTPUT_DIR/junit.xml" \
  "$OUTPUT_DIR/pytest.log" \
  > "$OUTPUT_DIR/artifacts.sha256"

exit "$pytest_status"
