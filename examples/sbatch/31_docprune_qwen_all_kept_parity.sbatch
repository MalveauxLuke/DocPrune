#!/usr/bin/env bash
#SBATCH --job-name=docprune-qwen-parity
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

set -euo pipefail

: "${RUNTIME_DIR:?set RUNTIME_DIR to the clean committed checkout}"
: "${RUNTIME_COMMIT:?set RUNTIME_COMMIT to its exact 40-character commit}"
: "${OUTPUT_DIR:?set OUTPUT_DIR to a fresh artifact directory}"
: "${MODEL_PATH:?set MODEL_PATH to the pinned cached Qwen snapshot}"
: "${PROBE_PAGE:?set PROBE_PAGE to the fixed processor-probe PNG}"
: "${QWEN_REVISION:?set QWEN_REVISION to the pinned model revision}"
: "${PYTHON:=/home/lmalveau/mamba-envs/docprune-sol/bin/python}"

test "${#RUNTIME_COMMIT}" -eq 40
test -d "$RUNTIME_DIR"
test -d "$MODEL_PATH"
test -f "$PROBE_PAGE"
test -x "$PYTHON"
test "$(git -C "$RUNTIME_DIR" rev-parse HEAD)" = "$RUNTIME_COMMIT"
test -z "$(git -C "$RUNTIME_DIR" status --porcelain --untracked-files=all)"
test ! -e "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
  > "$OUTPUT_DIR/gpu.txt"
git -C "$RUNTIME_DIR" show -s --format=fuller HEAD > "$OUTPUT_DIR/runtime-commit.txt"
sha256sum "$PROBE_PAGE" > "$OUTPUT_DIR/probe-page.sha256"

env \
  PYTHONPATH="$RUNTIME_DIR/src" \
  DOCPRUNE_REAL_MODEL_TEST=1 \
  DOCPRUNE_QWEN_MODEL_PATH="$MODEL_PATH" \
  DOCPRUNE_QWEN_PROBE_PAGE="$PROBE_PAGE" \
  DOCPRUNE_QWEN_REVISION="$QWEN_REVISION" \
  "$PYTHON" -m pytest -q \
    "$RUNTIME_DIR/tests/qwen2vl/test_model.py::test_real_model_all_kept_matches_stock_first_step_logits_without_download" \
    --junitxml "$OUTPUT_DIR/junit.xml" \
  | tee "$OUTPUT_DIR/pytest.log"

sha256sum \
  "$OUTPUT_DIR/gpu.txt" \
  "$OUTPUT_DIR/runtime-commit.txt" \
  "$OUTPUT_DIR/probe-page.sha256" \
  "$OUTPUT_DIR/junit.xml" \
  "$OUTPUT_DIR/pytest.log" \
  > "$OUTPUT_DIR/artifacts.sha256"
