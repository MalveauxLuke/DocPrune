#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/paths.env"

COHORT_SHA=$(sha256sum "$TASK9_COHORT" | awk '{print $1}')
FIXTURE_SHA=$(sha256sum "$TASK9_FIXTURE" | awk '{print $1}')
RUNTIME_COMMIT=$(git -C "$TASK9_REPO" rev-parse HEAD)
mkdir -p "$TASK9_ARTIFACT_ROOT/runs/production" "$TASK9_ARTIFACT_ROOT/logs" "$TMPDIR"
nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu --format=csv,noheader

run_worker() {
  local gpu_id=$1
  local batch_index
  for ((batch_index=gpu_id-3; batch_index<100; batch_index+=4)); do
    local root="$TASK9_ARTIFACT_ROOT/runs/production/batch-$(printf '%03d' "$batch_index")"
    mkdir -p "$root"
    CUDA_VISIBLE_DEVICES="$gpu_id" "$TASK9_ENV_PREFIX/bin/python" \
      "$TASK9_REPO/examples/run_task9_confirmation_batch.py" \
      --batch-index "$batch_index" --batch-size 1 --job-root "$root" \
      --runtime-dir "$TASK9_REPO" --runtime-commit "$RUNTIME_COMMIT" \
      --python "$TASK9_ENV_PREFIX/bin/python" \
      --config "$TASK9_REPO/configs/docprune-m3docvqa.toml" \
      --run-config "$TASK9_RUN_CONFIG" --index-manifest "$TASK9_INDEX_MANIFEST" \
      --fixture "$TASK9_FIXTURE" --fixture-sha256 "$FIXTURE_SHA" \
      --cohort "$TASK9_COHORT" --cohort-file-sha256 "$COHORT_SHA" \
      --mappings-dir "$TASK9_MAPPINGS" \
      >"$TASK9_ARTIFACT_ROOT/logs/batch-$(printf '%03d' "$batch_index").out" \
      2>"$TASK9_ARTIFACT_ROOT/logs/batch-$(printf '%03d' "$batch_index").err"
  done
}

# Batch 0 is the admitted smoke. Production covers 1..99 across CoRAL GPUs 4..7.
for gpu_id in 4 5 6 7; do
  run_worker "$gpu_id" &
done
wait
