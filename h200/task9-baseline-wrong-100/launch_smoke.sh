#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/paths.env"

GPU_ID=${1:?usage: launch_smoke.sh PHYSICAL_GPU_ID}
if [[ "$GPU_ID" -lt 4 || "$GPU_ID" -gt 7 ]]; then
  echo "CoRAL smoke GPU must be a physical GPU ID from 4 through 7" >&2
  exit 2
fi

nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
mkdir -p "$TASK9_ARTIFACT_ROOT/runs/smoke-000" "$TMPDIR"

COHORT_SHA=$(sha256sum "$TASK9_COHORT" | awk '{print $1}')
FIXTURE_SHA=$(sha256sum "$TASK9_FIXTURE" | awk '{print $1}')
RUNTIME_COMMIT=$(git -C "$TASK9_REPO" rev-parse HEAD)

CUDA_VISIBLE_DEVICES="$GPU_ID" "$TASK9_ENV_PREFIX/bin/python" \
  "$TASK9_REPO/examples/run_task9_confirmation_batch.py" \
  --batch-index 0 --batch-size 1 \
  --job-root "$TASK9_ARTIFACT_ROOT/runs/smoke-000" \
  --runtime-dir "$TASK9_REPO" --runtime-commit "$RUNTIME_COMMIT" \
  --python "$TASK9_ENV_PREFIX/bin/python" \
  --config "$TASK9_REPO/configs/docprune-m3docvqa.toml" \
  --run-config "$TASK9_RUN_CONFIG" --index-manifest "$TASK9_INDEX_MANIFEST" \
  --fixture "$TASK9_FIXTURE" --fixture-sha256 "$FIXTURE_SHA" \
  --cohort "$TASK9_COHORT" --cohort-file-sha256 "$COHORT_SHA" \
  --mappings-dir "$TASK9_MAPPINGS"
