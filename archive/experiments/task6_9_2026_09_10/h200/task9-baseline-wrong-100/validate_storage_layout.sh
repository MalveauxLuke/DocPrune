#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(git -C "$HERE" rev-parse --show-toplevel)
EXPECTED_ROOT=/mnt/data1/eunwooim/DocPrune/task9-h200-local-data

git -C "$REPO" check-ignore -q task9-h200-local-data/probe

# shellcheck disable=SC1091
source "$HERE/paths.env.example"

[[ "$TASK9_REPO" == /mnt/data1/eunwooim/DocPrune ]]
[[ "$TASK9_ARTIFACT_ROOT" == "$EXPECTED_ROOT" ]]
[[ "$TASK9_COHORT" == "$EXPECTED_ROOT/inputs/fixed/cohort.json" ]]
[[ "$TASK9_FIXTURE" == "$EXPECTED_ROOT/inputs/fixed/fixture.json" ]]
[[ "$TASK9_MAPPINGS" == "$EXPECTED_ROOT/inputs/mappings" ]]
[[ "$TASK9_RUN_CONFIG" == "$EXPECTED_ROOT/manifests/run-config.json" ]]
[[ "$TASK9_INDEX_MANIFEST" == "$EXPECTED_ROOT/manifests/index-manifest.json" ]]
[[ "$TASK9_ENV_PREFIX" == /mnt/data2/eunwooim/.conda/envs/docprune-h200 ]]
[[ "$HF_HOME" == /mnt/data2/eunwooim/hf-cache ]]
[[ "$TMPDIR" == /mnt/data2/eunwooim/tmp/docprune-task9 ]]

grep -Fq 'runs/smoke/batch-000' "$HERE/launch_smoke.sh"
grep -Fq 'runs/production' "$HERE/launch_production.sh"
grep -Fq 'task9-h200-local-data/' "$REPO/.gitignore"
grep -Fq 'task9-h200-local-data/' "$HERE/AGENTS.md"
grep -Fq 'task9-h200-local-data/' "$HERE/HANDOFF.md"
grep -Fq 'task9-h200-local-data/' "$HERE/TRANSFER_MANIFEST.md"
grep -Fxq '/.gitignore' "$HERE/SPARSE_CHECKOUT_PATHS.txt"

if grep -R -n -F '/mnt/data1/eunwooim/docprune-task9-baseline-wrong100-v1' \
  "$HERE" --exclude=validate_storage_layout.sh; then
  echo 'legacy outside-checkout Task 9 data root remains' >&2
fi
