#!/usr/bin/env bash
set -euo pipefail
correction_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
correction_python=${CORRECTION_PYTHON:-/mnt/data2/eunwooim/.conda/envs/docprune-h200/bin/python}
correction_data=${CORRECTION_DATA:-$correction_repo/task9-h200-local-data/correction-depth40}
if [[ ! -x "$correction_python" ]]; then
  echo "Set CORRECTION_PYTHON to the existing Python 3.10+ environment executable." >&2
  exit 1
fi
roots=(--reuse-root "$correction_repo/task9-h200-local-data")
for root in "$@"; do roots+=(--reuse-root "$root"); done
PYTHONPATH="$correction_repo/src${PYTHONPATH:+:$PYTHONPATH}" "$correction_python" \
  "$correction_repo/examples/package_correction_depth.py" check-assets \
  --package "$correction_repo/h200/correction-depth/recipe" \
  "${roots[@]}" --output "$correction_data/asset-check.json"
