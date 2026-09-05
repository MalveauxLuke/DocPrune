#!/usr/bin/env bash
# Validate the one CUDA-visible L40S without an early-closing GPU pipeline.

set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "usage: probe_task7_l40s_gpu.sh PYTHON" >&2
  exit 2
fi

readonly PYTHON="$1"
test -x "$PYTHON"

"$PYTHON" - <<'PY'
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("Task 7 requires exactly one CUDA-visible GPU")
device = torch.cuda.current_device()
if device != 0:
    raise SystemExit("single CUDA-visible Task 7 device must have logical index zero")
name = str(torch.cuda.get_device_properties(device).name).strip()
if not name or "L40S" not in name:
    raise SystemExit("canonical Task 7 native-boundary shard requires L40S")
print(name)
PY
