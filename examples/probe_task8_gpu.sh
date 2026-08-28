#!/usr/bin/env bash
# Publish one no-replace GPU identity without SIGPIPE-prone early-closing pipes.

set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "usage: probe_task8_gpu.sh PYTHON OUTPUT_JSON" >&2
  exit 2
fi

readonly PYTHON="$1"
readonly OUTPUT_JSON="$2"
test -x "$PYTHON"
case "$OUTPUT_JSON" in /*) ;; *) echo "GPU manifest path must be absolute" >&2; exit 2 ;; esac
test ! -e "$OUTPUT_JSON"

"$PYTHON" - "$OUTPUT_JSON" <<'PY'
import json
import os
import subprocess
import sys

import torch

output = sys.argv[1]
if not os.path.isabs(output):
    raise SystemExit("GPU manifest path must be absolute")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("Task 8 requires exactly one CUDA-visible GPU")
device = torch.cuda.current_device()
if device != 0:
    raise SystemExit("single CUDA-visible Task 8 device must have logical index zero")
properties = torch.cuda.get_device_properties(device)
name = str(properties.name).strip()
memory_total_mib = int(properties.total_memory) // (1024 * 1024)
if not name or memory_total_mib < 12000:
    raise SystemExit("Task 8 CUDA-visible GPU identity is invalid or undersized")
driver_output = subprocess.run(
    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
    check=True,
    capture_output=True,
    text=True,
).stdout
drivers = {line.strip() for line in driver_output.splitlines() if line.strip()}
if len(drivers) != 1:
    raise SystemExit("Task 8 node GPU driver identity is empty or inconsistent")
payload = json.dumps(
    {
        "driver_version": next(iter(drivers)),
        "memory_total_mib": memory_total_mib,
        "name": name,
    },
    sort_keys=True,
    separators=(",", ":"),
).encode()
descriptor = os.open(
    output,
    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
    0o640,
)
with os.fdopen(descriptor, "wb") as stream:
    stream.write(payload)
    stream.flush()
    os.fsync(stream.fileno())
PY
