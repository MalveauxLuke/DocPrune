#!/usr/bin/env python3
"""Fail closed on the GPU assigned to a reviewed evaluation policy."""

from __future__ import annotations

import sys


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: validate_gpu_policy.py POLICY GPU_NAME MEMORY_MIB", file=sys.stderr)
        return 2
    policy, gpu_name, memory_text = sys.argv[1:]
    try:
        memory_mib = int(memory_text)
    except ValueError:
        print("GPU memory must be an integer MiB value", file=sys.stderr)
        return 2

    if policy == "a100-80":
        accepted = "A100" in gpu_name and memory_mib >= 79_000
    elif policy == "quality-40gb":
        accepted = any(model in gpu_name for model in ("A100", "H100", "L40")) and memory_mib >= 39_000
    else:
        print(f"unknown GPU policy: {policy}", file=sys.stderr)
        return 2
    if not accepted:
        print(
            f"GPU {gpu_name!r} with {memory_mib} MiB violates policy {policy!r}",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
