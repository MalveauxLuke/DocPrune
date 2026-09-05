# H200 environment survey record

Status: **not yet performed**

The H200 agent fills this file only after running `SURVEY_COMMANDS.sh`. The
survey is observation-only; no environment, cache, repository, driver, GPU, or
storage changes are permitted before it is recorded.

## Identity and time

- Date/time/timezone:
- Hostname:
- Operator:
- Lab/channel notice status:

## Storage

- `/mnt/data1` mount, free space, ownership:
- `/mnt/data2` mount, free space, ownership:
- `/shared` mount and whether needed (expected: no):
- Confirmed project checkout root:
- Confirmed environment/cache root:
- Root filesystem free space (observation only):

## GPUs

- Driver and CUDA reported by `nvidia-smi`:
- Physical GPUs 4–7 model/memory/process state:
- GPUs 0–3 observed but not selected:
- Candidate smoke GPU:
- Channel notice required before smoke? Why:

## Software

- Shell/OS:
- Git:
- Conda/mamba/micromamba:
- Python candidates:
- Compiler/Ninja:
- Existing CUDA toolkit, if any:
- Existing environment/cache directories relevant to this user:

## Repository and artifacts

- Checkout exists? path/branch/commit/status:
- Sealed cohort present? path/file SHA-256/internal SHA-256:
- Fixed-page fixture/features/PDF subset present?:
- Qwen model revision present?:
- Completed source-built bundle/mappings present?:

## Findings before setup

- Confirmed path substitutions from the templates:
- Missing prerequisites:
- Compatibility risks:
- Proposed minimal setup actions:
- Explicit confirmation that no model/GPU job or environment mutation occurred:
