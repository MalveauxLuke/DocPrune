# DocPrune SOL Smoke-Recovery Handoff

## Authority and stopping boundary

This is the only active SOL handoff. Execute it in order. It authorizes:

1. creating new clean detached runtime worktrees;
2. updating the existing `docprune-sol` environment from the corrected YAML;
3. reinstalling the retained, checksum-verified FlashAttention wheel;
4. installing the pinned DocPrune and M3DocRAG sources;
5. running one structural GPU smoke; and
6. returning the recovery report.

Stop after the smoke report. This handoff does **not** authorize dataset
acquisition, processor probing, indexing, embedding, generation, evaluation,
training, or `11_docprune_m3docvqa.sbatch`.

Exact revisions:

```text
DocPrune: 64ea70c66a8f9e3dbce804d1fde265a4a2b8b09d
M3DocRAG: 29e6ac2294d6b87075a1d45b8a8df175b214248a
```

The original runtime at `/home/lmalveau/DocPrune-runtime-99dbece`, smoke job
`61567743`, and `/scratch/lmalveau/docprune/handoff-99dbece` are immutable
failure evidence. Do not edit, reset, remove, or reuse that runtime.

## Recorded recovery inputs

The existing environment is `/home/lmalveau/mamba-envs/docprune-sol`. The
retained FlashAttention inputs are:

```text
source SHA-256: 2e5b2bcff6d5cff40d494af91ecd1eb3c5b4520a6ce7a0a8b1f9c1ed129fb402
wheel SHA-256:  5f5d5a9b4a7a4bd8c2cdd7c58a2a58067d8249c2a14fc68f67f5ab334e1d0394
```

The source archive, wheel, build manifest, and checksum files must already
exist below the recorded artifact directory. A missing or mismatched artifact
is a stop condition; do not rebuild FlashAttention under this handoff.

## Phase 0: verify control state and create clean worktrees

Run these light Git operations from the login shell:

```bash
set -euo pipefail

export CONTROL_DIR="/home/lmalveau/DocPrune"
export PROJECT_DIR="/home/lmalveau/DocPrune-runtime-64ea70c"
export FAILED_PROJECT_DIR="/home/lmalveau/DocPrune-runtime-99dbece"
export M3DOCRAG_SOURCE_DIR="/home/lmalveau/src/m3docrag-docprune"
export M3DOCRAG_DIR="/home/lmalveau/src/m3docrag-runtime-29e6ac2"
export ENV_DIR="/home/lmalveau/mamba-envs/docprune-sol"
export RUN_ROOT="/scratch/lmalveau/docprune/handoff-64ea70c"
export EXPECTED_COMMIT="64ea70c66a8f9e3dbce804d1fde265a4a2b8b09d"
export FAILED_COMMIT="99dbece9f7cd09abdfe35c1ba6b61020218e6f1e"
export M3DOCRAG_COMMIT="29e6ac2294d6b87075a1d45b8a8df175b214248a"
export FLASH_ARTIFACT_DIR="/scratch/lmalveau/docprune/handoff-99dbece/artifacts/flash-attn-2.5.8-torch2.4.1-cu121-cp310-gcc12.1.0-job61554625"
export FLASH_WHEEL="$FLASH_ARTIFACT_DIR/wheels/flash_attn-2.5.8-cp310-cp310-linux_x86_64.whl"

test -d "$CONTROL_DIR/.git"
test "$(git -C "$CONTROL_DIR" remote get-url origin)" = \
  "https://github.com/MalveauxLuke/DocPrune.git"
test "$(git -C "$CONTROL_DIR" branch --show-current)" = "main"
test -z "$(git -C "$CONTROL_DIR" status --porcelain)"
git -C "$CONTROL_DIR" cat-file -e "$EXPECTED_COMMIT^{commit}"
git -C "$CONTROL_DIR" merge-base --is-ancestor "$EXPECTED_COMMIT" HEAD

test "$(git -C "$FAILED_PROJECT_DIR" rev-parse HEAD)" = "$FAILED_COMMIT"
test -z "$(git -C "$FAILED_PROJECT_DIR" status --porcelain)"

mkdir -p "$RUN_ROOT"

if [[ -d "$PROJECT_DIR/.git" || -f "$PROJECT_DIR/.git" ]]; then
  test "$(git -C "$PROJECT_DIR" rev-parse HEAD)" = "$EXPECTED_COMMIT"
  test -z "$(git -C "$PROJECT_DIR" status --porcelain)"
elif [[ -e "$PROJECT_DIR" ]]; then
  echo "unexpected PROJECT_DIR: $PROJECT_DIR" >&2
  exit 2
else
  git -C "$CONTROL_DIR" worktree add --detach "$PROJECT_DIR" "$EXPECTED_COMMIT"
fi

test -d "$M3DOCRAG_SOURCE_DIR/.git"
test "$(git -C "$M3DOCRAG_SOURCE_DIR" remote get-url origin)" = \
  "https://github.com/bloomberg/m3docrag.git"
git -C "$M3DOCRAG_SOURCE_DIR" cat-file -e "$M3DOCRAG_COMMIT^{commit}"
git -C "$M3DOCRAG_SOURCE_DIR" status --porcelain > \
  "$RUN_ROOT/preserved-m3docrag-source-status.txt"

if [[ -d "$M3DOCRAG_DIR/.git" || -f "$M3DOCRAG_DIR/.git" ]]; then
  test "$(git -C "$M3DOCRAG_DIR" rev-parse HEAD)" = "$M3DOCRAG_COMMIT"
  test -z "$(git -C "$M3DOCRAG_DIR" status --porcelain)"
elif [[ -e "$M3DOCRAG_DIR" ]]; then
  echo "unexpected M3DOCRAG_DIR: $M3DOCRAG_DIR" >&2
  exit 2
else
  git -C "$M3DOCRAG_SOURCE_DIR" worktree add --detach \
    "$M3DOCRAG_DIR" "$M3DOCRAG_COMMIT"
fi
```

Any failed path, remote, revision, ancestry, or cleanliness assertion stops the
handoff. Do not reset or delete an existing checkout.

## Phase 1: reuse and refresh the pinned environment

Run the following as one `lightwork` compute step. It keeps caches and build
outputs on scratch and installs M3DocRAG from a Git archive so packaging cannot
dirty its clean worktree.

```bash
export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR="$RUN_ROOT/cache/pip"
export XDG_CACHE_HOME="$RUN_ROOT/cache/xdg"
export TORCH_HOME="$RUN_ROOT/cache/torch"
export HF_HOME="$RUN_ROOT/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$HUGGINGFACE_HUB_CACHE"

env -u SLURM_JOB_ID -u SLURM_JOBID -u SLURM_STEP_ID -u SLURM_STEPID \
srun --export=ALL -p lightwork -q public -t 01:00:00 -c 4 --mem=32G \
  /bin/bash -lc '
set -euo pipefail
module load mamba/latest

export PYTHONNOUSERSITE=1
export PATH="$ENV_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_DIR/lib:$ENV_DIR/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

sha256sum -c "$FLASH_ARTIFACT_DIR/source-SHA256SUMS"
sha256sum -c "$FLASH_ARTIFACT_DIR/wheel-SHA256SUMS"

mamba env update -p "$ENV_DIR" \
  -f "$PROJECT_DIR/environments/docprune-sol.yml" --prune
export PATH="$ENV_DIR/bin:$PATH"

python -m pip install --no-deps --force-reinstall "$FLASH_WHEEL" \
  2>&1 | tee "$RUN_ROOT/install-flash-wheel.log"

M3DOCRAG_ARCHIVE="$RUN_ROOT/m3docrag-$M3DOCRAG_COMMIT.tar.gz"
git -C "$M3DOCRAG_DIR" archive --format=tar.gz \
  --prefix=m3docrag/ "$M3DOCRAG_COMMIT" > "$M3DOCRAG_ARCHIVE"
sha256sum "$M3DOCRAG_ARCHIVE" > "$RUN_ROOT/m3docrag-source.sha256"
python -m pip install --no-deps --force-reinstall "$M3DOCRAG_ARCHIVE" \
  2>&1 | tee "$RUN_ROOT/install-m3docrag.log"

python -m pip install --no-deps -e "$PROJECT_DIR" \
  2>&1 | tee "$RUN_ROOT/install-docprune.log"

python - <<"PY" | tee "$RUN_ROOT/environment-check.txt"
import flash_attn
import sys
import tokenizers
import tomli
import torch
import transformers

print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("tokenizers", tokenizers.__version__)
print("tomli", tomli.__version__)
print("flash_attn", flash_attn.__version__)

assert sys.version_info[:2] == (3, 10)
assert torch.__version__.startswith("2.4.1")
assert torch.version.cuda == "12.1"
assert transformers.__version__ == "4.46.3"
assert tokenizers.__version__ == "0.20.3"
assert tomli.__version__ == "2.4.1"
assert flash_attn.__version__ == "2.5.8"
PY

python -m pip freeze | LC_ALL=C sort > "$RUN_ROOT/environment-freeze.txt"
test -z "$(git -C "$PROJECT_DIR" status --porcelain)"
test -z "$(git -C "$M3DOCRAG_DIR" status --porcelain)"
'
```

If the allocation is preempted before package mutation completes, retry the
same command. Any hash, Mamba, installation, import, version, or cleanliness
failure stops the handoff and must be reported with the saved logs.

## Phase 2: structural GPU smoke

Submit exactly one smoke from the corrected runtime:

```bash
cd "$PROJECT_DIR"
SMOKE_ROOT="$RUN_ROOT/smoke"
mkdir -p "$SMOKE_ROOT"

JOB_ID="$(sbatch --parsable \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",ENV_DIR="$ENV_DIR",EXPECTED_COMMIT="$EXPECTED_COMMIT",RUN_ROOT="$SMOKE_ROOT" \
  examples/sbatch/10_docprune_smoke.sbatch)"
printf '%s\n' "$JOB_ID" | tee "$RUN_ROOT/smoke-job-id.txt"
```

Wait for the job to reach a terminal state, then record:

```bash
sacct -j "$JOB_ID" --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,NodeList \
  > "$RUN_ROOT/smoke-sacct.txt"
cat "$RUN_ROOT/smoke-sacct.txt"
```

Pass requires all of the following:

- the batch job state is `COMPLETED` with exit code `0:0`;
- `imports.txt` records PyTorch 2.4.1, CUDA 12.1, Transformers 4.46.3,
  and FlashAttention 2.5.8;
- all tests and Ruff pass;
- `inspect-top4.json` contains the supplement Table B top-4 values;
- both new worktrees remain clean at their exact commits.

On pass, record a machine-readable gate:

```bash
export JOB_ID
python - <<'PY' > "$RUN_ROOT/smoke-pass.json"
import json
import os

print(json.dumps({
    "schema_version": 1,
    "status": "passed",
    "job_id": os.environ["JOB_ID"],
    "docprune_commit": os.environ["EXPECTED_COMMIT"],
    "m3docrag_commit": os.environ["M3DOCRAG_COMMIT"],
    "flash_attention_wheel_sha256": "5f5d5a9b4a7a4bd8c2cdd7c58a2a58067d8249c2a14fc68f67f5ab334e1d0394",
}, sort_keys=True, indent=2))
PY
python -m json.tool "$RUN_ROOT/smoke-pass.json" >/dev/null
```

Do not create `smoke-pass.json` unless every pass condition is verified.

## Return and stop

Return:

- control, corrected runtime, failed runtime, and M3DocRAG commits;
- new environment-freeze and artifact checksum paths;
- smoke job ID, Slurm accounting, and pass/fail result;
- smoke output paths and any errors;
- both worktree cleanliness results.

Then stop. The staged M3DocVQA acquisition/probe handoff remains inactive
until this pass report is reviewed and `CURRENT_SOL_TASK.md` explicitly
activates it.
