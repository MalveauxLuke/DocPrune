# DocPrune SOL Handoff

> **Superseded:** This handoff stopped at structural smoke job `61567743` on
> the Python 3.10 `tomllib` incompatibility in runtime `99dbece`. It is retained
> as historical execution evidence and must not be resumed. The active authority
> is `DOCPRUNE_SOL_SMOKE_RECOVERY_HANDOFF.md`.

## Authority and stopping boundary

Execute this handoff in order. It authorizes:

1. synchronizing the GitHub repository;
2. constructing the pinned environment;
3. running the structural GPU smoke;
4. generating one processor-contract JSON report.

It does **not** authorize `11_docprune_m3docvqa.sbatch`, dataset-wide
embedding, generation, evaluation, or paper-parity claims. Stop after returning
the processor-contract report.

Pinned runtime implementation:

```text
99dbece9f7cd09abdfe35c1ba6b61020218e6f1e
```

Pinned M3DocRAG contract:

```text
29e6ac2294d6b87075a1d45b8a8df175b214248a
```

At the time this original handoff was prepared, no SOL or GPU command had been
run. Its later execution stopped at the failure recorded in the superseding
notice above.

## Phase 0: pull the repository first

Run these light Git operations on the login node. Keep `main` as the control
checkout containing this handoff and use a detached worktree for runtime.

```bash
set -euo pipefail

export REPOSITORY_URL="https://github.com/MalveauxLuke/DocPrune.git"
export CONTROL_DIR="$HOME/DocPrune"
export PROJECT_DIR="$HOME/DocPrune-runtime-99dbece"
export M3DOCRAG_DIR="$HOME/src/m3docrag-docprune"
export ENV_DIR="/home/$USER/mamba-envs/docprune-sol"
export RUN_ROOT="/scratch/$USER/docprune/handoff-99dbece"
export EXPECTED_COMMIT="99dbece9f7cd09abdfe35c1ba6b61020218e6f1e"
export M3DOCRAG_COMMIT="29e6ac2294d6b87075a1d45b8a8df175b214248a"

if [[ -d "$CONTROL_DIR/.git" ]]; then
  test -z "$(git -C "$CONTROL_DIR" status --porcelain)"
  test "$(git -C "$CONTROL_DIR" remote get-url origin)" = "$REPOSITORY_URL"
  git -C "$CONTROL_DIR" switch main
  git -C "$CONTROL_DIR" pull --ff-only origin main
elif [[ -e "$CONTROL_DIR" ]]; then
  echo "CONTROL_DIR exists but is not a Git checkout: $CONTROL_DIR" >&2
  exit 2
else
  git clone "$REPOSITORY_URL" "$CONTROL_DIR"
  git -C "$CONTROL_DIR" switch main
fi

git -C "$CONTROL_DIR" fetch --prune origin
test "$(git -C "$CONTROL_DIR" rev-parse HEAD)" = \
  "$(git -C "$CONTROL_DIR" rev-parse origin/main)"

if [[ -d "$PROJECT_DIR/.git" || -f "$PROJECT_DIR/.git" ]]; then
  test -z "$(git -C "$PROJECT_DIR" status --porcelain)"
  test "$(git -C "$PROJECT_DIR" rev-parse HEAD)" = "$EXPECTED_COMMIT"
elif [[ -e "$PROJECT_DIR" ]]; then
  echo "PROJECT_DIR exists but is not the expected worktree: $PROJECT_DIR" >&2
  exit 2
else
  git -C "$CONTROL_DIR" worktree add --detach "$PROJECT_DIR" "$EXPECTED_COMMIT"
fi

mkdir -p "$HOME/src"
if [[ -d "$M3DOCRAG_DIR/.git" ]]; then
  test -z "$(git -C "$M3DOCRAG_DIR" status --porcelain)"
  git -C "$M3DOCRAG_DIR" fetch --prune origin
else
  git clone https://github.com/bloomberg/m3docrag.git "$M3DOCRAG_DIR"
fi
git -C "$M3DOCRAG_DIR" checkout --detach "$M3DOCRAG_COMMIT"
test "$(git -C "$M3DOCRAG_DIR" rev-parse HEAD)" = "$M3DOCRAG_COMMIT"
```

If any cleanliness, remote, fast-forward, or commit check fails, stop and
report it. Do not reset, delete, overwrite, or repair an existing checkout.

## Phase 1: environment construction

Request a setup allocation before installing or importing packages:

```bash
salloc -p lightwork -q public -t 04:00:00 -c 4 --mem=32G
module load mamba/latest
mkdir -p "$RUN_ROOT"

if [[ -x "$ENV_DIR/bin/python" ]]; then
  mamba env update -p "$ENV_DIR" \
    -f "$PROJECT_DIR/environments/docprune-sol.yml" --prune
else
  mamba env create -p "$ENV_DIR" \
    -f "$PROJECT_DIR/environments/docprune-sol.yml"
fi

export PATH="$ENV_DIR/bin:$PATH"
export MAX_JOBS=4
python -m pip install flash-attn==2.5.8 --no-build-isolation
python -m pip install -e "$PROJECT_DIR" --no-deps
python -m pip install -e "$M3DOCRAG_DIR" --no-deps
python -m pip freeze | LC_ALL=C sort > "$RUN_ROOT/environment-freeze.txt"
```

If FlashAttention 2.5.8 fails to build against the pinned stack, preserve the
full log and stop. Do not upgrade or substitute Transformers, PyTorch,
FlashAttention, ColPali, or Python.

## Phase 2: structural GPU smoke

Exit the interactive setup allocation, then submit:

```bash
cd "$PROJECT_DIR"
sbatch \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",ENV_DIR="$ENV_DIR",EXPECTED_COMMIT="$EXPECTED_COMMIT",RUN_ROOT="$RUN_ROOT/smoke" \
  examples/sbatch/10_docprune_smoke.sbatch
```

Pass conditions:

- exact runtime commit matches;
- CUDA, Transformers 4.46.3, and FlashAttention 2.5.8 import;
- the full test suite and Ruff pass;
- top-4 inspection contains the supplement Table B values;
- the runtime worktree remains clean.

On failure, preserve the Slurm logs and `$RUN_ROOT/smoke`, report the exact
command and traceback, and stop. Retry only scheduler/preemption failures with
identical inputs.

## Phase 3: exact processor-contract probe

This probe loads processors and configuration only. It does not load model
weights, generate answers, or run an evaluation.

The supplement identifies `Qwen/Qwen2-VL-7B-Instruct`,
`vidore/colpali-v1`, and `m3docrag/m3docvqa` but gives no immutable revisions.
Qwen was resolved on 2026-08-15 to the reconstruction pin below. Resolve the
ColPali revision using the authenticated Hugging Face session; do not
substitute a differently named ColPali repository.

Set `M3DOCVQA_PAGE_ROOT` to the existing derived page-image directory on SOL.
The probe deterministically selects the lexicographically first PNG, JPEG, or
JPG. It never copies the image into the repository or report.

Run from a lightwork allocation:

```bash
set -euo pipefail
export PATH="$ENV_DIR/bin:$PATH"
export HF_HOME="${HF_HOME:-/scratch/$USER/hf_cache}"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TOKENIZERS_PARALLELISM=false
export QWEN_MODEL="Qwen/Qwen2-VL-7B-Instruct"
export QWEN_REVISION="eed13092ef92e448dd6875b2a00151bd3f7db0ac"
export COLPALI_MODEL="vidore/colpali-v1"
: "${M3DOCVQA_PAGE_ROOT:?set M3DOCVQA_PAGE_ROOT to the existing page-image directory}"

export COLPALI_REVISION="$(python - <<'PY'
from huggingface_hub import HfApi
info = HfApi().model_info("vidore/colpali-v1", revision="main")
if not info.sha or len(info.sha) != 40:
    raise SystemExit("ColPali did not resolve to a 40-character revision")
print(info.sha)
PY
)"

export PROBE_IMAGE="$(python - <<'PY'
import os
from pathlib import Path

root = Path(os.environ["M3DOCVQA_PAGE_ROOT"])
candidates = sorted(
    path for path in root.rglob("*")
    if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
)
if not candidates:
    raise SystemExit(f"no page images found under {root}")
print(candidates[0])
PY
)"
test -n "$PROBE_IMAGE"
test -f "$PROBE_IMAGE"

docprune-m3docvqa probe-processors \
  --page-image "$PROBE_IMAGE" \
  --qwen-model "$QWEN_MODEL" \
  --qwen-revision "$QWEN_REVISION" \
  --colpali-model "$COLPALI_MODEL" \
  --colpali-revision "$COLPALI_REVISION" \
  --output "$RUN_ROOT/processor-contract.json" \
  | tee "$RUN_ROOT/processor-contract.stdout.json"

python -m json.tool "$RUN_ROOT/processor-contract.json" >/dev/null
sha256sum "$PROBE_IMAGE" > "$RUN_ROOT/probe-image.sha256"
```

The SHA-256 file records identity only; do not copy or commit the page image.
If Hugging Face returns 401/403 or the exact ColPali ID does not resolve, save
the error and stop without substituting another model.

### Output schema

`$RUN_ROOT/processor-contract.json` has schema version 1:

```json
{
  "schema_version": 1,
  "resources": {
    "qwen": {"model": "...", "revision": "40 hex characters"},
    "colpali": {"model": "...", "revision": "40 hex characters"}
  },
  "page": {"raw_size_wh": [0, 0]},
  "qwen": {
    "grid_thw": [1, 0, 0],
    "merged_visual_token_count": 0,
    "patch_size": 0,
    "pixel_values_shape": [],
    "resized_size_hw": [0, 0],
    "spatial_merge_size": 2,
    "temporal_patch_size": 0
  },
  "colpali": {
    "attention_token_count": 0,
    "candidate_visual_token_count": 0,
    "image_token_id": 0,
    "image_token_positions": [],
    "inferred_visual_grid_hw": [0, 0],
    "pixel_values_shape": [],
    "sequence_length": 0
  },
  "mapping_checks": {
    "colpali_visual_grid_inferred": false,
    "qwen_merge_groups_valid": false,
    "raster_order_verified": false
  },
  "unresolved": [
    "ColPali raster order requires review against the pinned processor implementation."
  ]
}
```

The report intentionally excludes raw token IDs, image pixels, credentials,
weights, and dataset content. `raster_order_verified` remains false until the
pinned ColPali implementation is reviewed; this is expected and is why the
benchmark remains on hold.

## Return and stop

Return:

- control and runtime commit hashes;
- environment freeze path;
- smoke Slurm job ID and pass/fail result;
- `processor-contract.json` and `probe-image.sha256` paths;
- all errors or unresolved fields.

Then stop. `examples/sbatch/11_docprune_m3docvqa.sbatch` remains prepared but
unauthorized. A later approved handoff must add the reviewed integration
factory and require all-kept baseline equivalence before any benchmark.
