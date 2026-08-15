# DocPrune SOL Handoff

## Authority and boundary

This handoff authorizes environment construction and the structural GPU smoke
only. It does **not** authorize M3DocVQA benchmark submission yet. Stop after
the processor-contract report because the paper does not specify the ColPali
visual-token slice or exact resize-to-Qwen-grid mapping.

Runtime artifact commit:

```text
ee88169b50622ea940f72444a155674514476398
```

M3DocRAG contract commit:

```text
29e6ac2294d6b87075a1d45b8a8df175b214248a
```

No SOL or GPU command has been run by the preparing agent.

## Required locations

```bash
export PROJECT_DIR="$HOME/query-relevant-document-token-pruning"
export M3DOCRAG_DIR="$HOME/src/m3docrag-docprune"
export ENV_DIR="/home/$USER/mamba-envs/docprune-sol"
export RUN_ROOT="/scratch/$USER/docprune/handoff-ee88169"
export EXPECTED_COMMIT="ee88169b50622ea940f72444a155674514476398"
```

The repository has no Git remote. Transfer the prepared Git bundle to
`$HOME/incoming/docprune-ee88169.bundle`; do not reconstruct files manually.

## Phase 0: checkout and environment

Login node operations may create directories, clone, checkout, and submit. Run
all environment builds and imports inside a compute allocation.

```bash
mkdir -p "$HOME/incoming" "$HOME/src"
git clone "$HOME/incoming/docprune-ee88169.bundle" "$PROJECT_DIR"
git -C "$PROJECT_DIR" checkout --detach "$EXPECTED_COMMIT"
test "$(git -C "$PROJECT_DIR" rev-parse HEAD)" = "$EXPECTED_COMMIT"

git clone https://github.com/bloomberg/m3docrag.git "$M3DOCRAG_DIR"
git -C "$M3DOCRAG_DIR" checkout --detach 29e6ac2294d6b87075a1d45b8a8df175b214248a
```

Request setup compute:

```bash
salloc -p lightwork -q public -t 04:00:00 -c 4 --mem=32G
module load mamba/latest
mamba env create -p "$ENV_DIR" -f "$PROJECT_DIR/environments/docprune-sol.yml"
export PATH="$ENV_DIR/bin:$PATH"
export MAX_JOBS=4
mkdir -p "$RUN_ROOT"
python -m pip install flash-attn==2.5.8 --no-build-isolation
python -m pip install -e "$PROJECT_DIR" --no-deps
python -m pip install -e "$M3DOCRAG_DIR" --no-deps
python -m pip freeze | sort > "$RUN_ROOT/environment-freeze.txt"
```

If FlashAttention 2.5.8 fails to build against the pinned stack, save the full
log and stop. Do not opportunistically upgrade Transformers, PyTorch,
FlashAttention, or ColPali.

## Phase 1: structural GPU smoke

Submit from the exact checkout:

```bash
cd "$PROJECT_DIR"
sbatch --export=ALL,PROJECT_DIR="$PROJECT_DIR",ENV_DIR="$ENV_DIR",EXPECTED_COMMIT="$EXPECTED_COMMIT",RUN_ROOT="$RUN_ROOT/smoke" \
  examples/sbatch/10_docprune_smoke.sbatch
```

Pass gate:

- exact Git commit matches;
- CUDA, Transformers 4.46.3, and FlashAttention 2.5.8 import;
- all tests and Ruff pass;
- top-4 inspection emits the supplement Table B values;
- worktree remains clean.

On failure, preserve Slurm logs and `$RUN_ROOT/smoke`, record the command and
traceback, and stop. Retry only a transient scheduler/preemption failure with
the same inputs.

## Phase 2: resource and processor-contract report

The supplement names these resources but publishes no immutable revisions:

- `Qwen/Qwen2-VL-7B-Instruct`
- `vidore/colpali-v1`
- `m3docrag/m3docvqa`

As a reconstruction pin, Qwen was resolved on 2026-08-15 to
`eed13092ef92e448dd6875b2a00151bd3f7db0ac`. The two other exact resource URLs
returned HTTP 401 without credentials on the preparing machine; their identity
and revisions must be resolved from the user's authenticated Hugging Face
session. Do not substitute `vidore/colpali`, `colpali-v1.1`, or `colpali-v1.2`
without explicit owner approval.

From a GPU compute allocation, use one fixed M3DocVQA page and record:

1. exact resource IDs and 40-character revisions;
2. raw page size and Qwen's resized size;
3. Qwen `pixel_values` shape and `image_grid_thw`;
4. Qwen patch size, temporal patch size, and spatial merge size;
5. ColPali input IDs, attention mask, output embedding shape, image token ID,
   exact image-token indices, and inferred 2-D image grid;
6. whether the selected visual-only ColPali slice has exactly grid-height times
   grid-width tokens and preserves raster order;
7. the short-answer prompt and generation configuration used by the pinned
   M3DocRAG baseline.

Write the report to:

```text
/scratch/$USER/docprune/handoff-ee88169/processor-contract.json
```

The report must contain shapes, IDs, revisions, and hashes—not model weights,
page images, tokens, or credentials. Stop and return the report for review.

## Benchmark hold

`examples/sbatch/11_docprune_m3docvqa.sbatch` is a prepared but inactive runner.
It requires an approved `module:function` factory implementing the verified
processor contract and immutable `QA_REVISION`, `RETRIEVER_REVISION`, and
`DATASET_REVISION`. It must not be submitted under this handoff.

After the factory is reviewed, the next handoff must first require all-kept
baseline answer equivalence, then a single pruned sample with a monotonic trace
and no empty page, before top-1/top-2/top-4 runs. Numerical paper parity remains
unclaimed until those runs complete on frozen inputs.
