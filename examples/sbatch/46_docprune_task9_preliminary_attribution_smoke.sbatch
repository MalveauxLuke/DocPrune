#!/usr/bin/env bash
# One-question Task 9 preliminary attribution smoke: 256 fit + 32 global + 32 local.
#SBATCH --job-name=docprune-t9-prelim-q0
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --gres=gpu:1
#SBATCH --constraint="[a100_40|a100_80|l40s|l40|h100|h100_80|h200]"
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

: "${RUNTIME_DIR:?set the clean reviewed Task 9 runtime}"
: "${RUNTIME_COMMIT:?set its exact commit}"
: "${JOB_ROOT:?set a fresh existing Task 9 job root}"

readonly PYTHON=/home/lmalveau/mamba-envs/docprune-sol/bin/python
readonly ENV_DIR=/home/lmalveau/mamba-envs/docprune-sol
readonly PDFTOOLS_DIR=/home/lmalveau/mamba-envs/m3docvqa-acquisition
readonly M3DOCRAG_DIR=/home/lmalveau/src/m3docrag-task6-clean-20260828
readonly M3DOCRAG_COMMIT=29e6ac2294d6b87075a1d45b8a8df175b214248a
readonly FIXTURE=/scratch/lmalveau/docprune/task9-preliminary-mapping-inputs-5f7ff93-v1/fixture.json
readonly FIXTURE_SHA256=704ca552c79263a6398a596d82dccecc0c64b07b03dec5e1acbcb13ac686b930
readonly MAPPING=/scratch/lmalveau/docprune/task9-preliminary-mappings-ecd1a87-v1/00-5fb253761e72a6048b8b759f04c3b784.json
readonly MAPPING_SHA256=b7b3dbafc8bdcc3dbae5b0c78d555c5e0da082efccfbb851a8cf6c1b311de942
readonly ATTEMPT_ROOT=/scratch/lmalveau/docprune/benchmark-4e2473b/attempt-2
readonly RUN_CONFIG="$ATTEMPT_ROOT/run-configs/docprune-top4-task6-clean-m3-v1.json"
readonly INDEX_MANIFEST="$ATTEMPT_ROOT/indexes/docprune/top4/docprune/manifest.json"
readonly CONFIG="$RUNTIME_DIR/configs/docprune-m3docvqa.toml"
readonly QID=5fb253761e72a6048b8b759f04c3b784
readonly OUTPUT="$JOB_ROOT/output"
readonly GPU_MANIFEST="$JOB_ROOT/gpu.json"

test -d "$JOB_ROOT"
test ! -e "$OUTPUT"
test "$(git -C "$RUNTIME_DIR" rev-parse HEAD)" = "$RUNTIME_COMMIT"
test -z "$(git -C "$RUNTIME_DIR" status --porcelain --untracked-files=all)"
test "$(git -C "$M3DOCRAG_DIR" rev-parse HEAD)" = "$M3DOCRAG_COMMIT"
test -z "$(git -C "$M3DOCRAG_DIR" status --porcelain --untracked-files=all)"
test "$(sha256sum "$FIXTURE" | cut -d' ' -f1)" = "$FIXTURE_SHA256"
test "$(sha256sum "$MAPPING" | cut -d' ' -f1)" = "$MAPPING_SHA256"

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PATH="$PDFTOOLS_DIR/bin:$ENV_DIR/bin:$PATH"
export PYTHONPATH="$RUNTIME_DIR/src"
export LD_LIBRARY_PATH="$ENV_DIR/lib:$ENV_DIR/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HOME=/scratch/lmalveau/hf_cache
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_CACHE="$HUGGINGFACE_HUB_CACHE"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export DOCPRUNE_M3DOCRAG_ROOT="$M3DOCRAG_DIR"

"$RUNTIME_DIR/examples/probe_task8_gpu.sh" "$PYTHON" "$GPU_MANIFEST"

"$PYTHON" "$RUNTIME_DIR/examples/run_task9_regional_development.py" \
  --config "$CONFIG" \
  --run-config "$RUN_CONFIG" \
  --index-manifest "$INDEX_MANIFEST" \
  --fixture "$FIXTURE" \
  --fixture-sha256 "$FIXTURE_SHA256" \
  --mapping "$MAPPING" \
  --mapping-sha256 "$MAPPING_SHA256" \
  --expected-geometry-count 5549 \
  --expected-geometry-sha256 b21472333427046a117eebc1443bd7c899d164b5c46f3901b7e7d12860a99fc0 \
  --qid "$QID" \
  --boundary 13 \
  --fit-mask-count 256 \
  --holdout-mask-count 32 \
  --budget-local-holdout-mask-count 32 \
  --output "$OUTPUT" \
  --runtime-dir "$RUNTIME_DIR" \
  --runtime-commit "$RUNTIME_COMMIT"
