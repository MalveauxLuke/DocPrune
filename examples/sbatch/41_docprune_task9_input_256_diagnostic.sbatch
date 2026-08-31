#!/usr/bin/env bash
# One-question Task 9 B_input diagnostic: 256 fit plus 64 held-out masks.
#SBATCH --job-name=docprune-task9-input-256
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

: "${RUNTIME_DIR:?set the clean reviewed Task 9 runtime}"
: "${RUNTIME_COMMIT:?set its exact commit}"
: "${JOB_ROOT:?set the existing fresh Task 9 job root}"

readonly PYTHON=/home/lmalveau/mamba-envs/docprune-sol/bin/python
readonly ENV_DIR=/home/lmalveau/mamba-envs/docprune-sol
readonly PDFTOOLS_DIR=/home/lmalveau/mamba-envs/m3docvqa-acquisition
readonly M3DOCRAG_DIR=/home/lmalveau/src/m3docrag-task6-clean-20260828
readonly M3DOCRAG_COMMIT=29e6ac2294d6b87075a1d45b8a8df175b214248a
readonly FIXTURE=/scratch/lmalveau/docprune/task6-fixed-page-gate-v1/fixture-stage64-top4-v2.json
readonly FIXTURE_SHA256=32b3ddd6a1f608db509f002f9541dbc92317b59ac769f0b41fcb20bcc536652b
readonly MAPPING=/scratch/lmalveau/docprune/task8-region-mapping-f3f5de7-v1/region-token-mapping.json
readonly MAPPING_SHA256=a1c986232259c4979b6c3e423f866b8ff107f1f430ed96feaf5062ce42d04ab6
readonly MAPPING_INTERNAL_SHA256=993de7e7b9e1e0a728a3839c30821338a4e742abb43b8594133921362313857b
readonly ATTEMPT_ROOT=/scratch/lmalveau/docprune/benchmark-4e2473b/attempt-2
readonly RUN_CONFIG="$ATTEMPT_ROOT/run-configs/docprune-top4-task6-clean-m3-v1.json"
readonly RUN_CONFIG_SHA256=2233621303ccdf531267bb5bd2a7670775e54f90d04dede4fb19f264e8cad502
readonly INDEX_MANIFEST="$ATTEMPT_ROOT/indexes/docprune/top4/docprune/manifest.json"
readonly INDEX_MANIFEST_SHA256=ffa5979b3bf157adefcc132b0438af295ddafb2243377db4cdb8fcb8eaafe5da
readonly CONFIG="$RUNTIME_DIR/configs/docprune-m3docvqa.toml"
readonly CONFIG_SHA256=82463d2ef3296a199521f3f637b256341aad7938eb199cb55c6249debfea44aa
readonly QID=e1e6ed53f9ad11813845088f4cf2f6b1
readonly GEOMETRY_COUNT=3586
readonly GEOMETRY_SHA256=47b32cf6156dcee41da7eab1686219c760dd73803a135534546c1a50519086e8
readonly OUTPUT="$JOB_ROOT/output"

test -d "$JOB_ROOT"
test ! -e "$OUTPUT"
test -x "$PYTHON"
test "$(git -C "$RUNTIME_DIR" rev-parse HEAD)" = "$RUNTIME_COMMIT"
test -z "$(git -C "$RUNTIME_DIR" status --porcelain --untracked-files=all)"
test "$(git -C "$M3DOCRAG_DIR" rev-parse HEAD)" = "$M3DOCRAG_COMMIT"
test -z "$(git -C "$M3DOCRAG_DIR" status --porcelain --untracked-files=all)"
test "$(sha256sum "$FIXTURE" | cut -d' ' -f1)" = "$FIXTURE_SHA256"
test "$(sha256sum "$MAPPING" | cut -d' ' -f1)" = "$MAPPING_SHA256"
test "$(sha256sum "$RUN_CONFIG" | cut -d' ' -f1)" = "$RUN_CONFIG_SHA256"
test "$(sha256sum "$INDEX_MANIFEST" | cut -d' ' -f1)" = "$INDEX_MANIFEST_SHA256"
test "$(sha256sum "$CONFIG" | cut -d' ' -f1)" = "$CONFIG_SHA256"

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

"$RUNTIME_DIR/examples/probe_task7_l40s_gpu.sh" "$PYTHON" >/dev/null

"$PYTHON" "$RUNTIME_DIR/examples/run_task9_regional_development.py" \
  --config "$CONFIG" \
  --run-config "$RUN_CONFIG" \
  --index-manifest "$INDEX_MANIFEST" \
  --fixture "$FIXTURE" \
  --fixture-sha256 "$FIXTURE_SHA256" \
  --mapping "$MAPPING" \
  --mapping-sha256 "$MAPPING_SHA256" \
  --expected-geometry-count "$GEOMETRY_COUNT" \
  --expected-geometry-sha256 "$GEOMETRY_SHA256" \
  --qid "$QID" \
  --boundary input \
  --fit-mask-count 256 \
  --holdout-mask-count 64 \
  --output "$OUTPUT" \
  --runtime-dir "$RUNTIME_DIR" \
  --runtime-commit "$RUNTIME_COMMIT"

"$PYTHON" "$RUNTIME_DIR/examples/admit_task9_regional_development.py" \
  --root "$OUTPUT" \
  --runtime-commit "$RUNTIME_COMMIT" \
  --qid "$QID" \
  --boundary B_input \
  --fit-mask-count 256 \
  --holdout-mask-count 64 \
  --fixture-sha256 "$FIXTURE_SHA256" \
  --mapping-sha256 "$MAPPING_SHA256" \
  --mapping-internal-sha256 "$MAPPING_INTERNAL_SHA256" \
  --geometry-count "$GEOMETRY_COUNT" \
  --geometry-sha256 "$GEOMETRY_SHA256" \
  --original-visual-tokens 10032 \
  --post-btp-visual-tokens 4336 \
  --post-qtp-visual-tokens 3586 \
  --decoder-layer-count 28 \
  --gpu-substring L40S
