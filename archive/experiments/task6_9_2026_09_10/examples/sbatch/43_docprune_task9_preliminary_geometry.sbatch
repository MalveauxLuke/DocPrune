#!/usr/bin/env bash
# One fixed-page post-BTP+QTP geometry capture per Task 9 pilot array task.
#SBATCH --job-name=docprune-t9-geometry
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --array=0-47%8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:15:00
#SBATCH --no-requeue
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

: "${RUNTIME_DIR:?set the clean reviewed runtime checkout}"
: "${RUNTIME_COMMIT:?set the exact reviewed runtime commit}"
: "${BATCH_ROOT:?set a fresh existing batch root}"

readonly PYTHON=/home/lmalveau/mamba-envs/docprune-sol/bin/python
readonly ENV_DIR=/home/lmalveau/mamba-envs/docprune-sol
readonly PREPROCESSING_MANIFEST=/scratch/lmalveau/docprune/task9-preliminary-mapping-inputs-5f7ff93-v1/preprocessing-manifest.json
readonly PREPROCESSING_MANIFEST_SHA256=dcfa403263a1938837464da440aa8a2d1b53f52b59c54a2f4dae11fe4c2e09e9
readonly FIXTURE=/scratch/lmalveau/docprune/task9-preliminary-mapping-inputs-5f7ff93-v1/fixture.json
readonly FIXTURE_SHA256=704ca552c79263a6398a596d82dccecc0c64b07b03dec5e1acbcb13ac686b930
readonly SOURCE_RESULTS=/scratch/lmalveau/docprune/task9-preliminary-mapping-inputs-5f7ff93-v1/selected-source-results.jsonl
readonly SOURCE_RESULTS_SHA256=dd72ca9392e8d589bf63fb65d9e656ca4bfd0533d2c19a1ccffa41847878759b

test -d "$BATCH_ROOT"
test ! -L "$BATCH_ROOT"
test "$(git -C "$RUNTIME_DIR" rev-parse HEAD)" = "$RUNTIME_COMMIT"
test -z "$(git -C "$RUNTIME_DIR" status --porcelain --untracked-files=all)"
test "$(sha256sum "$PREPROCESSING_MANIFEST" | cut -d' ' -f1)" = "$PREPROCESSING_MANIFEST_SHA256"
test "$(sha256sum "$FIXTURE" | cut -d' ' -f1)" = "$FIXTURE_SHA256"
test "$(sha256sum "$SOURCE_RESULTS" | cut -d' ' -f1)" = "$SOURCE_RESULTS_SHA256"
readonly QUESTION_INDEX="${QUESTION_INDEX:-$SLURM_ARRAY_TASK_ID}"
read -r QID INPUT_MANIFEST INPUT_MANIFEST_SHA256 EXPECTED_GEOMETRY_COUNT < <(
  "$PYTHON" -c 'import json,sys; row=json.load(open(sys.argv[1]))["qid_inputs"][int(sys.argv[2])]; print(row["qid"], row["input_manifest"], row["input_manifest_sha256"], row["expected_post_qtp_visual_tokens"])' \
    "$PREPROCESSING_MANIFEST" "$QUESTION_INDEX"
)
readonly QID INPUT_MANIFEST INPUT_MANIFEST_SHA256 EXPECTED_GEOMETRY_COUNT
readonly JOB_ROOT="$BATCH_ROOT/$(printf '%02d' "$QUESTION_INDEX")-$QID"
readonly OUTPUT="$JOB_ROOT/geometry.json"
readonly GPU_MANIFEST="$JOB_ROOT/gpu.json"
mkdir "$JOB_ROOT"

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PATH="$ENV_DIR/bin:$PATH"
export PYTHONPATH="$RUNTIME_DIR/src"
export LD_LIBRARY_PATH="$ENV_DIR/lib:$ENV_DIR/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HOME=/scratch/lmalveau/hf_cache
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_CACHE="$HUGGINGFACE_HUB_CACHE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

"$RUNTIME_DIR/examples/probe_task8_gpu.sh" "$PYTHON" "$GPU_MANIFEST"
"$PYTHON" "$RUNTIME_DIR/examples/run_task9_preliminary_geometry_capture.py" \
  --fixture "$FIXTURE" \
  --fixture-sha256 "$FIXTURE_SHA256" \
  --input-manifest "$INPUT_MANIFEST" \
  --input-manifest-sha256 "$INPUT_MANIFEST_SHA256" \
  --source-results "$SOURCE_RESULTS" \
  --source-results-sha256 "$SOURCE_RESULTS_SHA256" \
  --qid "$QID" \
  --expected-geometry-count "$EXPECTED_GEOMETRY_COUNT" \
  --output "$OUTPUT" \
  --runtime-commit "$RUNTIME_COMMIT"

sha256sum "$OUTPUT" "$GPU_MANIFEST" > "$JOB_ROOT/artifacts.sha256"
