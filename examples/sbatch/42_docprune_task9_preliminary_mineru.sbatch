#!/usr/bin/env bash
# One fixed-page MinerU question per array task for the Task 9 random-48 pilot.
#SBATCH --job-name=docprune-t9-mineru
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --array=0-47%8
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

: "${RUNTIME_DIR:?set the clean reviewed runtime checkout}"
: "${RUNTIME_COMMIT:?set the exact reviewed runtime commit}"
: "${BATCH_ROOT:?set a fresh existing batch root}"

readonly PYTHON=/home/lmalveau/mamba-envs/docprune-sol/bin/python
readonly MINERU_ENV=/scratch/lmalveau/docprune/tool-envs/random-coverage-attribution-v2/mineru
readonly MINERU="$MINERU_ENV/bin/mineru"
readonly MINERU_PYTHON="$MINERU_ENV/bin/python"
readonly MINERU_SITE="$MINERU_ENV/lib/python3.11/site-packages/mineru"
readonly MINERU_SOURCE=/scratch/lmalveau/docprune/paper-code/random-coverage-attribution-v2/mineru
readonly MINERU_SOURCE_COMMIT=d9cd58add047c2364c1198eefcb1ee9cd63a971a
readonly MINERU_CLIENT_SHA256=3e955d440c1639e19335e1df9b9fde92722e74cb5c392865a7b7623378e2ff47
readonly MINERU_CONFIG_READER_SHA256=99c4a8a18be2688f5dac80d01b52098647967a5d493dc1594e53aef8f4e5252a
readonly MINERU_MIDDLE_CONVERTER_SHA256=06651d31202a48132725e23cb3baf7510bdcd8e7a114615917a63a3a1c812576
readonly PREPROCESSING_MANIFEST=/scratch/lmalveau/docprune/task9-preliminary-mapping-inputs-5f7ff93-v1/preprocessing-manifest.json
readonly PREPROCESSING_MANIFEST_SHA256=dcfa403263a1938837464da440aa8a2d1b53f52b59c54a2f4dae11fe4c2e09e9
readonly CONTRACT_ROOT=/scratch/lmalveau/docprune/task8-mineru-smoke-contract-d2429ba-v1
readonly MINERU_CONFIG="$CONTRACT_ROOT/mineru-config.json"
readonly MINERU_CONFIG_SHA256=997706d414d791432facc974db0ea0f8a03158d10e38605791a14df4c9ef3313
readonly MINERU_TOOL_MANIFEST="$CONTRACT_ROOT/mineru-tool-manifest.json"
readonly MINERU_TOOL_MANIFEST_SHA256=a0b815991a7380f7ef03076668c93cd54d930ec695b8ff3083b2057cdb3a4c29
readonly MODEL_SNAPSHOT=/scratch/lmalveau/docprune/tool-envs/random-coverage-attribution-v2/hf-cache/models--opendatalab--MinerU2.5-Pro-2604-1.2B/snapshots/d3f5e08d073c21466bbabe21c71bb1e9c2e595da
readonly MODEL_WEIGHTS=/scratch/lmalveau/docprune/tool-envs/random-coverage-attribution-v2/hf-cache/models--opendatalab--MinerU2.5-Pro-2604-1.2B/blobs/f2650d91aaa619534980445034f62cde27fc3fa0430aaf5c3302b91179cad0c5
readonly MODEL_WEIGHTS_SHA256=f2650d91aaa619534980445034f62cde27fc3fa0430aaf5c3302b91179cad0c5
readonly MODEL_INVENTORY=/scratch/lmalveau/docprune/tool-envs/random-coverage-attribution-v2/manifests/mineru-model-sha256.txt
readonly MODEL_INVENTORY_SHA256=5ec9100ca6d70984dfae755a8a89fd60ffac663fcc5ae9b31f4a958ebe2846e1

test -d "$BATCH_ROOT"
test ! -L "$BATCH_ROOT"
test "$(sha256sum "$PREPROCESSING_MANIFEST" | cut -d' ' -f1)" = "$PREPROCESSING_MANIFEST_SHA256"
read -r QID SMOKE_INPUT_MANIFEST SMOKE_INPUT_MANIFEST_SHA256 SMOKE_INPUT_DIR < <(
  "$PYTHON" -c 'import json,sys; row=json.load(open(sys.argv[1]))["qid_inputs"][int(sys.argv[2])]; print(row["qid"], row["input_manifest"], row["input_manifest_sha256"], row["input_root"])' \
    "$PREPROCESSING_MANIFEST" "$SLURM_ARRAY_TASK_ID"
)
readonly QID SMOKE_INPUT_MANIFEST SMOKE_INPUT_MANIFEST_SHA256 SMOKE_INPUT_DIR
readonly JOB_ROOT="$BATCH_ROOT/$(printf '%02d' "$SLURM_ARRAY_TASK_ID")-$QID"
readonly OUTPUT_DIR="$JOB_ROOT/mineru-output"
readonly GPU_MANIFEST="$JOB_ROOT/gpu.json"
mkdir "$JOB_ROOT"

export PYTHONPATH="$MINERU_SOURCE:$RUNTIME_DIR/src"
test -x "$PYTHON"
test -x "$MINERU"
test -x "$MINERU_PYTHON"
test "$(git -C "$RUNTIME_DIR" rev-parse HEAD)" = "$RUNTIME_COMMIT"
test -z "$(git -C "$RUNTIME_DIR" status --porcelain --untracked-files=all)"
test "$(git -C "$MINERU_SOURCE" rev-parse HEAD)" = "$MINERU_SOURCE_COMMIT"
test -z "$(git -C "$MINERU_SOURCE" status --porcelain --untracked-files=all)"
test "$(sha256sum "$MINERU_SOURCE/mineru/cli/client.py" | cut -d' ' -f1)" = "$MINERU_CLIENT_SHA256"
test "$(sha256sum "$MINERU_SITE/cli/client.py" | cut -d' ' -f1)" = "$MINERU_CLIENT_SHA256"
test "$(sha256sum "$MINERU_SOURCE/mineru/utils/config_reader.py" | cut -d' ' -f1)" = "$MINERU_CONFIG_READER_SHA256"
test "$(sha256sum "$MINERU_SITE/utils/config_reader.py" | cut -d' ' -f1)" = "$MINERU_CONFIG_READER_SHA256"
test "$(sha256sum "$MINERU_SOURCE/mineru/backend/vlm/model_output_to_middle_json.py" | cut -d' ' -f1)" = "$MINERU_MIDDLE_CONVERTER_SHA256"
test "$(sha256sum "$MINERU_SITE/backend/vlm/model_output_to_middle_json.py" | cut -d' ' -f1)" = "$MINERU_MIDDLE_CONVERTER_SHA256"
test "$(sha256sum "$SMOKE_INPUT_MANIFEST" | cut -d' ' -f1)" = "$SMOKE_INPUT_MANIFEST_SHA256"
test "$(sha256sum "$MINERU_CONFIG" | cut -d' ' -f1)" = "$MINERU_CONFIG_SHA256"
test "$(sha256sum "$MINERU_TOOL_MANIFEST" | cut -d' ' -f1)" = "$MINERU_TOOL_MANIFEST_SHA256"
test "$(readlink -f "$MODEL_SNAPSHOT/model.safetensors")" = "$MODEL_WEIGHTS"
test "$(sha256sum "$MODEL_WEIGHTS" | cut -d' ' -f1)" = "$MODEL_WEIGHTS_SHA256"
test "$(sha256sum "$MODEL_INVENTORY" | cut -d' ' -f1)" = "$MODEL_INVENTORY_SHA256"
sha256sum --quiet -c "$MODEL_INVENTORY"
test "$("$MINERU_PYTHON" -c 'import importlib.metadata; print(importlib.metadata.version("mineru"))')" = 3.0.9
test "$("$MINERU_PYTHON" -c 'import mineru; print(mineru.__file__)')" = "$MINERU_SOURCE/mineru/__init__.py"
"$MINERU_PYTHON" -c 'import importlib.util; raise SystemExit(0 if importlib.util.find_spec("vllm") is None and importlib.util.find_spec("lmdeploy") is None else 1)'

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PATH="$MINERU_ENV/bin:$PATH"
export MINERU_TOOLS_CONFIG_JSON="$MINERU_CONFIG"
export MINERU_MODEL_SOURCE=local
export MINERU_DEVICE_MODE=cuda
export MINERU_API_MAX_CONCURRENT_REQUESTS=1
export HF_HOME=/scratch/lmalveau/docprune/tool-envs/random-coverage-attribution-v2/hf-cache
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
export HF_HUB_CACHE="$HF_HOME"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

"$RUNTIME_DIR/examples/probe_task8_gpu.sh" "$MINERU_PYTHON" "$GPU_MANIFEST"
readonly GPU_MANIFEST_SHA256="$(sha256sum "$GPU_MANIFEST" | cut -d' ' -f1)"

"$PYTHON" "$RUNTIME_DIR/examples/run_task8_mineru_smoke.py" prepare \
  --job-root "$JOB_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --smoke-input-manifest "$SMOKE_INPUT_MANIFEST" \
  --smoke-input-manifest-sha256 "$SMOKE_INPUT_MANIFEST_SHA256" \
  --configuration "$MINERU_CONFIG" \
  --configuration-sha256 "$MINERU_CONFIG_SHA256" \
  --tool-manifest "$MINERU_TOOL_MANIFEST" \
  --tool-manifest-sha256 "$MINERU_TOOL_MANIFEST_SHA256" \
  --model-weights "$MODEL_WEIGHTS" \
  --model-weights-sha256 "$MODEL_WEIGHTS_SHA256" \
  --model-inventory "$MODEL_INVENTORY" \
  --model-inventory-sha256 "$MODEL_INVENTORY_SHA256" \
  --runtime-commit "$RUNTIME_COMMIT"

mkdir "$OUTPUT_DIR"
"$MINERU" \
  --path "$SMOKE_INPUT_DIR" \
  --output "$OUTPUT_DIR" \
  --backend vlm-auto-engine \
  --formula true \
  --table true

"$PYTHON" "$RUNTIME_DIR/examples/run_task8_mineru_smoke.py" finalize \
  --job-root "$JOB_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --gpu-manifest "$GPU_MANIFEST" \
  --gpu-manifest-sha256 "$GPU_MANIFEST_SHA256"
