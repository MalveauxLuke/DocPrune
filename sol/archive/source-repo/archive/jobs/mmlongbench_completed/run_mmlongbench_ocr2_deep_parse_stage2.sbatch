#!/usr/bin/env bash
#SBATCH --job-name=mmlb-ocr2-links-s2
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/lmalveau/mmlongbench_ocr2_deep_parse_links/logs/stage2_%j.out
#SBATCH --error=/scratch/lmalveau/mmlongbench_ocr2_deep_parse_links/logs/stage2_%j.err

set -euo pipefail
export PYTHONNOUSERSITE=1
: "${RUN_ROOT:?RUN_ROOT is required}"
PROJECT_DIR="/home/lmalveau/COLPALI_binary_classification"
ENV_DIR="/home/lmalveau/mamba-envs/deepseek-ocr2-sol"
PYTHON_BIN="$ENV_DIR/bin/python"
HF_CACHE_ROOT="/scratch/lmalveau/hf_cache"
export HF_HOME="$HF_CACHE_ROOT"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE_ROOT/hub"
export TRANSFORMERS_CACHE="$HF_CACHE_ROOT/transformers"
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH="$ENV_DIR/lib:$ENV_DIR/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$PROJECT_DIR"
echo "git_commit=$(git rev-parse HEAD)"
echo "run_root=$RUN_ROOT"
nvidia-smi
"$PYTHON_BIN" scripts/mmlongbench/prepare_deep_assisted.py --run-root "$RUN_ROOT"
"$PYTHON_BIN" scripts/document_parsing/run_deepseek_ocr.py \
  --run-root "$RUN_ROOT" --ocr-version 2 \
  --pages-manifest "$RUN_ROOT/stage2_requests.jsonl" \
  --output-root "$RUN_ROOT/stage2" --attempt-index 1
