#!/usr/bin/env bash
#SBATCH --job-name=seg-evid-s0
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/segment_evidence_stage0_%j.out
#SBATCH --error=logs/segment_evidence_stage0_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/COLPALI_binary_classification}"
DATASET_ROOT="${DATASET_ROOT:-/scratch/$USER/sciegqa_train_4k/evidence_option4_10431/combined_evidence/20260707T210746Z}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/$USER/segment_evidence_classifier/stage0/$(date -u +%Y%m%dT%H%M%SZ)}"
ENV_NAME="${ENV_NAME:-/home/lmalveau/mamba-envs/segment-evidence}"
HF_HOME="${HF_HOME:-/scratch/$USER/hf_cache}"
GROUP_LIMIT="${GROUP_LIMIT:-64}"

mkdir -p "$PROJECT_DIR/logs" "$OUTPUT_DIR" "$HF_HOME"
cd "$PROJECT_DIR"

module load mamba/latest
source activate "$ENV_NAME"
ENV_PREFIX="${CONDA_PREFIX:-$ENV_NAME}"

export ENV_PREFIX
export HF_HOME
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_HUB_CACHE="$HF_HOME/hub"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:$ENV_PREFIX/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
PYTHON_BIN="$ENV_PREFIX/bin/python"

echo "PROJECT_DIR=$PROJECT_DIR"
echo "DATASET_ROOT=$DATASET_ROOT"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "ENV_NAME=$ENV_NAME"
echo "HF_HOME=$HF_HOME"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none}"

"$PYTHON_BIN" scripts/run_segment_evidence_stage.py \
  --mode constants \
  --dataset-root "$DATASET_ROOT" \
  --output-dir "$OUTPUT_DIR"

"$PYTHON_BIN" scripts/run_segment_evidence_stage.py \
  --mode env-pin \
  --dataset-root "$DATASET_ROOT" \
  --output-dir "$OUTPUT_DIR"

"$PYTHON_BIN" scripts/run_segment_evidence_stage.py \
  --mode render-token-accounting \
  --dataset-root "$DATASET_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --split validation \
  --group-limit "$GROUP_LIMIT"

"$PYTHON_BIN" scripts/run_segment_evidence_stage.py \
  --mode frozen-score-64 \
  --dataset-root "$DATASET_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --split validation \
  --train-split train \
  --group-limit "$GROUP_LIMIT"

"$PYTHON_BIN" scripts/run_segment_evidence_stage.py \
  --mode microtrain-r1-20 \
  --dataset-root "$DATASET_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --split validation \
  --train-split train \
  --group-limit "$GROUP_LIMIT"

"$PYTHON_BIN" scripts/run_segment_evidence_stage.py \
  --mode microtrain-arm-a-20 \
  --dataset-root "$DATASET_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --split validation \
  --train-split train \
  --group-limit "$GROUP_LIMIT"

"$PYTHON_BIN" scripts/run_segment_evidence_stage.py \
  --mode microtrain-arm-b-20 \
  --dataset-root "$DATASET_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --split validation \
  --train-split train \
  --group-limit "$GROUP_LIMIT"

echo "Stage 0 outputs: $OUTPUT_DIR"
