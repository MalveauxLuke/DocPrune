#!/usr/bin/env bash
#SBATCH --job-name=seg-evid-cert
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/segment_evidence_certification_%j.out
#SBATCH --error=logs/segment_evidence_certification_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/COLPALI_binary_classification}"
DATASET_ROOT="${DATASET_ROOT:-/scratch/$USER/sciegqa_train_4k/evidence_option4_10431/combined_evidence/20260707T210746Z}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/$USER/segment_evidence_classifier/certification/$(date -u +%Y%m%dT%H%M%SZ)}"
ENV_NAME="${ENV_NAME:-/home/lmalveau/mamba-envs/segment-evidence}"
HF_HOME="${HF_HOME:-/scratch/$USER/hf_cache}"
GROUP_LIMIT="${GROUP_LIMIT:-64}"
EXTERNAL_LIMIT="${EXTERNAL_LIMIT:-32}"
EXTERNAL_DATASET="${EXTERNAL_DATASET:-vidore/arxivqa_test_subsampled}"
STAGE0_OUTPUT_DIR="${STAGE0_OUTPUT_DIR:-/scratch/$USER/segment_evidence_classifier/stage0/20260708T200747Z_r1}"

mkdir -p "$PROJECT_DIR/logs" "$OUTPUT_DIR" "$HF_HOME"
cd "$PROJECT_DIR"

module load mamba/latest
source activate "$ENV_NAME"
ENV_PREFIX="${CONDA_PREFIX:-$ENV_NAME}"

export ENV_PREFIX
export HF_HOME
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_HUB_CACHE="$HF_HOME/hub"
export PYTHONNOUSERSITE=1
export SEGMENT_EVIDENCE_HF_REVISIONS="${SEGMENT_EVIDENCE_HF_REVISIONS:-$STAGE0_OUTPUT_DIR/S0-env-pin/hf_revisions.json}"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:$ENV_PREFIX/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
PYTHON_BIN="$ENV_PREFIX/bin/python"

echo "PROJECT_DIR=$PROJECT_DIR"
echo "DATASET_ROOT=$DATASET_ROOT"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "ENV_NAME=$ENV_NAME"
echo "HF_HOME=$HF_HOME"
echo "GROUP_LIMIT=$GROUP_LIMIT"
echo "EXTERNAL_LIMIT=$EXTERNAL_LIMIT"
echo "EXTERNAL_DATASET=$EXTERNAL_DATASET"
echo "SEGMENT_EVIDENCE_HF_REVISIONS=$SEGMENT_EVIDENCE_HF_REVISIONS"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none}"

"$PYTHON_BIN" scripts/run_segment_evidence_stage.py \
  --mode certify-pipeline \
  --dataset-root "$DATASET_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --split validation \
  --train-split train \
  --group-limit "$GROUP_LIMIT" \
  --external-dataset "$EXTERNAL_DATASET" \
  --external-limit "$EXTERNAL_LIMIT"

echo "Stage 0.5 certification outputs: $OUTPUT_DIR"
