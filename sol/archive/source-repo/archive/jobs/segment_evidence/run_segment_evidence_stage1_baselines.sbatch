#!/usr/bin/env bash
#SBATCH --job-name=seg-evid-s1
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/segment_evidence_stage1_%j.out
#SBATCH --error=logs/segment_evidence_stage1_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/COLPALI_binary_classification}"
DATASET_ROOT="${DATASET_ROOT:-/scratch/$USER/sciegqa_train_4k/evidence_option4_10431/combined_evidence/20260707T210746Z}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/$USER/segment_evidence_classifier/stage1/$(date -u +%Y%m%dT%H%M%SZ)}"
ENV_NAME="${ENV_NAME:-/home/lmalveau/mamba-envs/segment-evidence}"
HF_HOME="${HF_HOME:-/scratch/$USER/hf_cache}"
BASELINES="${BASELINES:-B0 B0k B1 B2 B2b B3 B4-zs B5 B6}"

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
echo "BASELINES=$BASELINES"
echo "REPORT=reports/baselines.md"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none}"

"$PYTHON_BIN" scripts/run_segment_evidence_stage.py \
  --mode stage1-baselines \
  --dataset-root "$DATASET_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --splits train validation \
  --baselines $BASELINES

echo "Stage 1 baseline outputs: $OUTPUT_DIR"
echo "reports/baselines.md"
