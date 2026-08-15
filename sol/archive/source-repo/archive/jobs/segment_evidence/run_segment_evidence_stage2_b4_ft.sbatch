#!/usr/bin/env bash
#SBATCH --job-name=seg-evid-s2-b4ft
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --output=logs/segment_evidence_stage2_b4_ft_%j.out
#SBATCH --error=logs/segment_evidence_stage2_b4_ft_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/COLPALI_binary_classification}"
DATASET_ROOT="${DATASET_ROOT:-/scratch/$USER/sciegqa_train_4k/evidence_option4_10431/combined_evidence/20260707T210746Z}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/$USER/segment_evidence_classifier/stage2/$(date -u +%Y%m%dT%H%M%SZ)}"
STAGE0_OUTPUT_DIR="${STAGE0_OUTPUT_DIR:-/scratch/$USER/segment_evidence_classifier/stage0/20260708T200747Z_r1}"
ENV_NAME="${ENV_NAME:-/home/lmalveau/mamba-envs/segment-evidence}"
HF_HOME="${HF_HOME:-/scratch/$USER/hf_cache}"
PILOT_NAME="${PILOT_NAME:-B4-ft}"

case "$PILOT_NAME" in
  B4-ft) ;;
  *) echo "Unknown PILOT_NAME=$PILOT_NAME; expected B4-ft" >&2; exit 2 ;;
esac

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
echo "STAGE0_OUTPUT_DIR=$STAGE0_OUTPUT_DIR"
echo "ENV_NAME=$ENV_NAME"
echo "HF_HOME=$HF_HOME"
echo "PILOT_NAME=$PILOT_NAME"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none}"

"$PYTHON_BIN" scripts/run_segment_evidence_stage.py \
  --mode stage2-b4-ft-pilot \
  --dataset-root "$DATASET_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --stage0-output-dir "$STAGE0_OUTPUT_DIR" \
  --splits train validation \
  --pilot-name "$PILOT_NAME"

echo "Stage 2 B4-ft pilot outputs: $OUTPUT_DIR/stage2_pilots/$PILOT_NAME"
