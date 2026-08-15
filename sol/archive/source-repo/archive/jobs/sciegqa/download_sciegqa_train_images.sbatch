#!/usr/bin/env bash
#SBATCH --job-name=sciegqa-images
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/lmalveau/sciegqa_train_4k/logs/images_download_%j.out
#SBATCH --error=/scratch/lmalveau/sciegqa_train_4k/logs/images_download_%j.err

set -euo pipefail

ROOT="${ROOT:-/scratch/$USER/sciegqa_train_4k}"
RAW_DIR="$ROOT/raw"
METADATA_DIR="$RAW_DIR/hf_download_metadata/images_tar"
HF_HOME="${HF_HOME:-/scratch/$USER/hf_cache}"
HF_BIN="${HF_BIN:-/home/$USER/mamba-envs/colqwen25/bin/hf}"
REVISION="4ffb867c88e3264161920b4b2446d5ac6352269e"

mkdir -p "$RAW_DIR" "$METADATA_DIR" "$ROOT/logs" "$HF_HOME"
export HF_HOME PYTHONNOUSERSITE=1

date -u +%Y-%m-%dT%H:%M:%SZ | tee "$METADATA_DIR/started_at.txt"
printf '%s\n' "${SLURM_JOB_ID:-unknown}" | tee "$METADATA_DIR/slurm_job_id.txt"
printf '%s\n' "$REVISION" | tee "$METADATA_DIR/revision.txt"
df -h "/scratch/$USER"
"$HF_BIN" version | tee "$METADATA_DIR/hf_version.txt"
"$HF_BIN" env > "$METADATA_DIR/hf_env.txt"

# Re-running the same pinned local-dir download preserves Hugging Face's
# .incomplete/cache metadata and resumes instead of deleting partial state.
"$HF_BIN" download Yuwh07/SciEGQA-Train images.tar \
  --repo-type dataset \
  --revision "$REVISION" \
  --local-dir "$RAW_DIR"

test -f "$RAW_DIR/images.tar"
stat --printf='%s\n' "$RAW_DIR/images.tar" | tee "$METADATA_DIR/bytes.txt"
sha256sum "$RAW_DIR/images.tar" | tee "$METADATA_DIR/sha256.txt"
realpath "$RAW_DIR/images.tar" > "$METADATA_DIR/resolved_path.txt"
date -u +%Y-%m-%dT%H:%M:%SZ | tee "$METADATA_DIR/completed_at.txt"
