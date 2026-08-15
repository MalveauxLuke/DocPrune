#!/usr/bin/env bash
#SBATCH --job-name=doc-token-gpu-smoke
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/query-relevant-document-token-pruning}"
RUN_ROOT="${RUN_ROOT:-/scratch/$USER/query_relevant_document_token_pruning/smoke}"

mkdir -p "$RUN_ROOT"
cd "$PROJECT_DIR"

printf 'git_commit=%s\n' "$(git rev-parse HEAD)"
printf 'partition=%s\n' "${SLURM_JOB_PARTITION:-interactive}"
printf 'run_root=%s\n' "$RUN_ROOT"
nvidia-smi
python -V
