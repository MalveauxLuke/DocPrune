#!/usr/bin/env bash
#SBATCH --job-name=doc-token-setup
#SBATCH --partition=lightwork
#SBATCH --qos=public
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/query-relevant-document-token-pruning}"
RUN_ROOT="${RUN_ROOT:-/scratch/$USER/query_relevant_document_token_pruning/setup}"

mkdir -p "$RUN_ROOT"
cd "$PROJECT_DIR"

module load mamba/latest
mamba --version
mamba info --envs
python -V
git rev-parse HEAD
printf 'run_root=%s\n' "$RUN_ROOT"
