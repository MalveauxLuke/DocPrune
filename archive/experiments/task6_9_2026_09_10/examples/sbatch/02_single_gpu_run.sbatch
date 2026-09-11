#!/usr/bin/env bash
#SBATCH --job-name=doc-token-gpu
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/query-relevant-document-token-pruning}"
ENV_DIR="${ENV_DIR:-/home/$USER/mamba-envs/document-token-pruning}"
RUN_ROOT="${RUN_ROOT:-/scratch/$USER/query_relevant_document_token_pruning/runs/$SLURM_JOB_ID}"
HF_HOME="${HF_HOME:-/scratch/$USER/hf_cache}"
RUN_COMMAND="${RUN_COMMAND:-python -V}"

mkdir -p "$RUN_ROOT" "$HF_HOME"
cd "$PROJECT_DIR"

export HF_HOME
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$ENV_DIR/lib:$ENV_DIR/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

printf 'git_commit=%s\n' "$(git rev-parse HEAD)"
printf 'run_root=%s\n' "$RUN_ROOT"
printf 'environment=%s\n' "$ENV_DIR"
nvidia-smi
bash -lc "$RUN_COMMAND"
