#!/usr/bin/env bash
#SBATCH --job-name=doc-token-array
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-3%2
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/query-relevant-document-token-pruning}"
RUN_COMMAND="${RUN_COMMAND:-python -V}"
ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
SHARDS=(shard-0000 shard-0001 shard-0002 shard-0003)
SHARD_NAME="${SHARDS[$ARRAY_TASK_ID]}"
RUN_ROOT="${RUN_ROOT:-/scratch/$USER/query_relevant_document_token_pruning/arrays/${SLURM_ARRAY_JOB_ID:-manual}/$SHARD_NAME}"

mkdir -p "$RUN_ROOT"
cd "$PROJECT_DIR"

export SHARD_NAME RUN_ROOT
printf 'git_commit=%s\n' "$(git rev-parse HEAD)"
printf 'array_task_id=%s\n' "$ARRAY_TASK_ID"
printf 'shard=%s\n' "$SHARD_NAME"
printf 'run_root=%s\n' "$RUN_ROOT"
bash -lc "$RUN_COMMAND"
