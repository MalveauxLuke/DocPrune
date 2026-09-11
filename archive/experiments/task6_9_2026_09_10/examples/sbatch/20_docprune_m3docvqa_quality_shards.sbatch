#!/usr/bin/env bash
# Quality-only evaluation shards on any reviewed GPU with at least 40 GB.
# Mixed-hardware shards must use the quality-only merger; never aggregate efficiency.
#SBATCH --job-name=docprune-m3docvqa-quality-shards
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --array=0-38%8
#SBATCH --gres=gpu:1
#SBATCH --constraint=[a100_40|a100_80|h100|h100_80|l40|l40s]
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

: "${PROJECT_DIR:?set PROJECT_DIR to the exact control checkout}"
: "${ATTEMPT_ROOT:?set ATTEMPT_ROOT to the benchmark attempt root}"
: "${MODE:?set MODE to all-kept or docprune}"
: "${PAGES:?set PAGES to 1, 2, or 4}"
: "${SLURM_ARRAY_TASK_ID:?submit this wrapper as a Slurm array}"

case "$MODE" in all-kept|docprune) ;; *) echo "invalid MODE: $MODE" >&2; exit 2 ;; esac
case "$PAGES" in 1|2|4) ;; *) echo "PAGES must be 1, 2, or 4" >&2; exit 2 ;; esac
case "$SLURM_ARRAY_TASK_ID" in
  ''|*[!0-9]*) echo "SLURM_ARRAY_TASK_ID must be a nonnegative integer" >&2; exit 2 ;;
esac

export GPU_POLICY=quality-40gb
export SHARD_ID="$SLURM_ARRAY_TASK_ID"
export SHARD_PLAN_ROOT="$ATTEMPT_ROOT/shard-plan"
SHARD_NAME="$(printf 'shard-%04d' "$SHARD_ID")"
export CELL_ROOT="$ATTEMPT_ROOT/eval-shards/${MODE}/top${PAGES}/${SHARD_NAME}"
export RUN_CONFIG="$ATTEMPT_ROOT/run-configs/${MODE}-top${PAGES}.json"
export INDEX_MANIFEST="$ATTEMPT_ROOT/indexes/${MODE}/top${PAGES}/${MODE}/manifest.json"

CELL_LAUNCHER="$PROJECT_DIR/examples/sbatch/11_docprune_m3docvqa.sbatch"
test -f "$CELL_LAUNCHER"
exec "$CELL_LAUNCHER"
