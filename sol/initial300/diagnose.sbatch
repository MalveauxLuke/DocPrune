#!/usr/bin/env bash
#SBATCH --job-name=dp300-smoke-diagnostic
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24000M
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/lmalveau/docprune-initial300/20260915-v1/logs/batchdiag-%j.out
#SBATCH --error=/scratch/lmalveau/docprune-initial300/20260915-v1/logs/batchdiag-%j.err
set -euo pipefail
cd /home/lmalveau/DocPrune
source sol/initial300/environment.sh
(cd "$DP300_CODE" && sha256sum -c CODE_SHA256SUMS)
col_status=0
/usr/bin/time -v "$DP300_COL_PY" "$DP300_CODE/diagnose_colqwen_batch.py" --root "$DP300_ROOT" --base "$DP300_BASE" --adapter "$DP300_ADAPTER" || col_status=$?
miner_status=0
/usr/bin/time -v "$DP300_MINER_PY" "$DP300_CODE/extract_layout.py" --root "$DP300_ROOT" --model "$DP300_MINER_MODEL" --batch 4 --smoke || miner_status=$?
echo "DIAGNOSTIC_STAGE_EXIT_CODES colqwen=$col_status mineru=$miner_status"
(( col_status == 0 && miner_status == 0 ))
