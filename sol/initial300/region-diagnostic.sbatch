#!/usr/bin/env bash
#SBATCH --job-name=dp300-region-check
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24000M
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/lmalveau/docprune-initial300/20260915-v1/logs/regiondiag-%j.out
#SBATCH --error=/scratch/lmalveau/docprune-initial300/20260915-v1/logs/regiondiag-%j.err
set -euo pipefail
cd /home/lmalveau/DocPrune
source sol/initial300/environment.sh
(cd "$DP300_CODE" && sha256sum -c CODE_SHA256SUMS)
/usr/bin/time -v "$DP300_COL_PY" "$DP300_CODE/diagnose_colqwen_batch.py" --root "$DP300_ROOT" --base "$DP300_BASE" --adapter "$DP300_ADAPTER" --regions
