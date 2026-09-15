#!/usr/bin/env bash
#SBATCH --job-name=dp300-profiles
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/lmalveau/docprune-initial300/20260915-v1/logs/combine-%j.out
#SBATCH --error=/scratch/lmalveau/docprune-initial300/20260915-v1/logs/combine-%j.err
set -euo pipefail
cd /home/lmalveau/DocPrune
source sol/initial300/environment.sh
(cd "$DP300_CODE" && sha256sum -c CODE_SHA256SUMS)
/usr/bin/time -v "$DP300_COL_PY" "$DP300_CODE/combine.py" --root "$DP300_ROOT"
