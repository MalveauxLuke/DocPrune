#!/usr/bin/env bash
set -euo pipefail

date --iso-8601=seconds
hostname
id
uname -a
df -h / || true
for survey_path in /mnt/data1 /mnt/data2 /data1 /data2 /shared; do
  if [[ -e "$survey_path" ]]; then
    df -h "$survey_path" || true
    findmnt "$survey_path" || true
    ls -ld "$survey_path" || true
  else
    echo "MISSING: $survey_path"
  fi
done
nvidia-smi || true
nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu --format=csv,noheader || true
command -v git || true
git --version || true
command -v conda || true
command -v mamba || true
command -v micromamba || true
command -v python || true
python --version || true
command -v nvcc || true
nvcc --version || true
command -v ninja || true
ninja --version || true

if [[ -d /mnt/data1/eunwooim/DocPrune ]]; then
  git -C /mnt/data1/eunwooim/DocPrune status --short --branch
  git -C /mnt/data1/eunwooim/DocPrune rev-parse HEAD
fi
