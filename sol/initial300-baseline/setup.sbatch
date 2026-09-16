#!/usr/bin/env bash
#SBATCH --job-name=dp300-q3-setup
#SBATCH --partition=lightwork
#SBATCH --qos=public
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/lmalveau/docprune-initial300/20260915-v1/logs/%x-%j.out
#SBATCH --error=/scratch/lmalveau/docprune-initial300/20260915-v1/logs/%x-%j.err
set -euo pipefail
cd /home/lmalveau/DocPrune
source sol/initial300-baseline/environment.sh
/home/lmalveau/.conda/envs/docprune-colfeatures17/bin/python -m venv "$DP_ENV"
"$DP_ENV/bin/python" -m pip install --disable-pip-version-check 'torch==2.8.0' 'transformers==4.57.3' 'accelerate==1.10.1' 'Pillow==11.3.0' 'huggingface-hub==0.36.0'
"$DP_ENV/bin/hf" download Qwen/Qwen3-VL-8B-Instruct --revision "$DP_REVISION" --include '*.json' '*.safetensors' '*.txt' --max-workers 4
"$DP_ENV/bin/python" -c 'from transformers import Qwen3VLForConditionalGeneration, AutoProcessor; print("Qwen3 imports OK")'
"$DP_ENV/bin/python" -m pip freeze > "$DP_ROOT/baseline-environment.txt"
