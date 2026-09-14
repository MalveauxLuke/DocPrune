# Source on SOL. All runtime writes and downloads stay on scratch.
export ACQUISITION_SOL_ROOT=/scratch/lmalveau/docprune-adaptive-acquisition/20260914-smoke01
export HF_HOME="$ACQUISITION_SOL_ROOT/hf"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$ACQUISITION_SOL_ROOT/datasets"
export PIP_CACHE_DIR="$ACQUISITION_SOL_ROOT/pip-cache"
export UV_CACHE_DIR="$ACQUISITION_SOL_ROOT/uv-cache"
export TORCH_HOME="$ACQUISITION_SOL_ROOT/torch"
export XDG_CACHE_HOME="$ACQUISITION_SOL_ROOT/xdg-cache"
export CONDA_PKGS_DIRS="$ACQUISITION_SOL_ROOT/conda-pkgs"
export TMPDIR="$ACQUISITION_SOL_ROOT/tmp"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export ACQUISITION_SOL_PYTHON=/home/lmalveau/mamba-envs/docprune-acquisition-sol/bin/python
export ACQUISITION_SOL_SNAPSHOT="$HF_HOME/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5"
