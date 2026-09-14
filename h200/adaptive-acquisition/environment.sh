# H200 only. Source after inspecting mounts, storage and the existing Python env.
# This sets process-local paths; it installs nothing and creates no directories.
export HF_HOME=/mnt/data2/eunwooim/hf
export TRANSFORMERS_CACHE=/mnt/data2/eunwooim/hf/transformers
export HF_DATASETS_CACHE=/mnt/data2/eunwooim/hf/datasets
export PIP_CACHE_DIR=/mnt/data2/eunwooim/pip-cache
export UV_CACHE_DIR=/mnt/data2/eunwooim/uv-cache
export TORCH_HOME=/mnt/data2/eunwooim/torch
export XDG_CACHE_HOME=/mnt/data2/eunwooim/xdg-cache
export CONDA_PKGS_DIRS=/mnt/data2/eunwooim/conda-pkgs
export TMPDIR=/mnt/data2/eunwooim/tmp
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
