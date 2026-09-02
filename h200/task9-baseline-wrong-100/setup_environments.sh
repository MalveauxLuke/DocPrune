#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/paths.env"

MINERU_COMMIT=d9cd58add047c2364c1198eefcb1ee9cd63a971a
MINERU_MODEL=opendatalab/MinerU2.5-Pro-2604-1.2B
MINERU_MODEL_REVISION=d3f5e08d073c21466bbabe21c71bb1e9c2e595da
M3DOCRAG_COMMIT=29e6ac2294d6b87075a1d45b8a8df175b214248a

mkdir -p \
  "$TASK9_ARTIFACT_ROOT/tools" \
  "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" \
  "$TORCH_HOME" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$TMPDIR"

if [[ ! -x "$TASK9_ENV_PREFIX/bin/python" ]]; then
  conda env create --prefix "$TASK9_ENV_PREFIX" --file "$HERE/environment-h200.yml"
fi

if [[ ! -x "$TASK9_MINERU_ENV_PREFIX/bin/python" ]]; then
  conda env create --prefix "$TASK9_MINERU_ENV_PREFIX" \
    --file "$HERE/environment-mineru-h200.yml"
fi
if [[ ! -d "$TASK9_MINERU_SOURCE/.git" ]]; then
  git clone https://github.com/opendatalab/MinerU.git "$TASK9_MINERU_SOURCE"
fi
git -C "$TASK9_MINERU_SOURCE" fetch origin "$MINERU_COMMIT"
git -C "$TASK9_MINERU_SOURCE" checkout --detach "$MINERU_COMMIT"
"$TASK9_MINERU_ENV_PREFIX/bin/pip" install "$TASK9_MINERU_SOURCE"
"$TASK9_MINERU_ENV_PREFIX/bin/hf" download "$MINERU_MODEL" \
  --revision "$MINERU_MODEL_REVISION"

if [[ ! -d "$TASK9_M3DOCRAG_SOURCE/.git" ]]; then
  git clone https://github.com/bloomberg/m3docrag.git "$TASK9_M3DOCRAG_SOURCE"
fi
git -C "$TASK9_M3DOCRAG_SOURCE" fetch origin "$M3DOCRAG_COMMIT"
git -C "$TASK9_M3DOCRAG_SOURCE" checkout --detach "$M3DOCRAG_COMMIT"

PYTHONPATH="$TASK9_REPO/src" "$TASK9_ENV_PREFIX/bin/python" -c \
  'import sklearn, torch, transformers, docprune; print(torch.__version__, transformers.__version__, sklearn.__version__)'
"$TASK9_MINERU_ENV_PREFIX/bin/python" -c \
  'import torch, transformers, mineru; print(torch.__version__, transformers.__version__)'
git -C "$TASK9_MINERU_SOURCE" rev-parse HEAD
git -C "$TASK9_M3DOCRAG_SOURCE" rev-parse HEAD
