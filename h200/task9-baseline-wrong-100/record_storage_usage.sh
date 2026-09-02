#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/paths.env"

REPORT="$TASK9_ARTIFACT_ROOT/manifests/storage-usage.tsv"
mkdir -p "$(dirname -- "$REPORT")"

if [[ ! -s "$REPORT" ]]; then
  printf 'recorded_at_utc\tstage\tlabel\tpath\tbytes\tstate\n' >"$REPORT"
fi

stage=${1:-manual}
timestamp=$(date -u +'%Y-%m-%dT%H:%M:%SZ')

record_path() {
  local label=$1
  local path=$2
  local bytes state
  if [[ -e "$path" ]]; then
    bytes=$(du -sb -- "$path" | cut -f1)
    state=present
  else
    bytes=0
    state=missing
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$timestamp" "$stage" "$label" "$path" "$bytes" "$state" | tee -a "$REPORT"
}

record_path artifact_root "$TASK9_ARTIFACT_ROOT"
record_path transferred_bundle "$TASK9_TRANSFER_ROOT"
record_path fixed_inputs "$TASK9_FIXED_ROOT"
record_path mineru_outputs "$TASK9_MINERU_ROOT"
record_path geometry_outputs "$TASK9_GEOMETRY_ROOT"
record_path mappings "$TASK9_MAPPINGS"
record_path mineru_source "$TASK9_MINERU_SOURCE"
record_path m3docrag_source "$TASK9_M3DOCRAG_SOURCE"
record_path docprune_env "$TASK9_ENV_PREFIX"
record_path mineru_env "$TASK9_MINERU_ENV_PREFIX"
record_path conda_packages /mnt/data2/eunwooim/.conda/pkgs
record_path hf_cache "$HF_HOME"
record_path torch_cache "$TORCH_HOME"
record_path pip_cache "$PIP_CACHE_DIR"
record_path xdg_cache "$XDG_CACHE_HOME"
record_path task_tmp "$TMPDIR"

printf 'Storage snapshot appended to %s\n' "$REPORT"
