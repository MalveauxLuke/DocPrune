#!/usr/bin/env bash
set -euo pipefail
# Run inside the approved GPU allocation, from the pinned checkout root.
: "${SLURM_JOB_ID:?Run inside a SOL compute allocation}"
: "${COLFEATURES_ROOT:?Set a new task-specific scratch directory}"
: "${COLFEATURES_CODE_COMMIT:?Set the exact code commit from CURRENT_SOL_TASK.md}"
git diff --quiet "$COLFEATURES_CODE_COMMIT" HEAD -- . ":(exclude)sol/CURRENT_SOL_TASK.md"
test -z "$(git status --porcelain --untracked-files=no)"
mkdir -p "$COLFEATURES_ROOT"
export HF_HOME="$COLFEATURES_ROOT/hf-cache"
export TMPDIR="$COLFEATURES_ROOT/tmp"
mkdir -p "$TMPDIR" "$COLFEATURES_ROOT/logs"
python -m pytest -q tests/test_colfeatures17.py | tee "$COLFEATURES_ROOT/logs/task-tests.txt"
python scripts/colfeatures_package.py unpack transfer/colfeatures17/input "$COLFEATURES_ROOT/input"
python scripts/extract_colfeatures17.py "$COLFEATURES_ROOT/input" --audit-only > "$COLFEATURES_ROOT/logs/input-audit.json"
python scripts/extract_colfeatures17.py "$COLFEATURES_ROOT/input" --output "$COLFEATURES_ROOT/smoke" --limit 1
python scripts/verify_colfeatures17.py "$COLFEATURES_ROOT/smoke" --cases 1 | tee "$COLFEATURES_ROOT/logs/smoke-verification.json"
python scripts/extract_colfeatures17.py "$COLFEATURES_ROOT/input" --output "$COLFEATURES_ROOT/result"
python scripts/verify_colfeatures17.py "$COLFEATURES_ROOT/result" | tee "$COLFEATURES_ROOT/logs/result-verification.json"
python scripts/gather_colfeatures17.py "$COLFEATURES_ROOT/input" "$COLFEATURES_ROOT/gathered"
# Assemble without copying the model cache. Additional findings belong in this package.
mkdir "$COLFEATURES_ROOT/export"
cp -R "$COLFEATURES_ROOT/input" "$COLFEATURES_ROOT/result" "$COLFEATURES_ROOT/gathered" "$COLFEATURES_ROOT/logs" "$COLFEATURES_ROOT/export/"
cp docs/experiments/corrective-selection/COLFEATURES17.md "$COLFEATURES_ROOT/export/"
python scripts/colfeatures_package.py pack "$COLFEATURES_ROOT/export" "$COLFEATURES_ROOT/export-chunks"
python scripts/colfeatures_package.py unpack "$COLFEATURES_ROOT/export-chunks" "$COLFEATURES_ROOT/roundtrip"
printf 'Extraction and round-trip verification complete. Review availability.json, then follow the handoff to publish.\n'
