# H200 Stage 0 acquisition handoff

Status: steps 1–3 are prepared locally. No transfer, remote environment change,
GPU smoke or experimental reader scoring has been performed. GPU commands below
are a prepared next stage, not an instruction to launch without owner authority.

Read the repository's `AGENTS.md`, `h200-operations/README.md` and
`h200-operations/CORAL_POLICY.md`. For transfers also read the private runbook
named there. Confirm current mounts, free space, channel membership/relay and
CoRAL GPU occupancy. Preserve other users' files, environments and jobs.

## Code and input delivery

Use the existing `/mnt/data1/eunwooim/DocPrune` checkout. Code, tests and
instructions travel through Git: push from the Mac and fast-forward pull on H200.
Transfer the sealed input files separately with the documented `rsync -avP`
procedure to `task9-h200-local-data/inputs/stage0-adaptive-acquisition-v1/`.
Keep unrelated inputs and local changes untouched; do not use deletion flags.

The input package contains 17 sanitized cases and 68 original page PNGs. Its
`manifest.json` authenticates the files and provenance. Model weights, historical
masked scores and the Mac Python environment are not part of this input package.
The previously prepared standalone bundle is a local reference snapshot; the
H200 workflow uses Git-managed source in this checkout. File delivery does not
authorize tests, installations, model loading, smoke or experiment execution.

## Environment preparation

Reuse a compatible existing environment after inspection. The required observed
reader versions are Python 3.10, torch 2.4.1, transformers 4.49.0, numpy 1.26.4,
Pillow 10.4.0, qwen-vl-utils 0.0.8 and accelerate 1.1.0. The old generic project
Pillow/Transformers requirement is not an instruction to upgrade this reader.
The adapter imports the existing DocPrune stack; run the import check below too.
`runtime-constraints.txt` is a compatibility inventory, not a complete environment
installer. Do not alter a shared environment to satisfy it. A mismatch requires
a separately authorized isolated environment under `/mnt/data2/eunwooim`.

`h200/adaptive-acquisition/environment.sh` sets only this process's cache/temp variables; it does not install
packages or create directories. Inspect their existence and free storage before
source/use. Model loading is offline from the saved revision snapshot. The
snapshot location in `task9-h200-local-data/inputs/stage0-adaptive-acquisition-v1/runtime.json` was recorded historically and may be
supplied explicitly with `--snapshot` if a validated copy lives elsewhere.

Commands below execute **on H200**, from the existing DocPrune checkout root, using the
chosen existing environment's `python`:

```sh
source h200/adaptive-acquisition/environment.sh
python scripts/stage0_acquisition.py validate --package task9-h200-local-data/inputs/stage0-adaptive-acquisition-v1
python scripts/stage0_acquisition.py doctor --package task9-h200-local-data/inputs/stage0-adaptive-acquisition-v1
PYTHONPATH="$PWD/src" python -c 'import docprune.answerers; import docprune.acquisition_reader; import docprune.qwen2vl.model; print("reader imports OK")'
DOCPRUNE_ACQUISITION_PACKAGE="$PWD/task9-h200-local-data/inputs/stage0-adaptive-acquisition-v1" PYTHONPATH="$PWD/src" python -m unittest discover -s tests -v
```

`doctor` reads versions and snapshot metadata, without loading a model or querying
CUDA. Confirm Python 3.10, no missing packages/version mismatches, the expected
snapshot revision and all required metadata files before GPU admission. The
import check imports the Torch/Transformers code without loading weights.

## Step 4, after owner authorization

Inspect the current GPU state immediately before choosing one physical device
from CoRAL 4–7. The runner checks the selected device again and rejects occupied
devices. Normal assigned CoRAL usage needs no per-job announcement, but the
operator must be in the channel or have an active relay. The flag below records
that existing condition; it does not create it. Fill in an actually available
physical ID and new task output directories; placeholders are intentional.

```sh
python scripts/stage0_acquisition.py smoke --package task9-h200-local-data/inputs/stage0-adaptive-acquisition-v1 --cases Q12 \
  --output /mnt/data1/eunwooim/DocPrune/outputs/CHOSEN_SMOKE_DIRECTORY \
  --physical-gpu CHOSEN_CORAL_ID --coral-channel-or-relay --execute-gpu
```

A successful smoke writes a hashed `smoke.json`. Inspect parity, decoded-answer
identity, timing and memory conditions before progressing. The full scorer is
prepared but does not launch as a side effect of smoke:

```sh
python scripts/stage0_acquisition.py score --package task9-h200-local-data/inputs/stage0-adaptive-acquisition-v1 \
  --output /mnt/data1/eunwooim/DocPrune/outputs/CHOSEN_EXPERIMENT_DIRECTORY \
  --smoke-receipt /mnt/data1/eunwooim/DocPrune/outputs/CHOSEN_SMOKE_DIRECTORY/smoke.json \
  --physical-gpu CHOSEN_CORAL_ID --coral-channel-or-relay --execute-gpu
```

Rerun the same scoring command to resume matching completed units. Do not delete
journals, widen parity tolerances or overwrite identities to force a resume.
A stale/different smoke, changed code/environment/input or corrupt record stops
execution. A pre-existing completed physical score is reused only when requested
by the current arm. Output stores question/arm requests and results, physical
cache files, and question summaries. After 32-per-arm likelihood scoring, decoded
candidate review is still a separate required assessment.

## Local verification

The local CPU environment uses Python 3.12 with NumPy 1.26.4 for the controller.
It is not an H200 reader environment and is not transferred. Synthetic simulation
uses real question geometry and masks but artificial G/S responses. Passing it
is evidence about implementation behavior only.
