# M3DocVQA Dev Acquisition and Processor-Probe Handoff

## Inactive until smoke approval

This handoff is staged but **not active**. Do not execute it until:

1. `DOCPRUNE_SOL_SMOKE_RECOVERY_HANDOFF.md` has returned a reviewed passing
   smoke report;
2. `/scratch/lmalveau/docprune/handoff-64ea70c/smoke-pass.json` exists and
   validates; and
3. `sol/CURRENT_SOL_TASK.md` is committed with this handoff as the active task.

Once activated, it authorizes complete M3DocVQA dev metadata/PDF acquisition,
integrity validation, rendering one deterministic probe page, and generating
one processor-contract report. It does **not** authorize index construction,
embedding, answer generation, evaluation, training, or benchmarking.

Exact source revisions:

```text
DocPrune: 64ea70c66a8f9e3dbce804d1fde265a4a2b8b09d
M3DocRAG: 29e6ac2294d6b87075a1d45b8a8df175b214248a
Qwen: eed13092ef92e448dd6875b2a00151bd3f7db0ac
```

## Phase 0: verify the smoke gate and clean sources

```bash
set -euo pipefail

export PROJECT_DIR="/home/lmalveau/DocPrune-runtime-64ea70c"
export CONTROL_DIR="/home/lmalveau/DocPrune-control-bda6f3b"
export M3DOCRAG_DIR="/home/lmalveau/src/m3docrag-runtime-29e6ac2"
export ENV_DIR="/home/lmalveau/mamba-envs/docprune-sol"
export ACQ_ENV_DIR="/home/lmalveau/mamba-envs/m3docvqa-acquisition"
export DATA_ROOT="/scratch/lmalveau/docprune/datasets/m3docvqa"
export RUN_ROOT="/scratch/lmalveau/docprune/handoff-64ea70c"
export SMOKE_GATE="$RUN_ROOT/smoke-pass.json"
export EXPECTED_COMMIT="64ea70c66a8f9e3dbce804d1fde265a4a2b8b09d"
export CONTROL_COMMIT="bda6f3be448d05fd066032ceed90c700183e4b21"
export M3DOCRAG_COMMIT="29e6ac2294d6b87075a1d45b8a8df175b214248a"
export N_PROC=16

python -m json.tool "$SMOKE_GATE" >/dev/null
python - <<'PY'
import json
import os

with open(os.environ["SMOKE_GATE"], encoding="utf-8") as stream:
    gate = json.load(stream)
assert gate["status"] == "passed"
assert gate["docprune_commit"] == os.environ["EXPECTED_COMMIT"]
assert gate["m3docrag_commit"] == os.environ["M3DOCRAG_COMMIT"]
PY

test "$(git -C "$PROJECT_DIR" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$PROJECT_DIR" status --porcelain)"
test "$(git -C "$CONTROL_DIR" rev-parse HEAD)" = "$CONTROL_COMMIT"
test -z "$(git -C "$CONTROL_DIR" status --porcelain)"
test "$(git -C "$M3DOCRAG_DIR" rev-parse HEAD)" = "$M3DOCRAG_COMMIT"
test -z "$(git -C "$M3DOCRAG_DIR" status --porcelain)"
```

Any gate, commit, or cleanliness failure stops the handoff.

## Phase 1: construct the isolated acquisition environment

Create the environment and Chromium browser from a compute allocation. This
environment is separate from the validated DocPrune model environment.

```bash
mkdir -p "$DATA_ROOT/cache" "$DATA_ROOT/setup"
export PLAYWRIGHT_BROWSERS_PATH="$DATA_ROOT/cache/ms-playwright"
export PIP_CACHE_DIR="$DATA_ROOT/cache/pip"
export XDG_CACHE_HOME="$DATA_ROOT/cache/xdg"

env -u SLURM_JOB_ID -u SLURM_JOBID -u SLURM_STEP_ID -u SLURM_STEPID \
srun --export=ALL -p lightwork -q public -t 01:00:00 -c 2 --mem=8G \
  /bin/bash -lc '
set -euo pipefail
module load mamba/latest

if [[ -x "$ACQ_ENV_DIR/bin/python" ]]; then
  mamba install -y -p "$ACQ_ENV_DIR" -c conda-forge python=3.10 pip poppler
else
  mamba create -y -p "$ACQ_ENV_DIR" -c conda-forge python=3.10 pip poppler
fi

export PATH="$ACQ_ENV_DIR/bin:$PATH"
export PYTHONNOUSERSITE=1
python -m pip install \
  fire==0.7.1 \
  jsonlines==4.0.0 \
  loguru==0.7.3 \
  numpy==1.26.4 \
  pdf2image==1.17.0 \
  pillow==10.4.0 \
  playwright==1.55.0 \
  pdfrw==0.4 \
  requests==2.32.5 \
  tqdm==4.70.0 \
  2>&1 | tee "$DATA_ROOT/setup/pip-install.log"

python -m playwright install chromium \
  2>&1 | tee "$DATA_ROOT/setup/playwright-install.log"
python -m pip freeze | LC_ALL=C sort > "$DATA_ROOT/setup/environment-freeze.txt"
'
```

Any environment, dependency, or browser installation failure stops the
handoff. Do not install acquisition dependencies into `docprune-sol`.

## Phase 2: acquire metadata and create the dev split

Run metadata download, mapping, and split construction in a compute step:

```bash
env -u SLURM_JOB_ID -u SLURM_JOBID -u SLURM_STEP_ID -u SLURM_STEPID \
srun --export=ALL -p lightwork -q public -t 01:00:00 -c 2 --mem=8G \
  /bin/bash -lc '
set -euo pipefail
export PATH="$ACQ_ENV_DIR/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$M3DOCRAG_DIR/m3docvqa/src"
cd "$DATA_ROOT"

mkdir -p "$DATA_ROOT/multimodalqa"

# Download the five official source archives resumably.  Keep the .gz files:
# CorpusIdentity validates them against setup/mmqa-archives.sha256.
{
  curl --fail --location --retry 5 --retry-delay 2 --continue-at - \
    --output "$DATA_ROOT/multimodalqa/MMQA_dev.jsonl.gz" \
    "https://github.com/allenai/multimodalqa/raw/refs/heads/master/dataset/MMQA_dev.jsonl.gz"
  curl --fail --location --retry 5 --retry-delay 2 --continue-at - \
    --output "$DATA_ROOT/multimodalqa/MMQA_images.jsonl.gz" \
    "https://github.com/allenai/multimodalqa/raw/refs/heads/master/dataset/MMQA_images.jsonl.gz"
  curl --fail --location --retry 5 --retry-delay 2 --continue-at - \
    --output "$DATA_ROOT/multimodalqa/MMQA_tables.jsonl.gz" \
    "https://github.com/allenai/multimodalqa/raw/refs/heads/master/dataset/MMQA_tables.jsonl.gz"
  curl --fail --location --retry 5 --retry-delay 2 --continue-at - \
    --output "$DATA_ROOT/multimodalqa/MMQA_texts.jsonl.gz" \
    "https://github.com/allenai/multimodalqa/raw/refs/heads/master/dataset/MMQA_texts.jsonl.gz"
  curl --fail --location --retry 5 --retry-delay 2 --continue-at - \
    --output "$DATA_ROOT/multimodalqa/MMQA_train.jsonl.gz" \
    "https://github.com/allenai/multimodalqa/raw/refs/heads/master/dataset/MMQA_train.jsonl.gz"
} 2>&1 | tee "$DATA_ROOT/setup/download-mmqa.log"

sha256sum \
  "$DATA_ROOT/multimodalqa/MMQA_dev.jsonl.gz" \
  "$DATA_ROOT/multimodalqa/MMQA_images.jsonl.gz" \
  "$DATA_ROOT/multimodalqa/MMQA_tables.jsonl.gz" \
  "$DATA_ROOT/multimodalqa/MMQA_texts.jsonl.gz" \
  "$DATA_ROOT/multimodalqa/MMQA_train.jsonl.gz" \
  > "$DATA_ROOT/setup/mmqa-archives.sha256"

# Use the pinned M3DocRAG decompressor directly.  Unlike download_mmqa, this
# leaves the checked .gz archives intact after materializing their JSONL files.
python - <<'PY' 2>&1 | tee "$DATA_ROOT/setup/decompress-mmqa.log"
import os
from pathlib import Path

from m3docvqa.mmqa_downloader import decompress_gz_file

root = Path(os.environ["DATA_ROOT"]) / "multimodalqa"
for archive_name in (
    "MMQA_dev.jsonl.gz",
    "MMQA_images.jsonl.gz",
    "MMQA_tables.jsonl.gz",
    "MMQA_texts.jsonl.gz",
    "MMQA_train.jsonl.gz",
):
    archive_path = root / archive_name
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    decompress_gz_file(archive_path, archive_path.with_suffix(""))
PY

python "$M3DOCRAG_DIR/m3docvqa/main.py" generate_wiki_mapping \
  --text="$DATA_ROOT/multimodalqa/MMQA_texts.jsonl" \
  --image="$DATA_ROOT/multimodalqa/MMQA_images.jsonl" \
  --table="$DATA_ROOT/multimodalqa/MMQA_tables.jsonl" \
  --output="$DATA_ROOT/id_url_mapping.jsonl" \
  2>&1 | tee "$DATA_ROOT/setup/generate-wiki-mapping.log"

python "$M3DOCRAG_DIR/m3docvqa/main.py" create_splits \
  --split_metadata_file="$DATA_ROOT/multimodalqa/MMQA_dev.jsonl" \
  --split=dev \
  2>&1 | tee "$DATA_ROOT/setup/create-dev-split.log"

python - <<"PY" | tee "$DATA_ROOT/setup/metadata-counts.json"
import json
import os
from pathlib import Path

root = Path(os.environ["DATA_ROOT"])
question_count = sum(1 for line in (root / "multimodalqa/MMQA_dev.jsonl").open(encoding="utf-8") if line.strip())
with (root / "dev_doc_ids.json").open(encoding="utf-8") as stream:
    doc_ids = json.load(stream)
payload = {
    "dev_questions": question_count,
    "dev_document_ids": len(doc_ids),
    "unique_dev_document_ids": len(set(doc_ids)),
}
print(json.dumps(payload, sort_keys=True, indent=2))
assert payload == {
    "dev_questions": 2441,
    "dev_document_ids": 3366,
    "unique_dev_document_ids": 3366,
}
PY
'
```

Do not continue unless all three metadata counts match exactly.

## Phase 3: acquire every dev PDF as a Slurm array

Submit from a dedicated log directory so Slurm output stays on scratch:

```bash
mkdir -p "$DATA_ROOT/slurm-logs"
cd "$DATA_ROOT/slurm-logs"
export ATTEMPT="attempt-1"

ARRAY_JOB_ID="$(sbatch --parsable --array="0-$((N_PROC - 1))" \
  --export=ALL,BUILDER_DIR="$M3DOCRAG_DIR",ACQ_ENV_DIR="$ACQ_ENV_DIR",DATA_ROOT="$DATA_ROOT",N_PROC="$N_PROC",ATTEMPT="$ATTEMPT" \
  "$CONTROL_DIR/examples/sbatch/20_m3docvqa_download_array.sbatch")"
printf '%s\n' "$ARRAY_JOB_ID" | tee "$DATA_ROOT/$ATTEMPT-job-id.txt"
```

Wait for every array element to finish, then save accounting:

```bash
sacct -j "$ARRAY_JOB_ID" \
  --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,NodeList \
  > "$DATA_ROOT/$ATTEMPT-sacct.txt"
```

A nonzero downloader element is evidence to retain, but integrity—not the
append-only downloader logs—is the completion authority.

## Phase 4: run official and independent integrity checks

First run the pinned builder's requested `check_pdfs`, then run a fail-closed
validator that checks the expected ID set and every PDF structure:

```bash
export ATTEMPT="attempt-1"
env -u SLURM_JOB_ID -u SLURM_JOBID -u SLURM_STEP_ID -u SLURM_STEPID \
srun --export=ALL -p lightwork -q public -t 02:00:00 -c 4 --mem=16G \
  /bin/bash -lc '
set -euo pipefail
export PATH="$ACQ_ENV_DIR/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$M3DOCRAG_DIR/m3docvqa/src"
cd "$DATA_ROOT"

python "$M3DOCRAG_DIR/m3docvqa/main.py" check_pdfs \
  --pdf_dir="$DATA_ROOT/pdfs_dev" \
  2>&1 | tee "$DATA_ROOT/$ATTEMPT-check-pdfs.log"

python - <<"PY"
import json
import os
from pathlib import Path

from pdfrw import PdfReader

root = Path(os.environ["DATA_ROOT"])
attempt = os.environ["ATTEMPT"]
with (root / "dev_doc_ids.json").open(encoding="utf-8") as stream:
    expected_ids = set(json.load(stream))

pdf_paths = sorted((root / "pdfs_dev").glob("*.pdf"))
actual = {path.stem: path for path in pdf_paths}
missing = sorted(expected_ids - set(actual))
extra = sorted(set(actual) - expected_ids)
corrupt = []
page_count = 0

for doc_id in sorted(expected_ids & set(actual)):
    try:
        reader = PdfReader(str(actual[doc_id]))
        pages = len(reader.pages or [])
        if pages < 1:
            raise ValueError("PDF has no pages")
        page_count += pages
    except Exception as error:
        corrupt.append({"doc_id": doc_id, "path": str(actual[doc_id]), "error": str(error)})

question_count = sum(1 for line in (root / "multimodalqa/MMQA_dev.jsonl").open(encoding="utf-8") if line.strip())
within_page_target = abs(page_count - 41005) <= 4101
report = {
    "schema_version": 1,
    "attempt": attempt,
    "dev_questions": question_count,
    "expected_pdf_count": len(expected_ids),
    "actual_pdf_count": len(pdf_paths),
    "missing_pdf_ids": missing,
    "extra_pdf_ids": extra,
    "corrupt_pdfs": corrupt,
    "observed_page_count": page_count,
    "published_approximate_page_count": 41005,
    "within_ten_percent_of_published_page_count": within_page_target,
}
report_path = root / f"{attempt}-integrity.json"
report_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
(root / f"{attempt}-missing.txt").write_text("\n".join(missing) + ("\n" if missing else ""), encoding="utf-8")
(root / f"{attempt}-extra.txt").write_text("\n".join(extra) + ("\n" if extra else ""), encoding="utf-8")
(root / f"{attempt}-corrupt.txt").write_text("\n".join(item["path"] for item in corrupt) + ("\n" if corrupt else ""), encoding="utf-8")
print(json.dumps(report, sort_keys=True, indent=2))

assert question_count == 2441
assert len(expected_ids) == 3366
assert len(pdf_paths) == 3366
assert not missing
assert not extra
assert not corrupt
assert within_page_target
PY
'
```

### Authorized retry behavior

If validation fails only because PDFs are missing, extra, or corrupt, retain
the report and move only listed corrupt/extra files into an attempt-specific
quarantine directory under `$DATA_ROOT/quarantine`. Then resubmit the same
16-element array with the next immutable `ATTEMPT` label; the wrapper's
`--check_downloaded=True` skips retained readable PDFs. Run the full official
and independent validation again.

At most three total acquisition attempts are authorized. Stop after the third
failed integrity report. Do not delete PDFs or edit metadata/mapping files.

For attempt 1 failures, quarantine only paths named by the validator and then
repeat the Phase 3 submission and Phase 4 validation as `attempt-2`:

```bash
FAILED_ATTEMPT="attempt-1"
ATTEMPT="attempt-2"
QUARANTINE_ROOT="$DATA_ROOT/quarantine/$FAILED_ATTEMPT"
mkdir -p "$QUARANTINE_ROOT/corrupt" "$QUARANTINE_ROOT/extra"

while IFS= read -r path; do
  [[ -z "$path" ]] && continue
  case "$path" in
    "$DATA_ROOT"/pdfs_dev/*.pdf) ;;
    *) echo "unsafe corrupt path: $path" >&2; exit 2 ;;
  esac
  test -f "$path"
  mv -- "$path" "$QUARANTINE_ROOT/corrupt/"
done < "$DATA_ROOT/$FAILED_ATTEMPT-corrupt.txt"

while IFS= read -r doc_id; do
  [[ -z "$doc_id" ]] && continue
  path="$DATA_ROOT/pdfs_dev/$doc_id.pdf"
  test -f "$path"
  mv -- "$path" "$QUARANTINE_ROOT/extra/"
done < "$DATA_ROOT/$FAILED_ATTEMPT-extra.txt"

cd "$DATA_ROOT/slurm-logs"
ARRAY_JOB_ID="$(sbatch --parsable --array="0-$((N_PROC - 1))" \
  --export=ALL,BUILDER_DIR="$M3DOCRAG_DIR",ACQ_ENV_DIR="$ACQ_ENV_DIR",DATA_ROOT="$DATA_ROOT",N_PROC="$N_PROC",ATTEMPT="$ATTEMPT" \
  "$CONTROL_DIR/examples/sbatch/20_m3docvqa_download_array.sbatch")"
printf '%s\n' "$ARRAY_JOB_ID" | tee "$DATA_ROOT/$ATTEMPT-job-id.txt"
```

If attempt 2 fails on the same permitted conditions, apply the same commands
with `FAILED_ATTEMPT=attempt-2` and `ATTEMPT=attempt-3`. Every move is
recoverable from its quarantine directory. No fourth attempt is authorized.

## Phase 5: render one deterministic probe image

Run only after integrity passes. Select the lexicographically first expected
PDF and render only page 1 at 144 DPI:

```bash
export PROBE_ROOT="$DATA_ROOT/probe"
mkdir -p "$PROBE_ROOT"

env -u SLURM_JOB_ID -u SLURM_JOBID -u SLURM_STEP_ID -u SLURM_STEPID \
srun --export=ALL -p lightwork -q public -t 00:15:00 -c 2 --mem=8G \
  /bin/bash -lc '
set -euo pipefail
export PATH="$ACQ_ENV_DIR/bin:$PATH"
export PYTHONNOUSERSITE=1

python - <<"PY"
import hashlib
import json
import os
from pathlib import Path

from pdf2image import convert_from_path

root = Path(os.environ["DATA_ROOT"])
probe_root = Path(os.environ["PROBE_ROOT"])
with (root / "dev_doc_ids.json").open(encoding="utf-8") as stream:
    first_doc_id = sorted(json.load(stream))[0]
pdf_path = root / "pdfs_dev" / f"{first_doc_id}.pdf"
images = convert_from_path(str(pdf_path), dpi=144, first_page=1, last_page=1)
assert len(images) == 1
image_path = probe_root / f"{first_doc_id}-page-1.png"
images[0].save(image_path)

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

pdf_sha256 = sha256(pdf_path)
image_sha256 = sha256(image_path)
(probe_root / "selection.json").write_text(json.dumps({
    "selection_rule": "lexicographically first dev document ID, page 1",
    "document_id": first_doc_id,
    "pdf_path": str(pdf_path),
    "pdf_sha256": pdf_sha256,
    "image_path": str(image_path),
    "image_sha256": image_sha256,
}, sort_keys=True, indent=2) + "\n", encoding="utf-8")
(probe_root / "probe-pdf.sha256").write_text(f"{pdf_sha256}  {pdf_path}\n", encoding="utf-8")
(probe_root / "probe-image.sha256").write_text(f"{image_sha256}  {image_path}\n", encoding="utf-8")
print(image_path)
PY
'
```

The PDF remains in the validated corpus; only the single PNG is materialized.

## Phase 6: generate the processor contract

Use the validated DocPrune environment and the exact probe image:

```bash
export HF_HOME="/scratch/lmalveau/docprune/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export QWEN_MODEL="Qwen/Qwen2-VL-7B-Instruct"
export QWEN_REVISION="eed13092ef92e448dd6875b2a00151bd3f7db0ac"
export COLPALI_MODEL="vidore/colpali-v1.2"
export COLPALI_REVISION="961b51745de3e9adb3468ac5c9ccca0ac626c217"
export COLPALI_BACKBONE_MODEL="vidore/colpaligemma-3b-pt-448-base"
export COLPALI_BACKBONE_REVISION="30ab955d073de4a91dc5a288e8c97226647e3e5a"
export PROBE_IMAGE="$(python - <<'PY'
import json
import os
from pathlib import Path

selection = json.loads((Path(os.environ["PROBE_ROOT"]) / "selection.json").read_text(encoding="utf-8"))
print(selection["image_path"])
PY
)"

env -u SLURM_JOB_ID -u SLURM_JOBID -u SLURM_STEP_ID -u SLURM_STEPID \
srun --export=ALL -p lightwork -q public -t 01:00:00 -c 4 --mem=32G \
  /bin/bash -lc '
set -euo pipefail
export PATH="$ENV_DIR/bin:$PATH"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$ENV_DIR/lib:$ENV_DIR/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME"
cd "$PROJECT_DIR"

printf "%s\n" "$COLPALI_REVISION" > "$PROBE_ROOT/colpali-revision.txt"
printf "%s\n" "$COLPALI_BACKBONE_REVISION" > "$PROBE_ROOT/colpali-backbone-revision.txt"

docprune-m3docvqa probe-processors \
  --page-image "$PROBE_IMAGE" \
  --qwen-model "$QWEN_MODEL" \
  --qwen-revision "$QWEN_REVISION" \
  --colpali-model "$COLPALI_MODEL" \
  --colpali-revision "$COLPALI_REVISION" \
  --colpali-backbone-model "$COLPALI_BACKBONE_MODEL" \
  --colpali-backbone-revision "$COLPALI_BACKBONE_REVISION" \
  --output "$PROBE_ROOT/processor-contract.json" \
  | tee "$PROBE_ROOT/processor-contract.stdout.json"

python -m json.tool "$PROBE_ROOT/processor-contract.json" >/dev/null
test -z "$(git -C "$PROJECT_DIR" status --porcelain)"
'
```

If Hugging Face returns 401/403 or the exact ColPali ID does not resolve, save
the error and stop without substituting another model.

## Return and stop

Return:

- metadata and acquisition-environment freeze paths;
- every array job ID and accounting path;
- final integrity JSON with question, PDF, corrupt, missing, extra, and page
  counts;
- selected PDF/image metadata and SHA-256 paths;
- Qwen and ColPali immutable revisions;
- `processor-contract.json` and stdout paths;
- every error or unresolved mapping field.

Then stop. Indexing and `11_docprune_m3docvqa.sbatch` remain unauthorized.
