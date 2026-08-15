# SciEGQA-Train 4K Query Subset: SOL Handoff

## Objective

Build a reproducible 4,000-query subset from `Yuwh07/SciEGQA-Train` for
same-page segment hard-negative mining. Download the immutable annotation JSONL
and full `images.tar` to SOL scratch, analyze and select queries while the large
archive downloads, selectively extract only the required gold-page PNGs, and
run a one-document MinerU 3.4 smoke test before any large parsing run.

The query is the unit of selection. Multiple selected queries may share a page
or document. There is no document-disjointness requirement.

## Required reading before acting on SOL

Read these files from the SOL checkout before creating environments, scripts,
or jobs:

1. `AGENTS.md`
2. `SOLinstrucitons.md`
3. `sol/CURRENT_SOL_TASK.md`
4. `sol/README.md`
5. `sol/SOL_RUN_GUIDE.md`
6. Every note in `sol/bug_fixes/`, especially:
   - `COLQWEN_DOCSCOPED_ENV_FIX.md`
   - `VIDORE_ARXIVQA_DATASETS_FIX.md`
   - `VIDORE_ARXIVQA_OFFICIAL_MODEL_FIX.md`
7. Existing submit scripts, especially:
   - `sol/run_colqwen_top5.sbatch`
   - `sol/run_colqwen_verifier_v1.sbatch`
8. MinerU implementation and local runbook:
   - `docs/mineru_local_setup.md`
   - `scripts/run_mineru_local_api.sh`
   - `scripts/run_sciegqa_mineru.py`
   - `scripts/sciegqa_mineru/mineru_runner.py`
   - `scripts/sciegqa_mineru/canonicalize.py`

When this becomes the active SOL task, replace the stale contents of
`sol/CURRENT_SOL_TASK.md` with only the new current state, next action, job IDs,
run paths, and blockers. Do not append a historical diary.

## Immutable sources

- Dataset card: <https://huggingface.co/datasets/Yuwh07/SciEGQA-Train>
- Dataset files: <https://huggingface.co/datasets/Yuwh07/SciEGQA-Train/tree/main>
- Repository: `Yuwh07/SciEGQA-Train`
- Pinned revision: `4ffb867c88e3264161920b4b2446d5ac6352269e`
- Annotation file: `SciEGQA-Train.jsonl` (approximately 11 MB)
- Image archive: `images.tar` (approximately 77.1 GB)
- MinerU documentation: <https://opendatalab.github.io/MinerU/>
- MinerU version used by this repository: `3.4.0`

Never use mutable `main` as recorded provenance. Resolve and record the full
commit SHA in every manifest and job log.

## Facts already verified against the pinned JSONL

The pinned JSONL contains 30,780 rows and eight domains:

| Domain | All rows | Strict eligible rows |
|---|---:|---:|
| q-fin | 6,283 | 2,393 |
| q-bio | 5,457 | 1,913 |
| eess | 3,997 | 1,794 |
| physics | 3,595 | 1,500 |
| cs | 3,920 | 1,465 |
| econ | 3,381 | 1,164 |
| stat | 2,950 | 1,095 |
| math | 1,197 | 344 |
| **Total** | **30,780** | **11,668** |

“Strict eligible” means exactly one evidence page and exactly one valid evidence
box on that page, using the repository's `is_spsr_row()` validation contract.

An exactly even 500-per-domain 4K selection is impossible because only 344
strict math rows exist. The approved quota is therefore:

| Domain | Query quota |
|---|---:|
| q-fin | 523 |
| q-bio | 523 |
| eess | 522 |
| physics | 522 |
| cs | 522 |
| econ | 522 |
| stat | 522 |
| math | 344 |
| **Total** | **4,000** |

The two 523 allocations go to the domains with the greatest eligible
headroom. Do not oversample or duplicate math rows to manufacture equality.

### Important modality limitation

`subimg_type` is `[["image"]]` for all 11,668 strict rows. It does **not**
distinguish text, table, plot, figure, equation, or photograph evidence. Do not
report a modality-balanced subset based on this field.

Before page parsing, use query text only to derive a transparent **query-intent
proxy**. After MinerU parsing, audit that proxy against the actual overlapping
MinerU segment types. Keep the proxy and observed modality as separate fields.

## Storage layout

Keep code and small tracked fixtures in home. Keep datasets, caches, downloads,
and job outputs on scratch:

```text
/scratch/$USER/sciegqa_train_4k/
  raw/
    SciEGQA-Train.jsonl
    images.tar
    hf_download_metadata/
  analysis/
    source_profile.json
    source_profile.md
    intent_rules.json
    intent_review_sample.jsonl
  selection/
    queries.jsonl
    pages.jsonl
    documents.jsonl
    selected_image_members.txt
    selection_audit.json
  pages/
    <category>/<doc_name>/<doc_name>_<1-based-page>.png
  mineru_smoke/
  logs/
```

Before starting the archive download, run `df -h /scratch/$USER` and ensure
there is comfortable room for the 77.1 GB archive, partial-download state,
selected PNGs, MinerU models, and outputs. Treat 150 GB free as a prudent
minimum, not a guarantee of final usage.

## Concurrency model: do not wait for `images.tar`

Use independent Slurm jobs instead of keeping an agent or shell blocked:

1. Download the small JSONL first in a light compute allocation.
2. Submit a resumable, low-CPU archive-download job and record its job ID.
3. Immediately proceed with JSON profiling and selection implementation in a
   separate allocation/job.
4. In parallel, build the MinerU environment and submit a short GPU smoke job
   against a small tracked synthetic PDF fixture.
5. Monitor the archive job periodically with `squeue`, `sacct`, and its log;
   do not idle waiting for it.

Do not run the 77 GB download, JSON profiling, environment installation, or
MinerU inference on a login node.

## Phase 1: environment and JSON download

From an appropriate compute allocation:

```bash
salloc -p lightwork -q public -t 02:00:00 -c 4
module load mamba/latest

export PROJECT_DIR="$HOME/COLPALI_binary_classification"
export ROOT="/scratch/$USER/sciegqa_train_4k"
export RAW_DIR="$ROOT/raw"
export HF_HOME="/scratch/$USER/hf_cache"
export REVISION="4ffb867c88e3264161920b4b2446d5ac6352269e"
mkdir -p "$RAW_DIR" "$ROOT/analysis" "$ROOT/selection" "$ROOT/logs" "$HF_HOME"

hf download Yuwh07/SciEGQA-Train SciEGQA-Train.jsonl \
  --repo-type dataset \
  --revision "$REVISION" \
  --local-dir "$RAW_DIR"
```

Record:

- the full revision;
- JSONL byte size and SHA-256;
- `hf version` and `hf env`;
- start/completion timestamps;
- the resolved output path.

The download must fail closed if the revision or file is not exactly the one
requested.

## Phase 2: launch the archive download

Create a dedicated submit script, for example
`sol/download_sciegqa_train_images.sbatch`, following existing SOL patterns:

- `set -euo pipefail`;
- low CPU and memory;
- `lightwork`/`public` only if its low-duty bulk-I/O policy is appropriate on
  the live cluster; otherwise use a normal `public` CPU job;
- absolute project, environment, scratch, and log paths;
- `PYTHONNOUSERSITE=1`;
- scratch `HF_HOME`;
- the immutable revision;
- `hf download ... images.tar --repo-type dataset --revision ... --local-dir`;
- clear preflight output for interpreter, free space, and `hf` version;
- no deletion of partial Hugging Face download metadata, so resubmitting the
  same command resumes instead of restarting.

Submit it, capture the job ID, and continue immediately:

```bash
ARCHIVE_JOB_ID=$(sbatch --parsable sol/download_sciegqa_train_images.sbatch)
echo "$ARCHIVE_JOB_ID"
squeue -j "$ARCHIVE_JOB_ID"
```

Do not use `wget` against an Xet pointer. Use the current `hf` CLI. Never put a
token on the command line or in source; use `HF_TOKEN` only if authentication
is actually required.

## Phase 3: preliminary JSON analysis

While the archive job runs, create a deterministic profiling script and write
both machine-readable JSON and a short Markdown report. It must calculate:

- total rows and exact source fields;
- row counts by domain;
- strict SPSR counts by domain;
- unique document and `(category, doc_name, page)` counts;
- queries per document and per gold page;
- exact duplicate and near-duplicate query counts;
- evidence-box width, height, area, and aspect-ratio distributions;
- question length and answer length distributions;
- lexical query-intent and reasoning-operation distributions by domain;
- malformed/rejected rows with explicit reasons.

Reuse `scripts.sciegqa_mineru.selection.is_spsr_row()` rather than rewriting a
weaker eligibility check. Add `source_record_index` from the zero-based JSONL
row position before validation and selection.

If the pinned source does not reproduce the verified counts above, stop. Do
not silently adjust quotas.

## Phase 4: query-intent stratification

The source has no useful modality label. Derive two auditable fields from the
query string:

1. `query_intent` — the likely evidence presentation:
   - `table`
   - `plot_chart`
   - `figure_spatial`
   - `formula_symbol`
   - `citation_reference`
   - `textual_lookup`
2. `reasoning_operation` — what the question asks the model to do:
   - `comparison_extremum`
   - `counting`
   - `numeric_lookup`
   - `temporal_lookup`
   - `entity_lookup`
   - `other`

Start from explicit, versioned lexical rules such as “table/row/column,”
“plot/graph/chart/axis/peak,” “figure/image/panel/color/left/right,”
“equation/formula/symbol,” and “reference/cited/et al.” Profile collisions and
unmatched queries before fixing precedence. Store the final rule version and
matched rule in every selected row.

These are proxies, not ground truth. Manually inspect a deterministic,
domain-and-intent-stratified review sample before selection. Do not use an LLM
to silently assign labels. If an LLM is later approved, pin its model and
prompt and retain every raw response.

## Phase 5: deterministic 4K selection

Selection order:

1. Keep only strict SPSR rows.
2. Remove exact duplicate question instances using
   `(category, doc_name, evidence_page, normalized_query)` while retaining the
   lowest `source_record_index`. Record all dropped indices.
3. Apply the approved domain quotas.
4. Within each domain, allocate its quota proportionally across
   `query_intent` using largest-remainder allocation. Give a small nonzero floor
   to supported rare intents only when it can be done without replacement;
   record every adjustment.
5. Within each domain/intent stratum, use a fixed seed and round-robin over
   documents, then pages, so a few high-yield documents do not dominate.
6. Do not require document disjointness and do not prefer queries merely
   because they share a document.
7. Fill any exhausted stratum deterministically from the same domain and log
   the fallback path.

Use a single explicit seed, recommended `20260630`. Never sample from unordered
sets or filesystem iteration order. Stable-sort before seeded operations.

Every query row must preserve at least:

- stable `query_id`;
- source dataset and immutable revision;
- `source_record_index`;
- query and answer;
- category and `doc_name`;
- one-based `evidence_page`;
- absolute and normalized gold boxes;
- original `subimg_type`;
- derived `query_intent`, `reasoning_operation`, and matched rules;
- selection seed, stratum, and selection rank;
- exact expected archive-member path.

Expected page-member path from the dataset card:

```text
<category>/<doc_name>/<doc_name>_<1-based-evidence-page>.png
```

The final audit must prove:

- exactly 4,000 query rows;
- exact domain quotas shown above;
- every row is strict SPSR;
- no selected duplicate key;
- no replacement/oversampling;
- all IDs are unique and joins are complete;
- query-intent counts by domain;
- number of unique documents and unique gold pages;
- maximum and distribution of queries per document/page.

## Phase 6: selective image extraction

Do not extract all of `images.tar`.

After the archive job succeeds:

1. Verify the download with the Hugging Face cache/local-directory verification
   supported by the installed `hf` version.
2. Record `images.tar` byte size and SHA-256.
3. Inspect the archive prefix once with `tar -tf` and reconcile it with the
   dataset-card path convention.
4. Generate a unique, sorted `selected_image_members.txt` from selected pages.
5. Build one archive-member index and prove every requested member exists.
6. Extract only those members into `$ROOT/pages` using `tar -T` or an
   equivalently selective, fail-closed method.
7. Reject path traversal, duplicate members, missing members, corrupt PNGs, and
   images whose gold boxes fall outside actual dimensions.

Do not infer that the archive completed merely because `images.tar` exists.
Use Slurm exit state plus integrity verification.

## MinerU 3.4 infrastructure for SOL

### What already exists

The repository pins `mineru[all]==3.4.0` on Python 3.12 and provides:

- a localhost-only API launcher with bounded concurrency;
- a `hybrid-engine`, high-effort, image-analysis command path;
- retry/recovery for the API result-ZIP race;
- safe ZIP extraction;
- hashed content-list, middle, model, and native-layout artifacts;
- canonical segment geometry in `[0,1000]` coordinates;
- faithful `image_caption` extraction from `middle.json`.

The existing `environments/mineru34-macos.yml` documents versions but is not a
SOL Linux/CUDA environment. Do not blindly reuse a macOS-resolved environment.

### SOL environment requirements

Create the Linux environment from a compute allocation, preferably at an
absolute path under `/home/$USER/mamba-envs/`. Pin Python 3.12 and MinerU 3.4.0.
Before choosing vLLM/lmdeploy dependencies, inspect `nvidia-smi`, the allocated
GPU architecture, driver, and CUDA compatibility. MinerU officially supports
Volta-or-later CUDA GPUs; dependency compatibility must be demonstrated on SOL,
not assumed.

Every sbatch must follow the proven SOL pattern:

```bash
module load mamba/latest
source activate "$ENV_NAME"
ENV_PREFIX="${CONDA_PREFIX:-$ENV_NAME}"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:$ENV_PREFIX/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-/scratch/$USER/hf_cache}"
PYTHON_BIN="$ENV_PREFIX/bin/python"
```

Print `which python`, `sys.executable`, MinerU version, Torch version, CUDA
availability, GPU name, and free VRAM before starting the API.

### Tracked smoke document

Do **not** commit SciEGQA PDFs or page images merely to make a smoke fixture;
their redistribution status has not been established, and the repository's
current policy keeps generated source material out of git.

Instead, create and commit a small self-generated PDF fixture under
`archive/tests/fixtures/mineru_smoke/`. It should contain a paragraph, a simple table,
a small generated figure with a caption, and an equation. Keep it small and
record its SHA-256. This satisfies the need for a document available before
`images.tar` completes without creating a licensing problem.

### GPU smoke job

Create a short, dedicated MinerU sbatch based on existing scripts. It should:

1. Request one GPU, modest CPU/RAM, and a short wall time.
2. Activate the absolute MinerU environment and print the preflight described
   above.
3. Set scratch model/output paths and `CUDA_VISIBLE_DEVICES=0`.
4. Start `mineru-api` bound to `127.0.0.1` on a job-specific port with one
   concurrent request and one-page processing window.
5. Capture the API PID and install an EXIT trap that terminates it.
6. Poll `/health` with a bounded timeout; do not use an arbitrary long sleep.
7. Parse only the tracked fixture with `hybrid-engine`, high effort, image,
   formula, and table analysis enabled.
8. Require exactly one content-list, middle, model, and native-layout artifact.
9. Canonicalize the result and assert that text/table/figure-caption geometry
   is finite and within `[0,1000]`.
10. Record environment freeze, model repository/revision, command, job ID,
    timestamps, artifact paths, sizes, and SHA-256 hashes.
11. Exit nonzero on any failed invariant.

The first API start may download MinerU model assets into scratch HF cache. Run
this concurrently with `images.tar`; do not serialize the downloads.

The existing PDF runner assumes PDF input and a source page index. The 4K
dataset will use selected PNG pages, so add a dedicated image-input command
path rather than passing fake PDF page indices. MinerU supports image input,
but the repository wrapper must preserve the same provenance and artifact
validation contract.

After the synthetic smoke passes and at least one real selected PNG is
available, run a second one-image smoke test. Do not launch the full 4K MinerU
run as part of this handoff.

## Same-page hard-negative contract for the later phase

For each selected query:

- parse its unique gold page once;
- reuse segments for all selected queries sharing that page;
- preserve atomic MinerU segments and exact boxes;
- preserve `image_caption` as a separate segment linked to its figure;
- positive candidates are defined against the gold evidence region under an
  explicit overlap/containment rule;
- hard negatives come only from other segments on the same page;
- never label a linked caption or figure as negative merely because the source
  annotation covered only its paired component without auditing that case.

The overlap rule and linked-bundle labeling policy require separate user
approval before full dataset generation.

## Deliverables from the SOL agent

The agent should produce:

1. A small JSON profiling/selection implementation with focused tests.
2. A resumable archive-download sbatch.
3. A MinerU Linux environment specification.
4. A tracked synthetic smoke PDF and checksum.
5. A short MinerU GPU smoke sbatch.
6. `source_profile.json` and `source_profile.md`.
7. The 4K selection manifests and `selection_audit.json`.
8. Selectively extracted and validated unique gold pages on scratch.
9. A smoke-run manifest containing the job ID and hashed MinerU artifacts.
10. A concise update to `sol/CURRENT_SOL_TASK.md` containing only live state,
    next action, job IDs, paths, and blockers.

Large archives, page images, model caches, logs, and MinerU outputs remain on
scratch and must not be added to git.

## Stop conditions

Stop and report instead of guessing if:

- the immutable revision is unavailable or source counts differ;
- there is insufficient scratch space;
- the archive member layout differs from the dataset card;
- the archive job exits unsuccessfully or integrity cannot be established;
- strict eligibility or deduplication leaves fewer rows than an approved quota;
- query-intent rules have large ambiguous/unmatched strata that would distort
  selection;
- the SOL GPU/CUDA stack cannot support the pinned MinerU environment;
- the synthetic fixture cannot be legally and reproducibly committed;
- any proposed action would modify the old frozen verifier dataset.

## Completion boundary

This handoff is complete when the 4K selection and required unique pages are
validated, the full archive remains available/resumable on scratch, and the
synthetic plus one-real-page MinerU smoke tests pass. Full MinerU processing of
all selected pages is a later, separately approved run.
