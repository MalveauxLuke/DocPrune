# Evidence-DINO-Units Stages 0–4: Binding SOL Execution Contract

**Status:** Prepared locally; execution has not started on SOL.

**Owner approval:** The project owner approved the research-only Stage 0–4
scope and the ten experiment invariants on 2026-07-28.

This is the binding implementation contract for the current SOL agent. Execute
Stages 0, 1, 2, 3, and 4 in order. Complete each deliverable and advancement
gate before advancing. Stop after Stage 4.

## 1. Authority and conflict resolution

Read in this order:

1. root `AGENTS.md`;
2. `docs/SOL_INSTRUCTIONS.md`;
3. `sol/AGENTS.md`;
4. `sol/CURRENT_SOL_TASK.md`;
5. this binding contract;
6. `sol/task_spec/evidence_dino_units_training_plan_final.md` through Stage 4;
7. `agent-context/INDEX.md`;
8. `agent-context/CURRENT_TASK.md`;
9. `agent-context/SOL.md`; and
10. `agent-context/DATA_AND_ARTIFACTS.md`.

Authority order:

1. direct project-owner instructions;
2. root and nested `AGENTS.md`;
3. this binding Stage 0–4 contract;
4. explicitly marked **Binding Stage 0–4 SOL update** passages in the full
   plan;
5. all other unmarked full-plan text; and
6. archived material as nonbinding procedural evidence.

If two active sources conflict outside the explicitly versioned updates, stop
and report the exact passages. Do not choose silently.

## 2. Exact project identity

```text
source checkout:     ~/COLPALI_binary_classification
experiment worktree: ~/Evidence-DINO-Units
experiment branch:   codex/evidence-dino-units-dataset-stages-0-4
remote:              origin
scratch root:        /scratch/$USER/evidence_dino_units
acquisition env:     /home/$USER/mamba-envs/evidence-dino-acquire
full plan:           sol/task_spec/evidence_dino_units_training_plan_final.md
binding contract:    sol/task_spec/evidence_dino_units_stages_0_4.md
path manifest:       sol/task_spec/evidence_dino_units_workspace_paths.txt
sparse list:         sol/task_spec/evidence_dino_units_sparse_checkout.txt
```

The source checkout contains the repository-wide project information. The
experiment worktree is a separate sibling folder and becomes the SOL agent's
main workspace. It is not `main`.

## 3. Bootstrap and Git gate

Use the exact bootstrap in
`sol/task_spec/evidence_dino_units_workspace_bootstrap.md`.

Before reading data or creating an environment, record:

```bash
cd "$HOME/Evidence-DINO-Units"
git status --short --branch
git branch --show-current
git rev-parse HEAD
git rev-parse origin/codex/evidence-dino-units-dataset-stages-0-4
git merge-base --is-ancestor \
  origin/main \
  origin/codex/evidence-dino-units-dataset-stages-0-4
git remote -v
```

Required conditions:

- the current branch is exactly
  `codex/evidence-dino-units-dataset-stages-0-4`;
- the worktree is clean before implementation;
- local `HEAD` equals the fetched remote experiment-branch commit;
- the experiment branch contains the pushed handoff on `origin/main`;
- no other worktree is removed, reset, cleaned, or modified; and
- no force push is used.

Write the observed source, base, and execution commits into:

```text
data/manifests/repository_revisions.json
reports/stage_1/repository_bootstrap.md
```

Stop on a dirty target, wrong branch, detached HEAD, remote mismatch, or
unavailable pushed branch.

## 4. Compute, storage, and Git boundaries

### 4.1 Login node

Permitted:

- Git fetch/worktree/status/commit/push;
- reading and editing small files;
- `sinfo`, `squeue`, `sacct`, and job submission;
- lightweight manifest inspection; and
- checking paths and quotas.

Forbidden:

- environment creation or package installation;
- model or dataset downloads;
- hashing the archive/image corpus;
- archive validation or extraction;
- image decoding/indexing;
- metadata normalization or split generation; and
- any sustained Python computation.

### 4.2 Allocation patterns

For environment setup and small CPU validation:

```bash
salloc -p lightwork -q public -t 02:00:00 -c 4
```

For metadata audits expected to finish within four hours:

```bash
salloc -p htc -q public -t 04:00:00 -c 8 --mem=64G
```

For the image download, stream validation/extraction, full indexing, and
hashing, create `public` CPU SBATCH jobs with measured walltime and storage
requirements. Do not request a GPU for Stages 0–4.

Before every submission:

```bash
sinfo
squeue -u "$USER"
```

After submission, confirm the job starts, record the job ID, and stop polling
unless monitoring was explicitly requested.

### 4.3 Physical paths

Set:

```bash
export PROJECT_DIR="$HOME/Evidence-DINO-Units"
export ARTIFACT_ROOT="/scratch/$USER/evidence_dino_units"
export HF_HOME="$ARTIFACT_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export XDG_CACHE_HOME="$ARTIFACT_ROOT/cache/xdg"
export TMPDIR="$ARTIFACT_ROOT/tmp/${SLURM_JOB_ID:-interactive}"
export PYTHONNOUSERSITE=1
```

Large physical paths:

```text
$ARTIFACT_ROOT/data/raw
$ARTIFACT_ROOT/data/images
$ARTIFACT_ROOT/data/metadata
$ARTIFACT_ROOT/data/splits
$ARTIFACT_ROOT/cache
$ARTIFACT_ROOT/logs
$ARTIFACT_ROOT/reports/overlays
$ARTIFACT_ROOT/tmp
```

Git-safe paths:

```text
$PROJECT_DIR/data/manifests
$PROJECT_DIR/data/metadata/audits
$PROJECT_DIR/data/splits/manifests
$PROJECT_DIR/docs
$PROJECT_DIR/configs
$PROJECT_DIR/environments
$PROJECT_DIR/reports
$PROJECT_DIR/src
$PROJECT_DIR/tests
```

Do not commit:

- the six raw JSONL source files;
- Visual-CoT archive parts;
- extracted or selected images;
- Parquet tables containing local absolute paths unless packaged as an
  explicitly approved small audit fixture;
- model weights;
- environments or caches;
- generated overlays/contact sheets;
- Slurm logs; or
- machine-specific absolute symlinks.

## 5. Required stage-report format

Create:

```text
reports/stage_0/status.md
reports/stage_1/status.md
reports/stage_2/status.md
reports/stage_3/status.md
reports/stage_4/status.md
```

Every report records:

- state: `not_started`, `in_progress`, `blocked`, `gate_failed`, or `complete`;
- exact start/end timestamps in UTC;
- Git branch and commit;
- input revisions and hashes;
- environment lock/hash;
- commands and Slurm job IDs;
- observed counts rather than planning counts;
- outputs and their hashes;
- exclusions/quarantines with reason codes;
- gate evidence;
- deviations and owner approvals; and
- next permitted action.

Do not mark a stage complete from job exit status alone.

---

# Stage 0 — Experiment contract

## Stage 0 objective

Demonstrate understanding of the research target and freeze the owner-approved
invariants before implementation.

## Required invariant text

Create `docs/experiment_contract.md` containing these ten invariants verbatim:

1. Visual-CoT DUDE is part of Visual-CoT from the start.
2. No separate DUDE dataset is downloaded, joined, or used.
3. Multiple Visual-CoT boxes remain separate; never replace them with one
   enclosing rectangle.
4. Raw annotations are immutable and always recoverable.
5. Splits are grouped by visual identity, never by question row.
6. The answer is never provided to the selector as input.
7. A row without a defensible candidate mapping never receives fabricated
   negative unit labels.
8. TextVQA/TextCaps are conditional until their candidate route passes an
   oracle audit.
9. MMLongBench-Doc is not used for training, prompt selection, threshold
   selection, source weighting, architecture selection, or checkpoint
   selection.
10. Actual visual/OCR token counts are measured; crop area is not assumed to
    equal token savings.

Also record:

```text
owner: project owner
owner approval date: 2026-07-28
intended use: research-only
license status: unresolved for redistribution, commercial use, and weight publication
current execution scope: Stages 0–4 only
```

## Required conceptual explanation

In `reports/stage_0/status.md`, explain without code:

- phrase grounding: localizing a region named by an input phrase;
- answer-bearing localization: using the question to find a region containing
  answer information not necessarily repeated in the question; and
- semantically complete evidence selection: collecting every region/context
  needed to support the answer, which may exceed the annotated answer box.

The explanation must state that Visual-CoT boxes primarily supervise
answer-bearing localization and do not guarantee complete evidence.

## Stage 0 advancement gate

Advance only when:

- `docs/experiment_contract.md` exists;
- all ten invariants are verbatim;
- approval date/use/license status are explicit;
- the conceptual distinctions are correct; and
- no implementation, download, or environment work preceded the contract.

Commit the Stage 0 contract and report before Stage 1.

---

# Stage 1 — Reproducible workspace and locked revisions

## Stage 1 objective

Create the project structure, acquisition environment, immutable source
registry, revision locks, and license audit without downloading model weights
or the large image archive.

## Required logical tree

```text
Evidence-DINO-Units/
├── external/
│   ├── Visual-CoT/
│   ├── GroundingDINO/
│   └── MMLongBench-Doc/
├── data/
│   ├── raw/visual_cot_repo/
│   ├── images/visual_cot/
│   ├── metadata/{normalized,audits}/
│   ├── ocr/deepseek_ocr2/
│   ├── candidates/
│   ├── mapped/
│   ├── splits/
│   ├── manifests/
│   └── eval/mmlongbench_sealed/
├── configs/
├── src/{data,candidates,models,training,evaluation}/
├── tests/
├── experiments/
├── checkpoints/
├── reports/
├── environments/
└── docs/
```

Large runtime paths map to scratch as defined in Section 4.3. Directory
creation does not authorize later-stage population.

## Acquisition environment

From a compute allocation:

```bash
module load mamba/latest
ENV_PREFIX="/home/$USER/mamba-envs/evidence-dino-acquire"
mamba create --yes --prefix "$ENV_PREFIX" -c conda-forge \
  python=3.12 \
  huggingface_hub \
  pyyaml \
  orjson \
  pandas \
  pyarrow \
  pillow \
  imagehash \
  tqdm
source activate "$ENV_PREFIX"
python -V
hf --version
python -m pip freeze | sort \
  > "$PROJECT_DIR/reports/stage_1/acquisition_environment.txt"
```

After versions are observed and imports pass, create a reproducible
`environments/evidence-dino-acquire-sol.yml` using exact tested versions. Do
not guess package versions before observing the resolved environment.

## Immutable registry

Create `data/manifests/source_registry.yaml` containing:

```yaml
project_use: research-only
sources:
  visual_cot:
    repo: deepcs233/Visual-CoT
    type: dataset
    revision: 223d2d8c1146fda2bb918801b8276c587b78b61c
    selected_metadata:
      - metadata/docvqa_cot_train.jsonl
      - metadata/dude_cot_train.jsonl
      - metadata/infographicsvqa_cot_train.jsonl
      - metadata/sroie_cot_train.jsonl
      - metadata/textvqa_cot_train.jsonl
      - metadata/textcap_cot_train.jsonl
  deepseek_ocr2:
    repo: deepseek-ai/DeepSeek-OCR-2
    type: model
    revision: aaa02f3811945a91062062994c5c4a3f4c0af2b0
    download_stage: 6
  grounding_dino_tiny:
    repo: IDEA-Research/grounding-dino-tiny
    type: model
    revision: a2bb814dd30d776dcf7e30523b00659f4f141c71
    download_stage: 5
  mmlongbench_doc:
    repo: yubo2333/MMLongBench-Doc
    type: dataset
    revision_request: 2ff6aa9
    download_stage: 14
```

Resolve all Hub revisions through the API exactly once and write full hashes
to `data/manifests/revisions.lock.json`. A later revision requires a new lock
and lineage; never overwrite silently.

## Reference clones

From a compute/lightwork allocation:

```bash
mkdir -p "$PROJECT_DIR/external" "$PROJECT_DIR/data/manifests"
GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none \
  https://github.com/deepcs233/Visual-CoT.git \
  "$PROJECT_DIR/external/Visual-CoT"
git clone --filter=blob:none \
  https://github.com/IDEA-Research/GroundingDINO.git \
  "$PROJECT_DIR/external/GroundingDINO"
```

Do not clone MMLongBench code until Stage 14. Do not install the Visual-CoT
training stack. Record exact clone commits in:

```text
data/manifests/github_revisions.lock
```

## License audit

Save the pinned Visual-CoT README/license material and source terms without
adding large repository contents to Git. Write `docs/license_audit.md` with:

- reviewed source/revision;
- observed license statement;
- the Apache-2.0 versus CC BY-NC 4.0 discrepancy;
- constituent-source terms status;
- owner: project owner;
- intended use: research-only;
- redistribution: not approved;
- commercial use: not approved;
- model-weight publication: not approved; and
- required follow-up before any of those actions.

## Stage 1 required outputs

```text
data/manifests/source_registry.yaml
data/manifests/revisions.lock.json
data/manifests/github_revisions.lock
data/manifests/repository_revisions.json
docs/license_audit.md
environments/evidence-dino-acquire-sol.yml
reports/stage_1/acquisition_environment.txt
reports/stage_1/repository_bootstrap.md
reports/stage_1/status.md
```

## Stage 1 advancement gate

Advance only when:

- every external source has an immutable revision or explicitly recorded
  future-stage resolution;
- no run manifest references moving `main`;
- clone commits and environment versions are recorded;
- the license discrepancy has an owner and unresolved distribution status;
- no model weights, image archive, DeepSeek output, or MMLongBench data were
  downloaded; and
- a clean checkout can validate every Git-safe Stage 1 output.

---

# Stage 2 — Six-source Visual-CoT metadata acquisition and audit

## Stage 2 objective

Download only the six pinned metadata JSONL files and README, audit every row,
and freeze a deterministic 600-record visual-review manifest.

## Download

Run from an allocation with the acquisition environment active:

```bash
RAW_REPO="$ARTIFACT_ROOT/data/raw/visual_cot_repo"
mkdir -p "$RAW_REPO" "$PROJECT_DIR/data/manifests"

hf download deepcs233/Visual-CoT \
  metadata/docvqa_cot_train.jsonl \
  metadata/dude_cot_train.jsonl \
  metadata/infographicsvqa_cot_train.jsonl \
  metadata/sroie_cot_train.jsonl \
  metadata/textvqa_cot_train.jsonl \
  metadata/textcap_cot_train.jsonl \
  README.md \
  --repo-type dataset \
  --revision 223d2d8c1146fda2bb918801b8276c587b78b61c \
  --local-dir "$RAW_REPO"
```

No other DUDE data is allowed.

## Raw checksums and physical counts

```bash
find "$RAW_REPO/metadata" -type f -name '*.jsonl' -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "$PROJECT_DIR/data/manifests/visual_cot_metadata.sha256"

for name in \
  docvqa_cot_train.jsonl \
  dude_cot_train.jsonl \
  infographicsvqa_cot_train.jsonl \
  sroie_cot_train.jsonl \
  textvqa_cot_train.jsonl \
  textcap_cot_train.jsonl
do
  printf '%-36s ' "$name"
  wc -l < "$RAW_REPO/metadata/$name"
done | tee "$PROJECT_DIR/data/manifests/visual_cot_line_counts.txt"
```

The exact parsed count from `dude_cot_train.jsonl` is the sole authoritative
DUDE count.

## Metadata audit schema

Implement focused code under `src/data/` with pytest coverage. For every file
and aggregate, report:

```text
physical_line_count
blank_line_count
parsed_record_count
parse_failure_count
schema_key_frequencies
missing_required_field_counts
box_count_histogram
invalid_xyxy_count
out_of_bounds_box_count
zero_area_after_clip_count
unique_image_identifier_count
questions_per_image_distribution
duplicate_normalized_question_image_count
exact_duplicate_record_count
observed_source_values
observed_split_values
cross_source_filename_collision_count
```

Required fields are question, answer, image identifier, width, height, and
`bboxs`. Preserve malformed and rejected row identities with reason codes; do
not silently skip them.

Write:

```text
$ARTIFACT_ROOT/data/metadata/audits/metadata_audit.json
$PROJECT_DIR/data/metadata/audits/metadata_audit.md
$PROJECT_DIR/data/metadata/audits/invalid_rows.jsonl
$PROJECT_DIR/data/manifests/metadata_audit_manifest.json
```

The Git-safe summary/manifest must include hashes of scratch-only detailed
outputs.

## Deterministic 600-record review sample

For each source, select exactly 100 technically parseable rows. Calculate:

- number of boxes;
- total and maximum box-area fraction;
- normalized question length;
- image reuse count;
- repeated versus unique question;
- source-specific provisional direct-extraction versus computed/comparison
  heuristic.

Use source-local quantile bins for numeric fields. Traverse nonempty combined
strata round-robin. Within a stratum, order by:

```text
sha256(
  "evidence-dino-review-v1"
  + NUL + source
  + NUL + source_revision
  + NUL + source_row_index
  + NUL + source_image_id
  + NUL + question_normalized
)
```

Continue round-robin until 100 unique rows are selected. Do not replace failed
visual-review records silently later; record a frozen-record failure and use a
separately identified supplemental review row without changing the original
manifest.

Write:

```text
data/metadata/audits/review_sample_600.jsonl
data/metadata/audits/review_sample_600_summary.json
data/manifests/review_sample_600.sha256
```

The manifest is metadata-only in Stage 2. Visual dispositions are deferred
until Stage 3.

## Stage 2 required outputs

```text
data/manifests/visual_cot_metadata.sha256
data/manifests/visual_cot_line_counts.txt
data/manifests/metadata_audit_manifest.json
data/manifests/review_sample_600.sha256
data/metadata/audits/metadata_audit.md
data/metadata/audits/invalid_rows.jsonl
data/metadata/audits/review_sample_600.jsonl
data/metadata/audits/review_sample_600_summary.json
reports/stage_2/status.md
```

## Stage 2 advancement gate

Advance to Stage 3 only when:

- JSONL parse success is 100% for nonblank lines or every parse failure has a
  blocking explicit disposition;
- every invalid/missing annotation has a reason-coded disposition;
- the observed DUDE count comes only from `dude_cot_train.jsonl`;
- no separate DUDE source exists in the registry or filesystem;
- all six raw checksums and counts are frozen;
- the deterministic sample contains 100 unique rows per source; and
- the status report explicitly labels the 600-row visual report deferred, not
  complete.

---

# Stage 3 — Image archive, deterministic joins, and visual review

## Stage 3 objective

Download and verify all 13 Visual-CoT image archive parts, extract and index
the image tree, resolve selected metadata rows, and close the 600-record visual
review.

## Storage preflight

Run from the target filesystem:

```bash
mkdir -p "$ARTIFACT_ROOT"
df -h "$ARTIFACT_ROOT"
python - <<'PY'
import shutil
from pathlib import Path

root = Path("/scratch") / Path.home().name / "evidence_dino_units"
free = shutil.disk_usage(root).free
required = 350 * 1024**3
print({"root": str(root), "free_bytes": free, "required_bytes": required})
if free < required:
    raise SystemExit("Stage 3 requires at least 350 GiB free")
PY
```

Also inspect the applicable user quota/file-count policy. Stop if 350 GiB
cannot be reserved without threatening other active work.

## Dry run

```bash
RAW_REPO="$ARTIFACT_ROOT/data/raw/visual_cot_repo"
hf download deepcs233/Visual-CoT \
  --repo-type dataset \
  --revision 223d2d8c1146fda2bb918801b8276c587b78b61c \
  --include 'cot_images_tar_split/cot_images_*' \
  --local-dir "$RAW_REPO" \
  --dry-run \
  | tee "$PROJECT_DIR/reports/stage_3/image_download_dry_run.txt"
```

Review the exact 13 listed files and total bytes before submission.

## Full download

Submit a CPU `public` job that runs:

```bash
hf download deepcs233/Visual-CoT \
  --repo-type dataset \
  --revision 223d2d8c1146fda2bb918801b8276c587b78b61c \
  --include 'cot_images_tar_split/cot_images_*' \
  --local-dir "$RAW_REPO"
```

Require exactly:

```text
cot_images_00
cot_images_01
cot_images_02
cot_images_03
cot_images_04
cot_images_05
cot_images_06
cot_images_07
cot_images_08
cot_images_09
cot_images_10
cot_images_11
cot_images_12
```

Hash immediately:

```bash
sha256sum "$RAW_REPO"/cot_images_tar_split/cot_images_{00..12} \
  > "$PROJECT_DIR/data/manifests/visual_cot_image_parts.sha256"
```

Do not delete the parts.

## Validate and extract

In a CPU `public` job:

```bash
PART_ROOT="$RAW_REPO/cot_images_tar_split"
IMAGE_ROOT="$ARTIFACT_ROOT/data/images/visual_cot"
mkdir -p "$IMAGE_ROOT"

LC_ALL=C cat "$PART_ROOT"/cot_images_{00..12} | tar -tf - >/dev/null
LC_ALL=C cat "$PART_ROOT"/cot_images_{00..12} | tar -xf - -C "$IMAGE_ROOT"
```

Stop on a missing part, unreadable stream, unsafe archive member path,
extraction collision, or filesystem exhaustion.

Require source directories resolving to:

```text
cot/docvqa
cot/infographicsvqa
cot/dude
cot/sroie
cot/textvqa
cot/openimages
```

Do not use an unconstrained basename search to compensate for a missing source
folder.

## Image index schema

Create scratch `image_index.parquet` with:

```text
absolute_path
relative_path
basename
extension
file_size
sha256
width
height
perceptual_hash
source_folder_guess
decode_status
```

Sort rows by normalized relative path. Image decoding failures are retained
with a reason, not dropped.

## Ordered metadata-image join

For each metadata row, attempt:

1. exact source-relative path;
2. exact normalized path after removing one known archive prefix;
3. source-constrained basename;
4. source-constrained basename plus declared width/height; or
5. unresolved.

Never use an unconstrained ambiguous basename. Write:

```text
$ARTIFACT_ROOT/data/metadata/metadata_image_join.parquet
$PROJECT_DIR/data/metadata/audits/image_join_summary.json
$PROJECT_DIR/data/metadata/audits/ambiguous_image_joins.jsonl
$PROJECT_DIR/data/metadata/audits/missing_image_joins.jsonl
$PROJECT_DIR/data/manifests/image_index_manifest.json
```

The manifest records index/join row counts, schemas, hashes, code commit,
source archive hash-manifest hash, and physical scratch paths.

## Selected working image tree

Generate `required_images.txt` from normalized metadata. Create the selected
tree using hard links when source and target share a filesystem; otherwise use
untracked relative runtime symlinks. Never commit absolute links.

```text
$ARTIFACT_ROOT/data/images/selected_visual_cot/docvqa
$ARTIFACT_ROOT/data/images/selected_visual_cot/infographicsvqa
$ARTIFACT_ROOT/data/images/selected_visual_cot/dude
$ARTIFACT_ROOT/data/images/selected_visual_cot/sroie
$ARTIFACT_ROOT/data/images/selected_visual_cot/textvqa
$ARTIFACT_ROOT/data/images/selected_visual_cot/openimages
```

## 600-record visual inspection

Resolve every frozen review row. Render:

- the original image;
- question;
- answer;
- source and stable row ID;
- declared and decoded dimensions; and
- every raw box separately with stable labels.

Store overlays/contact sheets on scratch. Write Git-safe:

```text
data/metadata/audits/review_sample_600_dispositions.jsonl
data/metadata/audits/review_sample_600_report.md
data/manifests/review_sample_600_overlays_manifest.json
```

Each disposition records:

```text
review_id
source
source_row_index
image_sha256
overlay_sha256
image_join_correct
coordinate_transform_correct
all_boxes_visible
annotation_issue_codes[]
review_notes
review_status
```

All 600 require a terminal reviewed status. Supplemental rows do not erase
frozen failures.

## Stage 3 advancement gate

Advance only when:

- all 13 part hashes are frozen;
- stream validation and extraction succeed;
- every selected source directory exists;
- at least 99.9% of technically valid rows resolve to one readable image;
- every ambiguous/unresolved/unreadable row is quarantined;
- all 600 frozen rows have terminal visual dispositions;
- each source contributes exactly 100 reviewed frozen rows;
- coordinate overlays are visually correct for every accepted sample; and
- the deferred Stage 2 visual deliverable is explicitly closed.

---

# Stage 4 — Canonical records and leakage-safe grouped splits

## Stage 4 objective

Normalize accepted records, assign supervision classes, build visual-identity
groups across sources, and freeze deterministic training/development/audit
splits without constructing candidates or opening MMLongBench.

## Canonical record schema

Every normalized row contains:

```text
example_id
source
source_revision
source_row_index
source_split
source_document_id_optional
source_page_id_optional
source_image_id
image_path
image_sha256
image_phash
width
height
image_style
question
question_normalized
answers[]
answer_variants[]
raw_boxes[]
normalized_boxes[]
answer_type
supervision_class
candidate_generator
candidate_generator_revision
candidates[]
positive_candidate_ids[]
contextual_positive_ids[]
ignored_candidate_ids[]
hard_negative_candidate_ids[]
mapping_confidence
local_loss_weight
page_label
split_group_id
split
leakage_flags[]
```

During Stage 4:

```text
candidate_generator = null
candidate_generator_revision = null
candidates = []
positive_candidate_ids = []
contextual_positive_ids = []
ignored_candidate_ids = []
hard_negative_candidate_ids = []
mapping_confidence = null
local_loss_weight = null
```

Supervision-class multipliers belong in the Stage 4 policy/config and report;
they do not become candidate-level labels before mapping exists.

## Coordinate contract

- preserve raw `xyxy` arrays exactly;
- store normalized coordinates separately;
- store every resize/pad transform and inverse;
- preserve every box rather than unioning;
- require integer-coordinate round-trip error below one source pixel; and
- retain overlay evidence for every preprocessing route.

## Supervision classes

Assign exactly one:

```text
strong_semantic_localization
layout_localization
direct_ocr_localization
incomplete_or_computed_evidence
```

Record the plan's source/class local-loss multipliers as future configuration.
Rows with incomplete evidence must not imply that every unboxed region is a
negative.

## Visual-identity grouping

Build a disjoint-set graph over:

- exact image SHA-256;
- validated perceptual-hash near-duplicate edges;
- source document ID when available;
- source image ID;
- known shared underlying image IDs; and
- shared TextVQA/TextCaps OpenImages identities.

Each connected component is one `split_group_id`. Generate the stable group ID
from the lexicographically sorted immutable member identities and split seed:

```text
evidence-dino-split-v1
```

Do not connect images solely from a weak filename similarity. Record edge type,
threshold, and evidence in the cluster table.

## Deterministic split assignment

Hash the stable `split_group_id` with the fixed seed. Use the first unsigned
64-bit value modulo 100:

```text
00–89 -> train
90–94 -> development_validation
95–99 -> sealed_internal_audit
```

Group constraints control actual row percentages. Do not move individual
questions to force exact ratios. If extreme group imbalance makes a split
unusable, stop and report it rather than selecting a new seed after inspection.

The sealed audit manifest may be created and hashed, but its examples must not
be repeatedly inspected or used for architecture, threshold, scene-source, or
checkpoint decisions.

## Required outputs

Scratch:

```text
$ARTIFACT_ROOT/data/metadata/normalized/<source>.parquet
$ARTIFACT_ROOT/data/splits/visual_identity_clusters.parquet
$ARTIFACT_ROOT/data/splits/split_assignments.parquet
```

Git-safe:

```text
data/manifests/normalized_records_manifest.json
data/manifests/split_manifest.json
data/metadata/audits/supervision_class_report.md
data/metadata/audits/coordinate_roundtrip_report.json
data/splits/manifests/visual_identity_clusters.json
data/splits/manifests/train_groups.txt
data/splits/manifests/development_validation_groups.txt
data/splits/manifests/sealed_internal_audit_groups.txt
data/splits/manifests/leakage_audit.json
reports/stage_4/status.md
```

Manifests record row/group counts by source and supervision class, schemas,
hashes of scratch tables, code commit, split seed, pHash policy, and every
quarantine.

## MMLongBench quarantine

Record the resolved immutable MMLongBench revision in the Stage 1 lock and
reserve `data/eval/mmlongbench_sealed/`. Do not acquire MMLongBench. Do not
download, mount, inspect, render, hash, or use its annotations or documents.

## Stage 4 advancement gate

The task is complete only when:

- all accepted rows satisfy the canonical schema;
- raw boxes are unchanged and normalized boxes round-trip below one source
  pixel;
- future candidate/mapping fields have the required null/empty values;
- no exact or validated near-duplicate identity crosses splits;
- TextVQA/TextCaps shared images are in the same component/split;
- split manifests reproduce byte-for-byte from reversed input order;
- the 5% audit-group assignment is sealed;
- scratch table hashes and Git-safe manifests agree; and
- MMLongBench remains unacquired.

Stop after documenting and committing this gate.

---

# Hard boundary after Stage 4

Do not execute Stage 5.

Do not download Grounding-DINO weights or run T0-BOX.

Do not download or run DeepSeek-OCR-2.

Do not construct semantic, fallback, scene, or mapped candidates.

Do not train any selector, detector, reranker, or answerer.

Do not acquire MMLongBench.

Do not serve a viewer or website.

## Handback

After Stage 4:

1. run focused and full repository tests;
2. verify every Git-safe manifest/report reference and checksum;
3. scan staged files for forbidden extensions and oversized artifacts;
4. update `sol/CURRENT_SOL_TASK.md` with exact commits, stage states, job IDs,
   scratch paths, observed counts, and blockers;
5. commit only to
   `codex/evidence-dino-units-dataset-stages-0-4`;
6. push the experiment branch;
7. verify the remote branch commit with `git ls-remote`; and
8. stop for a new owner-approved Stage 5 handoff.

Do not merge or push experiment implementation to `main`.

## Stop conditions

Stop and report evidence if:

- the exact branch/worktree/remote commit cannot be established;
- the owner-approved invariants or license status are missing;
- a required pinned revision cannot be resolved;
- a source file/schema/path differs materially from the contract;
- any separate DUDE corpus appears;
- repeated preparation is nondeterministic;
- scratch has less than 350 GiB free before Stage 3;
- the archive is incomplete or unsafe;
- the image join or coordinate gate fails;
- duplicate groups cross splits;
- a required visual review cannot be completed;
- Git-safe handback would require committing large/forbidden artifacts; or
- authentication prevents a complete experiment-branch push.
