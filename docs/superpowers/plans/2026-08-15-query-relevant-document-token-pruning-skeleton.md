# Query-Relevant Document Token Pruning Skeleton Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a small sibling research repository that preserves reusable document-QA governance, research, navigation, and SOL knowledge without carrying old-project implementation or generated artifacts.

**Architecture:** Use a manifest-first extraction from the unchanged source repository into a fresh destination repository. Keep active governance intentionally generic, isolate inherited authority as historical reference, preserve the approved papers and SOL examples, and verify the result against a recorded source fingerprint and explicit exclusion rules.

**Tech Stack:** Git, POSIX shell, `rsync`, `find`, `shasum`, `bash -n`, Markdown

## Global Constraints

- Source is read-only: `/Users/god/Documents/COLPALI binary classification`.
- Destination is `/Users/god/Documents/Query-Relevant Document Token Pruning`.
- Active governance may know only that this repository concerns token compression for document QA and that no execution is active.
- The top-level README must state the project goal, probability target, and central research question without selecting an architecture.
- Preserve all instruction, navigation, SOL/HPC, handoff, research, reference, and approved paper material named by the design.
- Retain exactly seven canonical paper PDFs; omit their rendered/extracted derivatives and supplemental archives.
- Do not copy the source `.git`, implementation modules, coupled tests, datasets, checkpoints, caches, viewers, experiment outputs, or scratch material.
- Historical source material is nonbinding until a new-project task explicitly activates it.
- Use `apply_patch` for authored file changes; use `rsync` or `cp` only for byte-preserving copies.
- Do not push or create a remote.

---

### Task 1: Capture the source baseline and exact extraction inventory

**Files:**

- Create: `work/source-retained-files.txt`
- Create: `work/source-retained-sha256.txt`
- Create: `work/source-state.txt`
- Create: `docs/SOURCE_INVENTORY.md`

**Interfaces:**

- Consumes: the source filesystem and Git state
- Produces: a stable retained-surface manifest and checksum set used by Task 6

- [ ] **Step 1: Confirm repository identities and destination state**

Run:

```bash
SOURCE_REPO='/Users/god/Documents/COLPALI binary classification'
DEST_REPO='/Users/god/Documents/Query-Relevant Document Token Pruning'
test "$(git -C "$SOURCE_REPO" rev-parse --show-toplevel)" = "$SOURCE_REPO"
test "$(git -C "$DEST_REPO" rev-parse --show-toplevel)" = "$DEST_REPO"
test "$(git -C "$DEST_REPO" branch --show-current)" = main
git -C "$SOURCE_REPO" status --short --branch
git -C "$DEST_REPO" status --short --branch
```

Expected: both identities match; the source reports its pre-existing `main` state; the destination contains only committed planning documentation.

- [ ] **Step 2: Generate the retained-surface file manifest mechanically**

Create `work/source-retained-files.txt` from the approved categories, excluding all implementation and generated paths:

```bash
mkdir -p work
SOURCE_REPO='/Users/god/Documents/COLPALI binary classification'
(
  cd "$SOURCE_REPO"
  printf '%s\n' AGENTS.md README.md LICENSE .gitignore .rgignore .cursorignore .cursorindexingignore .gitattributes
  find agent-context docs reports archive sol -type f \
    \( -name '*.md' -o -name '*.txt' -o -name '*.yml' -o -name '*.yaml' -o -name '*.sbatch' -o -name '*.sh' \)
  find configs data environments experiments external src -type f \
    \( -name 'README.md' -o -name '*.yml' -o -name '*.yaml' \)
  printf '%s\n' \
    tmp/pdfs/dude.pdf \
    tmp/pdfs/mmlongbench_doc.pdf \
    tmp/pdfs/slidevqa.pdf \
    tmp/pdfs/tat_dqa.pdf \
    tmp/pdfs/ufmg.tot.pdf \
    tmp/pdfs/vgent_2512_11099v1/vgent_2512.11099v1.pdf \
    work/paper_review/m3grounder.pdf \
    work/paper_review/boundingdocs.md \
    work/paper_review/dude.md
) | LC_ALL=C sort -u > work/source-retained-files.txt
test -s work/source-retained-files.txt
```

Expected: the manifest contains only the explicit text/config/SOL surface, seven PDFs, and two authored paper notes.

- [ ] **Step 3: Record source state and retained-surface hashes**

Run:

```bash
SOURCE_REPO='/Users/god/Documents/COLPALI binary classification'
{
  git -C "$SOURCE_REPO" status --short --branch
  git -C "$SOURCE_REPO" rev-parse HEAD
  du -sk "$SOURCE_REPO"
  wc -l work/source-retained-files.txt
} > work/source-state.txt
(
  cd "$SOURCE_REPO"
  while IFS= read -r retained_file; do
    test -f "$retained_file"
    shasum -a 256 "$retained_file"
  done < "$OLDPWD/work/source-retained-files.txt"
) > work/source-retained-sha256.txt
test -s work/source-state.txt
test -s work/source-retained-sha256.txt
```

Expected: both records are non-empty, and every manifest entry exists.

- [ ] **Step 4: Write the initial source inventory**

Use `apply_patch` to create `docs/SOURCE_INVENTORY.md` with these sections and the measured values from Steps 2–3:

```markdown
# Source Inventory

## Source snapshot

- Source: `/Users/god/Documents/COLPALI binary classification`
- Branch and revision: recorded before extraction
- Working state: recorded verbatim before extraction
- Source size: recorded in KiB before extraction
- Retained-surface manifest: generated before any copy

## Retain classification

| Source area | Treatment |
|---|---|
| Root governance and navigation | Preserve originals under archival governance; create minimal active replacements |
| `agent-context/` | Preserve research; archive old active task/context files |
| `docs/`, `reports/`, archival Markdown | Retain as nonbinding research/reference material |
| `sol/` instructions, handoffs, task specs, and wrappers | Retain as historical material; expose neutral examples separately |
| Lightweight README/config examples | Retain or distill without active old-project authority |
| Seven approved paper PDFs and two authored notes | Retain under `references/` with checksums |

## Remove classification

Implementation modules, coupled tests, Git history, datasets, dataset PDFs,
embeddings, checkpoints, outputs, viewers, caches, logs, worktrees, scratch
trees, rendered paper pages, extracted paper text, HTML captures, and
supplemental archives are intentionally absent.

## Adapted active governance

List every active file replaced by a minimal new-project version and its exact
archived source counterpart.

## Retained papers

List source path, destination path, bytes, and SHA-256 for all seven PDFs.

## Retained SBATCH material

Record the historical wrapper count and the four distilled example paths.
```

- [ ] **Step 5: Verify the inventory change**

Run:

```bash
DEST_REPO='/Users/god/Documents/Query-Relevant Document Token Pruning'
git -C "$DEST_REPO" diff --check
test -s "$DEST_REPO/docs/SOURCE_INVENTORY.md"
```

Expected: exit 0 with no whitespace errors.

- [ ] **Step 6: Commit the baseline inventory document**

Run:

```bash
DEST_REPO='/Users/god/Documents/Query-Relevant Document Token Pruning'
git -C "$DEST_REPO" add docs/SOURCE_INVENTORY.md
git -C "$DEST_REPO" diff --cached --check
git -C "$DEST_REPO" commit -m 'docs: record source extraction inventory'
```

Expected: only the authored inventory document is committed; mechanical
baseline files remain outside the destination repository.

---

### Task 2: Preserve source knowledge without inheriting active authority

**Files:**

- Copy: source governance originals to `archive/source-repo-governance/`
- Copy: source directory-convention originals to `archive/source-repo-governance/`
- Copy: `agent-context/research/` to the same destination path
- Copy: retained `docs/specifications/`, `docs/superpowers/`, `reports/`, and documentation-only `archive/` content
- Copy: source SOL history to `sol/archive/source-repo/`
- Copy: two authored notes to `references/research-notes/`
- Copy: seven PDFs to `references/papers/`
- Create: `references/README.md`
- Create: `references/papers/README.md`
- Create: `docs/specifications/README.md`

**Interfaces:**

- Consumes: Task 1 allowlist and source hashes
- Produces: the preserved historical/research surface used by navigation and verification

- [ ] **Step 1: Copy exact source governance originals into archival paths**

Run byte-preserving copies:

```bash
SOURCE_REPO='/Users/god/Documents/COLPALI binary classification'
DEST_REPO='/Users/god/Documents/Query-Relevant Document Token Pruning'
mkdir -p "$DEST_REPO/archive/source-repo-governance/agent-context/modules"
mkdir -p "$DEST_REPO/archive/source-repo-governance/sol"
mkdir -p "$DEST_REPO/archive/source-repo-governance/docs"
cp -p "$SOURCE_REPO/AGENTS.md" "$DEST_REPO/archive/source-repo-governance/AGENTS.md"
cp -p "$SOURCE_REPO/README.md" "$DEST_REPO/archive/source-repo-governance/README.md"
for governance_file in ARCHITECTURE.md COMMANDS.md CURRENT_TASK.md DATA_AND_ARTIFACTS.md GOTCHAS.md INDEX.md SOL.md; do
  cp -p "$SOURCE_REPO/agent-context/$governance_file" "$DEST_REPO/archive/source-repo-governance/agent-context/$governance_file"
done
rsync -a "$SOURCE_REPO/agent-context/modules/" "$DEST_REPO/archive/source-repo-governance/agent-context/modules/" \
  --include='*/' --include='*.md' --exclude='*'
for sol_file in AGENTS.md CURRENT_SOL_TASK.md README.md; do
  cp -p "$SOURCE_REPO/sol/$sol_file" "$DEST_REPO/archive/source-repo-governance/sol/$sol_file"
done
rsync -a "$SOURCE_REPO/docs/" "$DEST_REPO/archive/source-repo-governance/docs/" \
  --include='*/' --include='*.md' --include='*.txt' --exclude='*'
for scaffold_area in configs data environments experiments external src; do
  mkdir -p "$DEST_REPO/archive/source-repo-governance/$scaffold_area"
  rsync -a "$SOURCE_REPO/$scaffold_area/" "$DEST_REPO/archive/source-repo-governance/$scaffold_area/" \
    --include='*/' --include='README.md' --include='*.yml' --include='*.yaml' --exclude='*'
done
```

Expected: every active source authority file has an exact archived counterpart.

- [ ] **Step 2: Copy documentation-only research and archive trees**

Run:

```bash
SOURCE_REPO='/Users/god/Documents/COLPALI binary classification'
DEST_REPO='/Users/god/Documents/Query-Relevant Document Token Pruning'
mkdir -p "$DEST_REPO/agent-context/research" "$DEST_REPO/docs/specifications" "$DEST_REPO/reports" "$DEST_REPO/archive"
rsync -a "$SOURCE_REPO/agent-context/research/" "$DEST_REPO/agent-context/research/" \
  --include='*/' --include='*.md' --include='*.txt' --exclude='*'
rsync -a "$SOURCE_REPO/docs/specifications/" "$DEST_REPO/docs/specifications/" \
  --include='*/' --include='*.md' --include='*.txt' --exclude='*'
rsync -a "$SOURCE_REPO/docs/superpowers/" "$DEST_REPO/docs/superpowers/" \
  --include='*/' --include='*.md' --include='*.txt' --exclude='*'
rsync -a "$SOURCE_REPO/reports/" "$DEST_REPO/reports/" \
  --include='*/' --include='*.md' --include='*.txt' --exclude='*'
rsync -a "$SOURCE_REPO/archive/" "$DEST_REPO/archive/" \
  --include='*/' --include='*.md' --include='*.txt' --include='*.yml' --include='*.yaml' --exclude='*'
```

Expected: authored text and lightweight legacy environment references survive; archived code, fixtures, images, and applications do not.

- [ ] **Step 3: Preserve every SOL instruction, handoff, and wrapper historically**

Run:

```bash
SOURCE_REPO='/Users/god/Documents/COLPALI binary classification'
DEST_REPO='/Users/god/Documents/Query-Relevant Document Token Pruning'
mkdir -p "$DEST_REPO/sol/archive/source-repo"
rsync -a "$SOURCE_REPO/sol/" "$DEST_REPO/sol/archive/source-repo/" \
  --include='*/' --include='*.md' --include='*.txt' --include='*.sbatch' --include='*.sh' --exclude='*'
```

Expected: 29 historical `.sbatch` files exist beneath `sol/archive/source-repo/archive/jobs/`, and source task specifications are present but not active.

- [ ] **Step 4: Copy the approved paper set and authored notes**

Run:

```bash
SOURCE_REPO='/Users/god/Documents/COLPALI binary classification'
DEST_REPO='/Users/god/Documents/Query-Relevant Document Token Pruning'
mkdir -p "$DEST_REPO/references/papers" "$DEST_REPO/references/research-notes"
cp -p "$SOURCE_REPO/tmp/pdfs/dude.pdf" "$DEST_REPO/references/papers/dude.pdf"
cp -p "$SOURCE_REPO/tmp/pdfs/mmlongbench_doc.pdf" "$DEST_REPO/references/papers/mmlongbench-doc.pdf"
cp -p "$SOURCE_REPO/tmp/pdfs/slidevqa.pdf" "$DEST_REPO/references/papers/slidevqa.pdf"
cp -p "$SOURCE_REPO/tmp/pdfs/tat_dqa.pdf" "$DEST_REPO/references/papers/tat-dqa.pdf"
cp -p "$SOURCE_REPO/tmp/pdfs/ufmg.tot.pdf" "$DEST_REPO/references/papers/ufmg-tot.pdf"
cp -p "$SOURCE_REPO/tmp/pdfs/vgent_2512_11099v1/vgent_2512.11099v1.pdf" "$DEST_REPO/references/papers/vgent-2512.11099v1.pdf"
cp -p "$SOURCE_REPO/work/paper_review/m3grounder.pdf" "$DEST_REPO/references/papers/m3grounder.pdf"
cp -p "$SOURCE_REPO/work/paper_review/boundingdocs.md" "$DEST_REPO/references/research-notes/boundingdocs.md"
cp -p "$SOURCE_REPO/work/paper_review/dude.md" "$DEST_REPO/references/research-notes/dude.md"
```

Expected: exactly seven PDFs and two Markdown notes exist in the promoted reference area.

- [ ] **Step 5: Author the reference and historical-status entry points**

Use `apply_patch` to add:

```markdown
# References

This area preserves document-QA papers, authored research notes, and the
source repository's research registry. Retention does not make any inherited
architecture or experiment binding for this project.
```

Create `references/papers/README.md` as a table with paper, source path,
destination filename, byte count, and SHA-256 from the copied files. Create
`docs/specifications/README.md` stating that inherited specifications are
nonbinding research history and that new work requires a new current task.

- [ ] **Step 6: Verify byte preservation**

Run:

```bash
SOURCE_REPO='/Users/god/Documents/COLPALI binary classification'
DEST_REPO='/Users/god/Documents/Query-Relevant Document Token Pruning'
cmp "$SOURCE_REPO/AGENTS.md" "$DEST_REPO/archive/source-repo-governance/AGENTS.md"
cmp "$SOURCE_REPO/sol/AGENTS.md" "$DEST_REPO/archive/source-repo-governance/sol/AGENTS.md"
test "$(find "$DEST_REPO/references/papers" -maxdepth 1 -type f -name '*.pdf' | wc -l | tr -d ' ')" = 7
find "$DEST_REPO/references/papers" -maxdepth 1 -type f -name '*.pdf' -exec shasum -a 256 {} + | LC_ALL=C sort
```

Expected: `cmp` exits 0 and exactly seven paper hashes are printed.

- [ ] **Step 7: Commit the preserved reference surface**

Run:

```bash
DEST_REPO='/Users/god/Documents/Query-Relevant Document Token Pruning'
git -C "$DEST_REPO" add agent-context/research archive docs/specifications docs/superpowers references reports sol/archive
git -C "$DEST_REPO" diff --cached --check
git -C "$DEST_REPO" commit -m 'docs: preserve source research and SOL references'
```

Expected: the commit contains only preserved or indexed reference material.

---

### Task 3: Create the deliberately minimal active repository surface

**Files:**

- Create: `README.md`
- Create: `AGENTS.md`
- Create: `agent-context/INDEX.md`
- Create: `agent-context/CURRENT_TASK.md`
- Create: `docs/NAVIGATION.md`
- Copy: `docs/SOL_INSTRUCTIONS.md`
- Create: `sol/AGENTS.md`
- Create: `sol/README.md`
- Create: `sol/CURRENT_SOL_TASK.md`
- Create: `.gitignore`
- Create: `.rgignore`
- Create: `.cursorignore`
- Create: `.cursorindexingignore`
- Create: `.gitattributes`
- Copy: `LICENSE`

**Interfaces:**

- Consumes: approved project statement and Task 2 historical/reference paths
- Produces: the only active authority and navigation surface

- [ ] **Step 1: Write the top-level project README**

Use `apply_patch` to create this core content, followed by links to navigation,
references, SOL guidance, SBATCH examples, and the source inventory:

```markdown
# Query-Relevant Document Token Pruning

This repository is a research skeleton for token compression in document QA.
The goal is to reduce the number of visual/document tokens sent to an
expensive multimodal answerer by pruning content that is irrelevant to the
specific query while preserving all information required to answer correctly.

The central target is:

\[
P(\text{token/region needed for answer}\mid \text{document}, q)
\]

The central research question is where query relevance should be estimated.
No model architecture, dataset, experiment, or cluster job is active yet.
```

- [ ] **Step 2: Write minimal active agent governance**

Use `apply_patch` to create `AGENTS.md` with only these project-specific facts:

```markdown
# AGENTS.md

## Scope

This repository concerns token compression for document QA.

## Context routing

Read `agent-context/INDEX.md` first. Treat inherited specifications, research,
and SOL material as nonbinding reference unless `agent-context/CURRENT_TASK.md`
explicitly activates work.

## Working rules

- Preserve immutable sources and keep generated or large artifacts outside Git.
- Make small, reviewable changes and surface material assumptions.
- Do not start an experiment, model run, or SOL job without a current task and
  explicit handoff.
- Keep navigation files current when adding durable research material.
- Never add secrets, model weights, datasets, caches, or generated outputs.
```

- [ ] **Step 3: Write minimal active task and context routing**

Use `apply_patch` to create:

```markdown
# Agent Context Index

This repository concerns token compression for document QA.

- `CURRENT_TASK.md`: current authority and next action.
- `../README.md`: project orientation.
- `../docs/NAVIGATION.md`: repository map.
- `../references/README.md`: nonbinding inherited research.
- `../sol/README.md`: SOL entry point.

Historical source-repository governance is preserved under
`../archive/source-repo-governance/` and is not active authority.
```

and:

```markdown
# Current Task

## Scope

This repository concerns token compression for document QA.

## State

The repository is a documentation-first skeleton. No implementation,
experiment, model execution, or SOL job is active.

## Next action

Create and approve a project-specific task before implementation or execution.
```

- [ ] **Step 4: Create neutral SOL entry points and retain general SOL rules**

Copy `docs/SOL_INSTRUCTIONS.md` byte-for-byte. Use `apply_patch` to create
neutral `sol/AGENTS.md`, `sol/README.md`, and `sol/CURRENT_SOL_TASK.md` that
state only the project scope, require a future explicit handoff, route to
`examples/sbatch/`, and mark `sol/archive/source-repo/` as historical.

The current task file must contain:

```markdown
# Current SOL Task

## State

There is no active SOL task for this token-compression document-QA repository.

## Next action

Stop. Do not submit a job until an approved handoff records the exact checkout,
commit, environment, inputs, scratch root, resources, command, outputs, and
recovery authority.
```

- [ ] **Step 5: Create artifact-safe ignore files**

Use `apply_patch` to create ignore files covering caches, environments,
worktrees, secrets, logs, outputs, datasets, checkpoints, weights, archives,
images, and PDFs, with the single PDF exception `!references/papers/*.pdf`.
Create `.gitattributes` with text normalization and `*.pdf binary`.

- [ ] **Step 6: Copy the license and write navigation**

Run:

```bash
SOURCE_REPO='/Users/god/Documents/COLPALI binary classification'
DEST_REPO='/Users/god/Documents/Query-Relevant Document Token Pruning'
cp -p "$SOURCE_REPO/LICENSE" "$DEST_REPO/LICENSE"
cp -p "$SOURCE_REPO/docs/SOL_INSTRUCTIONS.md" "$DEST_REPO/docs/SOL_INSTRUCTIONS.md"
```

Use `apply_patch` to create `docs/NAVIGATION.md` with sections for active
authority, inherited nonbinding research, paper references, SOL guidance,
SBATCH examples, and archived source governance.

- [ ] **Step 7: Verify the active knowledge boundary**

Run:

```bash
DEST_REPO='/Users/god/Documents/Query-Relevant Document Token Pruning'
rg -n 'R0|R1|M0|M1|HierDoc|Evidence-DINO|overlap-first|MMLongBench|BoundingDocs|DeepSeek' \
  "$DEST_REPO/AGENTS.md" \
  "$DEST_REPO/agent-context/INDEX.md" \
  "$DEST_REPO/agent-context/CURRENT_TASK.md" \
  "$DEST_REPO/sol/AGENTS.md" \
  "$DEST_REPO/sol/README.md" \
  "$DEST_REPO/sol/CURRENT_SOL_TASK.md" && exit 1 || true
rg -n 'token compression.*document QA|document QA.*token compression' \
  "$DEST_REPO/AGENTS.md" "$DEST_REPO/agent-context/INDEX.md" "$DEST_REPO/agent-context/CURRENT_TASK.md"
```

Expected: no inherited project terms appear in active governance, and the generic project scope is present.

- [ ] **Step 8: Commit active governance and navigation**

Run:

```bash
DEST_REPO='/Users/god/Documents/Query-Relevant Document Token Pruning'
git -C "$DEST_REPO" add .cursorignore .cursorindexingignore .gitattributes .gitignore .rgignore AGENTS.md LICENSE README.md agent-context/INDEX.md agent-context/CURRENT_TASK.md docs/NAVIGATION.md docs/SOL_INSTRUCTIONS.md sol/AGENTS.md sol/README.md sol/CURRENT_SOL_TASK.md
git -C "$DEST_REPO" diff --cached --check
git -C "$DEST_REPO" commit -m 'docs: establish minimal token compression governance'
```

Expected: active governance is committed separately from inherited references.

---

### Task 4: Add lightweight reusable scaffolding and discoverable SBATCH examples

**Files:**

- Create: `configs/README.md`
- Create: `data/README.md`
- Create: `environments/README.md`
- Create: `experiments/README.md`
- Create: `external/README.md`
- Create: `src/README.md`
- Create: `src/candidates/README.md`
- Create: `src/data/README.md`
- Create: `src/evaluation/README.md`
- Create: `src/models/README.md`
- Create: `src/training/README.md`
- Copy: `examples/environments/legacy/deepseek-ocr1-sol.yml`
- Copy: `examples/environments/legacy/deepseek-ocr2-sol.yml`
- Create: `examples/environments/legacy/README.md`
- Create: `examples/sbatch/README.md`
- Create: `examples/sbatch/00_lightwork_setup.sbatch`
- Create: `examples/sbatch/01_gpu_smoke.sbatch`
- Create: `examples/sbatch/02_single_gpu_run.sbatch`
- Create: `examples/sbatch/03_job_array.sbatch`

**Interfaces:**

- Consumes: general SOL rules and patterns from the retained source wrappers
- Produces: empty code/config surfaces plus four runnable, project-neutral job templates

- [ ] **Step 1: Write generic scaffolding READMEs**

Use `apply_patch` to make each directory's responsibility explicit without
naming an inherited model, dataset, metric, or experiment. State that large
data and generated artifacts stay outside Git, configurations must pin source
and revision information, and no implementation is active.

- [ ] **Step 2: Preserve legacy environment examples as reference-only**

Run:

```bash
SOURCE_REPO='/Users/god/Documents/COLPALI binary classification'
DEST_REPO='/Users/god/Documents/Query-Relevant Document Token Pruning'
mkdir -p "$DEST_REPO/examples/environments/legacy"
cp -p "$SOURCE_REPO/environments/deepseek-ocr1-sol.yml" "$DEST_REPO/examples/environments/legacy/deepseek-ocr1-sol.yml"
cp -p "$SOURCE_REPO/environments/deepseek-ocr2-sol.yml" "$DEST_REPO/examples/environments/legacy/deepseek-ocr2-sol.yml"
```

Use `apply_patch` to state in the directory README that these are inherited
compatibility examples, not active environment locks.

- [ ] **Step 3: Author the lightwork and GPU-smoke examples**

Use `apply_patch`. The lightwork script uses:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=doc-token-setup
#SBATCH --partition=lightwork
#SBATCH --qos=public
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$HOME/Query-Relevant-Document-Token-Pruning}"
RUN_ROOT="${RUN_ROOT:-/scratch/$USER/query_relevant_document_token_pruning/setup}"
mkdir -p "$RUN_ROOT"
cd "$PROJECT_DIR"
module load mamba/latest
mamba info --envs
python -V
git rev-parse HEAD
```

The GPU smoke script uses `htc`, one generic GPU, two hours, `nvidia-smi`, an
explicit `PROJECT_DIR`, and a scratch `RUN_ROOT`; it performs no model download
or training.

- [ ] **Step 4: Author the production GPU and job-array examples**

Use `apply_patch`. The production example uses `public`, one A100 80 GB GPU,
eight hours, explicit project/environment/run roots, cache variables under
scratch, `LD_LIBRARY_PATH`, Git revision logging, `nvidia-smi`, and:

```bash
RUN_COMMAND="${RUN_COMMAND:-python -V}"
bash -lc "$RUN_COMMAND"
```

The array example uses `#SBATCH --array=0-3%2`, maps the array index to a
four-item shard list, writes under `/scratch/$USER`, records the Git revision,
and runs `RUN_COMMAND` with the shard name exported.

- [ ] **Step 5: Write the SBATCH usage README**

Document that users must inspect `sinfo`, choose resources appropriate to the
workload, create or verify environments from a compute allocation, keep large
outputs on scratch, and replace the default no-op command only after a binding
task/handoff exists. Cite the historical source wrappers from which each
pattern was distilled.

- [ ] **Step 6: Validate all retained and new SBATCH scripts**

Run:

```bash
DEST_REPO='/Users/god/Documents/Query-Relevant Document Token Pruning'
find "$DEST_REPO" -type f -name '*.sbatch' -print0 | while IFS= read -r -d '' sbatch_file; do
  bash -n "$sbatch_file"
done
test "$(find "$DEST_REPO/examples/sbatch" -maxdepth 1 -type f -name '*.sbatch' | wc -l | tr -d ' ')" = 4
```

Expected: exit 0 and exactly four discoverable examples.

- [ ] **Step 7: Commit reusable scaffolding and examples**

Run:

```bash
DEST_REPO='/Users/god/Documents/Query-Relevant Document Token Pruning'
git -C "$DEST_REPO" add configs data environments examples experiments external src
git -C "$DEST_REPO" diff --cached --check
git -C "$DEST_REPO" commit -m 'docs: add neutral scaffolding and SOL examples'
```

Expected: the commit contains only generic scaffolding and reusable examples.

---

### Task 5: Repair navigation and document intentional omissions

**Files:**

- Modify: `README.md`
- Modify: `docs/NAVIGATION.md`
- Modify: `agent-context/INDEX.md`
- Modify: `docs/SOURCE_INVENTORY.md`
- Create: `archive/source-repo-governance/SKELETON_STATUS.md`
- Create: `sol/archive/source-repo/README_SKELETON_STATUS.md`

**Interfaces:**

- Consumes: all retained and adapted paths from Tasks 2–4
- Produces: discoverable navigation with explicit nonbinding/historical boundaries

- [ ] **Step 1: Add archive status notices**

Use `apply_patch` to state that original source governance and SOL material are
preserved for provenance, retain their original wording, and are never current
authority in this repository.

- [ ] **Step 2: Audit Markdown links**

Run a read-only relative-link audit over Markdown files. Ignore web URLs,
anchors, mail links, and absolute source/SOL paths. Produce a list of missing
relative targets grouped into:

1. retained target that needs a corrected link;
2. intentionally omitted implementation/artifact target; or
3. historical target meaningful only in the source repository.

Use `rg` for the first pass and a short read-only checker only if parsing nested
relative paths requires it.

- [ ] **Step 3: Repair active navigation links and annotate historical gaps**

Use `apply_patch` for active files. Correct links when the retained target
exists. In inherited documents, replace an unavailable relative link only when
the intended retained target is unambiguous; otherwise preserve the path as
plain text and record the omission in `docs/SOURCE_INVENTORY.md`.

- [ ] **Step 4: Finalize the inventory counts**

Record:

- retained and omitted source-area classifications;
- active files and their exact archived originals;
- seven paper source/destination checksums;
- historical and example SBATCH counts;
- final repository file count and size; and
- the maximum retained file size.

- [ ] **Step 5: Verify active navigation**

Check every relative link in `README.md`, `docs/NAVIGATION.md`,
`agent-context/INDEX.md`, `references/README.md`, `references/papers/README.md`,
`sol/README.md`, and `examples/sbatch/README.md`. Expected: zero missing targets.

- [ ] **Step 6: Commit navigation and final inventory documentation**

Run:

```bash
DEST_REPO='/Users/god/Documents/Query-Relevant Document Token Pruning'
git -C "$DEST_REPO" add README.md agent-context/INDEX.md archive/source-repo-governance/SKELETON_STATUS.md docs/NAVIGATION.md docs/SOURCE_INVENTORY.md sol/archive/source-repo/README_SKELETON_STATUS.md
git -C "$DEST_REPO" diff --cached --check
git -C "$DEST_REPO" commit -m 'docs: finalize skeleton navigation and inventory'
```

Expected: active navigation and measured inventory are committed together.

---

### Task 6: Prove source immutability and skeleton cleanliness

**Files:**

- Modify: `docs/SOURCE_INVENTORY.md` only if measured final counts differ from its draft
- Commit: all retained and authored skeleton files

**Interfaces:**

- Consumes: Task 1 baseline, completed destination, and design verification contract
- Produces: a clean committed skeleton plus fresh completion evidence

- [ ] **Step 1: Recompute the source retained-surface checksums**

Run:

```bash
SOURCE_REPO='/Users/god/Documents/COLPALI binary classification'
(
  cd "$SOURCE_REPO"
  while IFS= read -r retained_file; do
    shasum -a 256 "$retained_file"
  done < "$OLDPWD/work/source-retained-files.txt"
) > work/source-retained-sha256.after.txt
cmp work/source-retained-sha256.txt work/source-retained-sha256.after.txt
git -C "$SOURCE_REPO" status --short --branch
```

Expected: `cmp` exits 0 and source Git state matches the recorded pre-copy state.

- [ ] **Step 2: Verify required governance and preservation paths**

Run:

```bash
DEST_REPO='/Users/god/Documents/Query-Relevant Document Token Pruning'
for required_file in \
  AGENTS.md README.md agent-context/INDEX.md agent-context/CURRENT_TASK.md \
  docs/NAVIGATION.md docs/SOL_INSTRUCTIONS.md docs/SOURCE_INVENTORY.md \
  sol/AGENTS.md sol/README.md sol/CURRENT_SOL_TASK.md \
  archive/source-repo-governance/AGENTS.md \
  archive/source-repo-governance/sol/AGENTS.md \
  references/README.md references/papers/README.md \
  examples/sbatch/README.md; do
  test -f "$DEST_REPO/$required_file"
done
```

Expected: exit 0.

- [ ] **Step 3: Verify artifact and implementation exclusions**

Run:

```bash
DEST_REPO='/Users/god/Documents/Query-Relevant Document Token Pruning'
for forbidden_dir in .pytest_cache .superpowers .worktrees checkpoints logs outputs pilot_data scripts sol_results tests tmp viewer work mmlongbench_doc mmdocir_colqwen_pilot mmdocir_colqwen_pilot_mpdocvqa mmdocir_colqwen_diverse_longdoc_pilot; do
  test ! -e "$DEST_REPO/$forbidden_dir"
done
test -z "$(find "$DEST_REPO" -path "$DEST_REPO/.git" -prune -o -type f \
  \( -name '*.py' -o -name '*.js' -o -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' -o -name '*.safetensors' -o -name '*.parquet' -o -name '*.zip' -o -name '*.tar' -o -name '*.tar.gz' \) -print)"
test "$(find "$DEST_REPO" -path "$DEST_REPO/.git" -prune -o -type f -name '*.pdf' -print | wc -l | tr -d ' ')" = 7
```

Expected: no forbidden directory/file is found and the only PDFs are the seven approved papers.

- [ ] **Step 4: Verify size, syntax, links, and active boundary**

Run the complete SBATCH syntax check from Task 4, the active-link check from
Task 5, and the active-knowledge scan from Task 3. Then run:

```bash
DEST_REPO='/Users/god/Documents/Query-Relevant Document Token Pruning'
find "$DEST_REPO" -path "$DEST_REPO/.git" -prune -o -type f -size +50M -print
du -sh "$DEST_REPO"
find "$DEST_REPO" -path "$DEST_REPO/.git" -prune -o -type f -print | wc -l
git -C "$DEST_REPO" diff --check
git -C "$DEST_REPO" status --short
```

Expected: no file over 50 MiB, a small repository relative to the 44 GB source,
no whitespace errors, and only the intended uncommitted skeleton files.

- [ ] **Step 5: Commit any measured final-inventory correction**

Run:

```bash
DEST_REPO='/Users/god/Documents/Query-Relevant Document Token Pruning'
git -C "$DEST_REPO" branch --show-current
git -C "$DEST_REPO" diff --check
if ! git -C "$DEST_REPO" diff --quiet -- docs/SOURCE_INVENTORY.md; then
  git -C "$DEST_REPO" add docs/SOURCE_INVENTORY.md
  git -C "$DEST_REPO" diff --cached --check
  git -C "$DEST_REPO" commit -m 'docs: record final skeleton measurements'
fi
```

Expected: `main` contains no unrelated staged or uncommitted files.

- [ ] **Step 6: Run fresh post-commit verification**

Repeat Steps 1–4 after the commit and run:

```bash
DEST_REPO='/Users/god/Documents/Query-Relevant Document Token Pruning'
git -C "$DEST_REPO" status --short --branch
git -C "$DEST_REPO" log -8 --oneline
```

Expected: clean `main`, a concise local commit series, unchanged source
fingerprints, and all completion checks passing.
