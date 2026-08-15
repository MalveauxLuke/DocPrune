# Document-Understanding Repository Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate the document-understanding repository onto one MMLongBench-preserving branch, archive inactive experiments and jobs, organize active scripts by responsibility, and add human- and agent-facing navigation.

**Architecture:** Use `codex/mmlongbench-academic-ocr2-gold` as the ancestry base because it already contains the entire pushed 313-page OCR/deep-parse line plus the academic-gold corpus. Preserve the two later dirty-worktree commits by cherry-picking them into `codex/repo-consolidation`. Keep MMLongBench source data, results, viewers, and reproducibility contracts in the canonical line; move completed non-MMLLongBench experiments and inactive Slurm jobs under `archive/` without deleting their Git history.

**Tech Stack:** Git worktrees, Python 3.12, pytest, static HTML/CSS/JavaScript viewers, Slurm `sbatch`, Markdown documentation.

## Global Constraints

- Do not modify or merge the separate `main-project` planner line.
- Preserve every MMLongBench document, manifest, OCR result, viewer, prompt arm, test, and SOL contract.
- Preserve user-owned dirty work before removing any worktree or branch.
- Push only named feature branches; do not push or rewrite `main` without separate approval.
- Prefer `git mv` so history remains traceable.
- Move inactive material to `archive/`; delete only generated caches and redundant `.bak-*` snapshots already preserved by Git.
- Keep active `agent-context` files short and route full specifications to `docs/specifications/`.
- Run focused tests after each move and the full suite before publishing the consolidation branch.

---

### Task 1: Preserve and reconcile the active branch lines

**Files:**
- Modify through cherry-pick: `agent-context/**`
- Modify through cherry-pick: `scripts/sciegqa_parser_compare/final_labeling.py`
- Modify through cherry-pick: `viewer/mmlongbench_ocr2/**`
- Modify through cherry-pick: `tests/test_sciegqa_final_labeling.py`
- Modify through cherry-pick: `tests/test_mmlongbench_ocr2_viewer_static.py`
- Modify: `tests/test_mmlongbench_ocr2_sol.py`

**Interfaces:**
- Consumes: commits `72ddd01` and `eb15c7c` from the protected source branches.
- Produces: one consolidation branch containing the academic corpus, 313-page OCR results, deep-parse viewer, heading/bullet fixes, and current evidence-routing specifications.

- [ ] **Step 1: Cherry-pick the current design-contract commit**

```bash
git cherry-pick 72ddd01
```

- [ ] **Step 2: Cherry-pick the OCR viewer and segmentation-fix commit**

```bash
git cherry-pick eb15c7c
```

- [ ] **Step 3: Replace the stale SOL test expectation**

Update `tests/test_mmlongbench_ocr2_sol.py` so the active-task assertion checks for `mmlongbench_academic_gold_deepseek_ocr2.md` and the archived 313-page contract remains covered separately.

- [ ] **Step 4: Run the reconciliation tests**

```bash
python -m pytest -q \
  tests/test_mmlongbench_ocr2_sol.py \
  tests/test_mmlongbench_academic_gold.py \
  tests/test_sciegqa_final_labeling.py \
  tests/test_mmlongbench_ocr2_viewer_static.py
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit the reconciled line**

```bash
git add agent-context scripts tests viewer docs sol
git commit -m "chore: consolidate MMLongBench development lines"
```

### Task 2: Create human-facing repository navigation

**Files:**
- Modify: `README.md`
- Create: `docs/NAVIGATION.md`
- Create: `docs/ACTIVE_PROJECTS.md`
- Create: `archive/README.md`
- Modify: `agent-context/ARCHITECTURE.md`
- Modify: `agent-context/DATA_AND_ARTIFACTS.md`

**Interfaces:**
- Consumes: the reconciled source/data/result inventory.
- Produces: one clear entry point for humans and one source-of-truth directory map.

- [ ] **Step 1: Write the root README**

Document the research objective, active MMLongBench corpora, result viewers, local quick start, SOL boundary, test command, and navigation links. Explicitly distinguish source, frozen data, returned OCR results, generated sites, and archives.

- [ ] **Step 2: Add the repository map**

`docs/NAVIGATION.md` must list each active directory, owner/purpose, whether it is source or artifact, and its primary entry points.

- [ ] **Step 3: Add active-project status**

`docs/ACTIVE_PROJECTS.md` must identify MMLongBench academic OCR as the active SOL task, the 313-page and deep-parse runs as completed retained experiments, and hierarchical evidence routing as design-stage work.

- [ ] **Step 4: Add the archive policy**

`archive/README.md` must state that archived code/jobs are preserved for reproducibility, are excluded from default agent context, and are not maintained by the active test suite.

- [ ] **Step 5: Validate links and commit**

```bash
git diff --check
rg -n "MMLongBench|Navigation|archive" README.md docs/NAVIGATION.md docs/ACTIVE_PROJECTS.md archive/README.md
git commit -am "docs: add repository navigation"
```

### Task 3: Organize active MMLongBench and parsing scripts

**Files:**
- Create: `scripts/mmlongbench/__init__.py`
- Move: `scripts/mmlongbench_*.py` to `scripts/mmlongbench/`
- Move: `scripts/prepare_mmlongbench_*.py` to `scripts/mmlongbench/`
- Move: `scripts/package_mmlongbench_*.py` to `scripts/mmlongbench/`
- Move: `scripts/build_mmlongbench_*.py` to `scripts/mmlongbench/`
- Move: `scripts/mmlongbench_ocr2_viewer/` to `scripts/mmlongbench/viewer/`
- Move: `scripts/mmlongbench_ocr2_deep_parse_viewer/` to `scripts/mmlongbench/deep_parse_viewer/`
- Create: `scripts/document_parsing/__init__.py`
- Move: `scripts/sciegqa_parser_compare/deepseek_runner.py` to `scripts/document_parsing/deepseek_runner.py`
- Move: `scripts/sciegqa_parser_compare/grounding.py` to `scripts/document_parsing/grounding.py`
- Move: `scripts/sciegqa_parser_compare/final_labeling.py` to `scripts/document_parsing/semantic_sections.py`
- Move: `scripts/sciegqa_mineru/schema.py` to `scripts/document_parsing/schema.py`
- Move: `scripts/run_sciegqa_deepseek_ocr.py` to `scripts/document_parsing/run_deepseek_ocr.py`
- Modify: active MMLongBench tests, viewers, SOL wrappers, specifications, and agent context containing old paths.

**Interfaces:**
- Consumes: existing public functions and CLI arguments unchanged.
- Produces: responsibility-based import paths while preserving model prompts, hashes, corpus IDs, and output schemas.

- [ ] **Step 1: Move files with `git mv`**

Create the two package directories and move active files without changing behavior.

- [ ] **Step 2: Rewrite Python imports**

Replace `scripts.sciegqa_parser_compare.*`, `scripts.sciegqa_mineru.schema`, and top-level MMLongBench imports with the new package paths.

- [ ] **Step 3: Rewrite command and documentation paths**

Update active `sol/*.sbatch`, MMLongBench task contracts, README/navigation files, tests, and viewer build instructions.

- [ ] **Step 4: Prove no active old path remains**

```bash
rg -n "scripts/(mmlongbench_|prepare_mmlongbench_|package_mmlongbench_|build_mmlongbench_|run_sciegqa_deepseek_ocr)|scripts\.sciegqa_parser_compare|scripts\.sciegqa_mineru\.schema" \
  README.md agent-context docs scripts sol tests viewer
```

Expected: matches exist only inside archived historical documents.

- [ ] **Step 5: Run MMLongBench and parser tests**

```bash
python -m pytest -q \
  tests/test_mmlongbench_academic_gold.py \
  tests/test_mmlongbench_ocr2_pilot.py \
  tests/test_mmlongbench_ocr2_deep_parse_pilot.py \
  tests/test_mmlongbench_ocr2_viewer_bundle.py \
  tests/test_mmlongbench_ocr2_deep_parse_viewer_bundle.py \
  tests/test_deepseek_grounding.py \
  tests/test_sciegqa_final_labeling.py
```

- [ ] **Step 6: Commit the active script structure**

```bash
git commit -am "refactor: organize MMLongBench and parsing scripts"
```

### Task 4: Archive inactive scripts, tests, viewers, and Slurm jobs

**Files:**
- Create: `archive/code/colqwen/`
- Create: `archive/code/sciegqa/`
- Create: `archive/code/segment_evidence/`
- Create: `archive/tests/`
- Create: `archive/apps/`
- Create: `sol/archive/jobs/{colqwen,sciegqa,segment_evidence,vidore,mmlongbench_completed}/`
- Move: non-MMLLongBench source not required by the active transitive dependency set.
- Move: corresponding inactive tests and static viewers.
- Move: completed or superseded `sol/*.sbatch` wrappers.
- Delete: tracked `sol/**/*.bak-*` files after their canonical/archive copies are verified.

**Interfaces:**
- Consumes: dependency inventory and current SOL contract.
- Produces: an active tree containing only maintained MMLongBench/parsing code and active SOL entry points.

- [ ] **Step 1: Archive legacy ColQwen, verifier, Vidore, and MMDocIR scripts**

Move their source and tests by domain; retain README pointers to their last reports and environment files.

- [ ] **Step 2: Archive completed SciEGQA/MinerU pipelines**

Do not move the four shared parsing modules already transferred into `scripts/document_parsing/`.

- [ ] **Step 3: Archive the previous segment-evidence runner**

Move `run_segment_evidence_stage.py`, `segment_evidence_stage0.py`, their tests, environment, and Slurm wrappers together.

- [ ] **Step 4: Archive inactive Slurm wrappers**

Keep only the DeepSeek environment setup and the active academic-MMLongBench OCR submission path in `sol/`; place completed 313-page/deep-parse wrappers under `sol/archive/jobs/mmlongbench_completed/`.

- [ ] **Step 5: Remove redundant backup files**

Verify each `.bak-*` file has a canonical or archived copy, then remove the tracked backup filename.

- [ ] **Step 6: Validate active imports and tests**

```bash
python -m pytest -q
git diff --check
```

- [ ] **Step 7: Commit the archive move**

```bash
git commit -am "chore: archive inactive experiments and jobs"
```

### Task 5: Slim agent context and separate specifications

**Files:**
- Move: `agent-context/modules/segment_reranker_corpus.md` to `docs/specifications/segment_reranker_corpus.md`
- Move: `agent-context/hierarchical_evidence_routing/design.md` to `docs/specifications/hierarchical_evidence_routing.md`
- Move: `agent-context/modules/deepseek_semantic_segmentation.md` to `docs/specifications/deepseek_semantic_segmentation.md`
- Create: short replacements under `agent-context/modules/`
- Modify: `agent-context/INDEX.md`
- Reset: `agent-context/CURRENT_TASK.md`

**Interfaces:**
- Consumes: full preserved specifications.
- Produces: compact agent routing summaries linked to durable human-readable specifications.

- [ ] **Step 1: Move full contracts into documentation**

Use `git mv`; do not truncate or discard specification content.

- [ ] **Step 2: Write compact context summaries**

Each summary must contain scope, authoritative implementation files, focused tests, invariants, and one link to the complete specification.

- [ ] **Step 3: Update the index and exclusions**

Ensure default agent search skips `archive/`, generated results payloads, and completed plans while active source and short context remain visible.

- [ ] **Step 4: Commit context cleanup**

```bash
git diff --check
git commit -am "docs: slim agent context and preserve full specifications"
```

### Task 6: Verify, publish, and prune branches

**Files:**
- Modify only if verification reveals an in-scope path/reference defect.

**Interfaces:**
- Consumes: the complete consolidation branch.
- Produces: a pushed `codex/repo-consolidation` branch and a verified branch-retirement list.

- [ ] **Step 1: Run the complete verification suite**

```bash
python -m pytest -q
git diff --check
git status --short
```

Expected: tests pass; only explicitly ignored local artifact links may remain untracked.

- [ ] **Step 2: Push the consolidation branch**

```bash
git push -u origin codex/repo-consolidation
```

- [ ] **Step 3: Verify branch containment**

Confirm `COLQWEN_binary_classification`, `codex/mmlongbench-ocr2-313`, and `codex/mmlongbench-academic-ocr2-gold` have no unrepresented commits or dirty files.

- [ ] **Step 4: Remove inactive worktrees and local branches**

Remove only clean, represented worktrees. Leave `main-project` and its worktree untouched.

- [ ] **Step 5: Delete superseded remote branches**

After the consolidation push and containment proof, delete the three superseded document-understanding remote branches. Do not delete or update `main`.

## Self-review

- The plan retains all MMLongBench data/results and both completed experiment histories.
- The plan excludes the separate planner project.
- Active script moves preserve CLI arguments and artifact schemas.
- Archive moves are domain-scoped and reversible through Git history.
- Branch deletion is gated on dirty-worktree and commit-containment checks.
- No task pushes `main`.
