# Evidence-DINO-Units SOL Handoff Implementation Plan

> **Archived:** The handoff and repository contracts described here were
> completed. Current authority lives in `agent-context/CURRENT_TASK.md` and
> `sol/task_spec/evidence_dino_units_stages_0_4.md`. Unchecked boxes below are
> preserved as historical plan syntax, not unfinished work.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the repository's active handoff with a complete, auditable
Evidence-DINO-Units Stage 0–4 project and prepare a separate focused SOL
workspace on a dedicated experiment branch.

**Architecture:** Shared project status, archive routing, canonical
specifications, and the SOL bootstrap handoff are committed to `main`. The
experiment branch contains those shared changes plus a focused project surface;
SOL checks that branch out as the sibling worktree `~/Evidence-DINO-Units`.
Large source and derived artifacts live under
`/scratch/$USER/evidence_dino_units`, while code, manifests, reports, and
provenance remain in the Git worktree.

**Tech Stack:** Git worktrees and branches, Markdown contracts, Bash/Slurm
instructions, Python/pytest repository-contract checks, Hugging Face Hub CLI,
Mamba, SHA-256 manifests, JSONL/Parquet dataset artifacts.

## Global Constraints

- The active SOL task implements and executes Stages 0–4 only.
- Stage 5 and every later stage are explicitly out of scope.
- Visual-CoT DUDE means only `metadata/dude_cot_train.jsonl` and bundled
  `cot/dude` images.
- Keep all original full-plan passages that receive SOL-specific replacements;
  mark each original passage inactive for this task and place its binding
  replacement immediately beside it.
- The canonical full plan and the full SOL copy must be byte-identical.
- `sol/CURRENT_SOL_TASK.md` remains a concise state router; detailed execution
  belongs in `sol/task_spec/evidence_dino_units_stages_0_4.md`.
- `main` may receive and publish shared handoff/context changes by explicit
  owner authorization.
- Experiment implementation and sparse-worktree focusing remain on
  `codex/evidence-dino-units-dataset-stages-0-4`.
- Do not modify or add the unrelated untracked `work/` directory.
- No large data, images, archives, PDFs, models, environments, caches, or
  Slurm logs enter Git.
- Login nodes are limited to Git, inspection, editing, and submission; setup,
  downloading, extraction, hashing, indexing, and dataset processing use
  allocations.

---

### Task 1: Lock the approved design and repository contracts

**Files:**
- Modify: `docs/specifications/DETR_GroundingDINO_Datasetplan/evidence_dino_units_sol_project_design.md`
- Create: `tests/evidence_dino/test_handoff.py`

**Interfaces:**
- Consumes: the approved project design and existing repository archive rules.
- Produces: failing repository-contract checks that define every required
  handoff path, archive move, plan-copy invariant, and Stage 5 boundary.

- [ ] **Step 1: Finish the approved design amendment**

Record the owner's authorization to push shared documentation to `main`,
preserve experiment-only work on the dedicated branch, and require both the
original and replacement passages in every full-plan copy.

- [ ] **Step 2: Write the failing repository-contract tests**

Add tests that:

```python
def test_full_plan_copy_is_byte_identical() -> None: ...
def test_changed_full_plan_passages_show_original_and_binding_update() -> None: ...
def test_previous_agent_and_sol_tasks_are_archived() -> None: ...
def test_current_sol_task_binds_only_stages_zero_through_four() -> None: ...
def test_active_context_routes_to_evidence_dino() -> None: ...
def test_sol_workspace_bootstrap_contract_is_complete() -> None: ...
```

The copy test reads bytes directly. The plan-annotation test checks the named
replacement blocks and verifies a Stage 5 inactive marker. The archive test
checks paths, not prose history. The current-task test rejects authorization
to execute `T0-BOX`, DeepSeek-OCR-2, model training, or MMLongBench acquisition.

- [ ] **Step 3: Run the new tests and verify RED**

Run:

```bash
python -m pytest -q tests/evidence_dino/test_handoff.py --tb=short
```

Expected: failures for the missing SOL plan copy, missing binding contract,
unmoved old tasks, and unchanged active routing.

- [ ] **Step 4: Commit the approved design amendment and RED contract**

Run:

```bash
git branch --show-current
git add \
  docs/specifications/DETR_GroundingDINO_Datasetplan/evidence_dino_units_sol_project_design.md \
  tests/evidence_dino/test_handoff.py
git commit -m "test: define Evidence DINO handoff contract"
```

---

### Task 2: Version every changed passage in the authoritative plan

**Files:**
- Modify: `docs/specifications/DETR_GroundingDINO_Datasetplan/evidence_dino_units_training_plan_final.md`
- Create: `sol/task_spec/evidence_dino_units_training_plan_final.md`

**Interfaces:**
- Consumes: replacement requirements from the approved design.
- Produces: two byte-identical full plans containing visible original wording
  and binding Stage 0–4 SOL updates.

- [ ] **Step 1: Add the plan-wide execution overlay**

Immediately after the title/status, add a Stage 0–4 SOL overlay that lists the
changed locations:

```text
Stage 0 owner signoff and research-only status
Stage 1 workspace/storage split
Stage 1 acquisition environment
Stages 2–3 visual-review ordering
Stage 4 future candidate fields
Stage 4 MMLongBench quarantine
Stage 5+ inactive boundary
```

State that unmarked sections remain authoritative.

- [ ] **Step 2: Preserve and supersede the Stage 0 owner-signoff passage**

Keep the original Stage 0 deliverable and advancement gate. Add the binding
update recording owner approval on `2026-07-28`, research-only use, and the
continued prohibition on redistribution or commercial use without license
resolution.

- [ ] **Step 3: Preserve and supersede the Stage 1 layout/environment passages**

Keep the original `project/` layout and `.venv-acquire` commands. Mark only
their local execution form inactive for this SOL task. Add:

```text
home worktree: ~/Evidence-DINO-Units
scratch root: /scratch/$USER/evidence_dino_units
acquisition env: /home/$USER/mamba-envs/evidence-dino-acquire
```

Require environment creation through a compute allocation and prohibit
committing machine-specific absolute symlinks.

- [ ] **Step 4: Preserve and supersede the Stage 2 visual-review timing**

Keep Section 2.5 verbatim. Mark its immediate execution in Stage 2 inactive.
Add a binding replacement that freezes 100 deterministic records per source
in Stage 2 and performs all 600 visual reviews only after Stage 3 joins images.
Annotate the Stage 2 deliverable/gate accordingly without weakening either.

- [ ] **Step 5: Add the Stage 3 closure for the deferred review**

Add a binding Stage 3 subsection requiring:

- all 600 frozen overlays;
- source-stratified review dispositions;
- the existing coordinate correctness gate; and
- closure of the deferred Stage 2 visual deliverable.

- [ ] **Step 6: Preserve and supersede Stage 4 future-field and benchmark behavior**

Keep the canonical record schema. Add a binding statement that
`candidate_generator`, candidates, positive/contextual/ignored/hard-negative
IDs, mapping confidence, and local loss values remain null or empty until
Stages 6–7.

Keep Section 4.7. Add a binding statement that only the immutable
MMLongBench revision and sealed path reservation are recorded during Stages
0–4; the benchmark is not downloaded or inspected.

- [ ] **Step 7: Mark Stage 5+ inactive for this SOL task**

Before the Stage 5 heading, insert a clear boundary that the original Stage
5–19 content remains the future authoritative plan but is inactive and must
not be executed by the current SOL task.

- [ ] **Step 8: Create the byte-identical SOL full-plan copy**

Mechanically copy the revised canonical plan to:

```text
sol/task_spec/evidence_dino_units_training_plan_final.md
```

Verify:

```bash
cmp \
  docs/specifications/DETR_GroundingDINO_Datasetplan/evidence_dino_units_training_plan_final.md \
  sol/task_spec/evidence_dino_units_training_plan_final.md
sha256sum \
  docs/specifications/DETR_GroundingDINO_Datasetplan/evidence_dino_units_training_plan_final.md \
  sol/task_spec/evidence_dino_units_training_plan_final.md
```

Expected: `cmp` exits zero and both SHA-256 values match.

---

### Task 3: Write the binding Stage 0–4 SOL execution contract

**Files:**
- Create: `sol/task_spec/evidence_dino_units_stages_0_4.md`
- Create: `sol/task_spec/evidence_dino_units_workspace_paths.txt`

**Interfaces:**
- Consumes: the full plan, SOL policy, approved branch/worktree design, and
  executable ordering corrections.
- Produces: the sole binding implementation contract used by the SOL agent.

- [ ] **Step 1: Define authority, exact identity, and startup verification**

The contract must name:

```text
source checkout: ~/COLPALI_binary_classification
workspace: ~/Evidence-DINO-Units
branch: codex/evidence-dino-units-dataset-stages-0-4
artifact root: /scratch/$USER/evidence_dino_units
full plan: sol/task_spec/evidence_dino_units_training_plan_final.md
binding contract: sol/task_spec/evidence_dino_units_stages_0_4.md
```

Give exact `git fetch`, branch, worktree, remote-containment, status, and commit
recording commands. Fail closed on a dirty target, wrong branch, detached HEAD,
unfetched branch, or mismatch between the pushed and checked-out commit.

- [ ] **Step 2: Define the required reading and authority order**

Require this order:

1. root `AGENTS.md`;
2. `docs/SOL_INSTRUCTIONS.md`;
3. `sol/AGENTS.md`;
4. `sol/CURRENT_SOL_TASK.md`;
5. the binding Stage 0–4 contract;
6. the full plan through Stage 4;
7. current agent-context files.

State that the binding contract controls only where it explicitly supersedes
the preserved full-plan passages.

- [ ] **Step 3: Define allocation and storage rules**

Provide exact lightwork/public allocation patterns, free-space/quota checks,
environment location, cache paths, logs, download roots, extracted image
roots, and Git-safe return paths.

- [ ] **Step 4: Define Stage 0 outputs and gate**

Include the ten invariants verbatim, the owner signoff, the distinction among
phrase grounding, answer-bearing localization, and complete evidence, and the
required `docs/experiment_contract.md`.

- [ ] **Step 5: Define Stage 1 outputs and gate**

Specify the complete directory tree, immutable revisions, clone/download
boundaries, `source_registry.yaml`, GitHub/Hub locks, acquisition environment
lock, license files, license discrepancy status, and rerun lineage policy.

- [ ] **Step 6: Define Stage 2 commands, schemas, outputs, and gate**

Include the six exact metadata paths, `hf download` command, checksum and line
count commands, parse/audit fields, source-level/aggregate outputs,
deterministic 600-record sample manifest, exact DUDE rule, and failure
dispositions.

- [ ] **Step 7: Define Stage 3 jobs, schemas, outputs, and gate**

Include dry-run and full download commands, 350-GB preflight, all 13 part
names, checksums, stream validation/extraction, image index schema, ordered
join rules, selected tree, 600-overlay review, integrity audit, quarantine
behavior, and 99.9% resolution gate.

- [ ] **Step 8: Define Stage 4 schema, splits, outputs, and gate**

Include the canonical record fields, immutable raw boxes, transform
round-trip, supervision classes/weights, exact/near-duplicate clustering,
cross-source TextVQA/TextCaps grouping, deterministic 90/5/5 group split,
sealed audit handling, null future candidate fields, leakage tests, and no
MMLongBench acquisition.

- [ ] **Step 9: Define reports, commit policy, stop conditions, and handback**

Require per-stage status reports, exact commands/configuration, hashes, counts,
failures, exclusions, job IDs, scratch paths, environment versions, and
deviations. Commit only Git-safe code/manifests/reports to the experiment
branch. Stop after the Stage 4 gate and push the branch; do not merge to
`main`.

- [ ] **Step 10: Define the workspace path manifest**

List every home-worktree path, scratch path, Git-safe output, forbidden Git
artifact, and expected creator stage in a machine-readable tab-separated text
file.

---

### Task 4: Archive the previous SOL task and activate the new one

**Files:**
- Move: `sol/CURRENT_SOL_TASK.md` to `sol/archive/current_tasks/2026-07-28-mmlongbench-gold-pages-ocr2-prepared.md`
- Move: `sol/task_spec/mmlongbench_gold_pages_deepseek_ocr2.md` to `sol/archive/task_specs/mmlongbench_gold_pages_deepseek_ocr2.md`
- Move: `sol/run_deepseek_ocr.sbatch` to `sol/archive/jobs/mmlongbench_prepared_unlaunched/run_deepseek_ocr.sbatch`
- Move: `sol/setup_deepseek_ocr.sbatch` to `sol/archive/jobs/mmlongbench_prepared_unlaunched/setup_deepseek_ocr.sbatch`
- Create: `sol/CURRENT_SOL_TASK.md`
- Modify: `sol/AGENTS.md`
- Modify: `sol/README.md`

**Interfaces:**
- Consumes: the detailed task spec and preserved old handoff.
- Produces: one unambiguous active SOL task with discoverable history.

- [ ] **Step 1: Move the four superseded files with Git history**

Use `git mv` to the exact archive destinations. Preserve “prepared, not
launched” status in the archived current-task document.

- [ ] **Step 2: Write the concise current SOL state**

Include:

- prepared locally, not yet started on SOL;
- exact main handoff and experiment branches;
- source and target SOL directories;
- binding/full-plan paths;
- first bootstrap action;
- Stage 4 hard stop;
- no job/run IDs;
- blockers limited to pushed-commit availability and SOL storage/network
  preflight.

- [ ] **Step 3: Update SOL routing**

Make `sol/AGENTS.md` require branch/worktree/commit verification before reading
the task. Make `sol/README.md` list only the new active handoff and point old
OCR materials to their archive.

- [ ] **Step 4: Run the focused tests**

Run:

```bash
python -m pytest -q tests/evidence_dino/test_handoff.py --tb=short
```

Expected: remaining failures relate only to agent-context/current-project
routing or workspace scaffolding.

---

### Task 5: Archive the old agent task and make Evidence-DINO active

**Files:**
- Move: `agent-context/CURRENT_TASK.md` to `archive/agent-context/current_tasks/2026-07-28-hierarchical-segmentation-routing.md`
- Move: current inactive `agent-context/modules/*.md` files to `archive/agent-context/modules/` where their subsystem is paused
- Create: `agent-context/CURRENT_TASK.md`
- Create: `agent-context/modules/evidence_dino_units.md`
- Modify: `agent-context/INDEX.md`
- Modify: `agent-context/ARCHITECTURE.md`
- Modify: `agent-context/COMMANDS.md`
- Modify: `agent-context/DATA_AND_ARTIFACTS.md`
- Modify: `agent-context/GOTCHAS.md`
- Modify: `agent-context/SOL.md`
- Modify: `README.md`
- Modify: `docs/ACTIVE_PROJECTS.md`
- Modify: `docs/NAVIGATION.md`
- Modify: `docs/BRANCHES.md`
- Modify: `archive/README.md`

**Interfaces:**
- Consumes: archived old state and active Evidence-DINO specifications.
- Produces: concise, consistent human and agent routing on `main`.

- [ ] **Step 1: Archive the old current task and paused modules**

Move the current-task file and paused subsystem modules without deleting their
content. Keep `sol_job_monitoring.md` active because it remains operational.

- [ ] **Step 2: Write the new current task with the established structure**

Use:

```text
Goal
Scope
Canonical plans and inputs
Decisions made
Current state
Current blockers
Research sequence
Next steps for a fresh agent
```

Keep it concise and point detailed requirements to the canonical and SOL specs.

- [ ] **Step 3: Rewrite active project routing**

Place Evidence-DINO-Units first under `Active`. Move the four previous active
projects under `Paused`, preserving their exact specifications and state.
Update the status date to `2026-07-28`.

- [ ] **Step 4: Update every current router**

Ensure README, navigation, index, architecture, commands, data/artifacts,
gotchas, SOL, branch policy, and archive index agree about:

- active Evidence-DINO Stages 0–4;
- separate SOL workspace;
- scratch boundary;
- experiment branch;
- paused prior projects;
- Stage 5 stop;
- old-task archive paths; and
- `main` versus experiment-branch ownership.

- [ ] **Step 5: Run repository routing tests**

Run:

```bash
python -m pytest -q \
  tests/test_repository_structure.py \
  tests/evidence_dino/test_handoff.py \
  --tb=short
```

Expected: all pass.

---

### Task 6: Add the focused experiment workspace surface

**Files:**
- Modify: `.gitignore`
- Modify: `.rgignore`
- Create: `configs/README.md`
- Create: `data/README.md`
- Create: `external/README.md`
- Create: `src/data/README.md`
- Create: `src/candidates/README.md`
- Create: `src/models/README.md`
- Create: `src/training/README.md`
- Create: `src/evaluation/README.md`
- Create: `experiments/README.md`
- Create: `checkpoints/README.md`
- Create: `reports/README.md`
- Create: `environments/README.md`
- Create: `sol/task_spec/evidence_dino_units_workspace_bootstrap.md`
- Create: `sol/task_spec/evidence_dino_units_sparse_checkout.txt`

**Interfaces:**
- Consumes: the approved logical tree and storage contract.
- Produces: a Git-visible project skeleton and exact SOL command sequence.

- [ ] **Step 1: Add safe ignore rules**

Ignore runtime contents under raw/images/OCR/candidates/mapped/checkpoints,
external clones/models, caches, logs, and scratch-link locations while
allowing the directory README contracts, small manifests, audits, configs, and
reports.

- [ ] **Step 2: Add focused boundary READMEs**

Each README states:

- responsibility;
- first stage that may populate it;
- tracked versus scratch-only artifacts;
- required provenance; and
- forbidden early-stage behavior.

- [ ] **Step 3: Write the exact SOL bootstrap**

The bootstrap contract must:

1. verify `~/COLPALI_binary_classification`;
2. fetch the pushed experiment branch;
3. refuse a conflicting `~/Evidence-DINO-Units`;
4. create the sibling worktree on the experiment branch;
5. verify branch and remote containment;
6. create scratch roots only from a lightwork/compute context where needed;
7. record source/base/execution commits;
8. enter `~/Evidence-DINO-Units`; and
9. stop before dataset work until the contract read/signoff check is complete.

- [ ] **Step 4: Define the focused sparse worktree**

Create a cone-mode sparse-checkout list that exposes only:

```text
agent-context
archive/agent-context
configs
data
docs/specifications/DETR_GroundingDINO_Datasetplan
environments
experiments
external
reports
checkpoints
sol/task_spec
sol/archive/current_tasks
sol/archive/task_specs
sol/archive/jobs/mmlongbench_prepared_unlaunched
src
tests/evidence_dino
```

Cone-mode parent-file behavior retains the root files and current routing
documents while hiding MMLongBench data/results/viewers, legacy source, legacy
tests, unrelated specifications, and unrelated archive domains from the SOL
workspace. The content remains available on `main` and in repository history.

- [ ] **Step 5: Re-run the focused tests**

Run:

```bash
python -m pytest -q tests/evidence_dino/test_handoff.py --tb=short
```

Expected: all pass.

---

### Task 7: Verify, publish shared main state, and publish the experiment branch

**Files:**
- Verify: all files changed in Tasks 1–6

**Interfaces:**
- Consumes: complete shared handoff and focused experiment skeleton.
- Produces: pushed `main` shared state and pushed experiment branch with exact
  commit identities for SOL.

- [ ] **Step 1: Run full verification on the experiment branch**

Run:

```bash
git diff --check
cmp \
  docs/specifications/DETR_GroundingDINO_Datasetplan/evidence_dino_units_training_plan_final.md \
  sol/task_spec/evidence_dino_units_training_plan_final.md
python -m pytest -q --tb=short
```

Expected: clean diff, identical full-plan copies, zero test failures.

- [ ] **Step 2: Audit requirements and repository state**

Run:

```bash
git status --short
git diff --stat 379b641..HEAD
rg -n "Stage 5|T0-BOX|DeepSeek-OCR-2|MMLongBench" \
  sol/CURRENT_SOL_TASK.md \
  sol/task_spec/evidence_dino_units_stages_0_4.md
find sol -maxdepth 3 -type f | sort
find agent-context archive/agent-context -maxdepth 3 -type f | sort
```

Confirm Stage 5+ terms appear only as prohibitions/future boundaries in the
current task.

- [ ] **Step 3: Commit the shared handoff**

Run:

```bash
git branch --show-current
git add -A
git commit -m "docs: activate Evidence DINO dataset handoff"
```

- [ ] **Step 4: Fast-forward the shared handoff to main**

From the original main checkout:

```bash
git switch main
git status --short
git merge --ff-only codex/evidence-dino-units-dataset-stages-0-4
python -m pytest -q --tb=short
git push origin main
```

`main` is currently a strict ancestor of the experiment branch, so a
fast-forward preserves identical shared commit identities. Do not add or
modify the untracked `work/` directory.

- [ ] **Step 5: Record the shared starting identity**

At handoff time, `main` and the experiment branch intentionally point to the
same prepared commit. They diverge only after SOL begins Stage 0–4
implementation on the experiment branch. Record this exact starting commit in
the launch handoff.

- [ ] **Step 6: Push the experiment branch**

Run:

```bash
git push -u origin codex/evidence-dino-units-dataset-stages-0-4
git ls-remote --heads origin \
  main \
  codex/evidence-dino-units-dataset-stages-0-4
```

Record both remote commit hashes in `sol/CURRENT_SOL_TASK.md` if doing so does
not require another history rewrite; otherwise report them in the launch
handoff and update the task in the first SOL commit.

- [ ] **Step 7: Provide the minimal SOL launch prompt**

Use only:

```text
In ~/COLPALI_binary_classification, fetch the latest remote state, then read
AGENTS.md and sol/CURRENT_SOL_TASK.md. Follow the binding handoff exactly.
```

Do not duplicate the detailed instructions in the launch message.
