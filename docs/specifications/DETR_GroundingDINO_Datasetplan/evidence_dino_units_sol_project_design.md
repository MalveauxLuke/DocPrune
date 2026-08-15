# Evidence-DINO-Units SOL Project Design

**Status:** Approved on 2026-07-28.

**Implementation scope:** Prepare a focused repository and SOL handoff for
Stages 0–4 of
`evidence_dino_units_training_plan_final.md`. Do not execute or instruct SOL to
execute Stage 5.

## 1. Goal

Pause the repository's current segmentation and reranker work without deleting
it, make Evidence-DINO-Units the active research line, and give the SOL agent a
safe, self-contained workspace for contract, acquisition, normalization, and
split construction.

The initial SOL task ends when the Stage 4 deliverables and advancement gate
are complete. It does not perform Grounding-DINO adaptation, DeepSeek-OCR-2
inference, candidate construction, model training, or MMLongBench evaluation.

## 2. Approved project identity

- Git branch:
  `codex/evidence-dino-units-dataset-stages-0-4`
- SOL home worktree:
  `~/Evidence-DINO-Units`
- SOL heavy-artifact root:
  `/scratch/$USER/evidence_dino_units`
- Upstream repository:
  the same Git repository as `~/COLPALI_binary_classification`
- Intended use:
  research-only
- Stage 0 owner signoff:
  approved by the project owner on 2026-07-28

The branch is derived from the exact current local repository state. The SOL
agent must verify the exact worktree, branch, base commit, and execution commit
before reading the task or running a job.

The original approved design limited publication to the experiment branch.
That publication restriction was superseded by the project owner on
2026-07-28. Shared current-context, archive-routing, and SOL handoff
documentation may be committed and pushed to `main`. Experiment-only pruning
and implementation remain on the dedicated experiment branch.

## 3. Repository transition

### 3.1 Pause, do not delete

The following projects become paused:

- hierarchical segmentation and evidence routing;
- the MMLongBench unstructured segmentation lab;
- the MMLongBench gold-page quadrant-reranker baseline; and
- the segment-reranker corpus.

Their specifications and retained results remain available on `main`. The
experiment branch does not treat them as active dependencies.

Archive the previous `agent-context/CURRENT_TASK.md` under a dated
`archive/agent-context/current_tasks/` path, then replace it with a concise
Evidence-DINO-Units Stage 0–4 handoff that preserves the established section
structure.

Update these current-state routers together:

- `README.md`;
- `docs/ACTIVE_PROJECTS.md`;
- `docs/NAVIGATION.md`;
- `agent-context/INDEX.md`;
- `agent-context/ARCHITECTURE.md`;
- `agent-context/CURRENT_TASK.md`; and
- `agent-context/SOL.md`.

### 3.2 Archive the superseded SOL task

Archive the prepared but unlaunched MMLongBench gold-page task:

- `sol/CURRENT_SOL_TASK.md`;
- `sol/task_spec/mmlongbench_gold_pages_deepseek_ocr2.md`;
- `sol/run_deepseek_ocr.sbatch`; and
- `sol/setup_deepseek_ocr.sbatch`.

The task state and contract go into dated/current-task and task-spec archives.
The wrappers go into a named prepared-but-unlaunched job archive. Preserve
their history; do not describe the old task as launched or completed.

Keep and update:

- root `AGENTS.md`;
- `sol/AGENTS.md`;
- `sol/README.md`; and
- `docs/SOL_INSTRUCTIONS.md`.

### 3.3 Focused experiment surface

The experiment worktree uses this logical structure:

```text
Evidence-DINO-Units/
├── AGENTS.md
├── README.md
├── LICENSE
├── agent-context/
├── archive/
│   └── agent-context/current_tasks/
├── docs/
│   ├── ACTIVE_PROJECTS.md
│   ├── NAVIGATION.md
│   ├── SOL_INSTRUCTIONS.md
│   ├── experiment_contract.md
│   └── specifications/DETR_GroundingDINO_Datasetplan/
├── external/
│   ├── Visual-CoT/
│   ├── GroundingDINO/
│   └── MMLongBench-Doc/
├── data/
│   ├── raw/
│   ├── images/
│   ├── metadata/{normalized,audits}/
│   ├── ocr/
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
└── sol/
    ├── AGENTS.md
    ├── README.md
    ├── CURRENT_SOL_TASK.md
    ├── task_spec/
    │   ├── evidence_dino_units_training_plan_final.md
    │   └── evidence_dino_units_stages_0_4.md
    └── archive/
```

The initial handoff creates only the structure and files required for Stages
0–4. Empty later-stage directories establish interfaces but do not authorize
later-stage execution.

The following existing surfaces are absent from the focused experiment
workspace:

- retained MMLongBench `pilot_data/` and `sol_results/`;
- MMLongBench viewers and segmentation-lab implementation;
- legacy SciEGQA, ColQwen, Vidore, and MMLongBench archives;
- generated `tmp/`, `work/`, caches, logs, and viewer builds; and
- Stage 5+ implementations that have not yet passed their preceding gates.

The pruning occurs only on the dedicated experiment branch. It does not delete
these materials from `main`.

## 4. Storage boundary

Code, instructions, small configurations, immutable manifests, audit summaries,
checksums, and reports belong in the home worktree and Git when safe.

These belong under `/scratch/$USER/evidence_dino_units`:

- the Visual-CoT archive parts;
- extracted and selected images;
- Hub caches;
- large normalized/intermediate tables;
- temporary extraction files;
- generated overlays/contact sheets;
- checkpoints;
- run payloads; and
- scheduler logs.

The workspace bootstrap must create runtime links or explicit configured paths
without committing user-specific absolute symlinks. Scratch is temporary and
must never be the only location of a required manifest, report, or provenance
record.

No PDF, image archive, extracted image tree, model weight, cache, environment,
or Slurm log may be committed.

## 5. SOL handoff contract

`sol/CURRENT_SOL_TASK.md` remains short and contains:

- state;
- exact branch/worktree requirement;
- binding Stage 0–4 task-spec path;
- full-plan companion path;
- next action;
- stage boundary;
- latest jobs/runs; and
- blockers.

The active task-spec pair is:

1. `sol/task_spec/evidence_dino_units_stages_0_4.md` — binding execution
   contract; and
2. `sol/task_spec/evidence_dino_units_training_plan_final.md` — byte-identical
   authoritative full-plan companion.

The original canonical plan remains under
`docs/specifications/DETR_GroundingDINO_Datasetplan/`. A focused test or
checksum must prove that the SOL full-plan copy is byte-identical.

The SOL agent creates its sibling worktree from the pushed experiment branch.
Git/worktree inspection is permitted on the login node. Environment creation,
downloads, archive validation/extraction, indexing, hashing, normalization,
and split generation run only inside suitable allocations.

## 6. Stage 0–4 execution clarifications

These clarifications preserve the plan's intent while making it executable on
SOL.

### 6.1 Stage 0

Create `docs/experiment_contract.md` containing the ten invariants verbatim.
Record the research-only intended use and the 2026-07-28 owner signoff.

### 6.2 Stage 1

Use a pinned Mamba acquisition environment created from a compute allocation,
not the plan's generic local `.venv-acquire` example.

Resolve and record immutable external revisions. Do not silently refresh a
lock after experiment work begins.

The license audit must retain the Visual-CoT license discrepancy with an
explicit owner/status. Research-only approval does not authorize redistribution
of derived data or weights.

### 6.3 Stages 2 and 3 visual-review ordering

The authoritative plan currently requests a 600-row visual inspection in
Stage 2, before Stage 3 downloads the images. The executable order is:

1. Stage 2 deterministically selects and freezes 100 review records per source,
   for 600 records total.
2. Stage 2 completes metadata-only parsing, schema, box, count, and duplicate
   audits.
3. Stage 3 downloads, verifies, extracts, indexes, and joins the images.
4. Stage 3 renders and visually inspects all 600 frozen samples with their
   questions, answers, and separate raw boxes.
5. The Stage 2 visual-review deliverable and Stage 3 coordinate-overlay gate
   close together after that inspection.

The change moves the inspection, not its sample size or quality gate.

### 6.4 Stage 4

Do not download or inspect MMLongBench during this task. Record its immutable
revision and reserve the sealed path only. Acquisition remains Stage 14.

Fields for candidate generation and mapped labels remain empty/null in Stage 4
records. The agent must not fabricate candidates, positives, negatives, or
mapping confidence before Stages 6–7.

TextVQA and TextCaps visual identities must be grouped across sources before
splitting. No exact or near-duplicate visual identity may cross
train/development-validation/sealed-audit boundaries.

### 6.5 Preserve superseded plan text

Every complete copy of `evidence_dino_units_training_plan_final.md` must expose
both the original wording and the binding SOL update at each changed location.
Do not silently rewrite or delete the original passage.

Use this local structure:

```text
Original plan text — inactive for the Stage 0–4 SOL task
<original wording retained verbatim>

Binding Stage 0–4 SOL update
<replacement ordering or rule>
```

The inactive marker applies only to the explicitly identified passage and this
SOL task. Unmarked parts of the authoritative plan remain active. The canonical
copy and the full-plan copy under `sol/task_spec/` must remain byte-identical
after these annotations are added.

## 7. Stage boundary and stop rules

The SOL agent must stop after proving the Stage 4 advancement gate.

It must not:

- execute T0-BOX or any Stage 5 test;
- download Grounding-DINO weights for a model run;
- download or run DeepSeek-OCR-2;
- construct semantic/fallback candidates;
- map boxes to candidates;
- train any selector;
- acquire MMLongBench data;
- serve a viewer or website; or
- change a pinned revision, source set, split rule, sample size, or gate merely
  to make progress.

If a source cannot be resolved, a license/provenance requirement is
insufficient, the deterministic manifests differ, scratch is too small, or a
gate fails, stop and report the exact evidence.

## 8. Completion evidence

The handoff preparation is complete only when:

- the old agent and SOL tasks are archived and discoverable;
- current routing documents name Evidence-DINO-Units as active;
- all previously active projects are visibly paused;
- the full SOL plan copy is byte-identical to the canonical plan;
- the binding Stage 0–4 task spec contains the executable clarifications;
- root and SOL `AGENTS.md` files contain the required safety/context routing;
- shared handoff documentation is committed and pushed to the authorized
  `main`, while experiment-only changes are committed and pushed only to the
  dedicated branch; and
- a fresh checkout can identify the exact SOL worktree, scratch boundary,
  current task, and Stage 4 stop condition without reading historical files.
