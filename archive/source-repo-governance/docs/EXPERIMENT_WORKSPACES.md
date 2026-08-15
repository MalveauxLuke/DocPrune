# Experiment Workspaces

Status date: 2026-08-14

This registry records where experiments execute and how the central command
recovers their authority. A `destination-required` location is a hard migration
gate: do not remove the current implementation or artifacts until a real
destination is approved and verified. Architecture-registry proposals are not
approved experiments.

## Active workspaces

### Overlap-first document corpus V1

- Status: `complete`
- Execution location: completed SOL artifact; immutable package and source
  ledger are recorded below
- Git branch: `main` contains the final builder and control-plane record
- Authority: `reports/overlap_first_document_corpus.md`
- Manifest package: `/home/lmalveau/overlap_first_document_corpus/v1`
- Source ledger: `/home/lmalveau/overlap_first_document_corpus/run1`
- BoundingDocs images: `/scratch/$USER/boundingdocs_document_qa/derived/overlap_first_document_corpus/run1/images`
- OCR artifacts: `/scratch/$USER/evidence_dino_resolved_ocr`
- Recovery: read `agent-context/CURRENT_TASK.md`, the package `manifest.json`,
  and `reports/overlap_first_document_corpus.md`

### Query-planner main-project

- Status: `active`
- Execution location: protected local worktree
  `/Users/god/Documents/COLPALI_binary_classification-main`
- Git branch: `main-project`
- Authority: that worktree's own task and documentation state
- Recovery: inspect that worktree directly; never merge, rebase, prune, or
  modify it from the central-command cleanup

## Active workspaces requiring destinations

### Overlap-first V1 evidence-localization POC

- Status: `approved-design-review`
- Execution location: `destination-required`; do not run candidate generation,
  model inference, hard-negative mining, or training from the control checkout
- Git branch: `destination-required`
- Authority:
  `docs/specifications/overlap_v1_evidence_localization/README.md`
- Recovery: read `agent-context/CURRENT_TASK.md`, the POC architecture and
  experiment files, and `SOURCE_COVERAGE.md`; after written-spec and POC
  implementation-plan review, create a binding Stage 00 SOL handoff only when
  the destination, branch, scratch root, commit parity, and recovery authority
  are recorded

## Paused workspaces

### Evidence-DINO-Units

- Status: `paused-resumable`
- Execution location: retained sibling worktree `~/Evidence-DINO-Units`;
  retained SOL artifacts under `/scratch/$USER/evidence_dino_units`
- Git branch: `codex/evidence-dino-units-dataset-stages-0-4`
- Authority: `sol/task_spec/evidence_dino_units_stages_0_4.md`
- Recovery: use `sol/task_spec/evidence_dino_units_workspace_bootstrap.md` and
  verify the remote branch before resuming; do not reuse its paths for
  BoundingDocs

## Paused workspaces requiring destinations

### Hierarchical evidence routing

- Status: `paused-resumable`
- Execution location: `destination-required`; retained implementation is
  temporarily in this checkout
- Git branch: no dedicated execution branch is currently registered
- Authority: `docs/specifications/hierarchical_evidence_routing/design.md`
- Recovery: retain specification, code, tests, and referenced MMLongBench
  artifacts until the destination is approved and checksum-verified

### MMLongBench segmentation lab

- Status: `paused-resumable`
- Execution location: `destination-required`; retained implementation and
  outputs are temporarily in this checkout
- Git branch: no dedicated execution branch is currently registered
- Authority: `docs/specifications/mmlongbench_segmentation_lab/design.md`
- Recovery: preserve the 313 planned pages, 306 successful OCR pages, seven
  exclusions, and both segmentation strategies until migration

### MMLongBench gold-page reranker

- Status: `paused-resumable`
- Execution location: `destination-required`; the SOL acquisition task was
  prepared but never launched
- Git branch: no dedicated execution branch is currently registered
- Authority: `docs/specifications/mmlongbench_gold_page_quadrant_reranker_experiment.md`
- Recovery: resume from the specification and
  `sol/archive/current_tasks/2026-07-28-mmlongbench-gold-pages-ocr2-prepared.md`

### Segment-reranker corpus

- Status: `paused-resumable`
- Execution location: `destination-required`; no dataset implementation was
  started
- Git branch: no dedicated execution branch is currently registered
- Authority: `docs/specifications/segment_reranker_corpus.md`
- Recovery: retain the approved DUDE, SlideVQA, TAT-DQA, and held-out
  MMLongBench supervision contract until a destination is approved

## Registry rules

- Record actual local/SOL paths, remotes, branches, and immutable revisions
  when they exist.
- Never invent a destination to satisfy cleanup.
- Do not treat architecture-registry proposals as approved experiments.
- Do not remove source material until destination and recovery evidence are
  verified.
