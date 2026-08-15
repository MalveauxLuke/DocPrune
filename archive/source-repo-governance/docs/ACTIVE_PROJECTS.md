# Active Projects

Status date: 2026-08-14

This file records research status, not execution permission. Runtime locations
and unresolved migration gates are in
[`EXPERIMENT_WORKSPACES.md`](EXPERIMENT_WORKSPACES.md). Architecture-registry
proposals are nonbinding until a canonical project specification is separately
approved.

## Active

### Overlap-first V1 answer-anchor POC

- Status: `approved-design-review`; candidate/model execution has not started.
- Goal: compare mandatory stock/tuned pairwise Qwen R0/R1 with MiniVGent
  M0/M1 on one frozen V1 answer-anchor candidate universe.
- Later test: retain HierDoc H0/H1 and matched small-Qwen A1 for an
  answer-sufficiency experiment after stronger evidence-set labels exist.
- Immutable source: `/home/lmalveau/overlap_first_document_corpus/v1`.
- Authority:
  `docs/specifications/overlap_v1_evidence_localization/README.md`.
- Architecture/experiment split:
  `docs/specifications/overlap_v1_evidence_localization/architectures/` and
  `docs/specifications/overlap_v1_evidence_localization/experiment/`.
- Hard boundary: V1 is not modified; labels are answer anchors rather than
  complete evidence; no OCR-2 fine-tuning, page-policy training, synthetic
  multi-hop, answer-feedback RL, conflict resolution, or split redesign.
- Execution gate: written-spec and POC-plan review, destination, branch,
  scratch root, and a one-stage-at-a-time SOL handoff remain required.
- Workspace record:
  [Overlap-first V1 evidence-localization program](EXPERIMENT_WORKSPACES.md#overlap-first-v1-evidence-localization-program).

## Paused

### Evidence-DINO-Units

- Status: `paused-resumable`.
- Last state: Stage 0–4 dataset contract and SOL workspace prepared; no dataset
  execution started.
- Retained SOL workspace: `~/Evidence-DINO-Units`.
- Retained branch: `codex/evidence-dino-units-dataset-stages-0-4`.
- Retained scratch root: `/scratch/$USER/evidence_dino_units`.
- Resume from: `sol/task_spec/evidence_dino_units_stages_0_4.md`.
- Archived SOL task:
  `sol/archive/current_tasks/2026-08-07-evidence-dino-units-stages-0-4-prepared.md`.
- Workspace record: [Evidence-DINO-Units](EXPERIMENT_WORKSPACES.md#evidence-dino-units).

### Hierarchical segmentation and evidence routing

- Status: `paused-resumable`.
- Last state: page→halves→quadrants routing design and local manual-routing
  implementation were available for the retained MMLongBench OCR corpus.
- Resume from:
  `docs/specifications/hierarchical_evidence_routing/design.md`.
- Archived task:
  `archive/agent-context/current_tasks/2026-07-28-hierarchical-segmentation-routing.md`.
- Workspace record:
  [Hierarchical evidence routing](EXPERIMENT_WORKSPACES.md#hierarchical-evidence-routing).

### MMLongBench unstructured segmentation lab

- Status: `paused-resumable`.
- Input: `pilot_data/mmlongbench_ocr2_unstructured_313/`.
- Retained result:
  `sol_results/mmlongbench_ocr2_unstructured_313/20260717T182300Z/`.
- Last verified scope: 313 planned pages, 306 successful pages, seven
  exclusions, and two segmentation strategies.
- Resume from:
  `docs/specifications/mmlongbench_segmentation_lab/design.md`.
- Workspace record:
  [MMLongBench segmentation lab](EXPERIMENT_WORKSPACES.md#mmlongbench-segmentation-lab).

### MMLongBench gold-page quadrant-reranker baseline

- Status: `paused-resumable`.
- Last state: experiment specified; OCR-only SOL acquisition prepared but
  never launched.
- Resume from:
  `docs/specifications/mmlongbench_gold_page_quadrant_reranker_experiment.md`.
- Archived SOL task:
  `sol/archive/current_tasks/2026-07-28-mmlongbench-gold-pages-ocr2-prepared.md`.
- Workspace record:
  [MMLongBench gold-page reranker](EXPERIMENT_WORKSPACES.md#mmlongbench-gold-page-reranker).

### Segment-reranker corpus

- Status: `paused-resumable`.
- Last state: architecture approved; dataset implementation had not started.
- Prior sources: DUDE, SlideVQA, and TAT-DQA with MMLongBench held out.
- Resume from: `docs/specifications/segment_reranker_corpus.md`.
- Workspace record:
  [Segment-reranker corpus](EXPERIMENT_WORKSPACES.md#segment-reranker-corpus).

## Completed historical evidence

The BoundingDocs acquisition and overlap-first V1 packaging are complete; the
final immutable package and verification report remain the source of truth.
The retained MMLongBench OCR and deep-parse pilots remain historical evidence.
SciEGQA/MinerU pilots, ColQwen verifier experiments, Vidore runs, and the first
segment-evidence training stages remain archived by domain. The `main-project`
query-planner worktree is separately active and protected; see its
[workspace record](EXPERIMENT_WORKSPACES.md#query-planner-main-project).
