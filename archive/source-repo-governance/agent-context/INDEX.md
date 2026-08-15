# Agent Context Index

Read this file first. Open only the entry relevant to the task.

## Current research focus

- The overlap-first document corpus V1 is complete and verified.
- Read `CURRENT_TASK.md`, then
  [`modules/overlap_v1_evidence_localization.md`](modules/overlap_v1_evidence_localization.md),
  then
  [`../docs/specifications/overlap_v1_evidence_localization/README.md`](../docs/specifications/overlap_v1_evidence_localization/README.md).
- Package: `/home/lmalveau/overlap_first_document_corpus/v1`.
- The full canonical ledger has 12,856 documents and 66,551 questions; 65,787
  questions have resolved OCR-backed pages and 764 are quarantined.
- Primary DocVQA/DUDE splits are document-grouped 80/10/10.
  InfographicsVQA is isolated as `infographicsvqa_holdout` and never enters
  train, validation, or test.
- The active follow-on is a staged answer-anchor POC: frozen candidates, Qwen
  R0/R1, and MiniVGent M0/M1. HierDoc H0/H1 and the matched A1 control are
  preserved for a later answer-sufficiency experiment. Execution remains
  blocked until written-spec and implementation-plan review, workspace
  approval, and a Stage 00 SOL handoff.

## Core

- `CURRENT_TASK.md`: concise state, decisions, blockers, and next actions.
- [`../docs/EXPERIMENT_WORKSPACES.md`](../docs/EXPERIMENT_WORKSPACES.md):
  execution locations, unresolved destination gates, and recovery authority
  for active and paused projects.
- `ARCHITECTURE.md`: current repository/workspace and data flow.
- `COMMANDS.md`: focused verification and SOL bootstrap commands.
- `DATA_AND_ARTIFACTS.md`: home/scratch/Git boundaries.
- `SOL.md`: SOL routing, branch, and stage boundary.
- `GOTCHAS.md`: operational failures to avoid.

## Active modules

- `modules/evidence_dino_units.md`: completed overlap-first V1 dataset,
  artifact locations, split policy, and exclusions.
- `modules/overlap_v1_evidence_localization.md`: active POC arms, staged
  sequence, boundaries, authority, and execution gate.
- `modules/overlap_v1_segment_reranker.md`: compatibility pointer explaining
  how the former baseline survives as mandatory R0/R1 controls.
- `modules/sol_job_monitoring.md`: bounded polling when monitoring is explicitly
  requested.

## On-demand research

- `research/architecture_registry/README.md`: comparison ledger and detailed
  primary-source reviews of promising architectures across COLPALI, plus an
  index of nonbinding project proposals and controlled experiment records.
  Open only when evaluating paper mechanisms, architecture alternatives, or
  future improvements. Registry entries are nonbinding until the relevant
  canonical project specification is separately approved and updated.
- `research/architecture_registry/proposals/`: future project treatments.
  Current records cover DeepSeek-OCR-2 generative localization and a
  document-specific multimodal reranker baseline, plus loose query-rewriting
  and negative contrastive query-rewriting directions.
- [`../docs/specifications/overlap_v1_evidence_localization/`](../docs/specifications/overlap_v1_evidence_localization/):
  owner-approved binding design family separating candidate/model
  architectures from staged experiment planning. No SOL stage is active yet.
- [`../docs/specifications/overlap_v1_evidence_localization/OWNER_REVIEW.md`](../docs/specifications/overlap_v1_evidence_localization/OWNER_REVIEW.md):
  one-page owner review approving implementation-plan preparation while
  retaining the Stage 00-only execution gate.
- [`../docs/superpowers/plans/2026-08-14-overlap-v1-minivgent-poc.md`](../docs/superpowers/plans/2026-08-14-overlap-v1-minivgent-poc.md):
  consolidated R0/R1/M0/M1 implementation plan, organized by separately
  activated Stages 00-05; currently awaiting owner review.
- `research/architecture_registry/experiments/README.md`: format and approval
  boundary for compact experiment records; the binding design remains under
  `docs/specifications/`.
- [`../docs/specifications/deepseek_ocr2_minivgent_evidence_localization/`](../docs/specifications/deepseek_ocr2_minivgent_evidence_localization/):
  full nonbinding dataset and evaluation specification for controlled
  DeepSeek-OCR2 and MiniVGent evidence-localization experiments, plus a
  reader-friendly guide and a separate contentions/decision record. These
  documents do not alter the active BoundingDocs acquisition handoff or the
  approved paused Evidence-DINO-Units and segment-reranker specifications.
- [`../docs/specifications/deepseek_ocr2_minivgent_evidence_localization/single_hop_poc_research_report.md`](../docs/specifications/deepseek_ocr2_minivgent_evidence_localization/single_hop_poc_research_report.md):
  nonbinding recommendation to begin MiniVGent with an answer-anchor,
  single-hop POC, then add verified same-page multi-evidence before answer
  feedback, reinforcement learning, or genuine multi-hop training.
- `research/architecture_registry/proposals/minivgent_residual_candidate.md`:
  nonbinding bbox-free complement candidate for future MiniVGent candidate-miss
  handling; it is not part of the active reranker baseline.
- `research/architecture_registry/proposals/minivgent_answer_anchor_poc.md`:
  source proposal consolidated into the binding design. Its R0/R1/M0/M1
  components define the active POC; its HierDoc region-ID policy and matched
  A1 control are preserved for the later answer-sufficiency experiment.
- `research/architecture_registry/proposals/minivgent_qwen_implementation_readiness.md`:
  source-verified Qwen3-VL hidden-state, prompt, visual-grid, ROI, autograd,
  candidate-encoder, decoder, loss, dependency, and systems contracts for a
  MiniVGent implementation, with exact two- and four-block parameter counts;
  now preserved in the binding MiniVGent architecture.
- `research/architecture_registry/papers/hierdoc-2607.29638.md`:
  reviewed primary-source record for HierDoc's GRPO-trained page and semantic
  region set policies, including the exact boundary between the full
  long-document system and the V1 supplied-page adaptation.

## Paused and historical material

The earlier Evidence-DINO-Units OCR producer branch and scratch artifacts are
retained as provenance for V1. They are not an active execution task.

The previous hierarchical routing, document parsing, viewer, test, and corpus
modules are under `archive/agent-context/modules/`. The previous current task
is under `archive/agent-context/current_tasks/`.

Inactive code, tests, applications, plans, and reports remain under `archive/`
or temporarily in retained paths named by the workspace registry. Superseded
SOL tasks and wrappers are under `sol/archive/`. Do not load them unless a
current specification or approved migration plan names them.

When durable knowledge changes, update the relevant short module and its full
specification together. Keep full schemas and long execution instructions out
of `agent-context/`.
