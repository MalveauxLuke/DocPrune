# Evidence-DINO-Units

## Role

Evidence-DINO-Units supplied the Visual-CoT image and DeepSeek-OCR-2 foundation
for the completed overlap-first document corpus V1.

## Current state

- V1 package: `/home/lmalveau/overlap_first_document_corpus/v1`.
- 12,856 documents; 66,551 canonical questions; 65,787 usable questions;
  23,043 page positions; 23,006 unique OCR/image contents.
- Visual-CoT is the base; 8,085 BoundingDocs-only questions add information.
- DocVQA and DUDE use document-grouped 80/10/10 splits.
- InfographicsVQA is isolated in `infographicsvqa_holdout`.
- TextVQA, TextCaps, and SROIE are excluded from this merged document-QA V1.
- The corpus-building task is complete. The overlap-V1 answer-anchor POC is
  now active in design review; its R0/R1 controls retain the former
  segment-reranker baseline, its M0/M1 arms test MiniVGent, and V1 remains
  immutable.

## Sources of truth

- Full plan:
  `docs/specifications/DETR_GroundingDINO_Datasetplan/evidence_dino_units_training_plan_final.md`
- Approved project design:
  `docs/specifications/DETR_GroundingDINO_Datasetplan/evidence_dino_units_sol_project_design.md`
- Binding SOL dataset contract:
  `sol/task_spec/evidence_dino_units_stages_0_4.md`
- Current state: `agent-context/CURRENT_TASK.md`

## Durable invariants

- Preserve raw annotations and every original box.
- Group splits by visual identity across sources.
- Do not fabricate candidate labels before candidates exist.
- Treat answer-bearing localization as weaker than complete evidence
  supervision.
- Record actual counts/hashes rather than substituting planning estimates.
- Stop rather than weakening gates or changing pinned sources silently.
