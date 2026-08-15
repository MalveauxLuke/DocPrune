# Current Task State

## Goal

Prepare the owner-approved overlap-first V1 answer-anchor POC for staged SOL
execution. The initial comparison is the former pairwise segment-reranker as
Qwen R0/R1 versus MiniVGent M0/M1. HierDoc H0/H1 and the matched
autoregressive A1 control are preserved for a later answer-sufficiency test.

## Owner-approved program

1. Freeze one DeepSeek-OCR-2 semantic candidate revision and answer-anchor
   eligibility/oracle view.
2. Run stock and audited-hard-negative-tuned pairwise Qwen controls R0/R1.
3. Verify Qwen hidden states and MiniVGent tensors, ROI, losses, and systems
   behavior.
4. Run a bounded one-seed R0/R1/M0/M1 architecture screen.
5. Freeze decisions and run a three-seed confirmatory comparison with one
   internal-test opening.
6. Optionally run one locked InfographicsVQA robustness evaluation.

The entire single-hop answer-anchor POC is authorized, but stages activate one
at a time. Completing one stage never submits the next automatically. The
later HierDoc/A1 answer-sufficiency experiment is documented but not active.

## Binding design sources

- Program:
  `docs/specifications/overlap_v1_evidence_localization/README.md`
- Architectures:
  `docs/specifications/overlap_v1_evidence_localization/architectures/`
- Experiment planning:
  `docs/specifications/overlap_v1_evidence_localization/experiment/`
- Preservation audit:
  `docs/specifications/overlap_v1_evidence_localization/SOURCE_COVERAGE.md`
- Durable module:
  `agent-context/modules/overlap_v1_evidence_localization.md`

The older baseline, MiniVGent proposal/readiness record, HierDoc review, and
two implementation plans remain provenance until a POC implementation plan
passes written-spec review and supersedes their execution ordering.

## Boundaries

- Immutable source: `/home/lmalveau/overlap_first_document_corpus/v1`.
- Labels are accepted answer-bearing anchors, not complete evidence.
- Qwen R0/R1 is mandatory for a matched MiniVGent comparison; Jina is optional.
- HierDoc H0/H1, A1, answer-sufficiency training, and page routing belong to a
  later experiment and are not active POC arms.
- No V1 mutation, DeepSeek-OCR-2 fine-tuning, page-policy training, synthetic
  multi-hop, answer-feedback RL, split redesign, conflict adjudication, or
  complete-evidence claim.
- `infographicsvqa_holdout` remains sealed until the locked final robustness
  stage and cannot influence model choice.

## Current state

- The owner approved the POC-first staged design on 2026-08-14.
- Architecture and experiment specification files are consolidated and
  audited for source coverage.
- Owner review is recorded in
  `docs/specifications/overlap_v1_evidence_localization/OWNER_REVIEW.md`; it
  approves implementation-plan preparation, not execution.
- The consolidated staged implementation plan is drafted at
  `docs/superpowers/plans/2026-08-14-overlap-v1-minivgent-poc.md` and awaits
  owner review.
- No candidate universe, model lock, hard-negative manifest, model inference,
  or training run exists.
- No execution checkout, branch, scratch root, implementation-plan approval,
  or active SOL stage exists yet.
- Do not run candidate generation, model download/inference, mining, or
  training from the control checkout.

## Next action

Review the binding POC implementation plan. After approval, close the workspace
gate and activate only Stage 00 in a binding SOL handoff.
