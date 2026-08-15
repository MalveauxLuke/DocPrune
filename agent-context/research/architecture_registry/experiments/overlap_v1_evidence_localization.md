# Overlap-First V1 Evidence Localization

## Record

- Experiment ID: `overlap_v1_evidence_localization`
- Owner approval: 2026-08-14
- Status: `approved-design-review`
- Execution: no SOL stage active
- Binding specification:
  `../../../../docs/specifications/overlap_v1_evidence_localization/README.md`
- Architecture files:
  `../../../../docs/specifications/overlap_v1_evidence_localization/architectures/`
- Experiment files:
  `../../../../docs/specifications/overlap_v1_evidence_localization/experiment/`
- Preservation audit:
  `../../../../docs/specifications/overlap_v1_evidence_localization/SOURCE_COVERAGE.md`

## Hypothesis and controls

The POC tests whether audited pairwise adaptation, shared full-page Qwen
memory, and explicit MiniVGent candidate interaction improve supplied-page
answer-anchor localization under one frozen V1 candidate universe.

Active arms are R0/R1 and M0/M1. Qwen R0/R1 is mandatory. The primary
MiniVGent contrast is M1 versus M0 under matched initialization, rows,
objective, and schedule. HierDoc H0/H1 and A1 are retained for a later
answer-sufficiency experiment, not this POC.

## Stage state

The complete POC is authorized, but stages 00-05 activate one at a time.
The written specification and POC implementation plan must be reviewed and
the workspace gate closed before Stage 00 can become a binding SOL task.

## Current blockers

- written-spec owner review not yet recorded;
- no POC implementation plan;
- no execution checkout, branch, scratch root, or recovery authority;
- no candidate revision or model/environment locks; and
- no inference, mining, training, or benchmark result.

## Claim boundary

The experiment concerns answer-bearing anchors on one supplied page. It does
not establish complete evidence, genuine multi-hop reasoning, page routing, or
answer-feedback optimization.
