# Paper title

Use this template after checking the primary paper. The finished record should
link to the paper, keep paper summary to a short factual capsule, and reserve
most of the file for our analysis and interpretation.

Keep these statement types distinct:

- `paper_reported`: supported by the linked primary source;
- `repository_inference`: our interpretation of possible COLPALI fit;
- `unvalidated_proposal`: an experiment or change not yet run; and
- `repository_tested`: a local result with retained evidence.

Do not copy long paper passages. Do not claim local reproduction without
retained repository evidence.

## Registry metadata

- Paper:
- Stable identifier and reviewed version:
- Last reviewed:
- Applicability: `direct`, `adjacent`, or `architectural_analogue`
- Capability tags:
- Possible project consumers:
- Evidence levels used: `paper_reported`, `repository_inference`,
  `unvalidated_proposal`, and/or `repository_tested`
- Registry status: `unreviewed`, `reviewed`, `promising`, `scheduled`,
  `tested`, `adopted`, `rejected`, or `superseded`

## Paper link and factual capsule

- Primary paper:
- Official code, project page, or appendix checked:

In a short paragraph, identify the actual task, domain, central mechanism,
training/evaluation boundary, and what the paper demonstrates. State what it
does not demonstrate when that distinction matters for COLPALI. Link out for
methodological detail instead of reproducing the paper.

## Why this paper attracted attention

Record the project problem or architectural limitation that made this paper
worth retaining.

## Our analysis and interpretation

Explain our current view of the mechanism, including why it may or may not
transfer. Label each project-facing claim `repository_inference` or
`repository_tested`.

## Reusable mechanisms

Identify the smallest mechanisms that could transfer independently of the full
paper system.

## Possible COLPALI integration points

For each possible consumer, record:

- current subsystem or experiment;
- component that would change;
- proposed mechanism;
- evidence level; and
- why the mapping is plausible.

## Required adaptations

List task, data, representation, training, calibration, cost, and evaluation
changes required before the mechanism fits COLPALI.

## Advantages relative to the current subsystem

Compare against the actual current architecture or baseline, including reasons
to retain the simpler current design.

## Concerns and failure risks

Record domain mismatch, supervision mismatch, proposal ceilings, evidence
completeness, abstention, confirmation bias, cost, and implementation risks
when relevant.

## Open questions

Keep unresolved interpretation and design questions here. Remove a question
when a dated analysis note or retained experiment answers it.

## Smallest decisive experiment

Specify:

- one falsifiable hypothesis;
- changed component and valid control;
- frozen inputs and downstream settings;
- primary success and safety metrics;
- failure interpretation; and
- authorization state.

Label the experiment `unvalidated_proposal` until it is approved and run.

## Analysis and interpretation log

Add dated notes when our interpretation changes. Each note should explain what
new paper evidence, repository evidence, or design reasoning caused the change.

## Adoption status and decision history

Record dated status changes and their evidence. A paper becomes `adopted` only
after project-owner approval and a separate canonical-specification update.
