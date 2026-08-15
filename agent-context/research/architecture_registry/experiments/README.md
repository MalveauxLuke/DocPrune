# Architecture Registry Experiments

This directory is reserved for durable records of approved or executed
controlled experiments arising from the architecture registry.

> **Current state:** the overlap-first V1 answer-anchor POC
> is owner approved and awaiting written-spec review, a POC implementation
> plan, workspace approval, and a staged SOL handoff. Its canonical
> specification—not this directory—defines authority. No stage is scheduled or
> running.

## Records

- [Overlap-first V1 evidence localization](overlap_v1_evidence_localization.md):
  owner-approved R0/R1 versus M0/M1 staged POC; HierDoc H0/H1 and A1 are
  reserved for a later answer-sufficiency experiment; no SOL stage is active.
- [Overlap-first V1 segment-reranker baseline](overlap_v1_segment_reranker_baseline.md):
  retained predecessor record now subsumed as mandatory R0/R1 controls.

## Proposal versus experiment

- A file under `../proposals/` records an architecture hypothesis and a fair
  way to test it.
- A file here records one concrete, owner-approved experiment with frozen
  inputs, controls, metrics, execution state, and retained evidence.
- A paper record under `../papers/` records primary-source facts and our
  interpretation of a published architecture.

Do not convert a proposal into an experiment record until the user approves
the experiment and its scope.

## Required experiment record

Each experiment file should include:

- experiment ID, date, owner approval, and status;
- linked paper records, proposal records, and binding project specification;
- falsifiable hypothesis;
- control and treatment definitions;
- frozen candidate revision, dataset revision, splits, labels, and prompts;
- preregistered metrics and decision gates;
- environment, commands, checkpoint identifiers, and artifact locations;
- actual execution state and blockers;
- retained quantitative results and verification evidence;
- interpretation, including alternative explanations; and
- whether the result changes any canonical architecture decision.

Use statuses such as `approved`, `scheduled`, `running`, `blocked`,
`completed`, or `rejected`. A canonical project specification changes only
through a separate explicit approval and edit.
