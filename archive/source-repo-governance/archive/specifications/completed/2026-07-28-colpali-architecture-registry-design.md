# COLPALI Research Architecture Registry Design

**Status:** Completed and archived. The implemented registry is
`agent-context/research/architecture_registry/`.

**Date:** 2026-07-28

## Purpose

Create a repository-wide, on-demand research registry for promising paper
architectures and reusable mechanisms that may improve any COLPALI project.
The registry must make paper-linked project analysis discoverable without
turning unvalidated ideas into active architecture or bloating current-task
context.

The registry is broader than Evidence-DINO-Units. Evidence localization,
retrieval, routing, reranking, parsing, answering, evaluation, efficiency, and
grounded attribution may all use it.

## Location and structure

Create:

```text
agent-context/research/architecture_registry/
├── README.md
├── TEMPLATE.md
└── papers/
    └── vgent-2512.11099.md
```

Update `agent-context/INDEX.md` with an on-demand pointer to the registry.
The index entry must state that agents should open the registry only when
researching architecture alternatives, paper mechanisms, or future
improvements.

Individual active or paused project specifications may link to relevant paper
records later. The registry does not belong to any one project specification.

## Hybrid information model

### Comparison ledger

`README.md` is the short comparison and routing surface. It contains:

- the registry's purpose and nonbinding status;
- status and fit vocabularies;
- capability categories;
- a compact paper comparison table; and
- links to detailed paper records.

The comparison table records:

- paper and reviewed version;
- actual task and domain;
- reusable mechanism;
- relevant COLPALI capabilities or projects;
- applicability class;
- primary promise;
- primary limitation;
- evidence status; and
- registry status.

### Detailed paper records

Each paper receives one file named with a stable short name and arXiv ID when
available. The file links to the primary paper instead of reproducing a long
paper summary. It contains only the factual capsule required to identify the
task, mechanism, training/evaluation boundary, and what the paper actually
demonstrates.

Most of each record is reserved for COLPALI analysis:

1. why the paper attracted attention;
2. reusable mechanisms;
3. possible integration points across projects;
4. required adaptations;
5. advantages relative to the current relevant subsystem;
6. concerns, limitations, and failure risks;
7. open questions;
8. a smallest decisive experiment;
9. dated analysis and interpretation notes; and
10. adoption status and decision history.

Paper facts, repository inferences, and proposed experiments must be visibly
distinct. Architectural analogy must never be presented as demonstrated
document performance.

### Template

`TEMPLATE.md` defines the required record sections and allowed status values so
future entries remain comparable and provides explicit space for evolving
project analysis.

## Capability taxonomy

Papers may use multiple capability tags:

- page or document retrieval;
- candidate or proposal generation;
- question-conditioned representation;
- region, unit, or patch selection;
- multi-region and set reasoning;
- semantic and structural closure;
- answer sufficiency and necessity;
- supervision and pseudo-labeling;
- abstention, recovery, and uncertainty;
- parsing and document representation;
- answer generation and grounded attribution;
- token, latency, and memory efficiency; and
- evaluation and calibration.

This taxonomy is deliberately project-neutral. Project names are recorded
separately as possible consumers.

## Applicability and evidence labels

Every entry must classify applicability:

- `direct`: the paper evaluates substantially the same task and domain;
- `adjacent`: the paper evaluates a related document or multimodal task;
- `architectural_analogue`: a mechanism may transfer, but the evaluated task
  or domain is materially different.

Every project-facing claim must also identify its evidence level:

- `paper_reported`: directly stated or measured in the reviewed paper;
- `repository_inference`: a reasoned mapping to COLPALI;
- `unvalidated_proposal`: a change or experiment not yet implemented or run;
- `repository_tested`: implemented and measured locally with retained
  evidence.

## Registry statuses

- `unreviewed`: identified but the full primary source has not been reviewed;
- `reviewed`: methodology and evidence have been checked against the primary
  source;
- `promising`: reviewed and connected to a concrete COLPALI opportunity;
- `scheduled`: an approved controlled experiment exists;
- `tested`: the experiment ran and retained evidence exists;
- `adopted`: the relevant canonical project specification was explicitly
  updated and approved;
- `rejected`: a documented result or incompatibility rules out the proposal;
- `superseded`: a later entry or architecture replaces it.

Registry presence alone never means `adopted`.

## Adoption boundary

The registry is research memory, not an authority source for active execution.
A paper idea may affect implementation only after:

1. its actual task and evidence are verified from primary sources;
2. the target COLPALI component and expected advantage are explicit;
3. a controlled comparison and failure interpretation are specified;
4. the project owner approves the architecture or experiment change; and
5. the relevant canonical project specification is updated separately.

For Evidence-DINO-Units, registry entries do not modify the Stage 0–4 contract
and do not authorize Stage 5, candidate construction, model execution,
training, or benchmark acquisition.

## Initial VGent record

Seed the registry with a fully reviewed VGent record based on arXiv
`2512.11099`.

It must classify VGent as an `architectural_analogue`, not a document
understanding system. The record links to the paper, gives a compact factual
capsule of its natural-image visual-grounding task and modular
proposal-selection architecture, and then concentrates on:

- the distinction between object grounding and document evidence selection;
- how proposal self-attention and cross-attention could inform a future
  COLPALI candidate or unit selector;
- why document units, complete-evidence labels, OCR/layout features, and
  sufficiency evaluation would still be required;
- project-specific concerns and open questions;
- a minimal future comparison against the existing direct semantic-unit
  scorer, without authorizing that comparison now; and
- a dated space for later COLPALI interpretations.

Do not create placeholder paper files. Add RegionRAG, M3Grounder, and other
papers only when their primary sources have been reviewed for their own
records.

## Verification

The implementation is complete when:

- all three registry surfaces exist;
- the README and template agree on fields, statuses, and evidence labels;
- the VGent ledger row agrees with its detailed record;
- VGent is not described as a document-understanding result;
- `agent-context/INDEX.md` points to the registry as on-demand context;
- no active task or canonical architecture is silently changed;
- no unfinished placeholder sections remain; and
- a focused link/path and terminology check passes.

## Non-goals

- Rewriting the Evidence-DINO-Units canonical plan.
- Authorizing any Stage 5+ work.
- Treating paper publication claims as locally reproduced results.
- Creating a general bibliography without project relevance.
- Copying long paper text or maintaining secondary-source summaries.
- Pre-populating shallow entries for papers that have not been fully reviewed.
