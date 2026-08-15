# COLPALI Architecture Registry Implementation Plan

> **Archived:** Completed on 2026-07-28. The implemented, current registry is
> `agent-context/research/architecture_registry/`. Unchecked boxes below are
> preserved as historical plan syntax, not unfinished work.

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a repository-wide, on-demand registry for reviewed paper
architectures and seed it with a primary-source-linked VGent analysis.

**Architecture:** A concise registry README provides comparison and routing,
a template fixes the schema for future entries, and one file per reviewed
paper preserves project-specific analysis without reproducing the paper.
`agent-context/INDEX.md` exposes one on-demand pointer without adding the paper
records to default active context.

**Tech Stack:** Markdown, Git, `rg`, and focused shell validation.

## Global Constraints

- Keep the registry project-neutral and applicable across the broader COLPALI
  repository.
- Treat registry content as nonbinding research memory.
- Do not modify the Evidence-DINO-Units Stage 0–4 execution contract or
  authorize Stage 5+ activity.
- Keep paper-reported facts, repository inference, unvalidated proposals, and
  locally tested evidence visibly separate.
- Do not create unreviewed placeholder paper files.
- Preserve the existing untracked `work/` directory and unrelated repository
  content.
- Use VGent arXiv `2512.11099v1`, dated 2025-12-11, as the reviewed primary
  source for the initial record.

---

### Task 1: Create and index the architecture registry

**Files:**

- Create:
  `agent-context/research/architecture_registry/README.md`
- Create:
  `agent-context/research/architecture_registry/TEMPLATE.md`
- Create:
  `agent-context/research/architecture_registry/papers/vgent-2512.11099.md`
- Modify:
  `agent-context/INDEX.md`

**Interfaces:**

- Consumes:
  `../../specifications/completed/2026-07-28-colpali-architecture-registry-design.md`
  and the VGent primary paper at `https://arxiv.org/html/2512.11099`.
- Produces:
  an on-demand registry entry point, a reusable paper-record schema, one
  verified paper record, and a discoverable index link.

- [ ] **Step 1: Confirm the active repository boundary**

Run:

```bash
git branch --show-current
git status --short
sed -n '1,220p' agent-context/INDEX.md
sed -n '1,240p' \
  archive/specifications/completed/2026-07-28-colpali-architecture-registry-design.md
```

Expected:

- branch is `main`;
- `work/` may remain as an unrelated untracked path;
- the approved design is present;
- no existing registry path conflicts with the proposed files.

- [ ] **Step 2: Create the registry README**

Create `agent-context/research/architecture_registry/README.md` with:

1. a title and one-paragraph purpose;
2. an explicit notice that the registry is nonbinding and on-demand;
3. a “How to use this registry” section directing readers to the comparison
   ledger before opening detailed records;
4. the twelve capability categories from the approved design;
5. applicability definitions for `direct`, `adjacent`, and
   `architectural_analogue`;
6. evidence-level definitions for `paper_reported`,
   `repository_inference`, `unvalidated_proposal`, and
   `repository_tested`;
7. registry-status definitions for `unreviewed`, `reviewed`, `promising`,
   `scheduled`, `tested`, `adopted`, `rejected`, and `superseded`;
8. an adoption-boundary section stating that project-owner approval and a
   separate canonical-specification change are required before implementation;
9. a comparison table containing one VGent row; and
10. a relative link to `papers/vgent-2512.11099.md`.

The VGent ledger row must state:

- actual task: natural-image multi-target visual grounding;
- mechanism: frozen MLLM memory plus detector-proposal decoder;
- applicability: `architectural_analogue`;
- possible consumers: evidence candidate/unit selection and multi-region set
  reasoning;
- promise: set-aware selection over high-recall proposals;
- limitation: no document evidence, OCR/layout, or sufficiency evaluation;
- evidence level: `paper_reported` for the actual method and
  `repository_inference` for COLPALI fit; and
- status: `promising`.

- [ ] **Step 3: Create the reusable paper template**

Create `agent-context/research/architecture_registry/TEMPLATE.md` with these
required sections:

```markdown
# <Paper title>

## Registry metadata
## Paper link and factual capsule
## Why this paper attracted attention
## Our analysis and interpretation
## Reusable mechanisms
## Possible COLPALI integration points
## Required adaptations
## Advantages relative to the current subsystem
## Concerns and failure risks
## Open questions
## Smallest decisive experiment
## Analysis and interpretation log
## Adoption status and decision history
```

Under `Registry metadata`, require paper identity, reviewed version/date,
applicability, capability tags, possible project consumers, evidence levels,
registry status, and last-reviewed date.

The template must instruct authors to:

- rely on primary sources for paper facts;
- link to the paper rather than reproduce a long summary;
- keep the factual capsule short;
- label project mappings as inference;
- identify what the paper does not demonstrate;
- avoid copying long passages;
- avoid claiming local reproduction without retained repository evidence; and
- add a paper file only after reviewing enough of the primary source to support
  every factual capsule claim.

- [ ] **Step 4: Write the verified VGent record**

Create
`agent-context/research/architecture_registry/papers/vgent-2512.11099.md`.

Record:

- title: “VGent: Visual Grounding via Modular Design for Disentangling
  Reasoning and Prediction”;
- paper link and arXiv version `2512.11099v1`;
- applicability: `architectural_analogue`;
- status: `promising`;
- a compact factual capsule stating that VGent performs natural-image visual
  grounding with detector proposals, frozen MLLM hidden states, a
  cross-attentive proposal decoder, and proposal self-attention;
- the distinction between object grounding and document evidence selection;
- our analysis of proposal interaction, VLM-memory cross-attention, and global
  count signals for possible document-unit selection;
- required document-specific changes: parser-derived candidates, OCR/layout
  features, complete-evidence supervision, no-evidence behavior, structural
  closure, and fixed-answerer sufficiency evaluation;
- explicit concerns and open questions;
- a future controlled experiment comparing the current direct semantic-unit
  scorer with a VGent-style candidate decoder while holding candidates,
  training data, and evaluation fixed; and
- a dated analysis log with the initial COLPALI interpretation.

Label the controlled experiment `unvalidated_proposal` and explicitly state
that it is not authorized by the current Stage 0–4 task.

- [ ] **Step 5: Add the on-demand index pointer**

Modify `agent-context/INDEX.md` by adding:

```markdown
## On-demand research

- `research/architecture_registry/README.md`: comparison ledger and detailed
  primary-source reviews of promising architectures across COLPALI. Open only
  when evaluating paper mechanisms, architecture alternatives, or future
  improvements. Registry entries are nonbinding until the relevant canonical
  project specification is separately approved and updated.
```

Place this section after “Active modules” and before “Paused and historical
material” so it is discoverable but not part of mandatory active-task context.

- [ ] **Step 6: Run focused consistency checks**

Run:

```bash
test -f agent-context/research/architecture_registry/README.md
test -f agent-context/research/architecture_registry/TEMPLATE.md
test -f \
  agent-context/research/architecture_registry/papers/vgent-2512.11099.md
rg -n \
  "architectural_analogue|paper_reported|repository_inference|unvalidated_proposal|repository_tested" \
  agent-context/research/architecture_registry
rg -n \
  "unreviewed|reviewed|promising|scheduled|tested|adopted|rejected|superseded" \
  agent-context/research/architecture_registry/README.md \
  agent-context/research/architecture_registry/TEMPLATE.md
rg -n \
  "natural-image multi-target visual grounding|not a document|not.*document understanding|does not demonstrate" \
  agent-context/research/architecture_registry/papers/vgent-2512.11099.md
rg -n \
  "research/architecture_registry/README.md" \
  agent-context/INDEX.md
git diff --check
```

Expected:

- all files exist;
- the controlled vocabularies occur in both registry surfaces;
- VGent is explicitly bounded to visual grounding and does not claim
  demonstrated document understanding;
- the index points to the registry;
- no whitespace errors are reported.

- [ ] **Step 7: Review the complete diff**

Run:

```bash
git diff -- \
  agent-context/INDEX.md \
  agent-context/research/architecture_registry/README.md \
  agent-context/research/architecture_registry/TEMPLATE.md \
  agent-context/research/architecture_registry/papers/vgent-2512.11099.md
git status --short
```

Confirm:

- only the four intended registry/index files changed;
- the design/plan commits and unrelated `work/` path are not included;
- no active task, SOL contract, or canonical architecture changed;
- every VGent-to-COLPALI statement is labeled as inference or proposal.

- [ ] **Step 8: Commit the registry**

Run:

```bash
git branch --show-current
git add -- \
  agent-context/INDEX.md \
  agent-context/research/architecture_registry/README.md \
  agent-context/research/architecture_registry/TEMPLATE.md \
  agent-context/research/architecture_registry/papers/vgent-2512.11099.md
git commit -m "docs: add COLPALI architecture registry"
```

Expected:

- branch remains `main`;
- the commit contains exactly the four intended files;
- `work/` remains untracked and untouched.
