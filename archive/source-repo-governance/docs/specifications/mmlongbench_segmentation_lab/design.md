# MMLongBench Unstructured Segmentation Lab

## Status and authority

This document specifies a local, interactive website for inspecting the
retained unstructured MMLongBench-Doc OCR pages and manually exercising the
hierarchical evidence-routing procedure.

The routing behavior in
[`../hierarchical_evidence_routing/design.md`](../hierarchical_evidence_routing/design.md)
is authoritative. If this website specification and that routing contract ever
conflict, the routing contract wins and this specification must be updated.

The additive hybrid strategy and resizable-workspace implementation are also
recorded in
[`hybrid_routing_resizable_workspace_plan.md`](hybrid_routing_resizable_workspace_plan.md).
That follow-up plan is implementation history; the behavioral contract is
consolidated in this document.

All required OCR artifacts are already retained locally. A fresh agent working
on `main` on this computer builds and serves the segmentation lab locally. No
SOL work or model inference is required.

## Purpose

The website has four connected purposes:

1. Display the 306 successful pages from the retained unstructured
   MMLongBench OCR run.
2. Show the complete DeepSeek-OCR-2 units and the semantic sections produced by
   a selected segmentation strategy.
3. Let a human act as the binary reranker by assigning `yes` or `no` to every
   candidate created by the exact hierarchical-routing procedure.
4. Make segmentation and boundary-extension strategies replaceable so future
   experiments can change the rules without rewriting the viewer.

The website is a visual test ground. Human decisions simulate reranker output;
they do not become training labels unless explicitly exported and used by a
later, separately approved dataset pipeline.

## Dataset scope

The initial input and result are already present locally:

- input: `pilot_data/mmlongbench_ocr2_unstructured_313/`;
- OCR result:
  `sol_results/mmlongbench_ocr2_unstructured_313/20260717T182300Z/`;
- 10 manuals, brochures, handbooks, guidebooks, and device documents;
- 313 planned pages;
- 306 validated and packaged OCR pages;
- 7 explicitly excluded OCR failures.

The first website is page-centric. A build may intentionally select a smaller
document or page slice for rapid development, but selection must be explicit
and recorded in the generated manifest.

The builder consumes:

- `documents.jsonl` for document identity and page counts;
- `page_plan.jsonl` for document/page identity;
- the retained `pages.jsonl`, `runs.jsonl`, and `segments.jsonl` artifacts;
- the source PDFs used to reproduce page PNGs locally;
- the raw OCR artifact for each displayed page.

Follow the retained viewer's rendering contract: render the source PDF page at
the recorded scale, validate dimensions and hash against `pages.jsonl`, and
record any platform render drift rather than assuming PNG files were returned.

`page_plan.jsonl` remains the pre-render input. Returned `pages.jsonl` remains
the authoritative rendered-page manifest. The two must not be conflated.

## Chosen architecture

Use a static, generated website following the existing MMLongBench viewer
pattern:

```text
unstructured corpus + retained OCR
              |
              v
Python manifest builder
  - validates identities
  - applies a segmentation strategy
  - builds routing graph templates
              |
              v
static site
  - vanilla HTML/CSS/JavaScript
  - page PNGs and per-page JSON
  - browser-local review state
```

The Python builder is authoritative for segmentation, unit assignment,
candidate fingerprints, intersections, subtraction, and strict-progress
validation. The browser does not independently reinterpret bounding boxes or
reimplement segmentation rules.

The builder may precompute hidden derivation recipes containing the member IDs
that a future intersection or exclusive candidate would have. These recipes
are not candidate nodes and cannot receive decisions. The browser instantiates
and reveals `C` only after the human marks both `A` and `B` `yes`, and
instantiates and reveals `A - C` and `B - C` only after `C` is marked `yes`.

This architecture is preferred over:

- browser-only routing, which would duplicate the Python research logic and be
  difficult to test;
- a stateful web backend, which is unnecessary for a local visual aid and
  would make the retained viewer harder to reproduce.

## Lightweight implementation boundary

The first implementation uses:

- Python and PyMuPDF in the existing build pipeline;
- the current semantic-section builder;
- static HTML, CSS, and vanilla JavaScript;
- browser `localStorage` for automatic session recovery;
- JSON export and import for durable human annotations.

Do not add React, a JavaScript build system, a database, authentication, model
inference, or a production web server.

## Core data model

### DeepSeek unit

Every original DeepSeek unit remains recoverable with:

- stable `unit_id` within its page;
- page ID;
- type;
- reading order;
- content;
- normalized bounding box;
- original OCR provenance.

DeepSeek units are never cropped by an indiscriminate geometric split.

### Segmentation strategy

A segmentation strategy converts the complete DeepSeek units for one page into
routing atoms.

The strategy interface is conceptually:

```python
class SegmentationStrategy(Protocol):
    strategy_id: str
    version: str

    def build(
        self,
        page_id: str,
        atomic_units: Sequence[Mapping[str, object]],
    ) -> SegmentationResult: ...
```

`SegmentationResult` contains:

- `routing_atoms`;
- `terminal_candidates`;
- semantic diagnostics;
- source unit IDs for every atom;
- strategy ID, version, and configuration hash.

The initial strategy is `deepseek_semantic_v1`. It delegates to the canonical
`build_deepseek_semantic_sections` implementation and exposes:

- reliable complete semantic text sections as text routing atoms;
- every complete image, figure, and table unit as its own visual routing atom;
- paragraph/list routing fallbacks only where the semantic contract requires
  them;
- the canonical semantic sections and visual bundles as terminal candidates
  available after quadrant routing.

Future strategies register through the same interface. The viewer never
hardcodes the assumption that `deepseek_semantic_v1` is the only strategy.

The additive comparison strategy `deepseek_semantic_hybrid_v1` leaves
`deepseek_semantic_v1` unchanged. It first obtains the same canonical semantic
text sections, then partitions a section into maximal runs whose DeepSeek units
are consecutive in the complete page reading order. A visual or any other
unit between two section members therefore breaks the routing run. Uninterrupted
sections remain whole.

The hybrid uses every image, figure, and table as a standalone atomic routing
unit and terminal candidate. It does not create image-caption pairs.
Caption-like text excluded from a canonical text section remains available as
its own text routing unit. After this additive atom-construction step, the
existing routing graph, crossing, `C`, extension, fallback, and stopping rules
run without special cases.

### Routing atom

A routing atom is indivisible during page-to-half and half-to-quadrant splits.
It contains one or more DeepSeek unit IDs and uses the union of the actual
member boxes for geometry. It never uses a large rectangular envelope as its
claimed evidence area.

### Candidate

A candidate is a set of routing atoms represented by its underlying DeepSeek
unit IDs. Its canonical evidence fingerprint is:

```text
page_id + sorted(member_unit_ids)
```

The review-session key additionally includes:

```text
review_id + page_id + strategy_id + strategy_version + extension_mode
```

This keeps human decisions separate across named manual reviews and
experimental modes without changing evidence identity.

### Routing graph template

Each page, segmentation strategy, and extension mode produces a finite graph
containing:

- the page candidate;
- possible half-level children;
- possible quadrant-level children;
- hidden intersection derivation recipes;
- hidden exclusive-candidate derivation recipes;
- terminal typed candidates available after quadrant-level routing;
- provenance for every split, extension, intersection, and subtraction,
  including `crossing_atom_ids`, whether extension was applied, and the reason
  it was applied or suppressed.

The graph must record which nodes are conditional. JavaScript may reveal a
conditional node only through the transitions defined below.

## Boundary-extension modes

The lab initially supports two selectable modes. Sessions and decisions are
stored separately for each mode.

### Geometry mode

Initial membership is determined from complete routing-atom geometry. At each
cut:

1. Atoms crossing the cut are included in both raw children.
2. Compute `shared_crossing = raw_A intersection raw_B`.
3. If `shared_crossing` is nonempty, use the raw children without adding
   another atom.
4. Only if the raw children are disjoint, identify the nearest geometrically
   adjacent pair across the boundary.
5. Prefer pairs with orthogonal-axis overlap; break ties by primary-axis gap,
   then reading order, then stable atom ID.
6. For pair `(atom_from_first, atom_from_second)`, add `atom_from_second` to the
   first child and add `atom_from_first` to the second child.

Thus, when two nearby text atoms sit on opposite sides of a vertical cut and
the raw children are disjoint, both extended children contain both atoms. The
same rule applies to a horizontal cut.

Conditional extension is limited to one routing atom from each side. A
crossing atom already provides natural overlap and suppresses extension for
that split; no additional neighboring atom is copied.

### Reading-order mode

Initial membership still uses complete atom geometry. If any atom crosses the
cut, it belongs to both raw children and no additional extension is applied.
Only when the raw children are disjoint does boundary repair select the one
cross-boundary atom pair that is consecutive in DeepSeek reading order, with
geometric distance and stable atom ID as deterministic tie breakers.

For a horizontal cut, this will commonly match geometry because document
reading order normally proceeds top to bottom. For a vertical cut, it can preserve two
consecutive text atoms that geometry would place on different sides.

The same one-atom-per-side limit applies. Reading-order mode never crops a
unit, merges unit identity, or changes the underlying semantic segmentation.

The geometry and reading-order modes therefore differ only in how they select
the borrowed atom pair for a disjoint split. The legacy policy that extended
every split is retained in the canonical design's decision history and may be
implemented later as an ablation, but it is not the lab default.

### Strategy selection in the interface

The initial interface uses selectors for:

- segmentation strategy;
- boundary-extension mode.

Changing either selector loads or restores a separate review session. The
first implementation does not need to display geometry and reading-order
boards simultaneously. The data model must allow a later synchronized
comparison view without rebuilding the candidate graphs.

## Authoritative human-routing state machine

The page candidate begins as `yes`. It remains available as the page fallback
and as the permanent visual reference.

For every indiscriminate split, the interface follows this exact order.

### 1. Reveal routing children

Reveal the two routing children. They are raw children when natural crossing
overlap exists and conditionally extended children when the raw children are
disjoint:

```text
A = first routing child
B = second routing child
```

At page level they are top and bottom. At half level they are left and right.
Both begin `pending` and receive independent `Yes` and `No` controls.

### 2. Wait for both decisions

Do not derive an outcome until both `A` and `B` have explicit human decisions.

### 3. Exactly one `yes`

If exactly one child is `yes`:

- keep the positive child visible and eligible for descent;
- remove the negative child from the active board;
- do not reveal or request a decision for `C`.

### 4. Both `no`

If both children are `no`:

- remove both from the active board;
- stop descent from those children;
- retain the previously positive parent as the fallback.

### 5. Both `yes`

If both children are `yes`:

- keep both visible;
- reveal `C = A intersection B` alongside them;
- require an independent `yes` or `no` decision for `C`.

`C` must not become visible or decision-bearing before both child decisions are
`yes`.

### 6. `C` is `no`

If `C` is `no`:

- remove `C` from the active board;
- retain `A` and `B` as positive branches;
- permit both to descend.

### 7. `C` is `yes`

If `C` is `yes`:

- keep `C` visible as a positive branch;
- reveal nonempty `A - C` and `B - C` candidates;
- require independent decisions for the nonempty exclusive candidates;
- retain and descend every exclusive candidate marked `yes`;
- remove every exclusive candidate marked `no`.

Empty exclusive candidates are never displayed or scored.

### 8. Continue to quadrants

Every active half-scale branch may be split vertically with the same state
machine. More than one branch may remain positive and descend. All positive
branches remain visible next to one another; only candidates marked `no` leave
the active board.

An intersection may not repeat the split-and-extension operation that created
it. A half-level intersection advances to the vertical split. A quadrant-level
intersection advances to terminal typed candidates.

### 9. Stop indiscriminate splitting

Indiscriminate splitting stops after quadrant-scale candidates. The selected
terminal-candidate provider then exposes the complete semantic text sections,
visual atoms, and required paragraph fallbacks contained in each positive
quadrant-scale branch.

The initial terminal provider does not invent new image-caption pairs. Future
providers can add alternative fine-grained hypotheses through a separate
versioned interface.

## Deduplication and strict progress

The implementation must enforce every loop-prevention rule from the canonical
design:

1. Deduplicate candidates by canonical fingerprint before display.
2. Cache and reuse the human decision for an identical fingerprint within the
   same review, strategy, and extension-mode session.
3. If `A` and `B` are identical, display the candidate once and do not reveal
   an identical `C`.
4. Mark a split uninformative when it produces no distinct proper refinement.
5. Every descent must reduce membership, advance axis/level, or transition to
   typed candidates.
6. Advance an uninformative horizontal split to the vertical split.
7. Advance an uninformative vertical split to terminal typed candidates.
8. Never display empty exclusive candidates.
9. Preserve the page-to-halves-to-quadrants depth limit.

Because the user is the scorer, there is no model-call budget in the lab.
Nevertheless, the graph must report the number of distinct decisions so that
future experiments can estimate reranker cost.

## Potential improvements

The items in this section are research candidates, not current routing
behavior. They require explicit approval, implementation-plan updates, and
focused regression coverage before they become part of the state machine.

### Orthogonal full-page rescue after two negative horizontal children

The current both-negative rule stops descent from two distinct horizontal
children and retains the previously positive page fallback. A potential
high-recall alternative is to attempt one full-page vertical split when both
distinct horizontal children are explicitly `no`.

The proposed rescue would:

1. trigger only after the top and bottom candidates are both distinct proper
   refinements of the page and both have explicit `no` decisions;
2. retain the positive page fallback throughout;
3. split the original page vertically exactly once using the same whole-atom,
   crossing, extension, fingerprint, and deduplication rules;
4. reveal and score only distinct left/right candidates;
5. stop geometric localization and retain the page fallback if both vertical
   candidates are `no` or the vertical split is uninformative; and
6. record an explicit audit reason such as
   `orthogonal rescue after horizontal negatives`.

This rescue can recover evidence whose relevance depends on side-by-side,
column, or full-height visual context that top/bottom candidates obscure. It
costs at most two additional distinct candidate scores. It can also increase
cost and false-positive exposure on arbitrary nongold retrieved pages, so the
strongest initial use case is a known-positive or gold page where two
horizontal negatives indicate a localization failure rather than reliable
page rejection.

This proposal is separate from the implemented strict-progress behavior where
one horizontal child is identical to the page. In that existing case, the
parent is reused and advances to the vertical axis because the horizontal split
did not produce two proper refinements; no both-negative rescue is involved.

## Website layout

The interface uses four compact regions.

The desktop workspace occupies exactly the available browser viewport below
the header. Controls, reference content, active candidates, and audit history
scroll inside their own panes; additional decisions and candidates must not
increase the document height.

### Header and navigation

Show:

- document filename;
- source page number and document page count;
- document selector and previous/next document controls;
- page selector and previous/next successful-page controls;
- a short editable review name or target description;
- segmentation-strategy selector;
- boundary-extension selector;
- reset, export, and import controls.

Switching between pages preserves a separate session for each named
review/page pair.

### Reference-page pane

Show the complete document page at stable scale with switchable overlays for:

- atomic DeepSeek units;
- semantic routing atoms;
- current geometric cut;
- selected candidate membership;
- optional human evidence highlights.

Clicking an atomic unit in evidence-highlight mode toggles a visual highlight.
These highlights are stored separately from routing decisions and never affect
candidate construction or state transitions.

A draggable vertical separator changes the width allocated to the reference
pane and live routing board. Resizing changes only how much of the fixed-size
page is visible. The rendered page and overlay retain the same dimensions and
are clipped or reached through pane scrolling rather than rescaled. The
separator supports pointer, mouse, and keyboard operation and persists its
width independently from review state.

### Active candidate board

Display all retained positive and pending candidates in a horizontally
scrollable responsive row. Each candidate card shows:

- candidate label and provenance (`page`, `A`, `B`, `C`, `A-C`, `B-C`, or
  terminal type);
- hierarchy level and human-readable parent type;
- member-unit count;
- a page-position preview with nonmember units dimmed and member units outlined;
- `Yes` and `No` controls when pending;
- `Split` or `Show terminal candidates` when eligible;
- an explanation of why the candidate was revealed.

Positive ancestors remain available in a compact retained-fallback strip even
after descendants are revealed. Candidates marked `no` disappear from the
active board but remain in the audit history.

### Audit drawer

Record an ordered event log containing:

- candidate revealed;
- human decision;
- candidate removed;
- intersection revealed;
- exclusive candidate revealed;
- split declared uninformative;
- cached decision reused;
- terminal candidates exposed.

This log makes every manually simulated traversal reproducible.

Visible candidate cards, reference summaries, and audit entries use
human-readable candidate types rather than internal candidate fingerprints,
segment IDs, or routing-atom IDs. The exact identifiers remain in browser
session state and JSON export for machine-level reproducibility.

## Review-state persistence

Autosave review state to `localStorage` under a schema-versioned key. Provide
JSON export and import so annotations can move between browsers or become an
input to a later approved analysis pipeline.

An export contains:

- schema version;
- corpus snapshot identity;
- review ID, optional target description, and page ID;
- strategy ID, version, and configuration hash;
- extension mode;
- human evidence-highlight unit IDs;
- candidate decisions keyed by fingerprint;
- ordered event log;
- creation and update timestamps.

Import must reject a state whose corpus identity, strategy version, or page
identity does not match the active generated site. Do not silently reinterpret
old decisions under new segmentation rules.

## Generated-site organization

The builder writes an ignored site directory with:

```text
site/
  index.html
  app.js
  routing_session.js
  styles.css
  manifest.json
  pages/
    <page_id>.png
    <page_id>.json
  raw/
    <page_id>.txt
```

`manifest.json` contains corpus, strategy, document, page, and explicit build
selection indices. Per-page JSON contains OCR units, strategy outputs, and
routing graph templates. Pages are loaded on demand rather than embedding all
page payloads in one JavaScript file.

## Failure behavior

- Missing or invalid selected OCR artifacts fail the build with the page ID and
  source path. They are not silently replaced.
- Explicitly excluded OCR pages appear as unavailable only when the returned
  completion audit records their exclusion.
- Unknown segmentation strategies fail before site generation.
- Invalid candidate member IDs, duplicate IDs, non-progressing edges, or an
  illegal graph depth fail validation.
- The UI disables transitions that violate the authoritative state machine.
- Import errors leave the current browser session unchanged.
- A page with no reliable semantic section still exposes the documented atomic
  paragraph/list fallbacks and complete visual atoms.

## Acceptance criteria

The first implementation is complete only when all of the following hold:

1. A user can select a document and move through every successful OCR page.
2. The page begins `yes` and can be split into top/bottom routing candidates.
3. Exactly-one-positive, both-negative, both-positive/`C`-negative, and
   both-positive/`C`-positive outcomes follow the canonical order.
4. `C` is never displayed before both children are `yes`.
5. `A-C` and `B-C` are never displayed before `C` is `yes` and are omitted
   when empty.
6. Every positive branch remains visible; only negative candidates leave the
   active board.
7. Positive branches can continue through vertical quadrant splits.
8. No indiscriminate split occurs below quadrants.
9. Identical candidates share one decision and cannot create a loop.
10. Geometry and reading-order extension modes produce separately inspectable,
    reproducible graphs.
11. The initial DeepSeek semantic strategy can be replaced through the strategy
    registry without changing UI routing code.
12. Human evidence highlights do not alter the routing graph or decisions.
13. Review sessions survive reload and round-trip through JSON export/import.
14. The site remains static and makes no model or network calls.
15. A split with at least one naturally crossing atom applies no additional
    one-atom extension, while a disjoint split applies exactly one atom per
    side using the selected geometry or reading-order mode.
16. `deepseek_semantic_v1` remains unchanged while
    `deepseek_semantic_hybrid_v1` splits only semantic sections interrupted in
    complete-page reading order and creates no image-caption pairs.
17. Decisions never increase the desktop workspace height, and resizing the
    page/routing divider never changes the rendered page dimensions.

## Future query-linked gold-page phase

Query and gold-page navigation are intentionally deferred until the
segmentation and routing rules have been tested and refined on the available
unstructured pages.

The future adapter will:

- assign stable IDs to MMLongBench questions;
- map each question's evidence-page numbers onto retained page IDs;
- add query selection and multi-gold-page navigation;
- key review sessions by query/page as well as strategy and extension mode;
- preserve the same routing graph, manual decision state machine, and viewer
  components.

This future phase must extend the page-centric schema rather than replace it.
It is not part of the initial implementation.

## Implementation surface

The planned implementation is detailed in
[`implementation_plan.md`](implementation_plan.md). The expected source areas
are:

- `scripts/mmlongbench/segmentation_lab/` for schemas, strategies, graph
  construction, validation, and site packaging;
- `scripts/mmlongbench/build_segmentation_lab.py` for the CLI;
- `viewer/mmlongbench_segmentation_lab/` for static source assets;
- `tests/` for routing, manifest, static-viewer, and integration tests;
- ignored `tmp/` directories for generated sites.

The completed additive hybrid and resizable-workspace follow-up is detailed in
[`hybrid_routing_resizable_workspace_plan.md`](hybrid_routing_resizable_workspace_plan.md).

The implemented visual decision-tree design is specified in
[`visual_route_decision_tree_design.md`](visual_route_decision_tree_design.md).
