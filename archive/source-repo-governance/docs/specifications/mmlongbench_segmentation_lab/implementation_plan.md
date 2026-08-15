# MMLongBench Unstructured Segmentation Lab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a static local website over the retained unstructured MMLongBench OCR pages in which a human manually executes the canonical page-to-halves-to-quadrants evidence-routing procedure under replaceable segmentation and boundary-extension strategies.

**Architecture:** Python validates the retained 313-page unstructured corpus and its 306 successful OCR outputs, applies a registered segmentation strategy, and precomputes finite routing graph templates for geometry and reading-order extension modes. The preserved `deepseek_semantic_v1` baseline and additive `deepseek_semantic_hybrid_v1` comparison use the same routing graph and browser state machine. A vanilla-JavaScript static viewer loads one page graph at a time and reveals conditional candidates according to human `yes`/`no` decisions.

**Tech Stack:** Python 3, existing PyMuPDF/document-parsing modules, pytest, static HTML/CSS, vanilla JavaScript, browser `localStorage`.

## Global Constraints

- Treat `docs/specifications/hierarchical_evidence_routing/design.md` as authoritative.
- Never crop a DeepSeek unit during an indiscriminate split.
- Preserve complete semantic text segments in `deepseek_semantic_v1`.
- In `deepseek_semantic_hybrid_v1`, keep uninterrupted semantic text sections
  whole and split a section into contiguous routing runs only when another
  DeepSeek unit interrupts its complete-page reading order.
- Keep every image, figure, and table atomic in the hybrid; do not create
  image-caption pairs.
- Split page to horizontal halves, then active candidates vertically to quadrants, then stop.
- Put each complete atom crossing a cut in both raw children and suppress
  additional one-atom extension for that split.
- Apply one-atom boundary extension only when the raw children are disjoint;
  geometry and reading-order modes select the borrowed pair in that case.
- Reveal `C` only after both routing children are `yes`.
- Reveal nonempty `A-C` and `B-C` only after `C` is `yes`.
- Preserve every positive branch and the page/parent fallback.
- Keep geometry and reading-order extension sessions separate.
- Keep human evidence highlights independent of routing decisions.
- Work locally on `main`; the retained OCR bundle is already available and no
  SOL action or model inference is required.
- Add no frontend framework, database, authentication, or model call.

---

## Implemented hybrid follow-up

The additive hybrid design and resizable-workspace work were implemented after
the original six-task plan. The complete task history and verification
commands are in
[`hybrid_routing_resizable_workspace_plan.md`](hybrid_routing_resizable_workspace_plan.md).
The consolidated contract is in [`design.md`](design.md).

The hybrid strategy:

- obtains the same canonical DeepSeek semantic sections as the baseline;
- partitions each semantic text section into maximal runs that are consecutive
  in the complete page reading order;
- leaves uninterrupted semantic sections whole;
- uses standalone visual routing atoms and terminal candidates;
- retains omitted caption-like text as standalone text rather than pairing it
  with an image;
- covers every source unit; and
- then uses the unchanged crossing, boundary-extension, `C`, fallback,
  deduplication, and stopping rules.

The accompanying viewer follow-up bounds the desktop workspace to the viewport
and adds a persisted page/routing divider. Resizing changes the visible page
viewport but never rescales the rendered page or its overlays.

---

### Task 1: Define strategy and routing-graph schemas

**Files:**
- Create: `scripts/mmlongbench/segmentation_lab/__init__.py`
- Create: `scripts/mmlongbench/segmentation_lab/models.py`
- Create: `scripts/mmlongbench/segmentation_lab/strategies.py`
- Test: `tests/test_mmlongbench_segmentation_lab_strategies.py`

**Interfaces:**
- Produces: `RoutingAtom`, `TerminalCandidate`, `CandidateNode`, `RoutingGraph`, `SegmentationResult`, `SegmentationStrategy`, `DeepSeekSemanticV1`, and `get_strategy(strategy_id: str)`.
- Consumes: `build_deepseek_semantic_sections` from `scripts.document_parsing.semantic_sections`.

- [ ] Write tests proving that `DeepSeekSemanticV1` preserves all source unit IDs, emits complete semantic/visual atoms, records a version/configuration hash, and rejects duplicate or unknown member IDs.
- [ ] Run `python -m pytest -q tests/test_mmlongbench_segmentation_lab_strategies.py --tb=short` and confirm the new imports fail.
- [ ] Implement immutable dataclasses with explicit `to_dict()` methods. Use tuples for member IDs and boxes so fingerprints and serialized output are deterministic.
- [ ] Implement `DeepSeekSemanticV1.build(page_id, atomic_units)` by delegating grouping to the canonical builder, emitting semantic text sections plus every complete visual unit as routing atoms, and preserving canonical semantic sections/visual bundles as terminal candidates.
- [ ] Implement an explicit registry dictionary keyed by `strategy_id`; unknown IDs raise `ValueError`.
- [ ] Rerun the focused tests and confirm they pass.
- [ ] Commit only Task 1 files with `git commit -m "feat: define segmentation lab strategies"`.

### Task 2: Implement deterministic graph construction

**Files:**
- Create: `scripts/mmlongbench/segmentation_lab/routing_graph.py`
- Test: `tests/test_mmlongbench_segmentation_lab_routing.py`

**Interfaces:**
- Consumes: `RoutingAtom`, `CandidateNode`, and `RoutingGraph` from Task 1.
- Produces: `build_routing_graph(page_id, atoms, extension_mode) -> RoutingGraph` and `validate_routing_graph(graph) -> None`.

- [ ] Write fixtures containing disjoint text atoms across a horizontal boundary, disjoint consecutive text atoms across a vertical boundary, two full-height images, a boundary-crossing semantic atom, and an empty exclusive case.
- [ ] Write failing tests proving that a crossing semantic atom is retained in both raw children and suppresses extension, while disjoint children receive exactly one atom per side under geometry and reading-order selection. Also cover canonical fingerprints, `A/B/C` membership, `A-C/B-C`, identical-child deduplication, axis advancement, quadrant stopping, and graph strict progress.
- [ ] Run `python -m pytest -q tests/test_mmlongbench_segmentation_lab_routing.py --tb=short` and confirm failure because graph construction is absent.
- [ ] Implement normalized union-of-member-box geometry, deterministic initial side assignment, whole-atom inclusion in both children when an atom crosses the cut, and `shared_crossing` detection.
- [ ] Implement the shared-first decision: if `shared_crossing` is nonempty, use the raw children without extension; otherwise invoke the selected extension policy.
- [ ] Implement disjoint-only geometry extension using orthogonal overlap, primary-axis gap, reading order, and stable atom ID in that priority order.
- [ ] Implement disjoint-only reading-order extension using consecutive cross-boundary atoms, then geometry and stable atom ID as tie-breakers.
- [ ] Build hidden derivation recipes for potential `C`, `A-C`, and `B-C` membership; the browser must instantiate the candidate nodes only after their required decisions occur.
- [ ] Enforce identical-candidate deduplication, empty-exclusive omission, no repeated split operation, horizontal-to-vertical advancement, vertical-to-terminal advancement, and maximum quadrant depth.
- [ ] Rerun the focused routing tests and confirm they pass.
- [ ] Commit only Task 2 files with `git commit -m "feat: build segmentation routing graphs"`.

### Task 3: Build and validate the unstructured viewer manifest

**Files:**
- Create: `scripts/mmlongbench/segmentation_lab/viewer_bundle.py`
- Create: `scripts/mmlongbench/build_segmentation_lab.py`
- Test: `tests/test_mmlongbench_segmentation_lab_bundle.py`

**Interfaces:**
- Consumes: `get_strategy()` and `build_routing_graph()` from Tasks 1 and 2.
- Produces: `build_site(pilot_root, result_root, viewer_source, site_dir, strategy_ids, document_names=None, page_ids=None) -> dict`.

- [ ] Write tests with a temporary two-document fixture containing several successful pages and one explicitly excluded page. Assert that document/page navigation, page identity, strategy metadata, both extension graphs, page asset references, and explicit selection metadata are preserved.
- [ ] Write tests that reject missing OCR pages, mismatched page identities, unknown strategies, invalid graph members, and output directories that already exist.
- [ ] Run `python -m pytest -q tests/test_mmlongbench_segmentation_lab_bundle.py --tb=short` and confirm the builder imports fail.
- [ ] Implement strict JSONL loading and identity validation by adapting, not duplicating, the safe helpers in `scripts/mmlongbench/viewer/viewer_bundle.py` where practical.
- [ ] Emit a compact `manifest.json` plus one JSON payload per displayed page. Re-render page PNGs from the validated source PDFs using the existing viewer's PyMuPDF identity checks, and copy raw OCR using validated result-relative paths.
- [ ] Add CLI arguments `--pilot-root`, `--result-root`, `--viewer-source`, `--site-dir`, repeatable `--strategy`, repeatable `--document`, and repeatable `--page-id`.
- [ ] Record corpus snapshot hashes, selected document/page scope, strategy versions, extension modes, 306 packaged pages, and 7 explicit exclusions in the build summary.
- [ ] Rerun the focused bundle tests and confirm they pass.
- [ ] Commit only Task 3 files with `git commit -m "feat: package segmentation lab data"`.

### Task 4: Implement the browser routing-session controller

**Files:**
- Create: `viewer/mmlongbench_segmentation_lab/routing_session.js`
- Test: `tests/test_mmlongbench_segmentation_lab_static.py`

**Interfaces:**
- Consumes: one precomputed `RoutingGraph` and the active review/page/strategy/extension session key.
- Produces: `createSession`, `setDecision`, `expandCandidate`, `visibleCandidates`, `exportSession`, and `importSession` browser functions.

- [ ] Write static-source assertions for the required exported functions and prohibited network/model endpoints.
- [ ] Define browser-test fixtures for exactly-one-positive, both-negative, both-positive/`C`-negative, both-positive/`C`-positive, empty exclusive, identical candidate, and cached-decision reuse.
- [ ] Run `python -m pytest -q tests/test_mmlongbench_segmentation_lab_static.py --tb=short` and confirm the missing source fails.
- [ ] Implement a pure state transition module that waits for both sibling decisions, reveals conditional nodes only through allowed transitions, retains positive nodes, removes negative nodes from the active list, and appends every transition to an audit log.
- [ ] Implement schema-versioned `localStorage`, nonmutating JSON import validation, and JSON export.
- [ ] Ensure evidence-highlight unit IDs live in a separate field and are never read by routing transitions.
- [ ] Run the focused static tests and confirm they pass.
- [ ] Commit only Task 4 files with `git commit -m "feat: add manual routing session state"`.

### Task 5: Build the static review interface

**Files:**
- Create: `viewer/mmlongbench_segmentation_lab/index.html`
- Create: `viewer/mmlongbench_segmentation_lab/app.js`
- Create: `viewer/mmlongbench_segmentation_lab/styles.css`
- Create: `viewer/mmlongbench_segmentation_lab/README.md`
- Modify: `tests/test_mmlongbench_segmentation_lab_static.py`

**Interfaces:**
- Consumes: generated `manifest.json`, per-page JSON, page PNGs, and Task 4 routing-session functions.
- Produces: document/page navigation, overlays, candidate cards, decisions, audit history, and import/export controls.

- [ ] Extend static tests to require document/page/strategy/extension selectors, review-name input, overlay toggles, evidence-highlight mode, candidate board, decision buttons, retained-fallback strip, audit drawer, and reset/import/export controls.
- [ ] Run the focused static tests and confirm they fail against the absent UI.
- [ ] Implement lazy page loading and deep-link parameters for document, page, strategy, extension mode, and review ID.
- [ ] Render the full page once in the reference pane and use normalized SVG overlays for units, routing atoms, cuts, candidate members, and visual-only evidence highlights.
- [ ] Render all positive and pending candidates in a horizontally scrollable board. Keep positive ancestors in the retained-fallback strip; remove `no` candidates from the active board while preserving their audit events.
- [ ] Disable illegal controls until the routing-session controller exposes the corresponding transition.
- [ ] Add a concise README with the exact build and `python -m http.server` commands.
- [ ] Rerun the focused static tests and confirm they pass.
- [ ] Commit only Task 5 files with `git commit -m "feat: add MMLongBench segmentation lab UI"`.

### Task 6: Add integration coverage and repository navigation

**Files:**
- Create: `tests/test_mmlongbench_segmentation_lab_integration.py`
- Modify: `agent-context/modules/viewer.md`
- Modify: `agent-context/CURRENT_TASK.md`
- Modify: `agent-context/INDEX.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: the completed builder and static viewer.
- Produces: one generated fixture site proving the end-to-end contract and discoverable repository commands.

- [ ] Write an integration test that builds a fixture site, validates all referenced assets, loads both extension graphs, walks every state-machine outcome, confirms that all 306 packaged pages are navigable, and confirms that the 7 audited failures are not treated as successful pages.
- [ ] Run `python -m pytest -q tests/test_mmlongbench_segmentation_lab_integration.py --tb=short` and confirm it fails before final integration wiring.
- [ ] Add focused build/serve commands and the design pointer to repository navigation without duplicating the full specification.
- [ ] Run the four new segmentation-lab test modules together and confirm they pass.
- [ ] Run `python -m pytest -q tests/test_semantic_sections.py tests/test_mmlongbench_ocr2_viewer_bundle.py tests/test_mmlongbench_segmentation_lab_strategies.py tests/test_mmlongbench_segmentation_lab_routing.py tests/test_mmlongbench_segmentation_lab_bundle.py tests/test_mmlongbench_segmentation_lab_static.py tests/test_mmlongbench_segmentation_lab_integration.py --tb=short` and confirm all selected regression tests pass.
- [ ] Run `git diff --check` and inspect `git status --short` so generated sites and local annotation exports are not accidentally tracked.
- [ ] Commit only Task 6 files with `git commit -m "test: verify segmentation lab workflow"`.

## Final verification

- [ ] Build a small explicit document/page slice from `sol_results/mmlongbench_ocr2_unstructured_313/20260717T182300Z` into an ignored `tmp/` site directory.
- [ ] Serve it locally and manually verify one path for each `A/B/C` outcome in both extension modes.
- [ ] Verify a boundary-crossing atom appears in both raw children without any
  extra borrowed atom, and verify a disjoint split borrows exactly one atom per
  side under each extension mode.
- [ ] Confirm a named review retains independent state while switching pages.
- [ ] Confirm a strategy switch does not reuse incompatible decisions.
- [ ] Confirm exported state imports into the same corpus and is rejected under a changed corpus or strategy version.
- [ ] Confirm the browser makes no request except local static assets.
- [ ] Record the exact successful command and generated-site path in `agent-context/CURRENT_TASK.md` only while this implementation remains active.

## Potential improvements

These are deliberately outside the implemented plan.

### Orthogonal full-page rescue after two horizontal negatives

Potential behavior:

- When top and bottom are both distinct proper page refinements and both are
  explicitly `no`, retain the page fallback and attempt one full-page vertical
  split.
- Apply the existing whole-atom, crossing, extension, fingerprint,
  deduplication, and strict-progress rules without a new segmentation policy.
- Reveal only distinct left/right candidates.
- If both vertical candidates are `no` or the split is uninformative, stop
  geometric localization and keep only the page fallback.
- Record an explicit rescue trigger in graph provenance and the audit trail.

Before implementation:

- [ ] Decide whether rescue is mandatory only for known-positive/gold pages or
  also enabled for arbitrary retrieved pages.
- [ ] Add routing-graph and controller tests distinguishing true
  both-horizontal-negative rescue from the existing case where one horizontal
  child reuses the page fingerprint.
- [ ] Measure rescue frequency, recovered positive branches, added distinct
  decisions, and false-positive exposure during manual review.
- [ ] Obtain explicit approval before changing the authoritative state
  machine.
