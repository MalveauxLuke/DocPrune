# Hybrid Routing and Resizable Workspace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve `deepseek_semantic_v1` while adding a comparison strategy
that splits interrupted semantic text sections into contiguous routing runs,
and keep the static lab one viewport tall with a draggable page/routing
divider that never rescales the page.

**Architecture:** The new `deepseek_semantic_hybrid_v1` strategy reuses the
canonical DeepSeek semantic-section output, but constructs routing atoms from
maximal contiguous runs of each text section in the complete page reading
order. Visual units stay atomic and canonical image-caption bundles are not
used by the hybrid. The browser uses a fixed-height grid, independently
scrolling panes, and a pointer/keyboard-accessible divider whose persisted
width changes only the page viewport.

**Tech Stack:** Python 3, pytest, vanilla JavaScript, static HTML/CSS,
`localStorage`.

## Global Constraints

- Leave `deepseek_semantic_v1` behavior and identifiers unchanged.
- Do not modify the canonical DeepSeek semantic-section builder.
- Do not create image-caption pairs in the hybrid strategy.
- Preserve every DeepSeek source unit in at least one hybrid routing atom.
- Keep existing routing-graph, `C`, extension, fallback, and stopping rules
  unchanged.
- Keep the existing generated baseline site untouched.
- Work locally on `main`; do not use SOL.
- Add no production dependency.

---

### Task 1: Add the hybrid routing strategy

**Files:**
- Modify: `tests/test_mmlongbench_segmentation_lab_strategies.py`
- Modify: `scripts/mmlongbench/segmentation_lab/strategies.py`

**Interfaces:**
- Produces:
  `DeepSeekSemanticHybridV1.build(page_id, atomic_units) -> SegmentationResult`
- Registry key: `deepseek_semantic_hybrid_v1`

- [ ] **Step 1: Write failing strategy tests**

Add tests proving:

```python
heading -> text -> image -> text
```

becomes two semantic-text routing runs plus one atomic visual, while

```python
heading -> text -> text
```

remains one semantic-text routing atom. Assert that hybrid terminal candidates
contain no multi-unit visual bundle, every source unit is covered, and the
baseline result remains unchanged.

- [ ] **Step 2: Verify the tests fail for the missing strategy**

Run:

```bash
python -m pytest -q tests/test_mmlongbench_segmentation_lab_strategies.py --tb=short
```

Expected: failure importing or resolving `DeepSeekSemanticHybridV1`.

- [ ] **Step 3: Implement contiguous-run construction**

Add a helper with this contract:

```python
def _contiguous_member_runs(
    member_ids: tuple[str, ...],
    units_by_id: Mapping[str, Mapping[str, object]],
) -> tuple[tuple[str, ...], ...]:
    ...
```

Two section members remain in the same run only when their positions are
consecutive in the complete page reading order. Build one semantic routing atom
per run. Add standalone routing atoms and standalone terminal candidates for
visual units. Add standalone text routing atoms for source text units omitted
from canonical text sections, including caption-like text.

- [ ] **Step 4: Verify focused strategy tests pass**

Run the Step 2 command and require zero failures.

---

### Task 2: Verify hybrid membership on the reported page

**Files:**
- Modify: `tests/test_mmlongbench_segmentation_lab_integration.py`

**Interfaces:**
- Consumes: `get_strategy("deepseek_semantic_hybrid_v1")`
- Consumes: `build_routing_graph(..., "geometry")`

- [ ] **Step 1: Add a failing page-shaped regression**

Construct the six-unit sequence from
`mmlongbench_page_06715bbb9f895678bbe2` and assert:

```text
table | heading+intro | upper image | following text | lower image
```

The heading and intro remain together, the following text is separate, the
horizontal raw children are disjoint, and geometry extension borrows the upper
image into the bottom child and the following text into the top child.

- [ ] **Step 2: Run the regression and verify red**

```bash
python -m pytest -q tests/test_mmlongbench_segmentation_lab_integration.py -k hybrid --tb=short
```

- [ ] **Step 3: Make only the minimal strategy adjustment required**

Do not change `routing_graph.py`; correct hybrid atom construction if the test
reveals a mismatch.

- [ ] **Step 4: Verify the regression passes**

Run the Step 2 command and require zero failures.

---

### Task 3: Bound the workspace and add the divider

**Files:**
- Modify: `tests/test_mmlongbench_segmentation_lab_static.py`
- Modify: `viewer/mmlongbench_segmentation_lab/index.html`
- Modify: `viewer/mmlongbench_segmentation_lab/styles.css`
- Modify: `viewer/mmlongbench_segmentation_lab/app.js`

**Interfaces:**
- HTML: `#workspace-resizer`
- CSS variable: `--reference-pane-width`
- JavaScript:
  `initializeWorkspaceResize()`, `setReferencePaneWidth(width)`
- Persistence key: `mmlongbench-segmentation-lab:reference-pane-width:v1`

- [ ] **Step 1: Write failing static tests**

Require the divider, fixed viewport-height workspace, independent pane
overflow, a fixed-width `.page-stage`, and resize initialization/persistence.
Require keyboard arrow support and ensure the page image has no responsive
`width: 100%` rule.

- [ ] **Step 2: Verify the static tests fail**

```bash
python -m pytest -q tests/test_mmlongbench_segmentation_lab_static.py -k "layout or resize" --tb=short
```

- [ ] **Step 3: Implement the fixed workspace**

Make `.lab-grid` exactly the available viewport height and prevent body growth.
Keep controls, reference, candidates, and audit independently scrollable.
Render `.page-stage` at a stable stored width; narrowing its containing pane
must clip/scroll rather than rescale it.

- [ ] **Step 4: Implement the accessible divider**

Use pointer capture for dragging. Support `ArrowLeft` and `ArrowRight`. Clamp
the reference viewport to configured minimum and maximum widths, update
`--reference-pane-width`, and persist the value without touching routing
session identity or state.

- [ ] **Step 5: Verify static tests pass**

Run the Step 2 command and require zero failures.

---

### Task 4: Package, document, and verify

**Files:**
- Modify: `docs/specifications/mmlongbench_segmentation_lab/design.md`
- Modify: `agent-context/CURRENT_TASK.md`
- Modify: `agent-context/modules/viewer.md`

- [ ] **Step 1: Document the additive strategy and workspace behavior**

Record that the baseline remains authoritative and unchanged, while the hybrid
is an explicit comparison strategy. Record that resizing changes viewport
visibility, never page scale.

- [ ] **Step 2: Run focused regression tests**

```bash
python -m pytest -q \
  tests/test_mmlongbench_segmentation_lab_strategies.py \
  tests/test_mmlongbench_segmentation_lab_routing.py \
  tests/test_mmlongbench_segmentation_lab_static.py \
  tests/test_mmlongbench_segmentation_lab_integration.py \
  --tb=short
```

- [ ] **Step 3: Build a fresh two-strategy site**

```bash
python scripts/mmlongbench/build_segmentation_lab.py \
  --pilot-root pilot_data/mmlongbench_ocr2_unstructured_313 \
  --result-root sol_results/mmlongbench_ocr2_unstructured_313/20260717T182300Z \
  --viewer-source viewer/mmlongbench_segmentation_lab \
  --site-dir tmp/mmlongbench_segmentation_lab_hybrid_r2_20260723 \
  --strategy deepseek_semantic_v1 \
  --strategy deepseek_semantic_hybrid_v1 \
  --page-id mmlongbench_page_06715bbb9f895678bbe2
```

- [ ] **Step 4: Verify the live behavior**

Serve the fresh site. Confirm the hybrid membership on the reported page,
perform repeated decisions/splits without changing workspace height, and drag
the divider while confirming the page-stage dimensions remain constant.

- [ ] **Step 5: Run repository hygiene checks**

```bash
git diff --check
git status --short
```

Do not stage or commit generated `tmp/` output.
