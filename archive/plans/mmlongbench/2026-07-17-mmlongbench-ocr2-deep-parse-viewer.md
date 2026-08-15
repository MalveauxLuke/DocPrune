# MMLongBench OCR2 Deep-Parse Comparison Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a separate static viewer that compares two selectable DeepSeek-OCR-2 deep-parse prompt arms on each of the 13 frozen pilot pages.

**Architecture:** A Python builder validates and normalizes the frozen run into a portable `manifest.json`, copies the 13 existing page images and 47 crop images, and preserves every raw output. A dependency-free HTML/CSS/JavaScript site renders two synchronized page panes, arm-specific overlays, crop results, run warnings, prompts, inventories, and raw text.

**Tech Stack:** Python 3.11+, pytest, JSON/JSONL, static HTML/CSS/JavaScript, local HTTP server.

## Global Constraints

- Do not modify or rebuild `viewer/mmlongbench_ocr2/`.
- Do not rerun DeepSeek-OCR-2 or alter any frozen model response.
- Treat completed inference, parsed grounding, empty output, non-grounded output, truncation, and repetition as distinct states.
- Display all 13 pages, eight arms, 138 run records, and 47 crop results.
- Keep validation focused: manifest tests, static contract tests, one real build, and one browser smoke check.

---

### Task 1: Normalize the Frozen Experiment

**Files:**
- Create: `scripts/mmlongbench_ocr2_deep_parse_viewer/__init__.py`
- Create: `scripts/mmlongbench_ocr2_deep_parse_viewer/viewer_bundle.py`
- Create: `scripts/build_mmlongbench_ocr2_deep_parse_viewer.py`
- Create: `tests/test_mmlongbench_ocr2_deep_parse_viewer_bundle.py`

**Interfaces:**
- Consumes: the JSON/JSONL files under `sol_results/mmlongbench_ocr2_deep_parse_links/20260717T224039Z`, the source-page metadata in `pilot_data/mmlongbench_ocr2_unstructured_313/page_plan.jsonl`, and page PNGs from `viewer/mmlongbench_ocr2/pages/`.
- Produces: `build_viewer_manifest(result_root: Path, page_plan_path: Path) -> dict[str, Any]` and `build_site(result_root: Path, page_plan_path: Path, source_pages_dir: Path, viewer_source: Path, site_dir: Path) -> dict[str, Any]`.

- [ ] **Step 1: Write failing normalization tests**

Create fixtures for two pages, two full-page arms, one crop, unit mappings, prompts, and run warnings. Assert that the manifest keeps page/arm alignment, maps unit IDs to boxes, links the crop to its parent page, preserves raw response text, and reports empty output separately from parsed grounding.

```python
manifest = build_viewer_manifest(result_root, page_plan_path)
assert manifest["summary"]["page_count"] == 2
assert manifest["summary"]["run_count"] == 5
assert manifest["pages"][0]["arms"]["semantic"]["display_status"] == "parsed_grounding"
assert manifest["pages"][0]["arms"]["assisted"]["resolved_groups"][0]["units"][0]["bbox_norm_1000"] == [10, 20, 30, 40]
assert manifest["pages"][0]["crops"][0]["visual_id"] == "V001"
```

- [ ] **Step 2: Verify the tests fail for the missing module**

Run: `python -m pytest -q tests/test_mmlongbench_ocr2_deep_parse_viewer_bundle.py --tb=short`

Expected: collection fails because `scripts.mmlongbench_ocr2_deep_parse_viewer` does not exist.

- [ ] **Step 3: Implement strict normalization and portable artifact copying**

Implement JSONL loading, unique-ID validation, normalized-box validation, run/output joins, parse-status normalization, conservative group-ID parsing, crop-parent joins, raw-text loading, summary aggregation, and safe relative-path copying. Reject missing page images, unknown unit IDs in the canonical map, duplicate identities, path traversal, and mismatched run/output IDs. Preserve unresolved returned IDs as warnings rather than manufacturing boxes.

```python
def display_status(run: Mapping[str, Any], parsed: Mapping[str, Any], raw: str) -> str:
    if not raw.strip() or int(run.get("generated_token_count", 0)) <= 1:
        return "empty_output"
    if parsed.get("parse_status") == "parsed_grounding_segments":
        return "parsed_grounding"
    return "non_grounded_output"
```

- [ ] **Step 4: Run focused bundle tests**

Run: `python -m pytest -q tests/test_mmlongbench_ocr2_deep_parse_viewer_bundle.py --tb=short`

Expected: all Task 1 tests pass.

- [ ] **Step 5: Commit the manifest builder**

Run `git branch --show-current`, stage only the four Task 1 files, and commit with `feat: build OCR2 deep-parse viewer manifest`.

### Task 2: Build the Synchronized Comparison Interface

**Files:**
- Create: `viewer/mmlongbench_ocr2_deep_parse/index.html`
- Create: `viewer/mmlongbench_ocr2_deep_parse/styles.css`
- Create: `viewer/mmlongbench_ocr2_deep_parse/app.js`
- Create: `viewer/mmlongbench_ocr2_deep_parse/README.md`
- Create: `tests/test_mmlongbench_ocr2_deep_parse_viewer_static.py`

**Interfaces:**
- Consumes: `manifest.json` with `summary`, `arms`, and ordered `pages`; each page contains `image_url`, `arms`, and `crops`.
- Produces: a static two-pane comparison UI whose pane selectors have IDs `left-arm-select` and `right-arm-select`, whose synchronized stages use `.comparison-stage`, and whose diagnostic panels use `.arm-details`.

- [ ] **Step 1: Write failing static-contract tests**

Assert the HTML contains two independent arm selects, shared document/page navigation, two page stages, a crop dialog/panel, and prompt/raw-output disclosures. Assert JavaScript fetches `manifest.json`, validates boxes, renders both panes from the same page object, stores both arm choices in the URL, and handles `parsed_grounding`, `non_grounded_output`, and `empty_output`. Assert CSS uses a two-column comparison grid with a narrow-screen fallback.

```python
assert 'id="left-arm-select"' in html
assert 'id="right-arm-select"' in html
assert 'class="comparison-stage"' in html
assert 'params.set("left"' in app
assert 'params.set("right"' in app
assert "empty_output" in app
assert "grid-template-columns: repeat(2" in css
```

- [ ] **Step 2: Verify static tests fail because the viewer is absent**

Run: `python -m pytest -q tests/test_mmlongbench_ocr2_deep_parse_viewer_static.py --tb=short`

Expected: tests fail because the viewer files do not exist.

- [ ] **Step 3: Implement the dependency-free site**

Create a compact dark review interface with a sticky experiment header, source document/page navigation, summary warning chips, and two equal arm panes. Render normalized boxes into SVG, use matching colors for group members, highlight boxes and detail rows together, expose exact prompts and raw output, render inventory groups and unresolved IDs, and show `paper_deep_parse` as selectable parent boxes plus crop cards. Keep both image stages synchronized through shared scale/translation state and pointer/wheel handlers.

```javascript
const state = { manifest: null, pageIndex: 0, leftArm: "grounded_markdown_control",
  rightArm: "continuous_information_units", transform: { scale: 1, x: 0, y: 0 } };

function renderBoth() {
  const page = state.manifest.pages[state.pageIndex];
  renderPane("left", page, state.leftArm);
  renderPane("right", page, state.rightArm);
  applySharedTransform();
}
```

- [ ] **Step 4: Run the static-contract tests**

Run: `python -m pytest -q tests/test_mmlongbench_ocr2_deep_parse_viewer_static.py --tb=short`

Expected: all Task 2 tests pass.

- [ ] **Step 5: Commit the static viewer**

Run `git branch --show-current`, stage only the five Task 2 files, and commit with `feat: add OCR2 deep-parse comparison viewer`.

### Task 3: Build and Smoke-Test the Real Viewer

**Files:**
- Generate: `viewer/mmlongbench_ocr2_deep_parse/manifest.json`
- Generate: `viewer/mmlongbench_ocr2_deep_parse/pages/*.png`
- Generate: `viewer/mmlongbench_ocr2_deep_parse/crops/*.png`
- Generate: `viewer/mmlongbench_ocr2_deep_parse/raw/**/*.txt`
- Modify: `agent-context/modules/viewer.md`

**Interfaces:**
- Consumes: `build_site(...)` from Task 1 and the static source from Task 2.
- Produces: a portable locally served bundle and a durable launch note.

- [ ] **Step 1: Add a real-data acceptance test**

Add a test marked `integration` that builds into a temporary directory and asserts exactly 13 pages, 8 arms, 138 runs, 47 crops, and zero missing raw outputs. Assert the default arm pair and warning counts match the frozen run.

```python
assert manifest["summary"] == {
    **manifest["summary"], "page_count": 13, "arm_count": 8,
    "run_count": 138, "crop_count": 47,
}
```

- [ ] **Step 2: Verify the acceptance test fails before the real build contract is complete**

Run the single integration test and confirm it fails on a missing expected real-data field or count.

- [ ] **Step 3: Complete the real build and viewer documentation**

Run the builder with the frozen result root, page-plan path, existing page-image directory, static viewer source, and a fresh temporary destination; then replace only the generated artifacts inside the new viewer directory. Update `agent-context/modules/viewer.md` with the new builder, bundle path, run ID, and local serving command.

- [ ] **Step 4: Run focused verification**

Run both new test files plus the existing MMLongBench viewer static tests. Start a local server from `viewer/mmlongbench_ocr2_deep_parse/`, load `manifest.json`, and perform one browser smoke check of page navigation, independent arm selection, synchronized overlays, raw output, and a crop.

- [ ] **Step 5: Commit and report**

Run `git branch --show-current`, stage only the generated viewer artifacts, viewer module note, and acceptance-test change, then commit with `feat: publish OCR2 deep-parse pilot viewer`. Report the localhost URL and the observed result-quality summary without claiming that completed inference equals successful semantic grouping.
