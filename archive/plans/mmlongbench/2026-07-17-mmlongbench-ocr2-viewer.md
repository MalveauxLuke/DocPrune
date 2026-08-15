# MMLongBench DeepSeek-OCR-2 Segmentation Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local static viewer for the 306 validated MMLongBench DeepSeek-OCR-2 pages and their canonical semantic sections while omitting the seven failed pages.

**Architecture:** A Python bundle builder reads the frozen pilot PDFs and packaged SOL JSONL, re-renders only successful pages, verifies source identity, renderer version, dimensions, and per-page PNG agreement, applies the shared `build_deepseek_semantic_sections` function, and writes a portable manifest plus assets. A focused HTML/CSS/JavaScript viewer consumes that manifest and provides document/page navigation, atomic-versus-semantic overlays, linked-member highlighting, Markdown, diagnostics, and raw OCR access.

**Tech Stack:** Python 3, PyMuPDF, static HTML/CSS/JavaScript, pytest.

## Global Constraints

- Use run `sol_results/mmlongbench_ocr2_unstructured_313/20260717T182300Z/`.
- Include exactly the 306 packaged successful pages; omit all seven failed page IDs.
- Reuse `scripts.sciegqa_parser_compare.final_labeling.build_deepseek_semantic_sections`; do not duplicate semantic rules.
- Do not rerun OCR and do not deploy the site.
- Keep generated page PNGs, raw copies, and `manifest.json` outside tracked viewer source.
- Truthfully report 306 displayed of 313 attempted pages and seven upstream exclusions.
- Limit verification to focused unit/static tests, one real build, and one browser smoke check.

---

### Task 1: Portable Viewer Bundle Builder

**Files:**
- Create: `scripts/mmlongbench_ocr2_viewer/__init__.py`
- Create: `scripts/mmlongbench_ocr2_viewer/viewer_bundle.py`
- Create: `scripts/build_mmlongbench_ocr2_viewer.py`
- Create: `tests/test_mmlongbench_ocr2_viewer_bundle.py`

**Interfaces:**
- Consumes: `build_deepseek_semantic_sections(page_id, segments, max_center_distance=350.0)` and the pilot/result JSONL files.
- Produces: `build_viewer_manifest(...) -> dict[str, Any]`, `build_site(pilot_root: Path, result_root: Path, viewer_source: Path, site_dir: Path) -> dict[str, Any]`, and a CLI accepting `--pilot-root`, `--result-root`, `--viewer-source`, and `--site-dir`.

- [ ] **Step 1: Write failing manifest and site tests**

Create fixtures with two successful pages and one absent failed page. Assert that `build_viewer_manifest` returns only the two successful pages, groups them by source document, rejects duplicate page IDs and invalid boxes, and gives every canonical section member a known atomic segment. Add a site fixture with a one-page PDF and matching expected PNG hash; assert that `build_site` creates `manifest.json`, `pages/<page_id>.png`, and `raw/<page_id>.txt`.

```python
def test_manifest_omits_pages_absent_from_packaged_results():
    manifest = build_viewer_manifest(page_plan, pages, runs, segments, audit)
    assert [row["page_id"] for row in manifest["pages"]] == ["page-1", "page-3"]
    assert manifest["summary"] == {
        "document_count": 1,
        "attempted_page_count": 3,
        "displayed_page_count": 2,
        "excluded_page_count": 1,
        "atomic_segment_count": 4,
    }
```

- [ ] **Step 2: Run the focused test and confirm the import failure**

Run: `python -m pytest -q tests/test_mmlongbench_ocr2_viewer_bundle.py --tb=short`

Expected: FAIL because `scripts.mmlongbench_ocr2_viewer.viewer_bundle` does not exist.

- [ ] **Step 3: Implement validated manifest construction**

Implement JSONL indexing and validation, sort pages by frozen page-plan order, group ordered atomic segments by page, call the canonical section builder at the contract default of `350.0`, and store a portable page record containing:

```python
{
    "page_id": page_id,
    "document_filename": plan["document_filename"],
    "page_number": plan["page_number"],
    "document_page_count": plan["document_page_count"],
    "image_url": f"pages/{page_id}.png",
    "raw_artifact_url": f"raw/{page_id}.txt",
    "run": portable_run,
    "atomic_segments": ordered_segments,
    "semantic_sections": sections,
    "semantic_diagnostics": diagnostics,
    "page_mode": page_mode,
}
```

Reject page/run mismatches, duplicate IDs, boxes outside normalized `[0, 1000]` bounds, non-contiguous known section membership, and audit counts that do not reconcile with the packaged rows.

- [ ] **Step 4: Implement deterministic asset generation and CLI**

For each packaged page, open `pilot_root/documents/<document_filename>`, require the frozen PDF hash and packaged PyMuPDF version, and render the one-based `page_number` with `fitz.Matrix(2.0, 2.0)` and `alpha=False`. Require exact dimensions and record byte/SHA-256 agreement with SOL; retain operating-system-specific byte drift in the manifest instead of rejecting the page. Copy each referenced `grounded_output.txt` to `raw/<page_id>.txt`, copy the viewer source files, and atomically write `manifest.json`. Refuse an existing `site_dir`.

- [ ] **Step 5: Run the focused builder tests**

Run: `python -m pytest -q tests/test_mmlongbench_ocr2_viewer_bundle.py --tb=short`

Expected: all tests pass.

- [ ] **Step 6: Commit the builder**

```bash
git add scripts/mmlongbench_ocr2_viewer scripts/build_mmlongbench_ocr2_viewer.py tests/test_mmlongbench_ocr2_viewer_bundle.py
git commit -m "feat: build MMLongBench OCR2 viewer bundle"
```

### Task 2: Focused Static Viewer

**Files:**
- Create: `viewer/mmlongbench_ocr2/index.html`
- Create: `viewer/mmlongbench_ocr2/styles.css`
- Create: `viewer/mmlongbench_ocr2/app.js`
- Create: `viewer/mmlongbench_ocr2/README.md`
- Create: `tests/test_mmlongbench_ocr2_viewer_static.py`

**Interfaces:**
- Consumes: Task 1's `manifest.json` schema.
- Produces: a dependency-free static UI served by any HTTP server.

- [ ] **Step 1: Write failing static-source tests**

Assert that the HTML exposes document and page selectors, previous/next buttons, atomic/semantic overlay controls, summary metrics, Markdown, raw-output link, section legend, and detail panel. Assert that JavaScript fetches `manifest.json`, draws normalized SVG rectangles, filters pages by document, updates deep-link parameters, and uses text-safe DOM APIs rather than interpolating OCR content into `innerHTML`.

- [ ] **Step 2: Run the static test and confirm missing source files**

Run: `python -m pytest -q tests/test_mmlongbench_ocr2_viewer_static.py --tb=short`

Expected: FAIL because `viewer/mmlongbench_ocr2/` does not exist.

- [ ] **Step 3: Implement the accessible viewer shell and styling**

Create a two-pane layout derived from the existing parser viewer: a fixed-width inspection sidebar and flexible page canvas. Use a restrained dark-neutral interface, high-contrast overlay colors, responsive stacking below 900 px, keyboard-focus styles, and no external assets or libraries.

- [ ] **Step 4: Implement manifest-driven interaction**

Load the manifest, populate the document selector and successful pages only, preserve page-plan order, support `?page=<page_id>&mode=semantic`, render atomic boxes by segment type, render semantic member boxes with shared `S#` badges and connectors, and synchronize hover/focus between overlay members and legend entries. Show section kind, reading orders, member raw types, text/Markdown, page mode, and run diagnostics in the detail panel. Ensure the header always reports the manifest's 306/313/7 summary.

- [ ] **Step 5: Run the static-source tests**

Run: `python -m pytest -q tests/test_mmlongbench_ocr2_viewer_static.py --tb=short`

Expected: all tests pass.

- [ ] **Step 6: Commit the viewer source**

```bash
git add viewer/mmlongbench_ocr2 tests/test_mmlongbench_ocr2_viewer_static.py
git commit -m "feat: add MMLongBench OCR2 segmentation viewer"
```

### Task 3: Real 306-Page Bundle and Focused Acceptance

**Files:**
- Modify: `viewer/mmlongbench_ocr2/README.md`
- Generate, do not commit: `tmp/mmlongbench_ocr2_viewer_20260717T182300Z/site/`

**Interfaces:**
- Consumes: Tasks 1 and 2.
- Produces: a locally served review site and reproducible launch instructions.

- [ ] **Step 1: Run the focused Python/static test set once**

Run:

```bash
python -m pytest -q \
  tests/test_mmlongbench_ocr2_viewer_bundle.py \
  tests/test_mmlongbench_ocr2_viewer_static.py \
  tests/test_sciegqa_final_labeling.py --tb=short
```

Expected: all selected tests pass.

- [ ] **Step 2: Build the real viewer exactly once**

Run:

```bash
python scripts/build_mmlongbench_ocr2_viewer.py \
  --pilot-root pilot_data/mmlongbench_ocr2_unstructured_313 \
  --result-root sol_results/mmlongbench_ocr2_unstructured_313/20260717T182300Z \
  --viewer-source viewer/mmlongbench_ocr2 \
  --site-dir tmp/mmlongbench_ocr2_viewer_20260717T182300Z/site
```

Expected JSON summary: 10 documents, 313 attempted pages, 306 displayed pages, 7 excluded pages, and 4,218 atomic segments.

- [ ] **Step 3: Perform one manifest acceptance check**

Read the generated manifest once and assert the five expected counts, 306 unique page IDs, zero intersection with `completion_audit.json`'s excluded page IDs, existing page/raw assets, and semantic-section members drawn only from each page's atomic segment IDs.

- [ ] **Step 4: Perform one browser smoke check**

Serve the generated `site` directory with `python3 -m http.server 8000`, then verify in one browser session that the first page renders, next-page and document navigation work, atomic/semantic switching changes overlays, semantic hover highlights linked members, and the raw-output link opens.

- [ ] **Step 5: Finalize launch instructions and commit**

Record the exact build and serve commands plus the deep-link format in `viewer/mmlongbench_ocr2/README.md`.

```bash
git add viewer/mmlongbench_ocr2/README.md
git commit -m "docs: document MMLongBench viewer launch"
```
