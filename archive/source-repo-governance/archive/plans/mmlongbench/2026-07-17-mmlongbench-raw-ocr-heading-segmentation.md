# MMLongBench Raw OCR and Heading Segmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Display raw DeepSeek OCR inside the original viewer and make Markdown headings the only explicit text-section boundaries.

**Architecture:** Update the canonical Python semantic-section builder, then rebuild the static viewer manifest so all displayed sections use the new rules. Add a self-contained viewer tab controller that fetches the already-packaged raw artifact without altering the manifest schema.

**Tech Stack:** Python 3, pytest, static HTML/CSS/JavaScript, PyMuPDF viewer bundling.

## Global Constraints

- Preserve the untouched DeepSeek raw OCR artifacts.
- Treat only Markdown ATX headings `#` through `######` as explicit text boundaries.
- Merge consecutive heading blocks into the same semantic section.
- On unheaded pages, attach consecutive unordered bullets to the preceding paragraph; keep a leading bullet run together.
- Do not change visual/caption linking.
- Do not add dependencies.

---

### Task 1: Heading-only semantic segmentation

**Files:**
- Modify: `tests/test_sciegqa_final_labeling.py`
- Modify: `scripts/sciegqa_parser_compare/final_labeling.py`
- Modify: `agent-context/modules/deepseek_semantic_segmentation.md`

**Interfaces:**
- Consumes: atomic DeepSeek segments ordered by `reading_order`.
- Produces: `build_headed_text_candidates(page_id, segments)` candidates where consecutive headings and following text share one section.

- [ ] Add tests proving a bullet block stays with its heading, unheaded bullet runs attach to the preceding paragraph, a single `#` heading activates headed mode, and adjacent `#`/`##` blocks merge.
- [ ] Run the focused tests and confirm they fail under the existing `#{2,6}` and one-heading-per-section policy.
- [ ] Change heading recognition to `#{1,6}` and group adjacent heading positions before collecting section members.
- [ ] Run the focused and full final-labeling test file.
- [ ] Update the canonical segmentation contract.

### Task 2: Embedded raw OCR viewer

**Files:**
- Modify: `tests/test_mmlongbench_ocr2_viewer_static.py`
- Modify: `viewer/mmlongbench_ocr2/index.html`
- Modify: `viewer/mmlongbench_ocr2/app.js`
- Modify: `viewer/mmlongbench_ocr2/styles.css`

**Interfaces:**
- Consumes: `page.raw_artifact_url` and `page.run.document_markdown` from `manifest.json`.
- Produces: accessible `Markdown` and `Raw OCR` tabs, a literal raw-output panel, copy status, and the existing full-screen raw link.

- [ ] Add static tests for tab controls, raw text container, copy button, raw fetch, and text-safe rendering.
- [ ] Run the static tests and confirm they fail because the controls do not exist.
- [ ] Add the tab markup and styles.
- [ ] Add on-demand raw fetching, page-change invalidation, safe `textContent` rendering, tab state, and copy behavior.
- [ ] Run the viewer static tests.

### Task 3: Regenerate and inspect the active site

**Files:**
- Regenerate: `tmp/mmlongbench_ocr2_viewer_20260717T182300Z/site/manifest.json`
- Regenerate: copied static viewer files in the same site directory.

**Interfaces:**
- Consumes: frozen OCR artifacts under `sol_results/mmlongbench_ocr2_unstructured_313/20260717T182300Z`.
- Produces: the site currently served at `http://localhost:8000/`.

- [ ] Run the focused segmentation, viewer-bundle, and static-viewer tests.
- [ ] Rebuild the site using the pinned PyMuPDF runtime documented in `viewer/mmlongbench_ocr2/README.md`.
- [ ] Confirm the manifest still contains 306 displayed pages and references every packaged raw artifact.
- [ ] Open a brochure page and verify that bullets share their heading section, consecutive headings share one section, and Raw OCR matches the packaged artifact.
