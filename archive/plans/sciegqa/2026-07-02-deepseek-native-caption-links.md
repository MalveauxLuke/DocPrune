# DeepSeek-Native Caption Links Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make DeepSeek OCR2 reading-order/Markdown links the default final-label source while retaining the current geometry linker as a selectable comparison.

**Architecture:** Add a separate native reading-order linker and leave `build_visual_candidates` unchanged as the geometry implementation. Build and label one candidate set per strategy, default to the native set, and let the existing experiment selector switch strategy and gold-coverage threshold.

**Tech Stack:** Python, pytest, static JavaScript viewer, generated JSON manifest.

---

### Task 1: Lock native grouping behavior

**Files:**
- Modify: `tests/test_sciegqa_final_labeling.py`
- Modify: `scripts/sciegqa_parser_compare/final_labeling.py`

- [ ] Add a failing test where `Figure 9a shows...` remains prose while `Figure 9 Theoretical Maxima` + two contiguous figures + trailing `Note.` form one native visual bundle.
- [ ] Run the focused test and confirm it fails because the native linker is absent.
- [ ] Implement `build_native_visual_candidates` and native-only caption/note classification.
- [ ] Run the focused tests and confirm both native and unchanged geometry behavior pass.

### Task 2: Build selectable candidate strategies

**Files:**
- Modify: `tests/test_sciegqa_final_labeling.py`
- Modify: `scripts/sciegqa_parser_compare/final_labeling.py`

- [ ] Add a failing manifest test requiring native and geometry experiments/candidate sets.
- [ ] Generate both candidate sets, keep native as default, and calculate query labels against the strategy named by each experiment.
- [ ] Run final-labeling tests and confirm they pass.

### Task 3: Expose the comparison in the viewer

**Files:**
- Modify: `tests/test_sciegqa_parser_viewer.py`
- Modify: `viewer/sciegqa_parser_compare/app.js`
- Modify: `scripts/sciegqa_parser_compare/viewer_bundle.py`
- Modify: `viewer/sciegqa_parser_compare/README.md`
- Regenerate: `viewer/sciegqa_parser_compare/manifest.json`

- [ ] Add failing static assertions for strategy-aware experiment labels and candidate lookup.
- [ ] Update the experiment selector to show `DeepSeek native` and `Geometry` choices and render the selected candidate set.
- [ ] Regenerate the manifest, run targeted tests, and visually verify page 12 Figure 9 in both strategies.

### Task 4: Verify and hand off

- [ ] Run the final-labeling and viewer test files.
- [ ] Inspect the diff for scope and list any newly unreachable code without deleting it.
- [ ] Commit only after confirming the branch with `git branch --show-current`.

### Task 5: Correct native linking to follow DeepSeek raw structure

**Files:**
- Modify: `tests/test_sciegqa_final_labeling.py`
- Modify: `scripts/sciegqa_parser_compare/final_labeling.py`
- Modify: `viewer/sciegqa_parser_compare/app.js`
- Regenerate: `viewer/sciegqa_parser_compare/manifest.json`

- [ ] Add failing tests for `image → figure_title → image → figure_title`, two-image runs surrounded by titles, text-skipping image runs with a floating title, and equivalent tables.
- [ ] Replace caption-content inference with raw-type classification and top-down visual-run discovery. Each run searches upward before downward and skips at most two intervening raw text blocks.
- [ ] Link unclaimed floating titles with the unchanged 350-unit center-distance rule and merge connected visual groups.
- [ ] Render connectors for every native title-to-object edge while preserving singular geometry links.
- [ ] Regenerate the pilot manifest and verify the three supplied pages in the browser.
