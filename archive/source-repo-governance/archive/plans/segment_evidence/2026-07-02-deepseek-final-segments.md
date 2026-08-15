# DeepSeek-OCR-2 Final Segments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make DeepSeek-OCR-2 the exclusive source of final candidates while preserving a linked MinerU visual-only inspection view.

**Architecture:** The Python preprocessor infers caption anchors, derives parser-specific distance heuristics from the pilot, routes headed DeepSeek pages to sections and unheaded pages to atomic paragraphs, and stores MinerU visual links outside final labels. The existing viewer renders DeepSeek final bundles with `S#` links and MinerU-only bundles with `M#` links.

**Tech Stack:** Python 3, pytest, browser-native JavaScript/SVG, static HTML/CSS, checked-in JSON manifest.

---

### Task 1: Specify DeepSeek routing and caption inference

**Files:**
- Modify: `tests/test_sciegqa_final_labeling.py`
- Modify: `scripts/sciegqa_parser_compare/final_labeling.py`

- [ ] Add failing tests asserting:

```python
assert infer_caption_role(segment("c", "deepseek_ocr2", "unknown", 1, "Fig. 2: Result", [0, 10, 10, 20])) == "figure"
assert infer_caption_role(segment("c", "deepseek_ocr2", "text", 1, "Table 2. Values", [0, 10, 10, 20])) == "table"
assert infer_caption_role(segment("n", "deepseek_ocr2", "unknown", 1, "Note. values", [0, 10, 10, 20])) is None
```

Add an unheaded page fixture and assert each nonempty DeepSeek title/text/formula block becomes one `deepseek_paragraph`, caption blocks are excluded from paragraph candidates, and no `mineru_text_fallback` exists.

Add a headed fixture with DeepSeek visual and caption blocks and assert final kinds are exactly `document_preamble`, `headed_text`, and `visual_bundle`; assert every final candidate has `source_parser == "deepseek_ocr2"`.

- [ ] Run:

```bash
python -m pytest -q tests/test_sciegqa_final_labeling.py
```

Expected: failures showing missing caption inference and old MinerU fallback routing.

- [ ] Implement:

```python
_FIGURE_CAPTION = re.compile(r"^\s*(?:figure|fig\.|chart)\s*\w*\s*[:.]", re.I)
_TABLE_CAPTION = re.compile(r"^\s*table\s*\w*\s*[:.]", re.I)

def infer_caption_role(segment):
    if segment["type"] in {"figure_caption", "chart_caption"}: return "figure"
    if segment["type"] == "table_caption": return "table"
    text = _content(segment, "markdown").strip() or _content(segment, "text").strip()
    if _FIGURE_CAPTION.match(text): return "figure"
    if _TABLE_CAPTION.match(text): return "table"
    return None
```

Implement `build_deepseek_paragraph_candidates` as one candidate per nonempty `title`, `text`, or `formula` block that is not a caption anchor. Refactor visual linking to use inferred caption roles and retain original member boxes.

- [ ] Re-run the test file and require all tests to pass.

- [ ] Confirm `git branch --show-current`, then commit the Python implementation and tests.

### Task 2: Derive pilot heuristics and separate MinerU inspection links

**Files:**
- Modify: `scripts/sciegqa_parser_compare/final_labeling.py`
- Modify: `tests/test_sciegqa_final_labeling.py`

- [ ] Add failing tests for a deterministic heuristic:

```python
assert rounded_p95_distance([101, 149, 151]) == 175.0
assert rounded_p95_distance([10]) == 150.0
assert rounded_p95_distance([500]) == 350.0
```

Assert `build_final_labeling` stores:

```python
labeling["caption_link_heuristics"]["deepseek_ocr2"]
labeling["caption_link_heuristics"]["mineru"]
labeling["mineru_visual_links_by_page"]
```

and assert MinerU link candidate IDs never occur in `query_candidate_labels` or final `candidates_by_page`.

- [ ] Run the targeted tests and confirm the new assertions fail.

- [ ] Compute each parser's nearest compatible caption-object distances across all pilot pages, choose the rounded/clipped p95, build DeepSeek final visual links with its threshold, and store MinerU visual links separately with its threshold and diagnostics.

- [ ] Reduce labeling experiments to the five gold-coverage thresholds because paragraph-window and caption-distance grids no longer control final candidates. Remove fallback and caption-distance fields from summaries.

- [ ] Run `tests/test_sciegqa_final_labeling.py` and commit after it passes.

### Task 3: Render DeepSeek and MinerU linked visuals

**Files:**
- Modify: `tests/test_sciegqa_parser_viewer.py`
- Modify: `viewer/sciegqa_parser_compare/app.js`
- Modify: `viewer/sciegqa_parser_compare/styles.css`
- Modify: `viewer/sciegqa_parser_compare/README.md`

- [ ] Add failing static assertions for `renderMineruLinks`, `linkConnector`, `mineru_visual_links_by_page`, and `box-mineru-link`; update checked-in manifest expectations from 75 experiments to 5 and require every final candidate to use DeepSeek-OCR-2.

- [ ] Run `python -m pytest -q tests/test_sciegqa_parser_viewer.py` and confirm failure.

- [ ] Change `MinerU only` to the `mineru_links` layer. Render only stored MinerU visual bundles with deterministic `M#` badges. Do not render MinerU text blocks in that mode.

- [ ] Add an SVG connector helper:

```javascript
function linkConnector(fromBox, toBox, className, candidateId) {
  const line = document.createElementNS(SVG_NS, "line");
  line.setAttribute("x1", (fromBox[0] + fromBox[2]) / 2);
  line.setAttribute("y1", (fromBox[1] + fromBox[3]) / 2);
  line.setAttribute("x2", (toBox[0] + toBox[2]) / 2);
  line.setAttribute("y2", (toBox[1] + toBox[3]) / 2);
  line.setAttribute("class", className);
  line.dataset.candidateId = candidateId;
  return line;
}
```

Render connector lines for DeepSeek final `visual_bundle` members and MinerU inspection bundles. Keep `S#` and `M#` visually distinct and preserve all stored boxes.

- [ ] Replace fallback/caption experiment controls with the five coverage thresholds and display fixed DeepSeek/MinerU heuristic distances and sample counts in the summary.

- [ ] Document that final candidates are DeepSeek-only and `MinerU only` is a visual-link inspection view.

- [ ] Run viewer tests and commit after they pass except for the expected stale-manifest assertions.

### Task 4: Regenerate and verify

**Files:**
- Modify: `viewer/sciegqa_parser_compare/manifest.json`

- [ ] Regenerate twice and require identical SHA-256 hashes:

```bash
python scripts/build_sciegqa_final_labeling.py --manifest viewer/sciegqa_parser_compare/manifest.json --output viewer/sciegqa_parser_compare/manifest.json
first=$(shasum -a 256 viewer/sciegqa_parser_compare/manifest.json | awk '{print $1}')
python scripts/build_sciegqa_final_labeling.py --manifest viewer/sciegqa_parser_compare/manifest.json --output viewer/sciegqa_parser_compare/manifest.json
second=$(shasum -a 256 viewer/sciegqa_parser_compare/manifest.json | awk '{print $1}')
test "$first" = "$second"
```

- [ ] Run targeted tests, then `python -m pytest -q` and `git diff --check`.

- [ ] In the browser, verify one headed page, one unheaded paragraph page, one figure-caption page, one table-caption page, and one multi-panel page. Confirm final modes contain only DeepSeek candidates and `MinerU only` contains only linked MinerU visuals.

- [ ] Confirm zero browser warnings/errors, commit the regenerated manifest, and leave the updated viewer open.
