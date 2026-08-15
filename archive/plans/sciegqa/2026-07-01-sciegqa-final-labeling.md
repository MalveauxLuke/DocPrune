# SciEGQA Final Segment Labeling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Do not use subagents.

**Goal:** Add deterministic multimodal candidate generation, a 75-configuration gold-labeling experiment, and a final-label overlay to the checked-in 32-page/35-query SciEGQA comparison website.

**Architecture:** A pure Python module derives MinerU visual-caption bundles, DeepSeek-OCR-2 heading sections, and MinerU fallback paragraph windows from the existing manifest. A second pure layer computes multi-box overlap labels and experiment summaries, then an atomic CLI update embeds those results into the static manifest. The browser only selects and renders precomputed experiments; it never regenerates candidates or labels.

**Tech Stack:** Python 3.12, pytest, JSON, static HTML/CSS/JavaScript, SVG overlays, existing SciEGQA normalized bounding-box utilities.

---

## File responsibilities

Create:

- `scripts/sciegqa_parser_compare/final_labeling.py` — candidate generation, multi-box geometry, experiment grid, labels, summaries, and recommendation.
- `scripts/build_sciegqa_final_labeling.py` — atomic CLI for enriching a comparison manifest.
- `tests/test_sciegqa_final_labeling.py` — unit and pilot-level invariants.

Modify:

- `viewer/sciegqa_parser_compare/index.html` — final-label controls and experiment summary containers.
- `viewer/sciegqa_parser_compare/app.js` — experiment selection, filters, candidate rendering, and detail display.
- `viewer/sciegqa_parser_compare/styles.css` — label-specific non-obscuring outline styles.
- `viewer/sciegqa_parser_compare/README.md` — document final-label controls and regeneration command.
- `viewer/sciegqa_parser_compare/manifest.json` — checked-in deterministic experiment results.
- `tests/test_sciegqa_parser_viewer.py` — final-label manifest and static UI regression assertions.

Do not change parser runners, raw parser outputs, page images, existing parser segments, or SciEGQA query records.

## Task 1: Candidate-generation contracts

**Files:**

- Create: `scripts/sciegqa_parser_compare/final_labeling.py`
- Create: `tests/test_sciegqa_final_labeling.py`

- [ ] **Step 1: Write candidate-generation tests**

Create fixtures using the existing segment schema and assert:

```python
def test_deepseek_hash_headings_create_unsplit_semantic_sections():
    segments = [
        segment("h1", "deepseek_ocr2", "title", 0, "## 6 Heading", [10, 10, 90, 20]),
        segment("p1", "deepseek_ocr2", "text", 1, "one", [10, 20, 90, 30]),
        segment("p2", "deepseek_ocr2", "text", 2, "two", [10, 30, 90, 40]),
        segment("h2", "deepseek_ocr2", "title", 3, "### 6.1 Next", [10, 40, 90, 50]),
        segment("p3", "deepseek_ocr2", "text", 4, "three", [10, 50, 90, 60]),
    ]
    candidates = build_headed_text_candidates("page", segments)
    assert [row["member_segment_ids"] for row in candidates] == [
        ["h1", "p1", "p2"], ["h2", "p3"]
    ]
```

Also test that:

- one very long heading section remains one candidate;
- a `#` document title is not a qualifying `##` boundary;
- DeepSeek `figure`, `table`, caption, and `unknown` segments are excluded;
- a page with no qualifying heading returns no headed candidate;
- MinerU figure followed by figure-caption is one visual candidate;
- MinerU table followed or preceded by a table-caption is one visual candidate;
- a caption cannot be reused;
- unpaired objects and captions are recorded in diagnostics;
- candidate IDs are identical after reversing input order.

- [ ] **Step 2: Run the tests and verify failure**

Run:

```bash
python -m pytest tests/test_sciegqa_final_labeling.py -q
```

Expected: collection fails because `scripts.sciegqa_parser_compare.final_labeling` does not exist.

- [ ] **Step 3: Implement stable candidate construction**

Define these public constants:

```python
GOLD_THRESHOLDS = (0.50, 0.70, 0.80, 0.90, 0.95)
FALLBACK_PARAGRAPH_MAXIMA = (1, 2, 3, 4, 5)
TEXT_TYPES = frozenset({"text", "formula"})
FALLBACK_TEXT_TYPES = frozenset({"text", "formula", "unknown"})
VISUAL_TYPES = frozenset({"figure", "chart", "table"})
CAPTION_TYPES = frozenset({"figure_caption", "chart_caption", "table_caption"})
CAPTION_DISTANCE_LIMITS = (150.0, 250.0, 350.0)

```

Implement these exact public interfaces:

- `is_markdown_heading(segment: Mapping[str, Any]) -> bool`
- `build_visual_candidates(page_id: str, segments: Sequence[Mapping[str, Any]], max_center_distance: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]`
- `build_headed_text_candidates(page_id: str, segments: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]`
- `build_fallback_text_candidates(page_id: str, segments: Sequence[Mapping[str, Any]], max_paragraphs: int) -> list[dict[str, Any]]`

Use `make_stable_id()` from `scripts.sciegqa_mineru.schema`. Sort source segments by `(reading_order, segment_id)` before processing. A candidate record must contain:

```python
{
    "candidate_id": make_stable_id("candidate", page_id, kind, member_ids),
    "page_id": page_id,
    "kind": kind,
    "source_parser": source_parser,
    "source_parser_version": parser_version,
    "heading": heading_or_none,
    "text": joined_text,
    "markdown": joined_markdown,
    "member_segment_ids": member_ids,
    "member_bboxes_norm_1000": member_boxes,
    "member_reading_orders": member_orders,
    "member_types": member_types,
    "member_count": len(member_ids),
    "paragraph_count": paragraph_count,
    "word_count": len(joined_text.split()),
    "caption_link": caption_metadata_or_none,
}
```

Visual linkage assigns each object to its nearest compatible caption by
normalized box-center distance when that distance is within
`max_center_distance`. Figure/chart objects accept figure/chart captions;
tables accept table captions. Build one candidate per caption containing every
assigned object, allowing multi-panel figures without assigning an object to
more than one caption. Unassigned objects remain standalone candidates.

Heading detection uses `^\s*#{2,6}\s+`. A headed section includes the heading and following DeepSeek `text` and `formula` blocks until the next qualifying heading. It skips visual, caption, and unknown blocks without ending the section.

Fallback generation partitions MinerU text/formula/nonempty-unknown runs at
visual, caption, or empty-unknown segments, then emits every sliding window of
sizes `1..max_paragraphs` inside each run.

- [ ] **Step 4: Run candidate tests**

Run:

```bash
python -m pytest tests/test_sciegqa_final_labeling.py -q
```

Expected: candidate-generation tests pass.

- [ ] **Step 5: Commit candidate generation**

```bash
git branch --show-current
git add scripts/sciegqa_parser_compare/final_labeling.py tests/test_sciegqa_final_labeling.py
git commit -m "feat: derive SciEGQA final candidates"
```

## Task 2: Multi-box labels and experiment summaries

**Files:**

- Modify: `scripts/sciegqa_parser_compare/final_labeling.py`
- Modify: `tests/test_sciegqa_final_labeling.py`

- [ ] **Step 1: Write exact geometry and label tests**

Add hand-computed tests:

```python
def test_candidate_metrics_use_member_union_not_envelope():
    gold = [0, 0, 10, 10]
    boxes = [[0, 0, 4, 10], [6, 0, 10, 10]]
    result = candidate_geometry(gold, boxes)
    assert result == {
        "intersection_area": 80.0,
        "gold_coverage": 0.8,
        "candidate_precision": 1.0,
        "candidate_union_area": 80.0,
    }

def test_labels_keep_partial_overlap_out_of_binary_training():
    assert label_for_coverage(0.8, 0.8) == "positive"
    assert label_for_coverage(0.1, 0.8) == "partial_manual_review"
    assert label_for_coverage(0.0, 0.8) == "negative"
```

Test overlapping member boxes without double-counting, invalid boxes, 75 exact configurations, query-level unmatched status, anchor/all-linked summaries, and deterministic recommendation selection.

- [ ] **Step 2: Verify the new tests fail**

Run:

```bash
python -m pytest tests/test_sciegqa_final_labeling.py -q
```

Expected: failure for undefined geometry and experiment functions.

- [ ] **Step 3: Implement geometry and grid evaluation**

Add these exact public interfaces:

- `candidate_geometry(gold_box: Sequence[object], member_boxes: Sequence[Sequence[object]]) -> dict[str, float]`
- `label_for_coverage(gold_coverage: float, threshold: float) -> str`
- `build_experiment_grid() -> list[dict[str, Any]]`
- `build_final_labeling(manifest: Mapping[str, Any]) -> dict[str, Any]`

Use an x-sweep union algorithm so overlapping member boxes are counted once. Validate all input boxes through `BBox.validate((1000, 1000))`.

For each page:

1. Generate MinerU visual candidates for the experiment's caption-distance limit.
2. If OCR-2 has qualifying headings, generate headed candidates and no fallback candidates.
3. Otherwise generate fallback candidates separately for each paragraph maximum.

Store the union of all configuration-specific candidates under
`candidates_by_page`. Each experiment's label rows identify the exact candidate
variant active for that configuration.

Store per-experiment labels as:

```python
query_candidate_labels[experiment_id][query_id] = {
    "query_status": "matched" or "unmatched",
    "candidates": [
        {
            "candidate_id": candidate_id,
            "label": label,
            "gold_coverage": coverage,
            "candidate_precision": precision,
            "intersection_area": area,
        }
    ],
}
```

Only include fallback candidates whose paragraph count does not exceed the experiment maximum.

At threshold `0.80`, select the recommendation by:

```python
key = (
    -matched_query_count,
    candidate_count,
    median_candidate_word_count,
    fallback_max_paragraphs,
    caption_max_center_distance,
    experiment_id,
)
```

Include all required summary distributions and diagnostics from the specification.

- [ ] **Step 4: Run the full focused test file**

```bash
python -m pytest tests/test_sciegqa_final_labeling.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit experiment evaluation**

```bash
git branch --show-current
git add scripts/sciegqa_parser_compare/final_labeling.py tests/test_sciegqa_final_labeling.py
git commit -m "feat: evaluate SciEGQA labeling experiments"
```

## Task 3: Atomic manifest enrichment CLI

**Files:**

- Create: `scripts/build_sciegqa_final_labeling.py`
- Modify: `tests/test_sciegqa_final_labeling.py`

- [ ] **Step 1: Add CLI tests**

Test that the CLI:

- accepts `--manifest` and `--output`;
- supports equal input/output paths safely;
- preserves all preexisting manifest keys and parser records;
- writes `labeling.schema_version`, 75 experiments, a valid recommendation, candidates, and labels;
- produces byte-identical output when rerun on the same original manifest;
- removes an existing `labeling` object before recomputing, preventing recursive drift.

- [ ] **Step 2: Verify CLI tests fail**

```bash
python -m pytest tests/test_sciegqa_final_labeling.py -q
```

Expected: failure because the CLI is absent.

- [ ] **Step 3: Implement the CLI**

The entrypoint is:

```python
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    source = json.loads(args.manifest.read_text(encoding="utf-8"))
    source.pop("labeling", None)
    source["labeling"] = build_final_labeling(source)
    atomic_write_json(args.output, source)
    return 0
```

Write to a sibling temporary file, flush and `os.fsync()`, then `os.replace()` the destination. Do not mutate page images or raw artifacts.

- [ ] **Step 4: Run focused tests and CLI help**

```bash
python -m pytest tests/test_sciegqa_final_labeling.py -q
python scripts/build_sciegqa_final_labeling.py --help
```

Expected: tests pass and help lists both required arguments.

- [ ] **Step 5: Commit CLI**

```bash
git branch --show-current
git add scripts/build_sciegqa_final_labeling.py tests/test_sciegqa_final_labeling.py
git commit -m "feat: build final labeling manifest"
```

## Task 4: Final-label viewer controls and overlays

**Files:**

- Modify: `viewer/sciegqa_parser_compare/index.html`
- Modify: `viewer/sciegqa_parser_compare/app.js`
- Modify: `viewer/sciegqa_parser_compare/styles.css`
- Modify: `tests/test_sciegqa_parser_viewer.py`

- [ ] **Step 1: Add static viewer tests**

Assert that the viewer contains:

```python
assert "Final labeling only" in app
assert "Gold + final labeling" in app
assert 'id="labeling-experiment"' in html
assert 'id="label-filters"' in html
assert 'id="candidate-kind-filters"' in html
assert 'id="experiment-summary"' in html
assert "member_bboxes_norm_1000" in app
assert "query_candidate_labels" in app
assert "innerHTML" not in app
```

Extend the checked-in bundle test to require 75 experiments, 35 labeled queries per experiment, a valid recommended experiment, and valid candidate/member references.

- [ ] **Step 2: Verify viewer tests fail**

```bash
python -m pytest tests/test_sciegqa_parser_viewer.py -q
```

Expected: failure because the final-label controls are absent.

- [ ] **Step 3: Add semantic controls**

Add to `index.html`:

```html
<section id="labeling-controls">
  <h2>Final labeling experiment</h2>
  <label>Configuration <select id="labeling-experiment"></select></label>
  <div id="recommended-experiment"></div>
  <fieldset id="label-filters"><legend>Labels</legend></fieldset>
  <fieldset id="candidate-kind-filters"><legend>Candidate kinds</legend></fieldset>
  <dl id="experiment-summary"></dl>
</section>
```

Add the two approved modes to `modeLayers`. Extend state with:

```javascript
experimentId: null,
enabledLabels: new Set(["positive", "partial_manual_review", "negative"]),
enabledKinds: new Set(),
```

Render candidates by looking up the current experiment and query labels. For every visible candidate, append one SVG rectangle for every `member_bboxes_norm_1000` entry. Apply `data-candidate-id` and `data-label` to every member rectangle and show one shared payload in the details panel.

Populate configuration options and summaries using only `textContent`. Changing experiment, query, label filter, or kind filter calls `renderBoxes()` and `renderExperimentSummary()` without rebuilding candidates.

- [ ] **Step 4: Add non-obscuring styles**

```css
.box-final-positive { stroke: #047857; stroke-width: 3; }
.box-final-partial_manual_review { stroke: #d97706; stroke-dasharray: 9 4; }
.box-final-negative { stroke: #64748b; stroke-dasharray: 2 4; }
.box-final-member-active { stroke-width: 5; }
```

All final boxes retain `fill: none` and `vector-effect: non-scaling-stroke`.

- [ ] **Step 5: Run viewer tests**

```bash
python -m pytest tests/test_sciegqa_parser_viewer.py -q
```

Expected: pass after the manifest is regenerated in Task 5; static code assertions should pass immediately.

- [ ] **Step 6: Commit viewer implementation**

```bash
git branch --show-current
git add viewer/sciegqa_parser_compare/index.html viewer/sciegqa_parser_compare/app.js viewer/sciegqa_parser_compare/styles.css tests/test_sciegqa_parser_viewer.py
git commit -m "feat: visualize final SciEGQA labels"
```

## Task 5: Run pilot experiments and check in results

**Files:**

- Modify: `viewer/sciegqa_parser_compare/manifest.json`
- Modify: `viewer/sciegqa_parser_compare/README.md`
- Modify: `tests/test_sciegqa_parser_viewer.py`

- [ ] **Step 1: Generate a fresh enriched manifest**

```bash
cp viewer/sciegqa_parser_compare/manifest.json /tmp/sciegqa-parser-manifest.original.json
python scripts/build_sciegqa_final_labeling.py \
  --manifest /tmp/sciegqa-parser-manifest.original.json \
  --output viewer/sciegqa_parser_compare/manifest.json
```

- [ ] **Step 2: Verify deterministic regeneration**

```bash
python scripts/build_sciegqa_final_labeling.py \
  --manifest /tmp/sciegqa-parser-manifest.original.json \
  --output /tmp/sciegqa-parser-manifest.second.json
shasum -a 256 \
  viewer/sciegqa_parser_compare/manifest.json \
  /tmp/sciegqa-parser-manifest.second.json
```

Expected: identical SHA-256 hashes.

- [ ] **Step 3: Inspect experiment results**

Run a Python check that prints:

```text
candidate counts by kind
headed/unheaded pages
recommended experiment and tie-break fields
matched/unmatched queries for all 75 experiments
positive/partial/negative counts
unpaired caption diagnostics
```

Fail if there are not exactly 32 pages, 35 queries, and 75 experiments, or if any query/candidate/member reference is missing.

- [ ] **Step 4: Document controls and regeneration**

Add to the viewer README:

```bash
python scripts/build_sciegqa_final_labeling.py \
  --manifest viewer/sciegqa_parser_compare/manifest.json \
  --output viewer/sciegqa_parser_compare/manifest.json
```

Explain that the checked-in results are pilot-only and partial overlaps require manual review.

- [ ] **Step 5: Run focused and related tests**

```bash
python -m pytest \
  tests/test_sciegqa_final_labeling.py \
  tests/test_sciegqa_parser_viewer.py \
  tests/test_sciegqa_parser_metrics.py \
  tests/test_deepseek_grounding.py -q
```

Expected: all pass.

- [ ] **Step 6: Commit generated results**

```bash
git branch --show-current
git add viewer/sciegqa_parser_compare/manifest.json viewer/sciegqa_parser_compare/README.md tests/test_sciegqa_parser_viewer.py
git commit -m "data: add SciEGQA labeling experiments"
```

## Task 6: Browser and end-to-end verification

**Files:**

- Modify only if a verified defect is found in files already in scope.

- [ ] **Step 1: Start the static site**

```bash
cd viewer/sciegqa_parser_compare
python3 -m http.server 8000 --bind 127.0.0.1
```

- [ ] **Step 2: Verify application state**

Using the in-app browser, verify:

- the site loads without console errors;
- the recommended experiment is selected;
- both final-label modes render;
- switching experiments changes the summary and expected fallback candidates;
- label and candidate-kind filters update rectangle counts;
- switching among all 35 queries updates labels;
- original eight parser modes retain their exact rectangle counts.

- [ ] **Step 3: Verify exact final rectangles programmatically**

For each of 32 pages in the recommended experiment and each of the two final modes, assert in the DOM:

```text
actual final SVG rectangle count
= sum(member box counts for visible query candidates after filters)
```

This is 64 final-layer checks. Also inspect at least one headed-text page, one no-heading fallback page, one figure-caption page, one table page, one partial-review case, and one unmatched query visually.

- [ ] **Step 4: Run final verification**

```bash
python -m pytest \
  tests/test_sciegqa_final_labeling.py \
  tests/test_sciegqa_parser_viewer.py \
  tests/test_sciegqa_parser_metrics.py \
  tests/test_deepseek_grounding.py -q
git diff --check
git status --short
```

Expected: tests pass, diff check is clean, and only intentional scoped changes remain.

- [ ] **Step 5: Final commit if browser verification required fixes**

```bash
git branch --show-current
git add scripts/build_sciegqa_final_labeling.py scripts/sciegqa_parser_compare/final_labeling.py tests/test_sciegqa_final_labeling.py tests/test_sciegqa_parser_viewer.py viewer/sciegqa_parser_compare
git commit -m "fix: verify SciEGQA final labeling viewer"
```

Skip this commit when the worktree is already clean.
