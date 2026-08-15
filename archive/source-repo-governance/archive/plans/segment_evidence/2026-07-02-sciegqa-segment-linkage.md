# SciEGQA Segment Linkage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve document preambles as labelable semantic candidates and make every multi-box candidate visibly identifiable in the existing SciEGQA viewer.

**Architecture:** Extend the deterministic Python candidate builder with one page-level `document_preamble` candidate before the first accepted `##` heading. Keep stable candidate IDs as provenance, then derive experiment-specific, page-local `S#` display identifiers in JavaScript for repeated SVG badges, linked highlighting, and a segment legend.

**Tech Stack:** Python 3, pytest, static HTML/CSS, browser-native JavaScript and SVG, checked-in JSON manifest.

---

## File map

- Modify `scripts/sciegqa_parser_compare/final_labeling.py`: construct and route full preamble candidates.
- Modify `tests/test_sciegqa_final_labeling.py`: specify preamble membership, exclusions, routing, and overlap labels.
- Modify `viewer/sciegqa_parser_compare/app.js`: derive `S#` identities, render badges and legend, and link hover/focus state.
- Modify `viewer/sciegqa_parser_compare/index.html`: add the segment-legend container.
- Modify `viewer/sciegqa_parser_compare/styles.css`: style badges, legend entries, and linked emphasis.
- Modify `tests/test_sciegqa_parser_viewer.py`: enforce the checked-in viewer and manifest contract.
- Regenerate `viewer/sciegqa_parser_compare/manifest.json`: include preamble candidates and recomputed experiment summaries.
- Modify `viewer/sciegqa_parser_compare/README.md`: explain label colors, `S#` membership, and preamble candidates.

### Task 1: Specify document-preamble construction

**Files:**
- Modify: `tests/test_sciegqa_final_labeling.py`
- Test: `tests/test_sciegqa_final_labeling.py`

- [ ] **Step 1: Import the new candidate builder in the test module**

Add `build_document_preamble_candidate` to the existing import from `scripts.sciegqa_parser_compare.final_labeling`.

- [ ] **Step 2: Write failing tests for preamble membership and exclusions**

Add these tests:

```python
def test_document_preamble_preserves_leading_text_as_one_candidate() -> None:
    segments = [
        segment("title", "deepseek_ocr2", "title", 0, "# Paper", [10, 10, 90, 20]),
        segment("authors", "deepseek_ocr2", "text", 1, "Ada Lovelace", [10, 20, 90, 30]),
        segment("formula", "deepseek_ocr2", "formula", 2, "x = 1", [10, 30, 90, 40]),
        segment("noise", "deepseek_ocr2", "unknown", 3, "affiliation", [10, 40, 90, 50]),
        segment("figure", "deepseek_ocr2", "figure", 4, "image", [10, 50, 90, 70]),
        segment("caption", "deepseek_ocr2", "figure_caption", 5, "Figure 1", [10, 70, 90, 80]),
        segment("heading", "deepseek_ocr2", "title", 6, "## Abstract", [10, 80, 90, 90]),
        segment("body", "deepseek_ocr2", "text", 7, "answer", [10, 90, 90, 100]),
    ]

    candidate = build_document_preamble_candidate("page", segments)

    assert candidate is not None
    assert candidate["kind"] == "document_preamble"
    assert candidate["heading"] == "# Paper"
    assert candidate["member_segment_ids"] == [
        "title", "authors", "formula", "noise"
    ]
    assert candidate["member_bboxes_norm_1000"] == [
        [10, 10, 90, 20],
        [10, 20, 90, 30],
        [10, 30, 90, 40],
        [10, 40, 90, 50],
    ]


def test_document_preamble_requires_heading_and_nonempty_members() -> None:
    no_heading = [
        segment("title", "deepseek_ocr2", "title", 0, "# Paper", [1, 1, 9, 2])
    ]
    empty_leading = [
        segment("empty", "deepseek_ocr2", "unknown", 0, "", [1, 1, 9, 2]),
        segment("heading", "deepseek_ocr2", "title", 1, "## Abstract", [1, 2, 9, 3]),
    ]

    assert build_document_preamble_candidate("page", no_heading) is None
    assert build_document_preamble_candidate("page", empty_leading) is None
```

- [ ] **Step 3: Run the new tests and verify the intended failure**

Run:

```bash
python -m pytest -q \
  tests/test_sciegqa_final_labeling.py::test_document_preamble_preserves_leading_text_as_one_candidate \
  tests/test_sciegqa_final_labeling.py::test_document_preamble_requires_heading_and_nonempty_members
```

Expected: collection fails because `build_document_preamble_candidate` does not exist.

- [ ] **Step 4: Commit the red tests**

```bash
git branch --show-current
git add tests/test_sciegqa_final_labeling.py
git commit -m "test: specify document preamble candidates"
```

### Task 2: Implement and route document-preamble candidates

**Files:**
- Modify: `scripts/sciegqa_parser_compare/final_labeling.py`
- Modify: `tests/test_sciegqa_final_labeling.py`
- Test: `tests/test_sciegqa_final_labeling.py`

- [ ] **Step 1: Implement the focused preamble builder**

Near `build_headed_text_candidates`, add:

```python
PREAMBLE_TEXT_TYPES = frozenset({"title", "text", "formula"})


def build_document_preamble_candidate(
    page_id: str, segments: Sequence[Mapping[str, Any]]
) -> dict[str, Any] | None:
    ordered = _ordered(segments)
    heading_positions = [
        index for index, row in enumerate(ordered) if is_markdown_heading(row)
    ]
    if not heading_positions:
        return None
    members = []
    for row in ordered[: heading_positions[0]]:
        segment_type = str(row["type"])
        if segment_type in PREAMBLE_TEXT_TYPES:
            members.append(row)
        elif segment_type == "unknown" and _content(row, "text").strip():
            members.append(row)
    if not members:
        return None
    first_markdown = _content(members[0], "markdown").strip()
    heading = first_markdown if first_markdown.startswith("# ") else None
    return _make_candidate(
        page_id,
        "document_preamble",
        members,
        heading=heading,
    )
```

- [ ] **Step 2: Route the preamble only on headed pages**

In `build_final_labeling`, replace the headed-page assignment with the following behavior:

```python
headed = build_headed_text_candidates(page_id, deepseek)
if headed:
    page_modes[page_id] = "headed_text"
    preamble = build_document_preamble_candidate(page_id, deepseek)
    text_candidates = [*([preamble] if preamble else []), *headed]
else:
    page_modes[page_id] = "mineru_text_fallback"
    text_candidates = build_fallback_text_candidates(
        page_id,
        mineru,
        max(FALLBACK_PARAGRAPH_MAXIMA),
    )
```

Do not add special availability or label logic: non-fallback candidates are already active in every experiment and already use union geometry.

- [ ] **Step 3: Extend the routing fixture and assert full-candidate labeling**

Prepend these DeepSeek-OCR-2 rows to the headed-page fixture, with the existing `## Heading` rows shifted later in reading order:

```python
segment("doc", "deepseek_ocr2", "title", 0, "# Paper", [20, 0, 40, 2]),
segment("authors", "deepseek_ocr2", "text", 1, "Author", [20, 2, 40, 4]),
```

Update the headed-kind assertion to:

```python
assert {row["kind"] for row in headed} == {
    "visual_bundle", "document_preamble", "headed_text"
}
```

Add this query to the headed page fixture:

```python
{
    "query_id": "qp",
    "is_anchor": False,
    "source_bbox_norm_1000": [20, 0, 40, 4],
},
```

After selecting `experiment_id`, assert the preamble is a normal labeled candidate:

```python
preamble_id = next(
    row["candidate_id"]
    for row in headed
    if row["kind"] == "document_preamble"
)
preamble_labels = {
    row["candidate_id"]: row["label"]
    for row in labels["qp"]["candidates"]
}
assert preamble_labels[preamble_id] == "positive"
assert {
    label for candidate_id, label in preamble_labels.items()
    if candidate_id != preamble_id
} == {"negative"}
```

- [ ] **Step 4: Run candidate tests**

Run:

```bash
python -m pytest -q tests/test_sciegqa_final_labeling.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit candidate construction**

```bash
git branch --show-current
git add scripts/sciegqa_parser_compare/final_labeling.py tests/test_sciegqa_final_labeling.py
git commit -m "feat: preserve document preamble candidates"
```

### Task 3: Specify linked segment identities in the viewer

**Files:**
- Modify: `tests/test_sciegqa_parser_viewer.py`
- Test: `tests/test_sciegqa_parser_viewer.py`

- [ ] **Step 1: Add static viewer contract assertions**

Extend `test_static_viewer_uses_svg_textcontent_and_all_modes` with:

```python
assert 'id="segment-legend"' in html
assert "candidateDisplayRows" in app
assert "segment-badge" in app
assert "segment-legend-entry" in app
assert "candidate-muted" in app
assert "display_id" in app
```

- [ ] **Step 2: Require the checked-in pilot to contain the example preamble**

In `test_checked_in_viewer_bundle_is_self_contained`, locate `page_0082404e743dffd1c69e` and assert:

```python
polyominoes = next(
    page for page in manifest["pages"]
    if page["page_id"] == "page_0082404e743dffd1c69e"
)
preamble = next(
    candidate
    for candidate in labeling["candidates_by_page"][polyominoes["page_id"]]
    if candidate["kind"] == "document_preamble"
)
assert len(preamble["member_bboxes_norm_1000"]) >= 3
assert "Translational Aperiodic Sets" in preamble["markdown"]
assert "Chao Yang" in preamble["markdown"]
```

- [ ] **Step 3: Run the viewer tests and verify failure**

Run:

```bash
python -m pytest -q tests/test_sciegqa_parser_viewer.py
```

Expected: failure because the legend, badges, display rows, and regenerated preamble are absent.

- [ ] **Step 4: Commit the red viewer contract**

```bash
git branch --show-current
git add tests/test_sciegqa_parser_viewer.py
git commit -m "test: specify linked segment viewer"
```

### Task 4: Render deterministic segment badges and legend

**Files:**
- Modify: `viewer/sciegqa_parser_compare/index.html`
- Modify: `viewer/sciegqa_parser_compare/app.js`
- Modify: `viewer/sciegqa_parser_compare/styles.css`
- Test: `tests/test_sciegqa_parser_viewer.py`

- [ ] **Step 1: Add the legend container**

Insert this section after the final-label experiment controls:

```html
<section id="segment-legend" hidden>
  <h2>Semantic segments</h2>
  <p>Matching S-numbers belong to one segment.</p>
  <div id="segment-legend-items"></div>
</section>
```

- [ ] **Step 2: Derive stable display rows before filtering**

Add a pure helper that joins active experiment labels to page candidates, filters only by experiment availability, sorts by reading order and candidate ID, and assigns page-local identifiers:

```javascript
function candidateDisplayRows(page, query) {
  const labels = state.manifest.labeling.query_candidate_labels[state.experimentId][query.query_id];
  const labelIndex = new Map(labels.candidates.map(item => [item.candidate_id, item]));
  return state.manifest.labeling.candidates_by_page[page.page_id]
    .filter(candidate => labelIndex.has(candidate.candidate_id))
    .sort((left, right) => {
      const leftOrder = Math.min(...left.member_reading_orders);
      const rightOrder = Math.min(...right.member_reading_orders);
      return leftOrder - rightOrder || left.candidate_id.localeCompare(right.candidate_id);
    })
    .map((candidate, index) => ({
      candidate,
      labeling: labelIndex.get(candidate.candidate_id),
      display_id: `S${index + 1}`,
    }));
}
```

Do not number after label or kind filtering; hidden rows retain their position.

- [ ] **Step 3: Replace membership styling with linked emphasis**

Update `setCandidateActive` so every final box, badge, and legend entry with `data-candidate-id` receives either `candidate-active` or `candidate-muted` while a candidate is active. When the hover/focus ends, remove both classes from every node.

```javascript
function setCandidateActive(candidateId, active) {
  document.querySelectorAll("[data-candidate-id]").forEach(node => {
    const matches = node.dataset.candidateId === candidateId;
    node.classList.toggle("candidate-active", active && matches);
    node.classList.toggle("candidate-muted", active && !matches);
  });
}
```

- [ ] **Step 4: Render one badge for every member box**

Add an SVG text helper that places a badge near the member's upper-left corner without altering the stored geometry:

```javascript
function segmentBadge(box, displayId, candidateId) {
  const node = document.createElementNS(SVG_NS, "text");
  node.setAttribute("x", box[0] + 5);
  node.setAttribute("y", Math.max(18, box[1] - 6));
  node.setAttribute("class", "segment-badge");
  node.dataset.candidateId = candidateId;
  node.textContent = displayId;
  node.setAttribute("aria-hidden", "true");
  return node;
}
```

Append the badge immediately after each final candidate rectangle. Keep `pointer-events: none` on badges so the underlying member box remains the hover/focus target.

- [ ] **Step 5: Render the linked segment legend**

Use one shared payload and one eligibility function for boxes and legend rows:

```javascript
function candidateEnabled(row) {
  return state.enabledLabels.has(row.labeling.label)
    && state.enabledKinds.has(row.candidate.kind);
}

function candidatePayload(row, queryStatus) {
  return {
    candidate: row.candidate,
    labeling: row.labeling,
    display_id: row.display_id,
    experiment: currentExperiment(),
    query_status: queryStatus,
  };
}

function attachCandidateEvents(node, payload) {
  const candidateId = payload.candidate.candidate_id;
  node.dataset.candidateId = candidateId;
  node.addEventListener("mouseenter", () => {
    setCandidateActive(candidateId, true);
    showDetails(payload);
  });
  node.addEventListener("mouseleave", () => setCandidateActive(candidateId, false));
  node.addEventListener("focus", () => {
    setCandidateActive(candidateId, true);
    showDetails(payload);
  });
  node.addEventListener("blur", () => setCandidateActive(candidateId, false));
}

function renderSegmentLegend(rows, queryStatus) {
  const section = document.querySelector("#segment-legend");
  const items = document.querySelector("#segment-legend-items");
  items.replaceChildren();
  const visibleRows = rows.filter(candidateEnabled);
  section.hidden = false;
  for (const row of visibleRows) {
    const payload = candidatePayload(row, queryStatus);
    const preview = (
      row.candidate.text
      || row.candidate.markdown
      || "No extracted text"
    ).replace(/\s+/g, " ").slice(0, 100);
    const button = element(
      "button",
      `${row.display_id} · ${row.candidate.kind} · ${row.labeling.label} · ${preview}`,
    );
    button.type = "button";
    button.className = "segment-legend-entry";
    attachCandidateEvents(button, payload);
    items.append(button);
  }
}
```

Refactor `renderFinalBoxes` to iterate over `candidateDisplayRows(page, query)`, skip rows only through `candidateEnabled`, attach the shared payload to every member rectangle, and append a `segmentBadge` after every rectangle. Call `renderSegmentLegend` with the unfiltered display rows.

At the beginning of `renderBoxes`, clear `#segment-legend-items` and set `#segment-legend.hidden = true`; rendering the final layer shows it again. Update `showDetails` with explicit title selection:

```javascript
let title = "Gold evidence";
if (payload.parser) title = `${parserNames[payload.parser]} segment`;
if (payload.candidate) title = `Final ${payload.display_id} segment`;
panel.replaceChildren(
  element("h2", title),
  element("pre", JSON.stringify(payload, null, 2)),
);
```

- [ ] **Step 6: Add non-obscuring styles**

Add these styles:

```css
.segment-badge {
  fill: #111827;
  font: 700 18px system-ui, sans-serif;
  paint-order: stroke;
  stroke: white;
  stroke-width: 5px;
  stroke-linejoin: round;
  pointer-events: none;
}
.candidate-muted { opacity: .16; }
.box.candidate-active { stroke-width: 5; }
#segment-legend-items { display: grid; gap: .35rem; }
.segment-legend-entry {
  margin: 0;
  padding: .45rem .55rem;
  text-align: left;
  border: 1px solid #cbd5e1;
  border-radius: .35rem;
  background: #f8fafc;
}
.segment-legend-entry.candidate-active { border-color: #0f172a; box-shadow: 0 0 0 2px #0f172a; }
```

- [ ] **Step 7: Run viewer tests**

Run:

```bash
python -m pytest -q tests/test_sciegqa_parser_viewer.py
```

Expected: static UI assertions pass; only the checked-in preamble assertion remains red until manifest regeneration.

- [ ] **Step 8: Commit the viewer implementation**

```bash
git branch --show-current
git add viewer/sciegqa_parser_compare/index.html \
  viewer/sciegqa_parser_compare/app.js \
  viewer/sciegqa_parser_compare/styles.css
git commit -m "feat: show linked semantic segments"
```

### Task 5: Regenerate the pilot and document the interaction

**Files:**
- Modify: `viewer/sciegqa_parser_compare/manifest.json`
- Modify: `viewer/sciegqa_parser_compare/README.md`
- Test: `tests/test_sciegqa_final_labeling.py`
- Test: `tests/test_sciegqa_parser_viewer.py`

- [ ] **Step 1: Regenerate final labeling atomically**

Run:

```bash
python scripts/build_sciegqa_final_labeling.py \
  --manifest viewer/sciegqa_parser_compare/manifest.json \
  --output viewer/sciegqa_parser_compare/manifest.json
```

Expected: JSON output reports 75 experiments and the recommended experiment ID.

- [ ] **Step 2: Verify deterministic regeneration**

Capture a digest, run the same command a second time, and require the digest to remain identical:

```bash
first_hash=$(shasum -a 256 viewer/sciegqa_parser_compare/manifest.json | awk '{print $1}')
python scripts/build_sciegqa_final_labeling.py \
  --manifest viewer/sciegqa_parser_compare/manifest.json \
  --output viewer/sciegqa_parser_compare/manifest.json
second_hash=$(shasum -a 256 viewer/sciegqa_parser_compare/manifest.json | awk '{print $1}')
test "$first_hash" = "$second_hash"
```

Expected: `test` exits successfully and both hashes are identical.

- [ ] **Step 3: Update the viewer README**

Document these user-visible rules:

```markdown
- Label colors describe query overlap, not segment identity.
- Every repeated `S#` badge marks boxes belonging to one semantic candidate.
- Hover or focus a box or legend entry to emphasize all linked members.
- `document_preamble` is a full DeepSeek-OCR-2 candidate containing eligible title, author, and affiliation blocks before the first `##` section.
```

- [ ] **Step 4: Run the targeted suites**

Run:

```bash
python -m pytest -q \
  tests/test_sciegqa_final_labeling.py \
  tests/test_sciegqa_parser_viewer.py
```

Expected: all targeted tests pass.

- [ ] **Step 5: Commit the regenerated artifact and documentation**

```bash
git branch --show-current
git add viewer/sciegqa_parser_compare/manifest.json \
  viewer/sciegqa_parser_compare/README.md \
  tests/test_sciegqa_parser_viewer.py
git commit -m "feat: publish preamble labeling pilot"
```

### Task 6: End-to-end and visual verification

**Files:**
- Verify only; no planned source edits.

- [ ] **Step 1: Run repository verification**

Run:

```bash
python -m pytest -q
git diff --check
git status --short
```

Expected: the full test suite passes, `git diff --check` is silent, and the worktree is clean after commits.

- [ ] **Step 2: Start the checked-in viewer**

Run from `viewer/sciegqa_parser_compare`:

```bash
python3 -m http.server 8765
```

Open `http://127.0.0.1:8765/?page=0&mode=Gold%20%2B%20final%20labeling`.

- [ ] **Step 3: Verify the polyominoes preamble visually**

On `page_0082404e743dffd1c69e`, confirm:

- title, authors, and affiliations have one repeated `S#` identifier;
- the abstract has a different `S#` identifier;
- the preamble boxes preserve their original separate geometry;
- hovering one preamble member emphasizes all preamble members and dims the abstract and other candidates;
- the legend entry triggers the same emphasis and reports `document_preamble`;
- gold remains red and label colors remain unchanged.

- [ ] **Step 4: Retain exact DOM geometry assertions**

For all 32 pages in `Final labeling only` and `Gold + final labeling`, assert that the number of `.box[data-candidate-id]` rectangles equals the sum of member boxes for enabled candidates in the selected experiment. Assert that `.segment-badge` count equals the final member-box count, every visible candidate has one legend entry, and every badge text matches its candidate's legend identifier.

For all 32 pages in `All parsers`, retain the existing expected rectangle-count check and assert that no segment badges or legend entries are visible.

- [ ] **Step 5: Inspect browser errors**

Confirm the browser console contains no warnings or errors after page navigation, experiment switching, kind filtering, label filtering, box hover, and legend focus.

- [ ] **Step 6: Stop the local server and report evidence**

Stop the server after verification. Report the exact test totals, DOM assertion totals, manifest hash, and any changed recommended-experiment summary metrics without claiming results that were not observed.
