# SciEGQA Final Segment Labeling Design

## Objective

Extend the self-contained 32-page, 35-query SciEGQA parser-comparison viewer
with a deterministic final-candidate layer. Final candidates route visual
content through MinerU and semantically headed text through DeepSeek-OCR-2,
then compare each candidate with each SciEGQA gold box on its page.

This pilot uses only the checked-in comparison artifacts. It does not rerun
MinerU or DeepSeek-OCR-2 and does not process the full SciEGQA subset.

## Fixed segmentation policy

Final candidates are generated before VLM inference.

### Visual candidates

MinerU is authoritative for:

- figures;
- charts, represented by MinerU figure-like types;
- tables;
- figure, chart, and table captions.

A caption anchors one visual candidate. Every compatible nearby visual object
is assigned to its nearest caption, so a multi-panel figure may contribute
several visual boxes to one captioned candidate. The candidate retains each
source segment and bounding box independently. It never replaces them with one
enclosing rectangle.

Caption linkage uses compatible MinerU types and normalized box-center
distance. Figure and chart objects match figure or chart captions; tables match
table captions. A caption appears in only its own anchored candidate but may
own multiple visual panels. Unpaired visual objects remain valid candidates,
while unpaired captions remain visible in diagnostics and are not silently
attached to distant objects.

### Headed text candidates

DeepSeek-OCR-2 is authoritative for text on pages containing Markdown headings.
A heading is a DeepSeek segment whose Markdown begins with at least two `#`
characters followed by whitespace.

Each heading starts a candidate. The candidate contains the heading and all
subsequent text or formula blocks until the next qualifying heading or the end
of the page. Every qualifying heading is a boundary regardless of apparent
heading depth because OCR-2 can flatten section and subsection levels.

Heading candidates are semantic units. They are never split by columns,
coordinates, spatial gaps, token limits, or paragraph-count limits. DeepSeek
figure, table, caption, and unknown blocks are excluded from headed-text
candidates because MinerU owns those modalities.

### Unheaded text fallback

If a page contains no qualifying DeepSeek-OCR-2 headings, MinerU text, formula,
and nonempty `unknown` text segments become the fallback source. Empty unknown
regions remain excluded. The experiment varies the maximum number of
consecutive MinerU paragraphs in a candidate from one through five.

For a maximum of `N`, preprocessing emits consecutive reading-order windows of
sizes `1..N`. Windows slide by one segment so evidence crossing an arbitrary
window boundary is not lost. This heuristic applies only to unheaded pages and
never splits or modifies a DeepSeek heading candidate.

## Candidate contract

Every candidate records:

- stable candidate ID and configuration-independent base ID;
- page ID;
- source parser and exact parser revision;
- candidate kind: `visual_bundle`, `headed_text`, or `mineru_text_fallback`;
- heading text when present;
- concatenated Markdown and plain text in reading order;
- member segment IDs;
- original member bounding boxes in normalized 0–1000 coordinates;
- member reading-order positions and types;
- word count, paragraph count, and member count;
- caption-link provenance when applicable.

Candidate geometry is the union of member boxes. An enclosing display rectangle
is not part of the candidate contract.

## Gold comparison and labels

Each candidate is compared with every SciEGQA query on the same page.

For gold box `G` and candidate member-box union `C`:

```text
gold_coverage = area(G intersect C) / area(G)
candidate_precision = area(G intersect C) / area(C)
```

The experiment varies `gold_coverage_threshold`. Labels are:

- `positive`: `gold_coverage >= threshold`;
- `partial_manual_review`: `0 < gold_coverage < threshold`;
- `negative`: `gold_coverage == 0`.

A query receives query-level status `unmatched` when none of its candidates is
positive under the selected configuration. Continuous measurements are always
retained; changing a label threshold never discards geometry.

Candidate precision is diagnostic and does not control the initial label. This
preserves the approved assumption that a complete semantic unit is preferable
to a geometrically tight but semantically arbitrary block.

## Experiment grid

Run the Cartesian product:

```text
gold_coverage_threshold: 0.50, 0.70, 0.80, 0.90, 0.95
fallback_max_paragraphs:  1, 2, 3, 4, 5
caption_max_center_distance: 150, 250, 350
```

This produces 75 deterministic configurations over the existing pilot.

For every configuration report:

- total candidates by kind;
- positive, partial-review, and negative query-candidate pairs;
- matched and unmatched queries;
- anchor and all-linked-query results separately;
- positive candidate gold-coverage and candidate-precision distributions;
- candidate word, paragraph, and member-count distributions;
- visual objects with no caption and captions with no object;
- headed and unheaded page counts.

The experiment surfaces a Pareto frontier rather than claiming universal
optimality from 35 queries. The default pilot recommendation uses threshold
`0.80` and the paragraph/distance configuration that attains the best query
match rate observed at that threshold. Remaining ties prefer fewer candidates,
then lower median candidate word count, then the smaller paragraph maximum,
then the smaller caption distance.

## Viewer integration

Extend `viewer/sciegqa_parser_compare` rather than creating another site. The
existing gold, MinerU, OCR-1, and OCR-2 layers remain unchanged.

Add:

- `Final labeling only` mode;
- `Gold + final labeling` mode;
- experiment selector showing threshold and paragraph maximum;
- label filters for positive, partial manual review, and negative;
- candidate-kind filters;
- experiment summary and recommended-configuration indicator;
- selected candidate details including all metrics and provenance.

Each candidate renders one outline per member box. Member outlines share the
same candidate ID, label style, hover state, and focus state. They never fill or
obscure page content.

Suggested label styles:

- positive: solid dark green;
- partial manual review: dashed amber;
- negative: dotted neutral gray;
- SciEGQA gold: existing solid red.

The query selector determines which candidate labels are displayed. Switching
queries, experiments, or filters recomputes only the visible layer from the
precomputed manifest; the browser does not regenerate candidates or labels.

## Artifact and schema strategy

Candidate generation and experiment evaluation run in Python. The static
viewer consumes precomputed data. The checked-in manifest is regenerated to
include:

```text
labeling.experiments
labeling.recommended_experiment_id
labeling.candidates_by_page
labeling.query_candidate_labels
```

The preprocessing implementation remains independent of the viewer so the
same deterministic candidate and labeling logic can later run over the full
SciEGQA subset.

## Validation and failure behavior

Preprocessing fails closed on:

- duplicate candidate or experiment IDs;
- missing pages, queries, or referenced source segments;
- invalid or out-of-range member boxes;
- unknown experiment IDs;
- a visual object assigned to multiple caption candidates;
- a headed page that produces no headed-text candidate;
- labels inconsistent with stored coverage and threshold;
- a recommended configuration that does not satisfy the deterministic
  selection rule.

Malformed or unpaired content is preserved in diagnostics. It is not silently
deleted or reassigned.

## Testing

Focused tests cover:

- heading detection and section boundaries;
- preservation of very long heading sections;
- exclusion of OCR-2 visual modalities from text candidates;
- MinerU distance-based visual/caption linkage, multi-panel bundles, and
  unpaired diagnostics;
- no-heading fallback windows for paragraph maxima one through five;
- no-heading recovery of nonempty MinerU unknown text without admitting empty
  unknown regions;
- exact union-area geometry for multi-box candidates;
- positive, partial-review, negative, and unmatched states;
- deterministic experiment IDs and recommendation selection;
- viewer manifest joins and rejection of invalid references;
- final-label overlay modes, filters, shared candidate identity, and exact DOM
  rectangle counts;
- regression of every existing parser overlay mode.

The finished website is visually checked on all 32 pages for the recommended
configuration, with special attention to multi-box heading sections,
figure-caption pairs, table-caption pairs, and partial-overlap cases.

## Scope boundary

This work does not:

- rerun document parsers;
- alter MinerU or DeepSeek atomic outputs;
- split heading sections using nonsemantic heuristics;
- make candidate precision a label condition;
- manually resolve partial-overlap cases;
- train the downstream binary VLM;
- process or label the full 4,000-query subset.
