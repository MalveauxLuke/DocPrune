# SciEGQA Segment Linkage and Document Preamble Design

## Objective

Make final semantic candidates visually identifiable as groups without changing the meaning of the existing label colors. Preserve document preambles as normal evidence candidates instead of discarding content before the first `##` heading.

## Candidate construction

The DeepSeek-OCR-2 headed-text route continues to treat `##` through `######` headings as semantic section boundaries. Long headed sections remain intact; column transitions, spatial gaps, and token limits do not split them.

When at least one accepted `##` heading exists, eligible DeepSeek-OCR-2 blocks before the first accepted heading form one `document_preamble` candidate. Eligible members are title, text, and formula blocks plus nonempty unknown text blocks, kept in reading order. Visual objects and captions remain excluded because MinerU owns visual bundles. The candidate retains every original member bounding box and does not receive an enclosing rectangle.

The preamble is a full candidate. It uses the same union-geometry overlap calculation, positive/partial/negative labels, experiment thresholds, provenance fields, and downstream VLM eligibility as other candidates. If no accepted `##` heading exists, the existing MinerU paragraph-window fallback remains authoritative and no page-wide preamble candidate is created. If there are no eligible leading blocks, no empty candidate is emitted.

## Viewer behavior

Label colors retain their current meaning:

- green: positive
- orange: partial/manual review
- gray: negative
- red: SciEGQA gold evidence

For the active experiment, candidates receive deterministic page-local display identifiers (`S1`, `S2`, and so on) in reading order. The numbering is computed before label and kind filters, so filtering does not renumber candidates. Switching experiments may change numbering because the candidate set may change.

Every member box displays the same segment badge. While any member box is hovered or focused, all boxes with that candidate ID are emphasized and unrelated final candidates are visually de-emphasized. A segment legend lists each display identifier, candidate kind, current query label, and a short text preview. Hovering or focusing a legend entry produces the same linked highlighting and shows the existing structured candidate details.

Raw parser modes remain unchanged. Segment identifiers and the legend appear only in `Final labeling only` and `Gold + final labeling` modes.

## Data flow

1. The deterministic preprocessing code reads normalized DeepSeek-OCR-2 segments.
2. It builds the preamble, headed-text, MinerU fallback, and MinerU visual candidates.
3. Gold-overlap evaluation labels every eligible candidate, including the preamble.
4. The generated manifest stores the new candidate kind using the existing candidate schema.
5. The viewer derives display identifiers from the active experiment and renders repeated badges plus the linked legend.

Stable candidate IDs remain the provenance key. `S#` identifiers are presentation-only and must never be written back as dataset identity.

## Failure behavior

Invalid member geometry continues to fail manifest generation through existing bounding-box validation. Empty preambles are skipped. Missing query labels suppress the affected final candidate exactly as they do now. The viewer must not infer membership from color; all linking uses the stable `candidate_id`.

## Verification

Unit tests will cover preamble creation, member ordering, exclusion of visual blocks, absence of empty preambles, no-heading fallback behavior, and normal gold-overlap labeling. Viewer tests will cover deterministic display numbering, repeated badges, filter-stable numbering, legend contents, and linked hover/focus classes.

Targeted browser verification will inspect the current polyominoes example, confirm that title/authors/affiliations form one preamble candidate, and confirm that every box belonging to a candidate shares its badge and highlights together. Existing overlay rectangle-count assertions will be retained to detect geometry regressions.
