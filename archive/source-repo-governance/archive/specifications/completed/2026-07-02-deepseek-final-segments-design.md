# DeepSeek-OCR-2 Final Segments Design

## Scope

This design supersedes the candidate-source policy in the earlier final-labeling pilot. DeepSeek-OCR-2 is the exclusive source of final text and visual candidates. MinerU remains an optional visual-link inspection layer and never contributes final labels.

## Final candidates

- If DeepSeek-OCR-2 Markdown contains `##` through `######` headings, each heading and its following text blocks form one unsplit `headed_text` candidate.
- Eligible content before the first heading remains one `document_preamble` candidate.
- If no accepted heading exists, each nonempty DeepSeek-OCR-2 text, formula, or title block becomes one atomic `deepseek_paragraph` candidate. No overlapping MinerU windows are generated.
- DeepSeek-OCR-2 figure, chart, and table boxes are visual objects. Primary captions are detected from normalized Markdown/text prefixes `Figure`, `Fig.`, `Table`, and `Chart`. A `Note` block may extend an immediately preceding primary caption but cannot anchor a link by itself.
- Compatible captions and visual objects are linked using reading-order adjacency and normalized center distance. A linked object-caption group becomes one `visual_bundle` candidate while retaining every original member box. Multi-panel objects may share one caption.
- Empty blocks, incompatible caption types, and links beyond the parser-specific pilot distance are excluded and recorded as diagnostics. For each parser, the distance is the nearest compatible caption-object center-distance p95, rounded up to 25 normalized units and clipped to `[150, 350]`. The value and sample count are stored as provenance rather than presented as measured accuracy.

All final candidates use the existing union-geometry gold labels. `partial_manual_review` remains an annotation quarantine state, not a VLM target.

## Conditional adjacency

Unheaded pages expose atomic DeepSeek paragraphs only. Future binary VLM scores may conditionally activate immediate neighbor bundles when an atomic score is inconclusive. Preprocessing does not create or label those bundles before scores exist.

## Viewer

Final-label modes display only DeepSeek-OCR-2 candidates with the existing `S#` membership badges and label colors. DeepSeek visual bundles repeat the same `S#` on objects and captions and draw dashed connector lines between their centers.

The existing `MinerU only` view becomes a visual-link inspection view. It displays MinerU figure/chart/table-caption bundles with shared `M#` badges and visually distinct connector lines. It does not display MinerU text blocks or add another comparison mode. Users compare parsers by switching views.

## Pilot heuristics and verification

Caption-prefix coverage, compatible nearest-neighbor distance distributions, paired objects, unpaired objects, and orphan captions are calculated from the checked-in 32-page pilot. These are diagnostics, not caption-link accuracy, because the pilot has no link-level ground truth.

Tests cover headed routing, atomic unheaded routing, DeepSeek caption detection, object-caption compatibility, multi-panel linking, absence of MinerU final candidates, and viewer membership markers. Browser verification checks representative text, figure, table, and multi-panel pages in final and MinerU-only views without changing stored geometry.
