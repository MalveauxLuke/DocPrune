# MMLongBench OCR2 Deep-Parse Comparison Viewer Design

## Goal

Build a separate local static website for reviewing the completed DeepSeek-OCR-2
deep-parse semantic-link pilot. The viewer must make prompt-arm differences,
degenerate responses, and visual-to-text grouping behavior easy to compare on
the same source page without modifying the existing 306-page OCR viewer.

## Frozen Input

Use packaged run `20260717T224039Z` from
`sol_results/mmlongbench_ocr2_deep_parse_links/`. The package contains:

- 13 selected source pages;
- 91 full-page outputs across seven arms;
- 47 paper-style deep-parse outputs from visual crops on 11 pages;
- 138 completed run records in total;
- 47 crop records and 336 mapped OCR units; and
- zero records in `failures.jsonl`.

The website is a view over frozen artifacts. It must not rerun DeepSeek-OCR-2,
rewrite raw responses, or treat an inference marked `completed` as necessarily
useful output.

## Recommended Architecture

Add a generated viewer bundle under `viewer/mmlongbench_ocr2_deep_parse/` and a
small Python bundle builder under `scripts/`. Reuse source-page images from the
existing MMLongBench OCR viewer while copying the required images into the new
bundle so it remains independently serveable. The browser reads a precomputed
manifest and performs presentation only.

The builder normalizes the heterogeneous arms into one display schema:

- grounded full-page arms expose parsed regions and their raw response;
- inventory-assisted arms expose their supplied unit inventory, returned unit
  IDs, and any IDs that can be resolved back to OCR boxes;
- the paper deep-parse arm exposes each visual crop, its parent-page box, and
  its full description; and
- every arm exposes its exact prompt and run diagnostics.

Unparseable output is preserved as raw text with an explicit display status.
The builder must not invent semantic groups from malformed responses.

## Viewer Experience

The primary workspace contains two equal comparison panes. Both panes always
show the same selected source page, but each has an independent arm selector.
Page navigation, zoom, and pan are synchronized. Changing one arm does not
change the other.

Each grounded pane draws the selected arm's normalized boxes over the source
page. Boxes are selectable and reveal type, reading order, content, and raw
provenance. The viewer distinguishes actual semantic-group output from ordinary
OCR/object boxes through arm labeling and status badges; it does not imply that
every returned box is a valid semantic link.

For inventory-assisted arms, resolved unit IDs highlight their original OCR
boxes. Returned text, unresolved IDs, empty responses, and non-grounded outputs
remain visible in the detail panel. A visual unit may appear in multiple groups
without being deduplicated away.

For `paper_deep_parse`, the pane shows parent-page visual regions and a linked
crop gallery. Selecting a region reveals its crop and complete model
description. This arm is presented as crop interpretation, not as a full-page
segmentation result.

The page also includes:

- document and source-page navigation;
- prompt and raw-response disclosure panels;
- run status, latency, generated-token count, truncation, and repetition flags;
- per-arm counts for returned regions or groups; and
- a compact experiment overview showing useful-output and warning counts.

The default comparison is grounded Markdown control versus continuous
information units because both produced grounded output on all 13 pages and
form the clearest baseline-to-experiment comparison.

## Truthful Failure Presentation

The manifest and UI keep these states distinct:

- completed inference with parsed grounded regions;
- completed inference with non-grounded text;
- completed inference with an empty or one-token response;
- likely truncation;
- detected repetition; and
- missing or invalid source artifacts.

Observed pilot facts must remain visible: each semantic-grouping arm produced
grounded regions on 8 of 13 pages; OCR-inventory-assisted grouping had six
zero-token outputs; deep-parse-assisted grouping had seven; and crop deep parse
produced descriptions rather than grounded page regions. These are results,
not viewer build failures.

## Testing and Acceptance

Verification stays focused:

- unit-test manifest normalization, arm/page alignment, inventory-ID mapping,
  crop-parent linking, and malformed-artifact rejection with small fixtures;
- build the real 13-page bundle once and assert its frozen counts;
- perform one browser smoke check covering page navigation, independent arm
  selection, synchronized comparison, box selection, raw output, and a crop;
  and
- confirm the existing `viewer/mmlongbench_ocr2/` files are unchanged.

The work is complete when the separate local site displays all 13 pages, all
eight arms, all 138 run records, all 47 crop results, and every raw response
without silently converting weak or malformed outputs into successful semantic
groups.
