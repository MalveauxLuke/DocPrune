# MMLongBench DeepSeek-OCR-2 Segmentation Viewer Design

## Goal

Build a local static website for visually inspecting the successful outputs from
the MMLongBench-Doc DeepSeek-OCR-2 off-domain pilot. The viewer evaluates how
the canonical semantic-section rules behave on difficult, primarily
unstructured documents.

## Scope

- Use run `20260717T182300Z` from
  `sol_results/mmlongbench_ocr2_unstructured_313/`.
- Include only the 306 pages that passed the SOL quality gate.
- Omit the seven failed pages from the site, its navigation, and its manifest.
- Preserve a truthful run summary stating that 306 of 313 attempted pages are
  displayed and seven were excluded upstream.
- Perform semantic grouping locally with the canonical
  `build_deepseek_semantic_sections` implementation. Do not rerun OCR.
- Produce a generated static bundle that is served locally; do not deploy it.

## Recommended Approach

Adapt the existing SciEGQA parser-comparison viewer into a focused DeepSeek-only
viewer. Reuse its page-image overlay, linked-segment highlighting, raw-output
display, navigation, and generated-manifest pattern while removing controls for
Gold, MinerU, and DeepSeek-OCR-1.

An exact clone would retain irrelevant comparison controls. A simple gallery
would be faster but would not expose semantic membership, geometry, reading
order, or raw OCR needed for this experiment.

## Data Flow

1. Read the 306 packaged page, run, and atomic-segment records.
2. Re-render those pages from the frozen PDFs at the recorded 2.0 PyMuPDF scale
   because SOL intentionally returned text and JSON only.
3. Require the SOL PyMuPDF version, frozen source-PDF hash, page dimensions, and
   render parameters. Record exact PNG byte/hash agreement per page; allow
   platform-specific PNG drift when dimensions still match.
4. Apply the canonical DeepSeek semantic-section builder to each page's ordered
   atomic segments.
5. Write a portable `manifest.json`, page PNGs, and linked raw OCR artifacts
   into a generated site directory.
6. Serve that directory with a local static HTTP server.

## Viewer Experience

- Document selector covering all ten source PDFs.
- Previous/next navigation over the 306 successful pages only.
- Page canvas with normalized OCR bounding boxes.
- Toggle between atomic OCR blocks and canonical semantic sections.
- Shared badges and hover/focus highlighting for every member of a semantic
  section.
- Side panel containing section kind, member count, word count, reading order,
  member raw types, extracted Markdown, and run diagnostics.
- Raw OCR output link for the selected page.
- Compact header summary: ten documents, 306 displayed pages, seven upstream
  exclusions, and 4,218 atomic OCR segments.

## Failure Handling

The builder must fail with a clear error for missing packaged artifacts,
duplicate identifiers, invalid bounding boxes, source-PDF identity mismatch,
renderer-version mismatch, image-dimension mismatch, or semantic-section
records that reference unknown atomic segments. Exact PNG hash differences are
recorded rather than rejected because MuPDF output can vary across operating
systems even at the same version. The seven known quality-gate failures are not
treated as build errors because they are explicitly outside the approved
display set.

## Testing and Acceptance

Keep verification targeted:

- Unit-test manifest construction, semantic membership, omitted-page behavior,
  and malformed-input rejection with small fixtures.
- Run the focused viewer and semantic-section tests only.
- Build the real 306-page site once and confirm its counts and image hashes.
- Perform one browser smoke check covering initial load, navigation, overlay
  switching, and raw-output access.

The work is complete when the static site loads locally, exposes all 306
successful pages and their canonical semantic sections, contains none of the
seven failed page IDs, and reports the pilot counts accurately.
