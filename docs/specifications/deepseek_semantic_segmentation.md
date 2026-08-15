# DeepSeek-OCR-2 Semantic Segmentation Contract

## Purpose and scope

This document records the canonical segmentation contract now used by the
MMLongBench OCR viewer. The rules originated in the archived SciEGQA viewer and
4K segment-evidence dataset. They are intended for **academic, highly
structured page images**: scientific papers, reports, proceedings, and similar
documents with headings, paragraphs, figures, tables, captions, formulas, and
stable reading order.

The contract is not claimed to generalize unchanged to arbitrary webpages,
magazines, forms, receipts, slide decks, or natural-image scenes. Those layouts
may require different hierarchy, reading-order, and visual-linking rules.

The objective is page-conditional fine-grained evidence localization:

```text
known candidate page + query -> evidence-bearing semantic section
```

SciEGQA questions are often written with the gold page implicitly known. This
makes the dataset appropriate for the evidence-localization proof of concept,
but not by itself an open-corpus retrieval benchmark.

## Sources of truth

The Python implementation is authoritative. This document explains it but does
not replace it.

- [DeepSeek inference and provenance](../../scripts/document_parsing/deepseek_runner.py)
- [Grounded-output parsing](../../scripts/document_parsing/grounding.py)
- [Canonical semantic-section and labeling logic](../../scripts/document_parsing/semantic_sections.py)
- [Active segmentation and linking tests](../../tests/test_semantic_sections.py)
- [Active MMLongBench viewer builder](../../scripts/mmlongbench/viewer/viewer_bundle.py)
- [Archived 4K evidence builder](../../archive/code/sciegqa/sciegqa_evidence_poc.py)
- [Archived dataset integration tests](../../archive/tests/sciegqa/test_sciegqa_evidence_poc.py)
- [Archived parser viewer](../../archive/apps/sciegqa/sciegqa_parser_compare/)

`build_deepseek_semantic_sections` in `semantic_sections.py` is the active
shared builder. The archived dataset integration test records the former 4K
identity and 350-unit linking contract.

The browser is a visual audit surface over a precomputed manifest. JavaScript
does not independently recompute the segmentation.

## Pinned DeepSeek-OCR-2 inference

The current inference contract is:

- Model: `deepseek-ai/DeepSeek-OCR-2`
- Revision: `aaa02f3811945a91062062994c5c4a3f4c0af2b0`
- Prompt: `<image>\n<|grounding|>Convert the document to markdown.`
- Base size: `1024`
- Image size: `768`
- Crop mode: enabled
- Evaluation mode: enabled
- Maximum new tokens: `8192`

Every run records the input PNG hash, model and revision, prompt, inference
parameters, attempt number, raw-output hash, runtime provenance, GPU identity,
and Slurm job ID. A resume is accepted only when the stored provenance exactly
matches the requested inference and the raw artifact hash is valid.

DeepSeek-OCR-2 returns grounded blocks containing raw types, reading order,
content, and normalized bounding boxes. The pipeline preserves those atomic
blocks before deterministically constructing larger semantic sections.

## Text segmentation

### Pages with Markdown section headings

A DeepSeek Markdown block beginning with `#` through `######` starts a semantic
section. All such heading levels act as boundaries; the builder does not invent
a nested hierarchy from heading depth.

For each heading:

1. Include the heading block.
2. Include subsequent non-caption `text` and `formula` blocks.
3. Stop immediately before the next qualifying Markdown heading.
4. Emit one `headed_text` section.

The section is deliberately not split by:

- token or word limits;
- paragraph count;
- column transitions;
- page-coordinate gaps;
- candidate bounding-box size; or
- fixed sliding windows.

This preserves the academic semantic unit expressed by the author. The working
assumption is that an answer is more likely to be contained within a complete
section than within an arbitrary layout- or length-based fragment.

Content before the first `##`-`######` heading is preserved as one
`document_preamble` when nonempty. This retains titles, authors, affiliations,
abstract lead-in material, formulas, and other potentially query-relevant
front matter instead of silently discarding it.

### Pages without Markdown section headings

When no qualifying heading exists, the pipeline does not infer headings from
font size or geometry. Each nonempty DeepSeek `title`, `text`, or `formula`
block becomes one atomic `deepseek_paragraph` section.

There is no paragraph-length cap and no overlapping two- or three-paragraph
window generation. If a binary VLM score is inconclusive at inference time,
conditional adjacency can evaluate a neighboring section then; preprocessing
does not multiply overlapping training candidates in anticipation of that edge
case.

## Native visual and caption linking

Only DeepSeek-OCR-2 sections are used for final labeling. MinerU visual links
remain available in the viewer as an optional comparison and diagnostic layer.

### Recognized raw blocks

- Visual objects: `image`, `figure`, and `table`.
- Caption/title object: `figure_title`.
- `figure_title` is intentionally compatible with both image/figure objects and
  tables because DeepSeek-OCR-2 uses that raw type for observed table captions.
- Charts are currently represented as image/figure blocks; no synthetic
  `chart`, `table_title`, or `table_caption` raw type is invented.
- A table's HTML/Markdown content remains inside its single table block.

### Top-down run construction

Visual objects are processed in DeepSeek reading order from the top of the page.
A run begins at the first unprocessed visual object.

Consecutive visual objects of the same role form one run when they are separated
by at most two raw `text` blocks. The ignored text permits panel labels or small
intervening annotations without fragmenting a multi-panel figure. Ignored text
is never added to the visual bundle itself.

Images/figures and tables use the same algorithm, but unlike roles are not
collapsed into one primary run.

### Primary caption search

For each top-down visual run:

1. Search upward from the run first.
2. Ignore at most two raw `text` blocks while searching.
3. If the first eligible, unclaimed `figure_title` is found above, link it to
   every visual object in the run and end the primary search.
4. If no eligible title is found above, perform the corresponding downward
   search from the end of the run.
5. Stop once a title is found or both permitted search areas have been
   exhausted.
6. A primary title is claimed once and cannot become another run's primary
   title.

The upward-first rule is intentional for academic layouts that place a table or
figure title immediately above its object. Falling through to the downward
search supports the more common caption-below layout.

### Geometry fallback for floating titles

After all runs have completed primary searches, any unclaimed `figure_title`
is treated as a floating title. It links to every compatible image/figure or
table whose center lies within the configured maximum center distance.

The current shared dataset contract uses `350` normalized-page units. The pilot
viewer also records distance samples and a rounded, clipped 95th-percentile
diagnostic bounded to `150`-`350`; the final 4K shared builder is called at
`350`.

Fallback edges may connect multiple runs into one visual component. This is how
a shared global figure title or note can connect multiple already-captioned
subpanels without allowing that floating title to steal a primary link during
the top-down pass.

### Final visual section

Each connected component becomes one `visual_bundle` containing:

- its image/figure or table blocks;
- its linked `figure_title` blocks;
- the original member bounding boxes and reading orders; and
- provenance identifying primary versus geometry-fallback title links.

A visual with no linked title remains a valid standalone visual bundle and
produces a diagnostic. A title with no compatible visual also produces a
diagnostic.

## Geometry and query-relative labels

Candidate geometry is the **union of the actual member bounding boxes**. It is
not the rectangular envelope surrounding all members, because an envelope can
incorrectly claim whitespace or unrelated content between a figure and caption.

For query gold box `G` and candidate member-box union `C`:

```text
gold_coverage = area(G intersect C) / area(G)
candidate_precision = area(G intersect C) / area(C)
```

The 4K evidence dataset uses a gold-coverage threshold of `0.70`:

- `gold_coverage >= 0.70`: positive relation;
- `0 < gold_coverage < 0.70`: partial relation;
- `gold_coverage == 0`: negative relation / same-page hard negative.

Labels belong to `(query_id, section_id)` relations. Sections are shared by
multiple queries and must never receive one global positive/negative label.

For every retained query, exactly one section must be positive. Every other
section on the same gold page becomes that query's hard negative.

## Conservative failure and quarantine policy

DeepSeek inference that is truncated, malformed, repetitive, incomplete, or
otherwise invalid is retried once. If the second attempt is invalid, quarantine
the page and all queries attached to it.

Also quarantine the page and all its queries when:

- any query-section relation is partial;
- any query has zero positive sections; or
- any query has more than one positive section.

This intentionally sacrifices coverage rather than emitting questionable
supervision. No manual correction is inserted into the generated training
labels.

## Training and evaluation context

The immediate model task is a small VLM binary evidence scorer over
`query + semantic section`. The scorer should learn whether the section contains
sufficient evidence to answer the query, not whether it is merely topically
related. Visual bundles provide the relevant crop members plus structured
caption/table text; headed and paragraph sections provide their preserved
DeepSeek text.

Same-page non-gold sections are deliberately difficult negatives. The intended
future inference path scores atomic semantic sections first and uses conditional
adjacency only when the binary decision is inconclusive, avoiding systematic
overlapping candidates and unnecessary VLM calls.

The 4K train/validation/test split is page-grouped so that a page cannot cross
splits. This is essential because query-section relations from the same page are
strongly correlated. Answers are retained for provenance but are not model
inputs for evidence classification.

## What MinerU is and is not used for

MinerU helped establish the comparison and remains useful for visually
inspecting figure/table-caption extraction. It is not a source of final training
sections under the approved pipeline. Switching the optional MinerU overlay in
the 32-page viewer is a review action only; it must not alter DeepSeek final
labels.

## Change-control rule

Any change to heading boundaries, paragraph fallback, raw visual compatibility,
run formation, title claiming, geometry fallback, member-box union, or coverage
labeling is a dataset-definition change. Update the canonical builder, focused
tests, this document, and regenerated viewer/dataset manifests together. Do not
patch only the JavaScript viewer or duplicate the logic in a second dataset
implementation.
