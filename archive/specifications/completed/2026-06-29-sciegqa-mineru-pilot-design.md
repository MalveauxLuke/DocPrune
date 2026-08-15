# SciEGQA-Train MinerU Evidence-Page Pilot

Date: 2026-06-29

Status: Approved in chat; awaiting review of this written specification

## Objective

Build a reproducible 20-question SciEGQA-Train pilot that:

1. selects Single Page Single Region (SPSR) questions representative of the training distribution;
2. reconstructs each annotated evidence page from its source arXiv PDF;
3. parses each unique evidence page with MinerU 3.4;
4. preserves document, page, query, evidence, segment, and run provenance; and
5. provides a separate local web viewer with faithful Gold-only, MinerU-only, and Overlay views.

The pilot validates the document-processing and visualization substrate for the later segment-level VLM evidence scorer. It does not train that scorer or modify the validated ColQwen2 retrieval path.

## Research Findings

### SciEGQA

SciEGQA has two materially different components:

- SciEGQA-Bench contains 1,623 human-annotated examples.
- SciEGQA-Train contains 30,780 automatically generated examples from 3,671 papers.

This pilot uses SciEGQA-Train because the future training workload must reflect that distribution. Its released evidence annotations are treated as dataset-gold labels, while provenance explicitly records that they were generated automatically rather than manually cross-validated.

The Train JSONL contains these fields:

- `query`
- `answer`
- `doc_name`
- `evidence_page`
- `bbox`
- `rel_bbox`
- `category`
- `subimg_type`

SPSR is exactly recoverable from the released structure by requiring one evidence page, one page-level box list, and one box in that list. This produces 11,668 SPSR rows, matching the paper's reported 37.91% after rounding.

All released Train SPSR rows have `subimg_type == "image"`. The meaningful available stratification axis is therefore the released category, not modality.

The released category is not a canonical document-level label: 174 document IDs appear under multiple categories, including 103 documents with SPSR rows. Sampling must preserve the row's `source_category`, separately preserve arXiv's document categories, and prevent one document from satisfying multiple category quotas.

SciEGQA renders source PDFs at 300 DPI. For the inspected source paper `2411.02804v1`, a 300-DPI render produces 2481 x 3508 A4 pages. Converting the released absolute boxes with those dimensions reproduces `rel_bbox` exactly within floating-point precision.

### MinerU

MinerU 3.4 supports PDF parsing through pipeline, VLM, and hybrid backends. The default backend is hybrid with medium effort, but medium effort disables image/chart analysis. This is unsuitable for a scientific-document pilot that must retain figures and tables.

The approved backend is:

- MinerU package: `3.4.0`
- backend: `hybrid-engine`
- effort: `high`
- image analysis: enabled
- service: persistent `mineru-api` bound to localhost

MinerU 3.4 requires Python 3.10 through 3.13. The workstation's default Python 3.14.6 is incompatible, so MinerU must use a separate Python 3.12 environment. The existing ColQwen environment must not be modified.

The local machine is an Apple M1 Pro with 16 GB unified memory. It meets MinerU's documented minimum but not the recommended 32 GB. The implementation therefore begins with a one-page smoke test before processing the full pilot.

MinerU provides several artifacts with different purposes:

- `content_list.json`: flat reading-order content with 0-1000 normalized boxes;
- `middle.json`: detailed structured results for secondary development;
- `model.json`: raw model output;
- `layout.pdf`: native MinerU layout visualization; and
- extracted images, tables, Markdown, and related assets.

The viewer uses canonicalized `content_list.json` segments. Raw structured files and `layout.pdf` remain immutable verification artifacts.

## Approved Architecture

The system is an offline preprocessing pipeline plus a static local inspection application.

```text
SciEGQA-Train JSONL + revision
              |
              v
 deterministic SPSR selection -----> selection audit
              |
              v
 arXiv v1 PDF + 300-DPI pages ------> source hashes and dimensions
              |
              v
 localhost MinerU 3.4 service ------> raw MinerU artifacts
              |
              v
 canonical provenance records ------> derived viewer manifest
              |
              v
 static page-first inspection app
```

Parsing and visualization are deliberately separated. The viewer never starts MinerU, mutates parsing output, or depends on model availability. This keeps visual inspection deterministic and allows every displayed rectangle to be traced to an immutable source record.

## Dataset Selection

### Eligibility

A row is eligible only when:

1. `evidence_page` contains exactly one page;
2. `bbox` contains exactly one page-level list;
3. that list contains exactly one evidence box;
4. the corresponding `rel_bbox` has the same shape;
5. box coordinates are ordered and finite;
6. normalized coordinates are within `[0,1000]`; and
7. the source PDF and evidence page can be validated exactly.

No invalid box is clipped or repaired.

### Quotas

The 20 rows use the largest-remainder approximation of the Train SPSR category distribution:

| Source category | Questions |
|---|---:|
| q-fin | 4 |
| q-bio | 3 |
| eess | 3 |
| physics | 3 |
| cs | 2 |
| econ | 2 |
| stat | 2 |
| math | 1 |
| **Total** | **20** |

Within each category, selection favors one high-yield document to limit repeated model work. A document may be assigned to only one quota even if Train contains rows for that document under multiple categories. This should produce approximately eight source documents while preserving the approved category distribution.

Candidate documents are ordered deterministically by descending eligible SPSR count and then arXiv ID. Candidate rows are ordered by their zero-based source JSONL row index. Any rejection is written to the selection audit with a concrete reason; it is not silently discarded.

## Source-Page Reconstruction

The canonical source is the explicit arXiv `v1` PDF for each selected `doc_name`. Using `v1` prevents later arXiv revisions from shifting pages or layouts relative to SciEGQA's annotations.

For every selected document:

1. download the `v1` PDF;
2. record the URL, retrieval time, byte size, and SHA-256;
3. render selected pages at 300 DPI;
4. record image dimensions and SHA-256; and
5. verify that absolute and normalized SciEGQA boxes imply the rendered dimensions.

For every usable nonzero coordinate pair, the annotation-derived page dimension must be within 0.1 pixel of an integer, and that integer must equal the rendered dimension. A mismatch is a hard provenance failure. The pipeline must not resize the page to force alignment.

Source PDFs and rendered pages are generated data and are not committed to git.

## MinerU Processing

MinerU runs in a dedicated, pinned Python 3.12 environment. Model downloads use an explicit Hugging Face source, and the resolved local model snapshot is recorded. No secrets or API keys are required.

A persistent localhost-only `mineru-api` avoids reloading the model for each page. Each unique evidence page is parsed once, even when several questions reference it. Requests use the source page's zero-based page range, while canonical records retain both the source's one-based page number and zero-based index.

Processing parameters are explicit rather than inherited from defaults:

- `backend=hybrid-engine`
- `effort=high`
- `image_analysis=true`
- formulas enabled
- tables enabled

The first run is a one-page smoke test. The remaining unique pages run only after the service health check, output schema check, and coordinate validation pass.

MinerU API task state is process-local and temporary. Completed outputs are copied into the run artifact directory immediately and hashed before the next stage.

## Provenance Model

Canonical records are separated by entity and joined through stable IDs.

### Document

- `document_id`
- `arxiv_id`
- `arxiv_version`
- `arxiv_categories`
- `pdf_url`
- `pdf_sha256`
- `pdf_bytes`
- `page_count`

### Page

- `page_id`
- `document_id`
- `source_page_number` (one-based)
- `source_page_index` (zero-based)
- `render_dpi`
- `width_px`
- `height_px`
- `image_path`
- `image_sha256`

### Query

- `query_id`
- `dataset_name`
- `dataset_revision`
- `source_record_index`
- `source_category`
- `question`
- `answer`
- `page_id`
- `selection_rule`

### Evidence

- `evidence_id`
- `query_id`
- `page_id`
- `annotation_origin = "sciegqa_train_automatic"`
- `source_bbox_px`
- `source_bbox_norm_1000`
- `source_subimg_type`

### Segment

- `segment_id`
- `page_id`
- `reading_order`
- `type`
- `sub_type`
- `bbox_norm_1000`
- structured content fields appropriate to the type
- links to extracted assets
- links to raw MinerU artifacts

### Run

- `run_id`
- timestamps
- git revision
- environment lock or package inventory
- MinerU package version
- model snapshot
- backend and parameters
- input and output hashes
- status and explicit failure details

Stable IDs derive from immutable source identity, not filenames generated during a run.

## Coordinate Contract

All boxes use `[x0, y0, x1, y1]` with a top-left origin.

The cross-system coordinate contract is `[0,1000]` normalized page space:

```text
x_px = x_norm * page_width_px / 1000
y_px = y_norm * page_height_px / 1000
```

SciEGQA absolute pixel boxes and floating-point `rel_bbox` values are both preserved. MinerU's normalized integer boxes are preserved exactly as emitted. Canonicalization does not round source values beyond serialization requirements.

IoU and overlap are derived inspection values. They never alter, merge, expand, snap, or replace source geometry.

## Viewer Design

The approved layout is the page-first inspector:

- a large zoomable page canvas occupies the main area;
- a persistent side panel shows query, answer, document/page provenance, selected segment details, and layer controls;
- previous/next navigation traverses the 20 questions;
- shared pages are reused without duplicating image assets; and
- the current query's gold evidence remains query-specific.

The required layer modes are:

1. Gold only
2. MinerU only
3. Overlay

Gold evidence uses a fixed red treatment. MinerU uses a distinct palette keyed by segment type, with a visible legend. Color is not the only distinction: labels, line patterns, and tooltips identify each source.

The overlay is an SVG aligned to the page image's intrinsic aspect ratio. It uses the canonical coordinate transform directly, with non-scaling strokes for legibility during zoom. The application must not estimate coordinates from rendered CSS dimensions or manually offset boxes.

Selecting or hovering a MinerU box reveals its segment ID, type, reading order, coordinates, content preview, and derived overlap with the current gold region. Dense pages can filter MinerU boxes by type without changing the underlying data.

The viewer is static and local. It does not expose MinerU as a public service.

## Failure Handling

The pipeline fails explicitly when:

- the dataset revision or expected fields cannot be resolved;
- a selected source row is malformed;
- an arXiv `v1` PDF or page is unavailable;
- rendered dimensions disagree with annotation-derived dimensions;
- MinerU is unhealthy or returns an incomplete task;
- required raw output files are missing;
- a segment box is malformed or out of bounds; or
- canonical entity references do not resolve.

Network failures, parser failures, and annotation failures are not masked with approximate pages, fuzzy query matching, resized images, clipped boxes, or alternative parser output.

## Verification and Acceptance Criteria

### Automated checks

- exactly 20 selected queries;
- category counts exactly match the approved quotas;
- every query has exactly one page and one evidence region;
- no document satisfies more than one category quota;
- all references between documents, pages, queries, evidence, segments, and runs resolve;
- all source and output artifacts have hashes;
- all boxes are finite, ordered, and in bounds;
- SciEGQA absolute-to-normalized conversions reproduce released values within floating-point tolerance;
- normalized-to-pixel-to-normalized round trips remain within floating-point tolerance;
- page dimensions match annotation-derived dimensions;
- required MinerU artifacts exist and identify version/backend; and
- viewer records reproduce canonical geometry without mutation.

### Visual checks

- compare web MinerU boxes against MinerU's native `layout.pdf`;
- inspect every unique evidence page;
- inspect the query-specific gold box for all 20 questions;
- verify Gold-only, MinerU-only, and Overlay modes independently;
- verify labels and legends remain unambiguous on dense pages;
- verify zoom and responsive resizing do not introduce drift; and
- record MinerU segmentation errors as observed output rather than correcting them in the viewer.

Geometric fidelity and parsing quality are reported separately. Passing geometric validation means the viewer faithfully displays the source annotations and MinerU output; it does not claim MinerU's predicted segmentation is semantically correct.

## Scope Boundaries

This pilot includes local MinerU setup, deterministic Train sampling, source-page reconstruction, page parsing, provenance records, the static viewer, and verification.

It excludes:

- ColQwen2 retrieval changes;
- segment-level VLM evidence scoring;
- planner/expert orchestration;
- public deployment;
- full SciEGQA-Train materialization; and
- deletion or refactoring of the frozen verifier path.

SOL deployment is a follow-up after the local pilot is correct. It must use a separate MinerU environment and a focused GPU smoke test before any scaled job.

## Licensing and Distribution

MinerU 3.4 uses the MinerU Open Source License, based on Apache 2.0 with additional commercial thresholds and an attribution requirement for online third-party services. This pilot is local and does not expose such a service.

SciEGQA's repository and Hugging Face card do not clearly state a dataset-wide software-style license. Source PDFs also retain their own arXiv license terms. Consequently, downloaded PDFs, rendered pages, and dataset archives remain uncommitted local artifacts. Only code, configuration, tests, and small provenance records that do not redistribute source page content are candidates for git.

## Primary Sources

- [SciEGQA project](https://yuwenhan07.github.io/SciEGQA-project/)
- [SciEGQA-Train release](https://huggingface.co/datasets/Yuwh07/SciEGQA-Train)
- [SciEGQA repository](https://github.com/yuwenhan07/SciEGQA)
- [SciEGQA paper](https://arxiv.org/abs/2511.15090)
- [MinerU documentation](https://opendatalab.github.io/MinerU/)
- [MinerU 3.4 repository](https://github.com/opendatalab/MinerU)
- [MinerU output formats](https://opendatalab.github.io/MinerU/reference/output_files/)
- [MinerU license](https://github.com/opendatalab/MinerU/blob/master/LICENSE.md)
