# Task 5: SciEGQA MinerU Subset Build Evidence

Date: 2026-06-29

State: **OPEN - BLOCKED_AT_LIVE_GATE**

## Scope and implementation audit

Task 5 adds the provenance-preserving SciEGQA-Train subset builder and its
focused tests. The implementation was audited line by line against the approved
Task 5 boundaries:

- `DATASET_REPO` and `DATASET_FILENAME` name the approved Train source;
- `resolve_dataset()` resolves the requested Hub revision before downloading
  from the immutable commit SHA;
- document, page, query, and evidence records use stable IDs and joinable keys;
- queries preserve source row, revision, category, question, answer, page, and
  selection-rule provenance;
- evidence preserves exact pixel and normalized boxes, annotation origin, and
  source subimage type;
- entity lists use deterministic approved sorts;
- materialization uses the Task 4 v1 download, category, dimension, and render
  boundaries, relative artifact paths, and `PdfReader` page counts;
- `SourceDocumentError` triggers deterministic exclusion and reselection;
- dataset and entity manifests are written only after complete materialization;
- `dataset_source.json` records the immutable revision and source JSONL hash;
- `selection_audit.json` retains the approved strategy and records source
  rejections; and
- the CLI supports both Hub resolution and an explicit local JSONL boundary.

No production-code change was needed after this audit. No fallback, clipping,
resizing, fuzzy matching, metadata substitution, or retry behavior was added.

Implementation commit:
`d62aec0c1f523e4c663af699647d5bfe087aa6cd`

## TDD evidence

The prior agent did not leave its original RED transcript. It would be false to
present later evidence as that original run. A **reconstructed RED** was created
in a temporary out-of-repository copy with only
`scripts/build_sciegqa_mineru_subset.py` omitted:

```text
python -m pytest -q tests/test_sciegqa_mineru_end_to_end.py
ModuleNotFoundError: No module named 'scripts.build_sciegqa_mineru_subset'
1 error during collection
```

This proves the focused test imports the Task 5 boundary, but it is explicitly
not the original test-first transcript.

GREEN checks in the repository:

```text
python -m pytest -q \
  tests/test_sciegqa_mineru_end_to_end.py \
  tests/test_sciegqa_mineru_selection.py \
  tests/test_sciegqa_mineru_source_pages.py
82 passed, 17 subtests passed in 0.80s

python -m pytest -q
138 passed, 17 subtests passed in 4.01s
```

`git diff --check` was clean before the implementation commit.

## Immutable dataset and offline selection evidence

The live Hub resolution succeeded independently of the arXiv block:

- dataset: `Yuwh07/SciEGQA-Train`;
- file: `SciEGQA-Train.jsonl`;
- resolved revision:
  `4ffb867c88e3264161920b4b2446d5ac6352269e`;
- JSONL SHA-256:
  `7eb895fb913ffb607e3b41b01ec66e651685ad9404d230972fd987ba7f005e84`;
- source rows: 30,780;
- eligible SPSR rows: 11,668;
- selected queries: 20;
- selected documents: 8, unique across category assignments; and
- selected unique pages: 16.

The approved ordered category counts were reproduced exactly:

| Category | Count | Chosen document |
|---|---:|---|
| q-fin | 4 | 2412.17314 |
| q-bio | 3 | 2303.04443 |
| eess | 3 | 2412.19200 |
| physics | 3 | 2412.04371 |
| cs | 2 | 2411.06980 |
| econ | 2 | 2411.08350 |
| stat | 2 | 2403.12110 |
| math | 1 | 2509.25114 |

Selected source rows and pages:

| Source row | Category | Document | Page |
|---:|---|---|---:|
| 188 | cs | 2411.06980 | 2 |
| 189 | cs | 2411.06980 | 6 |
| 1361 | econ | 2411.08350 | 8 |
| 1362 | econ | 2411.08350 | 8 |
| 2674 | eess | 2412.19200 | 6 |
| 2675 | eess | 2412.19200 | 3 |
| 2676 | eess | 2412.19200 | 5 |
| 3215 | physics | 2412.04371 | 5 |
| 3216 | physics | 2412.04371 | 5 |
| 3217 | physics | 2412.04371 | 5 |
| 4038 | q-bio | 2303.04443 | 6 |
| 4039 | q-bio | 2303.04443 | 2 |
| 4040 | q-bio | 2303.04443 | 5 |
| 6981 | q-fin | 2412.17314 | 2 |
| 6982 | q-fin | 2412.17314 | 3 |
| 6983 | q-fin | 2412.17314 | 4 |
| 6984 | q-fin | 2412.17314 | 3 |
| 7210 | stat | 2403.12110 | 1 |
| 7211 | stat | 2403.12110 | 8 |
| 24195 | math | 2509.25114 | 12 |

Offline entity construction inferred 2550 x 3300 pages except document
`2411.08350` page 8 at 2481 x 3260 and document `2509.25114` page 12 at
2481 x 3508. These remain annotation-derived expectations until live renders
can be validated.

## Live build attempt and blocker

Command:

```text
$HOME/miniforge3/envs/mineru34/bin/python \
  scripts/build_sciegqa_mineru_subset.py \
  --output-dir outputs/sciegqa_mineru_pilot
```

The run downloaded source PDFs, then blocked before rendering while calling
`fetch_arxiv_categories()` against `https://export.arxiv.org/api/query`.
Independent probes observed both:

```text
HTTP 429, 14 bytes
Rate exceeded.

curl: (28) Operation timed out after 15008 milliseconds with 0 bytes received
HTTP 000
```

After a 60-second cooldown, the single approved probe still returned:

```text
HTTP 429, 14 bytes
Rate exceeded.
```

The unchanged build was stopped with Ctrl-C while blocked inside
`urllib.request.urlopen()` in `fetch_arxiv_categories()` and exited 130. At the
stop-approval snapshot there were 8 PDFs. A ninth PDF
(`2403.12110v1.pdf`) completed before termination, so the final generated state
was 9 PDFs, 0 PNGs, and 0 manifests. Because manifests are intentionally delayed
until complete materialization, no partial `source_rejections` audit was written.
All generated files remain under git-ignored `outputs/`.

## Pending live gate

The following required Task 5 evidence is still pending and must be appended to
this report after the arXiv metadata API recovers:

- successful unchanged build summary and persisted source rejections;
- exact document/query/evidence/page foreign-key and uniqueness validation;
- PDF byte count, SHA-256, and `PdfReader` page-count validation;
- source page range validation;
- strict PNG verification, stored dimensions, 300-DPI metadata, and hashes;
- normalized-to-pixel evidence coordinate agreement without clipping or resize;
- representative rendered-page/contact-sheet visual inspection; and
- final focused/full/status/diff verification against the completed artifacts.

Until those checks pass, Task 5 must not be described as complete.
