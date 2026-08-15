# SciEGQA MinerU Task 4 - Source Artifact Quality Fix

## Scope and Immutable Implementation

This correction hardens the existing Task 4 source-artifact pipeline without
changing its successful metadata contract or adding a renderer, retry, resize,
or PDF-parser fallback.

Implementation commit:
`6bbc16eea39355263d85d112dfaa9c10bf4e5ed4`

Changed implementation and test files:

- `scripts/sciegqa_mineru/source_pages.py`
- `tests/test_sciegqa_mineru_source_pages.py`

## Root-Cause Evidence

Failure probes against the prior implementation showed:

```text
pdf_after_read_failure= b''
page_after_dimension_failure_size= 69
asymmetric_zero_result= (1000, 1000)
```

The PDF destination was opened in truncating mode before the response read
completed. Poppler rendered directly to the canonical page prefix before PIL
and dimension validation. Geometry inference silently discarded asymmetric
zero pairs. These write and validation boundaries were the root causes.

## TDD Evidence

RED command:

```text
python -m pytest -q tests/test_sciegqa_mineru_source_pages.py
```

The new tests failed at the expected behaviors before the fix:

```text
26 failed, 18 passed, 1 subtests passed in 0.34s
```

The failures covered canonical PDF/page corruption, residual staging
artifacts, malformed IDs reaching network code, unvalidated PDF payloads,
lazy PNG inspection accepting truncation, missing suffix enforcement, invalid
geometry, and the floating-point consistency boundary.

Focused GREEN:

```text
python -m pytest -q tests/test_sciegqa_mineru_source_pages.py
............................                            [100%]
28 passed, 17 subtests passed in 0.11s
```

Full regression:

```text
python -m pytest -q
........................................................................ [ 54%]
....................................................... [ 96%]
....                                                                     [100%]
131 passed, 17 subtests passed in 2.05s
```

`git diff --check` also exited successfully.

## Atomic-Publish and Validation Review

- Modern SciEGQA arXiv IDs must exactly match four ASCII digits, a period,
  then four or five ASCII digits. URL construction, category fetch, and PDF
  download all use the same validation helper.
- PDF data streams in 1 MiB reads to a unique sibling temporary file. The
  response and file handles close before header validation, hashing, sizing,
  and `os.replace`.
- Empty or non-PDF staged payloads are rejected unless their first five bytes
  are exactly `%PDF-`.
- Response-read, payload-validation, and hash failures preserve existing PDF
  bytes and remove the staged file.
- Page paths require the exact `.png` suffix before Poppler is invoked.
- Poppler renders into a unique sibling temporary directory. PIL first runs
  `verify()`, then reopens and fully loads the PNG before exact dimension
  comparison, hashing, and `os.replace`.
- Subprocess, unreadable-image, readable-header/truncated-image, dimension,
  and hash failures preserve existing page bytes and remove all staging
  artifacts.
- Every geometry coordinate must be finite and nonnegative. Jointly zero
  pairs are skipped, asymmetric zero pairs fail, each axis needs a candidate,
  and the rounded dimension must be positive.
- The inclusive 0.1-pixel consistency boundary uses an explicit `1e-9`
  floating-point tolerance; values immediately above the boundary fail.
- The exact page-specific 300-DPI Poppler command and successful return fields
  remain unchanged.

## Independent Real-Poppler Probe

A synthetic one-page 72-by-72-point PDF was generated in a temporary
directory and rendered with the real local `pdftoppm` through the production
function. No production document or network request was used.

```text
real_poppler_success= 300 300 300
success_artifacts= ['synthetic.pdf', 'synthetic.png']
real_poppler_failure= rendered page size (300, 300) does not match annotation-derived (301, 300)
canonical_preserved= True
failure_artifacts= ['synthetic.pdf', 'synthetic.png']
```

The success produced the expected 300-by-300 pixel PNG at 300 DPI. The
intentional dimension failure preserved the prior canonical PNG and left no
temporary sibling artifacts.

## Deferred Evidence and Concerns

No production PDF was downloaded. Live arXiv response behavior remains part of
the later selected-document materialization task. Same-directory staging keeps
each `os.replace` on the destination filesystem; this implements atomic
visibility but does not add filesystem durability synchronization, which was
not part of this task.
