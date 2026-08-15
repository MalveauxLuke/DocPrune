# SciEGQA MinerU Pilot - Task 4 Code-Quality Review

## Scope and Verdict

Reviewed Task 4 from `b7b400e` through implementation `6a48e31` and evidence
report `b41f166`. The spec review had already passed; this review focused on
failure atomicity, artifact integrity, geometry edge cases, identifier safety,
resource handling, and test strength. No production document was downloaded.

**Ready: No.** The final-path writes and incomplete-image acceptance are
blocking for an accuracy-sensitive provenance pipeline. Re-review after the
high-severity findings and their regression tests are addressed.

## Findings

### High - Downloads and renders overwrite canonical artifacts non-atomically

`scripts/sciegqa_mineru/source_pages.py:99-114` opens the final PDF path with
`wb` before streaming has completed. An injected read failure after one chunk
replaced an existing artifact with only `b"new-partial"`. Likewise,
`scripts/sciegqa_mineru/source_pages.py:123-140` tells Poppler to write directly
to the canonical page prefix and leaves that result in place when PIL or the
dimension check fails. A rerun can therefore destroy a previously hashed PDF
or page; if the build is interrupted before its entity files are replaced, the
previous provenance records can point at changed or partial bytes.

Fix by downloading/rendering to unique sibling temporary paths, validating and
hashing those temporary artifacts, and publishing with `os.replace` only after
all checks pass. Remove temporary files on every exception. Add failure-
injection tests proving that an existing destination is byte-for-byte unchanged
after response, subprocess, PIL, dimension, and hashing failures.

### High - A structurally incomplete PNG is accepted and hashed as valid

`scripts/sciegqa_mineru/source_pages.py:129-140` reads only `Image.size`. PIL can
obtain dimensions from the header without validating the whole image. A valid
20 x 30 PNG truncated by five bytes failed `Image.verify()` but
`render_and_validate_page()` accepted it and returned its SHA-256. This creates
a direct false-success path for the page later shown in the evidence viewer.

Fully validate the staged image with `Image.verify()` (or an equivalently
strict full decode) before dimension comparison and hashing. Add a regression
test using a deliberately truncated PNG whose header still exposes the
expected dimensions.

### Medium - Output suffixes can make a stale file look like the fresh render

`scripts/sciegqa_mineru/source_pages.py:124-140` strips any suffix for Poppler,
which always appends `.png`, but then opens and hashes the original `page_path`.
With an existing 300 x 300 `suffix-mismatch.jpg`, an actual Poppler run created
`suffix-mismatch.png` while the helper successfully returned the unchanged
JPEG's hash. Task 5 plans to pass `.png`, which limits current exposure, but the
helper itself does not enforce that invariant and can report the wrong bytes.

Require `page_path.suffix == ".png"` (case policy explicit), and use the staged
PNG path as the single source for validation, hashing, and final publication.
Test a non-PNG path and a stale destination.

### Medium - Dimension inference silently discards contradictory zero pairs

`scripts/sciegqa_mineru/source_pages.py:48-75` skips a coordinate pair whenever
either representation is zero. Consequently, absolute `x0=0` with normalized
`x0=10` was ignored and the remaining endpoint produced a successful
`(1000, 1000)` result. A small positive pair can also round to a zero page
dimension and return successfully. The mathematically exact boundary candidate
`1000.1` is rejected because its binary-float distance from `1000` is slightly
greater than `0.1`, despite the code expressing an inclusive threshold.
Non-finite values fail incidentally through arithmetic/`round`, rather than
through a clear coordinate contract.

Check every pair for finite values, reject pairs where exactly one side is
zero, use jointly-zero pairs only as unavailable candidates, and require each
resolved integer dimension to be positive. Preserve the existing 0.1-pixel
consistency threshold with an explicit floating-point tolerance, and add
boundary tests immediately below, at, and above that threshold.

### Medium - PDF provenance is emitted without validating identifier or payload

`scripts/sciegqa_mineru/source_pages.py:22-23` and
`scripts/sciegqa_mineru/source_pages.py:37-44` interpolate unvalidated IDs into
URLs; `2411.02804v1` becomes `...v1v1`, and `../escape` remains a path-bearing
identifier that Task 5 would also use in local paths. In
`scripts/sciegqa_mineru/source_pages.py:99-114`, a mocked HTTP 200 HTML body was
saved and returned with `pdf_url`, `pdf_sha256`, and `pdf_bytes` as though it
were a PDF. Task 5's later `PdfReader` call would catch the payload, but the
source helper's success contract and final-path mutation are already false.

Validate the SciEGQA-supported unversioned arXiv ID form once and reuse it for
URL, metadata, download, and path construction. Before publication, reject
empty/non-PDF payloads and preferably parse the staged PDF; also verify the
metadata entry ID matches the requested v1 document. Add malformed-ID,
already-versioned-ID, HTML-body, and empty-body tests.

## Strengths

- Requests use explicit user-agent strings and bounded timeouts; `urlopen`
  failures propagate rather than being hidden.
- Response, destination, and PIL handles are all scoped with context managers,
  so Python resources close correctly on success and exceptions.
- The Poppler command is deterministic, page-specific, 300 DPI, and uses
  `check=True`; there is no resize, clipping, fallback renderer, or silent retry.
- Hash assertions in the success tests compare against independently computed
  content hashes rather than copied constants.
- The implementation is compact and keeps network and subprocess boundaries
  mockable.

## Verification Evidence

- `python -m pytest -q tests/test_sciegqa_mineru_source_pages.py`:
  `12 passed in 0.07s`.
- `python -m pytest -q`: `115 passed in 2.20s`.
- `git diff --check b7b400e..HEAD`: passed.
- A synthetic 72 x 72 point PDF rendered through the real Poppler path to an
  inspected 300 x 300 PNG with the expected content and metadata.
- Independent safe probes reproduced the interrupted-overwrite, HTML-as-PDF,
  incomplete-PNG, suffix/stale-file, asymmetric-zero, and zero-dimension cases.

The existing tests are useful success-contract checks, but they do not exercise
the artifact-preservation or content-integrity failures above. Those tests are
required before Task 5 starts materializing durable source provenance.
