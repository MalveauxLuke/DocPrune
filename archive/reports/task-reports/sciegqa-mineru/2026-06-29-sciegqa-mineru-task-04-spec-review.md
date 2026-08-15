# SciEGQA MinerU Pilot - Task 4 Spec Review

## Review Scope

Reviewed the complete Task 4 change from `b7b400e` through `b41f166`, including
implementation commit `6a48e31`, its tests, and its evidence report. This was a
read-only implementation audit; no production PDF or page artifact was fetched.

## Contract Evidence

- `scripts/sciegqa_mineru/source_pages.py:16-44` stores `doc_name`, constructs
  the exact arXiv v1 and export API URLs, requires an Atom entry, returns sorted
  unique category terms, and uses the required user-agent and 60-second timeout.
- `scripts/sciegqa_mineru/source_pages.py:47-75` derives each axis only from
  nonzero absolute/relative coordinate pairs, requires candidates on both axes,
  rounds their mean, and rejects candidates more than 0.1 from that result. The
  supplied annotation independently resolves to `2481 x 3508`.
- `scripts/sciegqa_mineru/source_pages.py:78-96` returns the exact page-specific,
  300-DPI, single-file PNG Poppler token list.
- `scripts/sciegqa_mineru/source_pages.py:99-114` creates the destination parent,
  sends the exact request with a 120-second timeout, reads in 1 MiB chunks, and
  returns exactly the v1 URL, SHA-256, and byte count.
- `scripts/sciegqa_mineru/source_pages.py:117-141` creates the page parent, uses
  the suffixless output prefix, invokes Poppler once with `check=True`, requires
  the exact expected PIL image size, and returns exactly width, height, 300 DPI,
  and image SHA-256 metadata. It has no retry, resize, fallback, or alternate
  renderer path.
- `tests/test_sciegqa_mineru_source_pages.py:40-200` covers all required success
  and explicit failure paths. The Task 4 diff contains only the source module,
  its tests, and the implementation evidence report; no PDF or rendered page was
  added.

## Independent Verification

- `python -m pytest -q tests/test_sciegqa_mineru_source_pages.py`:
  `12 passed in 0.07s`.
- `python -m pytest -q`: `115 passed in 2.26s`.
- Independent standard-library mocks verified exact request URLs, headers,
  timeouts, download read sizes, subprocess arguments/call count, and exact
  metadata key sets.
- Independent geometry probe returned `(2481, 3508)` from the supplied boxes.
- `git diff --check b7b400e..b41f166` passed; changed files are regular
  `rw-r--r--` files. Poppler 26.03.0 and Pillow 12.2.0 are available.

## Issues

None.

## Verdict

✅ Spec compliant
