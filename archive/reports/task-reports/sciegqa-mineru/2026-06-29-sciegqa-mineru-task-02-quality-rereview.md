# SciEGQA MinerU Task 2 Quality Re-review

## Reviewed scope

- Prior quality review: `ae82668`
- Implementation fix: `c4d0829`
- Fix evidence: `2942cb2`

The implementation change is limited to
`scripts/sciegqa_mineru/schema.py` and
`tests/test_sciegqa_mineru_schema.py`. The subsequent commit adds only the
quality-fix evidence report.

## Verification

- `python -m pytest tests/test_sciegqa_mineru_schema.py -q`: 16 passed.
- `python -m pytest -q`: 56 passed.
- `git diff --check ae82668..2942cb2`: clean.
- Independent current-code probes rejected zero, negative, NaN, positive
  infinity, and negative infinity on both target-dimension axes.
- The known 2481 by 3508 conversion still produced the expected pixel values;
  its normalized round trip differed by at most approximately `1.14e-13`.
- Independent JSONL probes preserved valid object order while skipping blank
  physical lines. Arrays, `null`, and scalar values after three physical lines
  each raised `JSONL row at line 4 must be an object`.
- An independent execution of the pre-fix `ae82668` module accepted both a
  zero-width target and an array JSONL row, confirming the new defect tests
  exercise behavior that the old code failed.

## Prior findings

### Invalid target dimensions

Resolved. `norm1000_to_pixels()` now rejects non-finite dimensions before the
positive-dimension check and rejects zero or negative dimensions before any
scaling. It introduces no clipping or rounding and leaves valid arithmetic
unchanged. Parameterized tests cover zero, negative, NaN, and infinity on each
axis.

### Non-object JSONL rows

Resolved. `read_jsonl()` enumerates physical source lines from one, continues
to skip blank lines, preserves valid row order, and rejects every decoded
non-dictionary row with a direct error containing its physical line number.
The tests cover array, null, and scalar JSON values after a blank line.

### Artifact digest assertion

Resolved. The test now compares `artifact_sha256()` with an independently
computed `hashlib.sha256` digest of the complete written bytes rather than
checking digest length alone.

## Scope and quality

The fixes address the two root contract failures directly. They add no new
dependencies, fallback behavior, geometry abstraction, or unrelated changes.
The tests are focused and would fail against the prior defective behaviors.

## Remaining issues

Critical: None.

Important: None.

Minor: None.

## Assessment

✅ **Approved.** Ready to proceed with Task 3.
