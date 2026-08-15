# SciEGQA MinerU Task 2 Quality-Fix Evidence

## Scope

Added strict target-dimension validation to normalized-to-pixel conversion,
rejected non-object JSONL rows with physical source line numbers, and
strengthened the artifact digest test to verify the complete hash value.

## Root Cause

- `norm1000_to_pixels` validated its normalized source box but scaled by the
  target width and height without validating those dimensions.
- `read_jsonl` appended every non-blank `json.loads` result without checking
  that it was a dictionary, contradicting its public row contract.

Direct reproduction showed zero, negative, NaN, and infinite dimensions
returning malformed boxes, while array, null, and scalar JSON rows were
returned as data.

## Evidence

- RED: `python -m pytest tests/test_sciegqa_mineru_schema.py -q` reported 11
  expected failures and 5 passes: all 8 invalid target-dimension combinations
  and all 3 non-object JSONL rows failed because no `ValueError` was raised.
- GREEN: `python -m pytest tests/test_sciegqa_mineru_schema.py -q` passed all 16
  tests in 0.05 seconds.
- Full suite: `python -m pytest -q` passed all 56 tests in 2.95 seconds.
- Independent probe: zero/negative dimensions on both axes reported
  `target dimensions must be positive`; NaN/infinity on both axes reported
  `target dimensions must be finite`.
- Independent probe: array, null, and scalar rows after a blank line each
  reported `JSONL row at line 3 must be an object`.
- Hygiene: `git diff --check` completed without output.

## Files

- `scripts/sciegqa_mineru/schema.py`
- `tests/test_sciegqa_mineru_schema.py`
- `docs/superpowers/task-reports/2026-06-29-sciegqa-mineru-task-02-quality-fix.md`

## Commits

- Fix: `c4d0829` (`fix: validate schema geometry and JSONL inputs`)
- Report: finalized in the subsequent report-only commit.

## Self-review

- Target dimensions are checked before scaling and no clipping, rounding, or
  fallback geometry was introduced.
- JSONL blank lines remain skipped, valid dictionary row order is unchanged,
  and physical line numbers include blank lines.
- Runtime row validation requires `dict`; it does not broaden acceptance to
  arbitrary `Mapping` implementations.
- The artifact test now compares the streamed digest against an independent
  `hashlib.sha256` digest of the full file bytes.
- No review document or file outside the assigned scope was changed.

## Concerns

None.
