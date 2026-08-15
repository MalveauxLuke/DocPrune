# SciEGQA MinerU Task 2 Code-Quality Review

## Reviewed scope

- Base: `416d103`
- Implementation: `2bf0919`, `5853f91`
- Evidence: `7ace11a`, `db1cdd2`
- Prior review-only commits were excluded from the implementation assessment.

## Verification

- `python -m pytest tests/test_sciegqa_mineru_schema.py -q`: 5 passed.
- `python -m pytest -q`: 45 passed.
- `git diff --check 416d103..db1cdd2`: clean.
- Independent probes covered zero, negative, NaN, and infinite conversion
  dimensions; non-object JSONL rows; Unicode JSONL round trips; repeat artifact
  hashing; and repeat stable-ID generation.

## Strengths

- The module is compact, stdlib-only, and exposes a small API suitable for the
  later dataset and viewer stages.
- `BBox` is immutable, conversion arithmetic preserves floating-point
  precision, and `validate()` rejects non-finite, degenerate, and out-of-bounds
  boxes without clipping.
- IDs are SHA-256-derived from source identity parts and do not incorporate a
  local filesystem path. Hashing is streamed rather than loading an artifact
  into memory.
- JSONL output is deterministic by key order, UTF-8-safe, and creates missing
  parent directories explicitly.

## Critical issues

None.

## Important issues

1. `scripts/sciegqa_mineru/schema.py:48-55` accepts invalid target page
   dimensions in `norm1000_to_pixels()`. The method validates only the source
   normalized box, so zero, negative, NaN, and infinite widths or heights all
   return degenerate, negative, or non-finite pixel boxes. This violates the
   strict no-clipping/no-invalid-geometry contract and is asymmetric with
   `pixels_to_norm1000()`, which validates its bounds. Validate `width` and
   `height` as finite and positive before conversion (preferably through one
   shared bounds validator), then add parameterized regression tests for all
   four invalid classes on both axes.

2. `scripts/sciegqa_mineru/schema.py:85-91` declares
   `list[dict[str, Any]]` but does not enforce dictionary rows. A JSONL file
   containing `[]`, `null`, or `42` returns a list, `None`, or integer in that
   declared collection. Downstream code will rely on keyed provenance records,
   so the mismatch defers a malformed-input failure until an unrelated access.
   Check each decoded row is a dictionary and raise a clear `ValueError` that
   includes the source line number; add regression tests for non-object rows.

## Minor issues

1. `tests/test_sciegqa_mineru_schema.py:41-48` verifies only that the artifact
   digest has 64 characters. This would not detect a constant or content-
   independent implementation. Assert the known SHA-256 for a fixed payload,
   or compare against `hashlib.sha256(path.read_bytes()).hexdigest()`.

## Recommendations

- Fix both Important issues before Task 3 relies on this contract, and keep the
  fixes inside the existing schema module and focused test file.
- Preserve the current direct, dependency-free API; no broader abstraction is
  warranted for these fixes.

## Assessment

Ready to proceed? **No**. The happy path is sound and all existing tests pass,
but invalid page dimensions can create corrupt geometry and malformed JSONL can
escape a public dictionary-row contract. Both are root-contract failures that
should be fixed and re-reviewed first.
