# SciEGQA MinerU Task 2 Specification Review

## Inspected revisions

- Base: `416d103`
- Implementation: `2bf0919`
- Implementation report: `7ace11a`
- Reviewed range: `416d103..7ace11a`

## Verification evidence

- `git diff --name-status 416d103..7ace11a` showed only the package metadata,
  schema helpers, focused tests, and Task 2 implementation report.
- `python -m pytest tests/test_sciegqa_mineru_schema.py -q` passed all 4 tests.
- `python -m pytest -q` passed all 44 tests.
- Independent Python probes confirmed schema version `1.0`, four-coordinate
  float conversion, strict rejection of zero-area/non-finite/out-of-bounds
  boxes, identity-preserving validation, the specified 2481x3508 round trip,
  exact compact-JSON SHA-256 stable IDs, sorted UTF-8 JSONL output, blank-line
  skipping, and artifact hashes matching `hashlib.sha256`.

## Issue

- `scripts/sciegqa_mineru/schema.py:85`: `read_jsonl` is declared as returning
  `list[Any]`, while the requested public contract is a list of dictionary
  rows. The approved implementation shape is `list[dict[str, Any]]`; the
  current annotation discards that schema guarantee. Change the return
  annotation (and, if desired, the local accumulator annotation) to preserve
  the requested dict-row contract. Runtime rejection of non-object JSON was
  not required by the approved Task 2 plan and is therefore not requested by
  this review.

## Report accuracy

The implementation report's focused and full-suite pass counts were
reproduced. Its scope and behavioral self-review are accurate except that the
JSONL reader's public type does not express the required dictionary-row
contract.

## Verdict

❌ Issues found
