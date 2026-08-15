# SciEGQA MinerU Task 2 Evidence

## Scope

Implemented the canonical geometry and provenance helpers for the SciEGQA
MinerU pilot. The work is limited to the package version constant, frozen
`BBox` geometry contract, deterministic IDs, artifact hashing, JSONL I/O, and
focused tests.

## Evidence

- RED: `python -m pytest tests/test_sciegqa_mineru_schema.py -q` exited 2 during
  collection with `ModuleNotFoundError: No module named
  'scripts.sciegqa_mineru'` before production files existed.
- GREEN: `python -m pytest tests/test_sciegqa_mineru_schema.py -q` passed all 4
  tests in 0.04 seconds.
- Full suite: `python -m pytest -q` passed all 44 tests in 2.18 seconds.
- Hygiene: `git diff --cached --check` completed without output.

## Files

- `scripts/sciegqa_mineru/__init__.py`
- `scripts/sciegqa_mineru/schema.py`
- `tests/test_sciegqa_mineru_schema.py`
- `docs/superpowers/task-reports/2026-06-29-sciegqa-mineru-task-02.md`

## Commits

- Implementation: `2bf0919` (`feat: define SciEGQA MinerU provenance schema`)
- Report: finalized in the subsequent report-only commit.

## Self-review

- `BBox` is frozen and performs no clipping, rounding, coordinate inference,
  or lossy serialization.
- Bounds are inclusive, while boxes require strict positive area (`x0 < x1`
  and `y0 < y1`) as approved after the assumptions review.
- Normalized and pixel conversions use direct floating-point scale operations;
  the representative coordinates round-trip within nine decimal places.
- Stable IDs derive only from the supplied source identity parts, not local
  paths.
- Hashing streams 1 MiB chunks, and JSONL I/O preserves row order while
  ignoring blank input lines.
- No dependencies or files outside the assigned scope were changed.

## Concerns

None.
