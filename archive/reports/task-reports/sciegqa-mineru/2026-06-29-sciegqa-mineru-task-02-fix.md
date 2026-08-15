# SciEGQA MinerU Task 2 Fix Evidence

## Scope

Corrected the public `read_jsonl` return annotation from `list[Any]` to
`list[dict[str, Any]]` and added a regression test for that exact contract.
Runtime JSON parsing behavior is unchanged.

## Evidence

- RED: `python -m pytest tests/test_sciegqa_mineru_schema.py -q` reported 1
  failure and 4 passes. The failure showed `list[typing.Any]` did not equal
  `list[dict[str, typing.Any]]`.
- GREEN: `python -m pytest tests/test_sciegqa_mineru_schema.py -q` passed all 5
  tests in 0.04 seconds.
- Full suite: `python -m pytest -q` passed all 45 tests in 2.26 seconds.
- Hygiene: `git diff --check` completed without output.

## Files

- `scripts/sciegqa_mineru/schema.py`
- `tests/test_sciegqa_mineru_schema.py`
- `docs/superpowers/task-reports/2026-06-29-sciegqa-mineru-task-02-fix.md`

## Commits

- Fix: `5853f91` (`fix: preserve JSONL dictionary row contract`)
- Report: finalized in the subsequent report-only commit.

## Self-review

- `typing.get_type_hints(read_jsonl)["return"]` now resolves exactly to
  `list[dict[str, Any]]`.
- Only the function return annotation and local accumulator annotation changed
  in production code.
- No runtime object rejection or unrelated behavior was added.
- No earlier report or review document was changed.

## Concerns

None.
