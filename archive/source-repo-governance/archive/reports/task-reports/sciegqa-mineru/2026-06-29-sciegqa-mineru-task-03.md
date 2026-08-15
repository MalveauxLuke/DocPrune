# Task 3: Deterministic SPSR Selection Evidence

## Scope

- Added the approved production category quotas in insertion order, totaling 20.
- Added strict SPSR geometry filtering for one evidence page, one absolute box,
  and one normalized box.
- Added deterministic high-yield document selection with one category per source
  document, lexicographic document tie-breaking, integer source-index ordering,
  and deterministic document exclusion.
- Added the exact selection audit contract.
- Did not access SciEGQA-Train or add dataset I/O, a CLI, randomization, or
  dependencies.

## TDD evidence

RED command:

```text
python -m pytest -q tests/test_sciegqa_mineru_selection.py
```

RED result (exit 2):

```text
E   ModuleNotFoundError: No module named 'scripts.sciegqa_mineru.selection'
1 error in 0.07s
```

The failure was the expected missing production module, before any selection
implementation existed.

Focused GREEN command:

```text
python -m pytest -q tests/test_sciegqa_mineru_selection.py
```

Focused GREEN result (exit 0):

```text
.............                                                            [100%]
13 passed in 0.04s
```

Full-suite command:

```text
python -m pytest -q
```

Full-suite result (exit 0):

```text
.....................................................................    [100%]
69 passed in 2.24s
```

## Determinism evidence

The focused tests establish:

- required selection `[0, 1, 3, 4]` for the cross-category fixture;
- required replacement `[5, 6, 3, 4]` with `excluded_docs={"cross"}`;
- lexicographic selection of `a-doc` when candidate yields tie;
- ascending integer source-index order within the chosen document;
- exact category counts and audit fields; and
- an error when no unused document can satisfy a category quota.

An additional probe ran the same fixture in forward and reverse input order.
Both runs returned:

```text
[0, 1, 3, 4] {'q-fin': 'cross', 'econ': 'econ-only', 'math': 'math-only'}
```

## Files

- `scripts/sciegqa_mineru/selection.py`
- `tests/test_sciegqa_mineru_selection.py`
- `docs/superpowers/task-reports/2026-06-29-sciegqa-mineru-task-03.md`

Immutable implementation commit:
`ce17d93b6b5b24d1e727bee99efa5439a8fbe320`

## Concerns

None within Task 3 scope. Real dataset compatibility remains intentionally
deferred to the later dataset-building task.
