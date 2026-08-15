# Task 3 Quality Fix Evidence

## Scope

- Enforced actual JSON-list shapes at every evidence-page and bounding-box
  nesting level.
- Enforced an integer evidence page while excluding booleans, and treated
  numeric overflow as structurally ineligible.
- Added deliberate validation errors for category, document, source-index, and
  quota metadata.
- Rejected duplicate source indices across all structurally eligible rows
  before grouping or selection.
- Replaced the mutable quota default with `None` and an internal mapping copy.
- Corrected the approved audit key to `selection_strategy`.
- Preserved the approved quota order, valid selections, document uniqueness,
  tie-breaking, and unsatisfied-quota error.

## TDD evidence

Focused RED command:

```text
python -m pytest -q tests/test_sciegqa_mineru_selection.py
```

Focused RED result (exit 1):

```text
33 failed, 12 passed in 0.21s
```

The failures directly exposed list-shape impersonation, invalid page types,
uncaught overflow, incidental metadata errors, permutation-sensitive duplicate
indices, invalid quotas, the mutable default, and the audit-key mismatch.

Focused GREEN command:

```text
python -m pytest -q tests/test_sciegqa_mineru_selection.py
```

Focused GREEN result (exit 0):

```text
.............................................                            [100%]
45 passed in 0.06s
```

Full-suite command:

```text
python -m pytest -q
```

Full-suite result (exit 0):

```text
........................................................................ [ 71%]
.............................                                            [100%]
101 passed in 3.36s
```

## Independent probes

Forward and reversed valid fixtures both produced the same selected indices
and audit:

```text
[0, 1, 3, 4]
{'q-fin': 'cross', 'econ': 'econ-only', 'math': 'math-only'}
```

Forward and reversed duplicate fixtures both raised:

```text
Duplicate source_record_index 9 among eligible rows
```

Additional malformed-input probes confirmed:

```text
malformed-string-box False
invalid-metadata Invalid source_record_index for source row at input position 0
invalid-quota 0 Invalid quota for q-fin: expected positive integer
invalid-quota -1 Invalid quota for q-fin: expected positive integer
invalid-quota True Invalid quota for q-fin: expected positive integer
invalid-quota 1.0 Invalid quota for q-fin: expected positive integer
```

`git diff --check` completed without output.

## Files

- `scripts/sciegqa_mineru/selection.py`
- `tests/test_sciegqa_mineru_selection.py`
- `docs/superpowers/task-reports/2026-06-29-sciegqa-mineru-task-03-quality-fix.md`

Immutable quality-fix commit:
`ec132995120d772b9280f7045dd459ec28652d48`

## Self-review

The implementation copies quota mappings, does not mutate caller state or the
production constant, validates quotas before consuming rows, validates all
selection metadata before grouping, and detects duplicate eligible indices
before category filtering. No dataset access, CLI, randomization, dependency,
dead code, or out-of-scope file change was introduced.

No remaining concerns within this quality-fix scope.
