# Task 3 Quota Immutability Fix Evidence

## Scope

- Wrapped the approved production quota literal in the standard-library
  `MappingProxyType`.
- Preserved public mapping access, insertion order, values, and total.
- Preserved internal copying for default selection and support for explicit
  caller-provided quota mappings.
- Added regression coverage for both item assignment and item deletion.

## TDD evidence

Focused RED command:

```text
python -m pytest -q tests/test_sciegqa_mineru_selection.py
```

Focused RED result (exit 1):

```text
............................................FF.                          [100%]
2 failed, 45 passed in 0.09s
```

Both failures were the expected `Failed: DID NOT RAISE <class 'TypeError'>`:
one for assignment and one for deletion. Test cleanup restored the original
dictionary after each RED case.

Focused GREEN command:

```text
python -m pytest -q tests/test_sciegqa_mineru_selection.py
```

Focused GREEN result (exit 0):

```text
...............................................                          [100%]
47 passed in 0.05s
```

Full-suite command:

```text
python -m pytest -q
```

Full-suite result (exit 0):

```text
........................................................................ [ 69%]
...............................                                          [100%]
103 passed in 2.21s
```

## Default and explicit mapping smoke

A synthetic valid fixture supplied every approved default quota from a unique
category document. Results:

```text
default-selected 20
default-counts {'q-fin': 4, 'q-bio': 3, 'eess': 3, 'physics': 3, 'cs': 2, 'econ': 2, 'stat': 2, 'math': 1}
quota-order [('q-fin', 4), ('q-bio', 3), ('eess', 3), ('physics', 3), ('cs', 2), ('econ', 2), ('stat', 2), ('math', 1)]
explicit-selected 4
```

`git diff --check` completed without output.

## Files and commit

- `scripts/sciegqa_mineru/selection.py`
- `tests/test_sciegqa_mineru_selection.py`
- `docs/superpowers/task-reports/2026-06-29-sciegqa-mineru-task-03-immutability-fix.md`

Immutable fix commit:
`9843fa01aa1901d1beca1f56529ba30250a09585`

No dead code, new dependency, or remaining concern was introduced within this
fix scope.
