# Task 3 Spec Compliance Review

✅ Spec compliant

## Inspected range

- Base: `e7f6c3320addd0ad421a007c5ec3bfa10b772492`
- Implementation: `ce17d93b6b5b24d1e727bee99efa5439a8fbe320`
- Evidence report: `af31e40314d176ab5260cd1306a7d2386d37f449`
- Reviewed diff: `e7f6c33..af31e40`

The diff contains only the permitted production module, focused tests, and Task
3 implementation report. `git diff --check` found no whitespace errors.

## Contract evidence

- `CATEGORY_QUOTAS` preserves the approved insertion order and sums to 20.
- `is_spsr_row` requires one evidence page, one nested absolute box, and one
  nested normalized box. It rejects missing or malformed structures through the
  required exception boundary, requires finite positive-area absolute geometry,
  and validates normalized geometry against `(1000, 1000)` with `BBox`.
- `select_rows` filters to SPSR-eligible rows before grouping by category and
  document, consumes quotas in insertion order, honors excluded and already-used
  documents, ranks candidates by descending eligible yield then lexical document
  name, and orders selected rows by integer `source_record_index`.
- The returned audit has the exact strategy string and required counts and
  document mapping. The unsatisfied-quota message is exactly
  `No unused document can satisfy {category} quota {quota}`.
- The approved fixture selects `[0, 1, 3, 4]`; excluding `cross` selects
  `[5, 6, 3, 4]`, with the required document choices and category counts.
- No dataset fetch, CLI, randomization, dependency, or broader abstraction was
  added.

## Independent verification

```text
python -m pytest -q tests/test_sciegqa_mineru_selection.py
13 passed in 0.04s

python -m pytest -q
69 passed in 3.43s
```

Additional independent probes passed for reversed and rotated input order,
document exclusion, descending-yield priority, lexical tie behavior, integer
source-index ordering, exact audit equality, exact error text, missing fields,
non-finite coordinates, degenerate boxes, and normalized out-of-bounds boxes.

No missing or extra work was found.
