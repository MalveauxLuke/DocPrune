# Task 3 Code-Quality Review

## Reviewed range

- Base: `e7f6c3320addd0ad421a007c5ec3bfa10b772492`
- Implementation: `ce17d93b6b5b24d1e727bee99efa5439a8fbe320`
- Implementation report: `af31e40314d176ab5260cd1306a7d2386d37f449`
- Spec review: passed in `993e16f`

## Strengths

- The approved happy path is compact, readable, and deterministic: category
  iteration follows quota insertion order, documents rank by eligible yield and
  lexical name, and rows with distinct integer indices sort consistently.
- Document exclusion is implemented through one shared unavailable-document
  set, so a valid source document cannot be assigned to two categories.
- Unsatisfied quotas fail explicitly with the required message instead of
  returning a partial selection.
- The audit is direct and easy to inspect, and valid quotas produce exact
  selected and per-category counts.
- Runtime is suitable for the 30,780-row Train split. An independent synthetic
  probe filtered and selected from 30,780 rows in about 114 ms on this machine.
- The implementation adds no dependency or unrelated abstraction.

## Critical findings

None.

## Important findings

### 1. Malformed sequence shapes can qualify as SPSR rows

**Location:** `scripts/sciegqa_mineru/selection.py:25-45`

The code checks only `len(...)` and then lets `BBox.from_list` iterate the
innermost value. Consequently, JSON-compatible strings can impersonate the
expected lists: `evidence_page = "1"`, `bbox = [["1234"]]`, and
`rel_bbox = [["1234"]]` all return `True` in independent probes. That means a
structurally malformed dataset row can enter the gold-evidence pilot and later
be treated as trustworthy provenance. Conversion can also leak
`OverflowError`, which is outside the current exception boundary.

**Fix:** Validate that the page/group/box layers have the expected non-string
sequence types and that a page value has the intended integer type before
geometry conversion. Extend the conversion boundary to reject numeric
conversion overflow. Add focused tests for strings at each nesting level and
for conversion overflow.

### 2. Duplicate indices make selection depend on input order, while invalid selection metadata leaks incidental exceptions

**Location:** `scripts/sciegqa_mineru/selection.py:56-61,81-84`

Python's stable sort preserves incoming order when two rows have the same
`source_record_index`. When a duplicate falls at the quota cutoff, permuting the
same rows selected `alpha` in one run and `beta` in another. This violates the
selection routine's deterministic contract. Invalid JSON-compatible metadata
also escapes through implementation details: a list-valued category or
document raises `TypeError` during grouping, and a non-numeric source index
raises `ValueError` during sorting. The current tests cover only valid strings
and unique integer indices.

**Fix:** Validate eligible rows' category, document name, and source index before
grouping; require the approved non-empty string/string/integer types (excluding
booleans). Detect duplicate source indices and raise one explicit validation
error rather than inventing a tie-breaker for records whose provenance identity
is ambiguous. Test reordered duplicates and each invalid metadata type.

### 3. Invalid quotas can produce a false audit instead of failing validation

**Location:** `scripts/sciegqa_mineru/selection.py:68-87`

Quotas are accepted without checking that each value is a positive integer.
Independent probes showed that quota `0` selects no row but still claims a
chosen document and consumes it; quota `-1` selected one row and reported count
`1`; quota `1.5` leaked a slicing `TypeError`; and `True` was silently treated as
one. These outcomes break the category-quota/count audit contract and can cause
a later category to fail because a document was consumed for a zero quota.

**Fix:** Validate category keys and require each quota to be a positive integer
but not a boolean before filtering or assigning any document. Add tests proving
that zero, negative, boolean, and non-integer quotas fail with one deliberate
error and cannot return an audit.

## Minor findings

### 1. The public mutable quota constant is also the function default

**Location:** `scripts/sciegqa_mineru/selection.py:11-20,51-54`

`select_rows` does not mutate the mapping, so there is no immediate internal
mutation bug. However, external mutation of `CATEGORY_QUOTAS` silently changes
the default behavior for all later calls in the process.

**Fix:** Use a `None` default and bind an immutable production quota mapping (or
copy a private production mapping) inside the function.

## Verification

```text
python -m pytest -q tests/test_sciegqa_mineru_selection.py
13 passed in 0.04s

python -m pytest -q
69 passed in 3.48s
```

Independent probes additionally covered all input permutations for distinct
indices, duplicate indices at a quota cutoff, malformed string shapes,
conversion overflow, invalid category/document/index values, invalid quotas,
and a 30,780-row synthetic input.

## Recommendations

Address the three important findings before Task 3 is used as the provenance
gate for Train rows. Keep the current selection strategy and audit shape; the
needed change is narrow input validation plus explicit duplicate detection, not
a redesign.

## Ready to proceed?

No. The valid-data path passes, but malformed structural values can be admitted,
duplicate provenance indices are input-order dependent, and invalid quotas can
return incorrect audit state.
