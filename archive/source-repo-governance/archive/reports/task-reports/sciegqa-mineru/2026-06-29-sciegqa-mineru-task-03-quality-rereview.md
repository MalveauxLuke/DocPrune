# Task 3 Quality-Fix Re-review

## Reviewed range

- Prior quality review: `365b6b6622a15748f485deb3775d3eb50a0026f2`
- Quality fix: `ec132995120d772b9280f7045dd459ec28652d48`
- Fix evidence: `b103e29b180b03cf68e68d079de6db298572800a`

## Resolution of prior findings

### Important 1: malformed SPSR shapes and escaping overflow

Resolved. Evidence pages and every bounding-box nesting layer now require JSON
lists, page values require integers excluding booleans, and conversion overflow
is rejected by `is_spsr_row`. Independent probes across malformed strings,
tuples, mappings, nulls, non-numeric coordinates, NaN, infinity, and overflow
all returned `False` without an escaping exception.

### Important 2: duplicate-index nondeterminism and invalid metadata

Resolved. Structurally eligible rows now require integer source indices
excluding booleans plus non-empty string categories and document names.
Duplicate eligible indices raise a deliberate error before grouping. Every
permutation of a duplicate fixture raised the same identity error, while all
5,040 permutations of the approved seven-row fixture returned `[0, 1, 3, 4]`
and the same audit.

### Important 3: invalid quota behavior and false audits

Resolved. Quota validation happens before row consumption, requires non-empty
string categories and positive integer values excluding booleans, and raises a
deliberate `ValueError`. Zero, negative, boolean, float, string, and invalid-key
probes all failed without mutating the caller mapping or returning an audit.

### Minor 1: public mutable production quotas controlling defaults

Partially resolved. The function signature now safely uses `None`, and an
explicit caller mapping is copied and left unchanged. However, the public
production mapping remains mutable and is copied at call time for the default
path. An independent probe cleared `CATEGORY_QUOTAS`, set only `q-fin: 1`, and
a subsequent `select_rows([row])` silently used that altered one-row default.
The probe restored the mapping afterward.

## Remaining finding

### Minor: external mutation still changes production default selection

**Location:** `scripts/sciegqa_mineru/selection.py:11-20,67`

Replacing the parameter default with `None` removes the Python mutable-default
object trap, but it does not close the root issue from the prior review:
`CATEGORY_QUOTAS` is still a public dictionary, and `dict(CATEGORY_QUOTAS)`
reads any external mutations on every default call. A test that only confirms
the function itself does not mutate the constant cannot detect this behavior.

**Fix:** Store the authoritative production quota mapping immutably, for
example with `MappingProxyType`, and copy that immutable mapping into
`active_quotas`. Add a test that attempts external mutation and confirms it is
rejected, while retaining the existing insertion-order assertion.

## Audit and regression checks

- The audit now uses the approved `selection_strategy` key and exact value.
- Valid quotas still return exact category counts and document assignments.
- Valid document exclusion and explicit unsatisfied-quota behavior remain
  covered by the focused suite.
- No document reuse, dependency change, unrelated abstraction, or scope creep
  was found in the fix diff.
- A synthetic 30,780-row run completed in about 129 ms and selected the exact
  requested count.
- `git diff --check 365b6b6..b103e29` produced no errors.

## Verification

```text
python -m pytest -q tests/test_sciegqa_mineru_selection.py
45 passed in 0.06s

python -m pytest -q
101 passed in 2.19s
```

## Verdict

Not yet approved. All three Important findings are resolved, but the prior
Minor root cause remains at `scripts/sciegqa_mineru/selection.py:11-20,67`.
