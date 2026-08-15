# Task 5 Quality Fix Evidence

Date: 2026-06-29

State: implementation quality fixes complete; live artifact build intentionally
not rerun in this task.

## Approved architecture

The subset builder now fails closed around one complete generated dataset:

1. The canonical output directory must be absent, including a dangling symlink,
   before source reading or materialization begins.
2. Each deterministic selection attempt uses a unique sibling staging directory
   on the same filesystem.
3. Only deterministic annotation, page-range, or exact rendered-dimension
   failures use `SourceDocumentError` and may exclude a selected document.
4. HTTP failures, timeouts, URL errors, storage failures, missing tools,
   subprocess failures, invalid PDF payloads, PDF-reader failures, and manifest
   or publication failures propagate immediately without reselection.
5. A rejected deterministic attempt is deleted before reselection. Repeated or
   nonselected document rejection fails deliberately.
6. A successful attempt writes and validates all six root manifests in staging,
   then publishes the entire directory with one `os.replace`.
7. A runtime, exhaustion, write, or publication failure leaves the canonical
   directory absent, removes the current stage, and atomically writes a sibling
   failure audit with revision, accumulated deterministic rejections, and the
   error type/message.
8. Local JSONL mode requires an explicit nonempty revision other than `main`;
   Hub mode resolves the requested revision (default `main`) to the immutable Hub
   SHA before IDs are built.

Optional `export.arxiv.org` category metadata was removed from this pilot's
materialization path by approved design. The required identity chain is the
immutable SciEGQA JSONL source row, its `doc_name`, the explicit arXiv v1 URL,
and the downloaded PDF/page/hash audit. No category-metadata fallback or retry
was added.

Every emitted query is validated against the immutable JSONL row at its
`source_record_index` for exact question, answer, category, document ID binding,
arXiv version and v1 URL, evidence page, pixel box, normalized box, and
`subimg_type`. Query/evidence cardinality, uniqueness, and page joins are also
checked before publication.

## TDD evidence

Targeted tests were added before production changes. Actual RED evidence
included:

- local JSONL without a revision returned `1` after falling into file I/O rather
  than the required argparse return code `2`;
- invalid source fields and same-page dimension conflicts were accepted;
- page bounds were not checked before rendering;
- runtime failures were converted into `SourceDocumentError`;
- canonical output was created incrementally rather than published atomically;
- failure audits and stage cleanup did not exist;
- a nonselected `SourceDocumentError` caused an infinite reselection loop (the
  RED run was terminated after this behavior was identified);
- whitespace-padded `main` bypassed local revision rejection;
- a dangling canonical symlink bypassed the existence gate; and
- a tampered document v1 URL passed the first source-round-trip validator.

Each later edge case was observed failing before its minimal implementation
change. No production change preceded its corresponding RED test.

Final GREEN commands and results:

```text
python -m pytest -q \
  tests/test_sciegqa_mineru_end_to_end.py \
  tests/test_sciegqa_mineru_selection.py \
  tests/test_sciegqa_mineru_source_pages.py
113 passed, 17 subtests passed in 2.11s

python -m pytest -q
169 passed, 17 subtests passed in 5.56s

python -m py_compile \
  scripts/build_sciegqa_mineru_subset.py \
  tests/test_sciegqa_mineru_end_to_end.py
PASS

git diff --check
PASS
```

## Independent probes

An independent temporary-directory probe, separate from pytest fixtures,
reported:

```text
PASS immutable-round-trip revision=4ffb867c88e3264161920b4b2446d5ac6352269e rows=30780 selected=20
PASS 429-fail-fast calls=1 canonical=absent stages=0
PASS rejection-cleanup attempts=2 canonical=complete stages=0
PASS same-page-conflict source-records=2,9
PASS page-bounds-before-render category-metadata-calls=0
PASS local-cli-floating-revision returncode=2
```

The immutable round-trip probe used the real cached Hub JSONL, selected the
approved 20 rows, built the entity records, and compared every required field
back to the original row. It did not call arXiv or run the live build.

## Files and commit

Changed implementation and tests:

- `scripts/build_sciegqa_mineru_subset.py`
- `tests/test_sciegqa_mineru_end_to_end.py`

Immutable implementation commit:
`eac34049aab5a61c38049c02c4951fe0843f241a`

This report is the only additional tracked file in the quality-fix task.

## Remaining concern

The live build was deliberately not rerun, and the existing ignored partial
`outputs/sciegqa_mineru_pilot` artifacts were not touched or moved. Full live
PDF/page/hash/geometry/visual verification and canonical manifest publication
remain a later gate after review of this fix.

First-page text extraction is not a hard identity requirement: the approved
probe found that one cached paper (`2411.08350`) does not expose its ID there.
Identity validation therefore relies on the explicit v1 URL/document binding,
immutable source-row join, and later PDF/page/hash audit.
