# Task 5 Code-Quality Review

Date: 2026-06-29

Implementation: `d62aec0` on base `62772e4`

Verdict: **Not ready**

Live gate: **Separately pending because the arXiv metadata service returned
HTTP 429/time-outs.** The service condition is external and is not itself a
code failure. This review performed no production network requests and does
not recommend adding a retry or fallback.

## Findings

### Critical

1. **Transient infrastructure failures are converted into source rejection
   and can silently change the selected dataset.**

   Files: `scripts/build_sciegqa_mineru_subset.py:170-201` and
   `scripts/build_sciegqa_mineru_subset.py:228-245`

   `materialize_document_pages()` catches every `Exception`, including HTTP
   429, time-outs, missing executables, permission errors, and disk failures,
   and wraps it as `SourceDocumentError`. `build_subset()` interprets that
   wrapper as evidence that the selected document is invalid, excludes the
   document, and chooses a lower-ranked document. If the service later
   recovers, the build can publish a 20-query subset whose document choices
   depend on a temporary outage rather than the approved deterministic source
   rules. Each exclusion also restarts materialization for all selected
   documents, multiplying otherwise successful source operations.

   An offline probe reproduced the exact classification boundary: a mocked
   `HTTPError(429)` from `fetch_arxiv_categories()` became a
   `SourceDocumentError` for `2401.00001`. The partial live artifact inventory
   is consistent with the consequence: it includes alternative documents but
   no completed canonical manifests.

   Fix: distinguish permanent, document-specific content/geometry validation
   failures from transport, runtime, and storage failures. Only the former may
   trigger exclusion. Propagate the latter and fail the run while preserving
   the approved selection. This is fail-fast behavior, not a retry. Add tests
   showing an invalid PDF can be rejected while HTTP 429, time-out,
   `FileNotFoundError` for `pdftoppm`, and write failures cannot reselect.

2. **The canonical output is neither cross-run safe nor transactionally
   published.**

   File: `scripts/build_sciegqa_mineru_subset.py:236-269`

   Source materialization writes into the final output tree before success,
   and the six root manifests are then overwritten sequentially. On a failed
   rerun, manifests from a prior run remain in place with no state marker to
   identify them as stale. On an I/O failure during publication, consumers can
   observe a mixed or incomplete canonical manifest set.

   Independent probes demonstrated both cases:

   - six pre-existing root manifests all remained byte-for-byte unchanged
     after a new build rejected its only candidate and failed; the final error
     only said `No unused document can satisfy cs quota 1`, and the accumulated
     rejection evidence was lost;
   - a simulated failure on the second JSONL write left only
     `dataset_source.json` and `documents.jsonl` from the new run.

   A later MinerU or viewer step has no reliable way to distinguish either
   state from an intended canonical build directory.

   Fix: construct and validate the complete run in a sibling staging
   directory, then promote it only as a complete generation. Either refuse a
   non-empty canonical destination or use an approved generation/pointer
   scheme; do not merge a new run into an old canonical directory. Persist a
   failed-run audit outside the canonical generation so rejection history is
   retained without making partial manifests look valid. Add existing-output,
   interrupted-publication, and rejection-exhaustion tests.

### Important

3. **Conflicting dimensions for the same page are silently discarded.**

   File: `scripts/build_sciegqa_mineru_subset.py:83-106`

   Each evidence row independently infers its page dimensions, but
   `pages.setdefault()` keeps whichever dimensions appeared first. A later
   annotation for the same document/page can infer a different size and still
   be emitted as evidence against the first page record. An offline probe
   supplied valid annotations inferring 2550 x 3300 and 2700 x 3300 for one
   page; the builder emitted one 2550 x 3300 page plus both evidence boxes with
   no error. The current 20-row cached Train selection happens to be
   consistent on all three shared pages, but the invariant is not enforced.

   Fix: when a page ID already exists, compare the complete page identity and
   inferred dimensions and raise a diagnostic naming both source record
   indices on any conflict. Add same-page agreement and disagreement tests.

4. **Local dataset mode records a floating label as if it were an immutable
   dataset revision.**

   Files: `scripts/build_sciegqa_mineru_subset.py:69-81`,
   `scripts/build_sciegqa_mineru_subset.py:250-256`, and
   `scripts/build_sciegqa_mineru_subset.py:280-293`

   Hub mode resolves `main` to an immutable commit before IDs and provenance
   are created. Local `--dataset-jsonl` mode bypasses resolution but defaults
   to the literal string `main`; that value is stored as `dataset_revision`
   and used in every stable ID. Different local file contents can therefore
   reuse the same entity identities while the manifest implies revision
   provenance it does not possess.

   Fix: require an explicit immutable source revision for local mode, or derive
   a clearly labelled local identity from the JSONL SHA-256 and store it
   separately from any requested Hub revision. Add a CLI test proving local
   content changes cannot retain the same source identity accidentally.

5. **Page numbers are not validated at the entity or PDF boundary.**

   Files: `scripts/build_sciegqa_mineru_subset.py:65-85` and
   `scripts/build_sciegqa_mineru_subset.py:174-193`

   A page number of zero is accepted and emitted with
   `source_page_index: -1`; an offline probe confirmed this. After the PDF page
   count is known, the builder also does not explicitly require
   `1 <= source_page_number <= page_count` before invoking `pdftoppm`. The
   renderer should usually fail, but that produces a tool-level error and,
   under the current broad exception handling, rejects the whole document
   rather than identifying a bad annotation precisely.

   Fix: reject non-positive page numbers before ID creation and validate the
   upper bound immediately after reading `page_count`. Include the document,
   page number, source record, and valid page range in the error.

### Minor

6. **The selected-row contract does not validate all fields it later indexes.**

   Files: `scripts/sciegqa_mineru/selection.py:35-66` and
   `scripts/build_sciegqa_mineru_subset.py:107-129`

   SPSR eligibility validates page and bbox cardinality but not the presence
   and shape of `query`, `answer`, or `subimg_type`. Malformed input can fail
   later with an unhelpful `KeyError` or `IndexError`, while `None` question or
   answer values become the strings `"None"`. The immutable cached Train file
   satisfies the expected schema, so this does not affect the current 20-row
   selection.

   Fix: validate these fields at the selection boundary and report the source
   record index and field name.

## Strengths

- Hub resolution pins the download to the returned immutable commit SHA.
- IDs and foreign keys are stable and joinable for valid selected input;
  duplicate eligible source indices fail explicitly.
- Document and page inputs are cloned before enrichment; the tests confirm no
  mutation of the caller's entity records.
- Strict arXiv ID validation prevents source-path traversal.
- Source PDF and PNG helpers publish individual artifacts atomically and
  preserve hashes, exact v1 URLs, 300-DPI rendering, and relative paths.
- Selection and entity ordering are deterministic, the default quota object is
  immutable, and the real cached Train source is small enough for the current
  in-memory pass (10,951,183 bytes and 30,780 rows).
- The focused tests cover direct script imports, joinability, deterministic
  reselection, relative artifact paths, exact source boundaries, and wrapped
  document failures. Full and focused suites are green.

## Verification

```text
python -m pytest -q \
  tests/test_sciegqa_mineru_end_to_end.py \
  tests/test_sciegqa_mineru_selection.py \
  tests/test_sciegqa_mineru_source_pages.py
82 passed, 17 subtests passed in 0.67s

python -m pytest -q
138 passed, 17 subtests passed in 2.66s

git diff --check 62772e4..HEAD
clean
```

Offline/mocked probes additionally covered HTTP 429 classification,
conflicting same-page dimensions, page zero, stale manifests after failed
reruns, partial manifest publication, cached Train-scale selection, and
shared-page dimension agreement. No production network call was made.

## Readiness

**Ready: No.** Findings 1 and 2 can change or ambiguously publish the canonical
dataset and should be fixed before Task 6 consumes it. Findings 3-5 should be
fixed in the same Task 5 boundary because they protect geometry and provenance
invariants. Finding 6 is minor but should receive a focused validation test.

The live artifact gate remains independently pending after these code fixes;
green offline tests do not substitute for successful PDF/page rendering and
visual geometry inspection.
