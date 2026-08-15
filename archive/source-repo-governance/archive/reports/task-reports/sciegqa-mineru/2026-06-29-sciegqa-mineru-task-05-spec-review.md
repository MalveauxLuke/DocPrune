# Task 5 Specification Review

Date: 2026-06-29

Review range: `62772e4..676afc7`

Verdict: **✅ Spec compliant (live validation pending)**

## Scope reviewed

The review covered the Task 5 implementation commit `d62aec0`, the partial
evidence report commit `676afc7`, and every changed file in the requested
range:

- `scripts/build_sciegqa_mineru_subset.py`
- `tests/test_sciegqa_mineru_end_to_end.py`
- `docs/superpowers/task-reports/2026-06-29-sciegqa-mineru-task-05.md`

No out-of-scope implementation file is present in the diff.

## Contract findings

- The dataset constants identify `Yuwh07/SciEGQA-Train` and
  `SciEGQA-Train.jsonl`. `resolve_dataset()` resolves `HfApi.dataset_info()`
  first and passes the returned immutable SHA to `hf_hub_download()`.
- Document, page, query, and evidence IDs are derived with `make_stable_id()`
  from immutable source identities. Records preserve the approved category,
  question, answer, selection rule, one-based page number, zero-based page
  index, exact pixel box, exact normalized box, annotation origin, and source
  subimage type. The four lists use the approved deterministic sorts.
- Materialization clones the document and page dictionaries before enrichment,
  groups pages by document, uses the Task 4 v1 download/category/render
  helpers and `PdfReader`, emits the exact relative PDF and page-image paths,
  and converts any document-scoped exception into `SourceDocumentError`.
  There is no fuzzy match, fallback source, clipping, resize, or metadata
  substitution.
- `build_subset()` reads strict JSONL objects, assigns the zero-based source
  index by enumeration, performs deterministic exclusion and reselection, and
  records concrete source rejection reasons. Dataset and entity manifests are
  not written until materialization succeeds. The dataset source record
  contains the immutable revision and JSONL SHA-256; the audit retains the
  current `selection_strategy` key; all four entity JSONLs and the exact
  summary fields are written.
- The CLI exposes the approved arguments and defaults, supports an explicit
  local JSONL boundary, resolves the Hub source otherwise, and runs directly
  as a script.
- Focused tests cover the direct CLI boundary, fixture build, stable and
  joinable provenance, materialization boundaries and relative paths,
  document-input cloning, source-failure wrapping, and deterministic
  reselection. Generated source artifacts remain under ignored `outputs/`.

## Independent verification

Focused suite:

```text
python -m pytest -q \
  tests/test_sciegqa_mineru_end_to_end.py \
  tests/test_sciegqa_mineru_selection.py \
  tests/test_sciegqa_mineru_source_pages.py

82 passed, 17 subtests passed in 0.66s
```

Full suite:

```text
python -m pytest -q

138 passed, 17 subtests passed in 2.73s
```

`git diff --check 62772e4..676afc7` was clean.

Independent mock probes confirmed that the resolved SHA, rather than the
mutable requested revision, is passed to `hf_hub_download()`. Additional
fixture probes confirmed document-page-query-evidence foreign keys and ID
uniqueness, and confirmed that a failed build followed by reselection
exhaustion leaves none of the six root manifests behind.

The locally cached immutable Train source independently reproduced:

- revision `4ffb867c88e3264161920b4b2446d5ac6352269e`;
- SHA-256 `7eb895fb913ffb607e3b41b01ec66e651685ad9404d230972fd987ba7f005e84`;
- 30,780 source rows and 11,668 eligible SPSR rows;
- exactly 20 selected queries across 8 uniquely assigned documents and 16
  unique pages; and
- the approved ordered category counts and chosen documents reported in the
  Task 5 evidence report.

## Live-gate assessment

The live `outputs/sciegqa_mineru_pilot` state contains 9 PDFs, 0 PNGs, and 0
root manifests, matching the partial evidence report. The report explicitly
labels its state `OPEN - BLOCKED_AT_LIVE_GATE`, distinguishes reconstructed RED
evidence from an original TDD transcript, identifies the arXiv metadata HTTP
429/time-out blocker, and lists the artifact and geometry checks still
pending. It does not claim Task 5 is complete.

The external 429 prevents live artifact acceptance but is not an
implementation specification failure. No production network download was
performed during this review.
