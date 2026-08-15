# Overlap-First Document Corpus Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic metadata-first ledger that canonicalizes exact parent-document/question overlaps across the three requested Visual-CoT manifests and the three selected BoundingDocs sources without discarding source annotations or silently resolving conflicts.

**Architecture:** A single Python CLI exposes independently tested normalization, adapter, grouping, conflict, accounting, and JSONL-writing functions. Visual-CoT filename stems and BoundingDocs native IDs produce `(document_family, parent_document_id)` keys; NFKC/casefold/whitespace-normalized questions are compared only inside those keys. Full generated manifests go to a caller-selected output directory, while a small observed-count report is retained in the repository.

**Tech Stack:** Python 3.12, standard library, PyArrow/Parquet, pytest.

## Global Constraints

- Read only `docvqa_cot_train.jsonl`, `dude_cot_train.jsonl`, and `infographicsvqa_cot_train.jsonl` from Visual-CoT.
- Read only BoundingDocs rows whose exact source is `DUDE`, `MP-DocVQA`, or `SP-DocVQA`.
- Do not use global question-text deduplication; canonical keys always include document family and parent document ID.
- Preserve all source observations and emit conflicts for incompatible answers, pages, or boxes.
- Do not run OCR, train, download data, or create final dataset splits.
- Do not create a branch or worktree.

---

### Task 1: Identity and question normalization

**Files:**
- Create: `scripts/build_overlap_first_corpus.py`
- Create: `tests/test_build_overlap_first_corpus.py`

**Interfaces:**
- Produces: `normalize_question(text: str) -> str`, `visual_parent_identity(dataset: str, image: str) -> tuple[str, str, int | None]`, and `bounding_parent_identity(source: str, doc_id: str) -> tuple[str, str, int | None]`.

- [ ] Write literal tests for DocVQA/DUDE suffix stripping, InfographicVQA stems, SP-DocVQA suffix stripping, and conservative NFKC/case/whitespace normalization.
- [ ] Run the focused tests and confirm failure because the module does not exist.
- [ ] Implement the smallest normalization functions that satisfy those cases and reject unsupported labels.
- [ ] Run the focused tests and confirm they pass.

### Task 2: Source adapters and annotation preservation

**Files:**
- Modify: `scripts/build_overlap_first_corpus.py`
- Modify: `tests/test_build_overlap_first_corpus.py`

**Interfaces:**
- Produces: `iter_visual_observations(path: Path, dataset: str) -> Iterator[dict]` and `iter_bounding_observations(raw_root: Path) -> Iterator[dict]`.
- Each observation includes canonical keys plus source record ID, native ID, original question, answers, raw and normalized boxes, answer pages, OCR reference, split, image reference, and provenance.

- [ ] Add small real JSONL and Parquet fixtures that assert exact preservation and stable line/QA identities.
- [ ] Run the adapter tests and confirm the missing-function failures.
- [ ] Implement strict adapters; malformed in-scope rows become excluded records rather than disappearing.
- [ ] Run the adapter tests and confirm they pass.

### Task 3: Canonical groups, overlaps, and conflicts

**Files:**
- Modify: `scripts/build_overlap_first_corpus.py`
- Modify: `tests/test_build_overlap_first_corpus.py`

**Interfaces:**
- Produces: `build_ledger(observations: Iterable[dict], excluded: list[dict]) -> dict[str, list[dict] | dict]`.
- Ledger keys: `documents`, `questions`, `overlap`, `unique`, `conflicts`, `unmatched`, `excluded`, and `summary`.

- [ ] Add literal fixtures proving identical text on different documents stays separate, same-document normalized questions merge, all annotations survive, and answer/page/box disagreements are emitted to conflicts.
- [ ] Run tests and verify expected failures.
- [ ] Implement grouping, pair/family counts, document-ecosystem unmatched classification, and strict accounting totals.
- [ ] Run tests and confirm all focused tests pass.

### Task 4: Deterministic CLI and real-data outputs

**Files:**
- Modify: `scripts/build_overlap_first_corpus.py`
- Modify: `tests/test_build_overlap_first_corpus.py`
- Create: `reports/overlap_first_document_corpus.md`

**Interfaces:**
- CLI accepts the Visual-CoT metadata root, BoundingDocs raw root, and output directory; writes canonical document/question plus overlap, unique, conflict, unmatched, and excluded JSONL manifests and `summary.json`.

- [ ] Add an integration fixture asserting sorted byte-identical outputs across two runs and complete input accounting.
- [ ] Run it and verify failure before CLI/output implementation.
- [ ] Implement atomic deterministic JSONL/JSON writing and summary validation.
- [ ] Run focused tests, then run the CLI twice on the real inputs and compare output hashes.
- [ ] Inspect at least three joined examples for DocVQA/MP-DocVQA, DocVQA/SP-DocVQA, and DUDE, plus InfographicVQA unmatched examples.
- [ ] Write the short measured report, including unavailable image hashes/OCR fields and the stale BoundingDocs card count.

### Task 5: Focused completion verification

**Files:**
- Verify: `scripts/build_overlap_first_corpus.py`
- Verify: `tests/test_build_overlap_first_corpus.py`
- Verify: `reports/overlap_first_document_corpus.md`

- [ ] Run `git diff --check`.
- [ ] Run only `tests/test_build_overlap_first_corpus.py` with third-party pytest plugin autoload disabled.
- [ ] Confirm summary accounting equals 60,243 Visual-CoT rows plus 36,528 selected BoundingDocs questions, minus only explicitly excluded rows.
- [ ] Confirm a second real-data run is byte-identical and report all generated paths without adding large manifests to Git.
