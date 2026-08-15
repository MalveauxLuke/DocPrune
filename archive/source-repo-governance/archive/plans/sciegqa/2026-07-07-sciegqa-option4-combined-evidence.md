# SciEGQA Option 4 Combined Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce one deterministic, provenance-complete SciEGQA Option 4 evidence dataset from 10,431 queries on 6,630 pages by reusing hash-validated original OCR, completing the new-page OCR retry gate, and applying the exact original semantic-section, labeling, quarantine, and fresh 70/15/15 split process.

**Architecture:** Keep the existing shared semantic-section implementation in `scripts/sciegqa_parser_compare/final_labeling.py` unchanged. Extend the evidence orchestration layer to prepare a clean new-page run from the completed primary shards, merge two already validated OCR batches into a self-contained combined run, record separate selection and split seeds, and emit a batch-comparability audit. All large artifacts remain under `/scratch/lmalveau`.

**Tech Stack:** Python 3.12, pytest, JSONL manifests, SHA-256 provenance, Slurm/SOL, pinned DeepSeek-OCR2.

## Global Constraints

- Use dataset revision `4ffb867c88e3264161920b4b2446d5ac6352269e` and model revision `aaa02f3811945a91062062994c5c4a3f4c0af2b0`.
- Reuse original OCR only after validating page images, model/prompt/parameters, raw artifacts, hashes, segments, and boxes.
- Give each expansion page at most one primary attempt and one retry; never submit a third attempt.
- Call only the shared `build_deepseek_semantic_sections()` implementation with `max_center_distance=350.0`.
- Use member-union gold coverage and threshold `0.70` exactly.
- Quarantine an entire page for terminal OCR failure, any partial relation, zero positives, or multiple positives.
- Keep all same-page hard negatives and create no cross-page negatives.
- Use fresh split seed `20260707`, grouped by page and approximately stratified by domain × query intent.
- Preserve the old split only as provenance; do not reuse it.
- Keep large outputs, OCR, caches, and manifests under `/scratch/lmalveau`.

---

### Task 1: Baseline and artifact audit

**Files:**
- Read: `sol/SCIEGQA_4K_DEEPSEEK_EVIDENCE_HANDOFF.md`
- Read: `sol/SCIEGQA_OPTION4_NEW_PAGES_HANDOFF.md`
- Read: `/scratch/lmalveau/sciegqa_train_4k/evidence_poc/20260703T090303Z`
- Read: `/scratch/lmalveau/sciegqa_train_4k/evidence_option4_10431/ocr_input`

**Interfaces:**
- Consumes: original finalized run, Option 4 manifests, completed expansion primary shards.
- Produces: verified baseline counts/hashes and a failing-fast discrepancy report.

- [ ] Verify branch, ancestry, repository status, source hashes, 10,431/6,630 cardinality, 5,822/4,609 decomposition, and zero page overlap.
- [ ] Run the focused baseline tests in `/home/lmalveau/mamba-envs/mineru34-sol/bin/python`.
- [ ] Validate all 2,919 primary expansion attempts and record exact retry reasons.

### Task 2: Generalize run preparation and split provenance

**Files:**
- Modify: `scripts/sciegqa_evidence_poc.py`
- Modify: `scripts/build_sciegqa_evidence_poc.py`
- Modify: `tests/test_sciegqa_evidence_poc.py`

**Interfaces:**
- Produces: `prepare_run(..., expected_query_count, expected_page_count, selection_seed, split_seed)` and matching CLI flags.

- [ ] Add failing tests proving non-4K run preparation accepts explicit counts and records separate seeds.
- [ ] Run the tests and confirm the expected failure.
- [ ] Implement the minimal parameterization while preserving original defaults.
- [ ] Run focused tests and confirm they pass.

### Task 3: Add validated combined-run construction

**Files:**
- Modify: `scripts/sciegqa_evidence_poc.py`
- Modify: `scripts/build_sciegqa_evidence_poc.py`
- Modify: `tests/test_sciegqa_evidence_poc.py`

**Interfaces:**
- Consumes: Option 4 selection, enriched original/new page manifests, original/new merged OCR runs and parse audits.
- Produces: `prepare_combined_run(...)` with 6,630 enriched pages, 10,431 queries, selected OCR copied into a self-contained raw tree, terminal records, combined parse audit, and provenance.

- [ ] Add failing tests for disjoint batch membership, exact query decomposition, source artifact hash rejection, image hash rejection, duplicate IDs, terminal-page preservation, copied-artifact rehashing, and stable combined ordering.
- [ ] Run the tests and confirm the expected failures.
- [ ] Implement combined preparation by calling `validate_ocr_attempt()` on every reused selected OCR record before copying.
- [ ] Verify copied raw, markdown, and diagnostics hashes after copying.
- [ ] Run focused tests and confirm they pass.

### Task 4: Preserve exact semantics and add batch audit

**Files:**
- Modify: `scripts/sciegqa_evidence_poc.py`
- Modify: `tests/test_sciegqa_evidence_poc.py`
- Do not modify semantic rules in: `scripts/sciegqa_parser_compare/final_labeling.py`

**Interfaces:**
- Consumes: combined selected OCR and Option 4 queries.
- Produces: unchanged semantic sections/labels plus `audits/batch_comparability.json`.

- [ ] Add a failing integration test proving the dataset builder calls the shared section builder with `350.0` and does not use a copied implementation.
- [ ] Add failing tests for per-batch source, OCR-valid, terminal, retained, quarantined, query, section-kind, and proportion metrics with a 0.05 absolute-difference flag.
- [ ] Implement batch audit generation without altering semantic-section membership or labeling rules.
- [ ] Make `finalize_run()` use `split_seed` while retaining backward compatibility with old runs.
- [ ] Run the golden final-labeling regressions and evidence tests.

### Task 5: Commit reproducible orchestration state

**Files:**
- Commit only relevant code, tests, submit scripts, handoffs, and this plan.
- Exclude unrelated logs and artifacts.

- [ ] Review the diff for accidental semantic-rule changes and unrelated files.
- [ ] Run the complete focused test suite.
- [ ] Confirm branch `COLQWEN_binary_classification` and commit the reproducible code state before new SOL jobs.

### Task 6: Complete expansion OCR gate and retry

**Files/paths:**
- Create under scratch: `/scratch/lmalveau/sciegqa_train_4k/evidence_option4_10431/new_page_evidence/<RUN_ID>`
- Reuse primary: `/scratch/lmalveau/sciegqa_train_4k/evidence_option4_10431/ocr_input/deepseek_ocr2_attempts/attempt_1`

- [ ] Prepare the clean 4,609-query/2,919-page run with split seed `20260707`.
- [ ] Build and verify the retry manifest from the exact original quality gate.
- [ ] Submit one retry array only for invalid pages and record its job ID immediately.
- [ ] Wait for terminal completion and validate retry coverage.
- [ ] Deterministically merge attempts, select first valid attempts, preserve failed artifacts, and mark terminal failures.

### Task 7: Build and finalize the combined dataset

**Files/paths:**
- Create under scratch: `/scratch/lmalveau/sciegqa_train_4k/evidence_option4_10431/combined_evidence/<RUN_ID>`

- [ ] Prepare the combined run from the original finalized OCR and new merged OCR.
- [ ] Hash-check all reused source images and OCR artifacts, then verify all copied artifact hashes.
- [ ] Verify 6,630 pages, 10,431 queries, 5,822 original-page queries, 4,609 expansion-page queries, and disjoint batch page IDs.
- [ ] Run finalization twice and require byte-identical section, label, quarantine, split, and audit manifests.
- [ ] Verify exact shared semantic-section rules through golden tests and output invariants.
- [ ] Verify every retained query has exactly one positive, no partial survives, every negative is same-page, and no page/query/pair crosses splits.
- [ ] Verify the fresh 70/15/15 split and batch-comparability audit.

### Task 8: Final verification and handoff

**Files:**
- Modify: `sol/CURRENT_SOL_TASK.md`
- Update or create a concise final Option 4 evidence handoff under `sol/` only if needed for paths/hashes.

- [ ] Run the full focused test suite from a clean command.
- [ ] Inspect final manifest counts, SHA-256 values, determinism audit, parse audit, labeling audit, split audit, and batch audit.
- [ ] Update `sol/CURRENT_SOL_TASK.md` with only current state, next action, job/run IDs, paths, and blockers.
- [ ] Report the final deliverable and any quarantined/terminal counts without claiming training has begun.
