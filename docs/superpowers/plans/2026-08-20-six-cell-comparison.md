# Six-Cell Comparison Validation and Reporting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development to implement this plan task-by-task.

**Goal:** Validate exactly six independently sealed all-kept/DocPrune runs and emit signed JSON plus deterministic Markdown absolute-and-delta reports.

**Architecture:** `docprune.comparison` owns matrix normalization, independent corpus/page checks, pair identity checks, and canonical report materialization. It reuses `validate_benchmark_run` only as one independent input gate and never treats its signed claims as proof of cross-run comparability. The CLI exposes a narrow `compare` boundary with six explicit run paths and atomic non-overwriting outputs.

**Tech Stack:** Python 3.10+, dataclasses, canonical JSON/SHA-256, pytest, existing `docprune.evaluation` and `docprune.metrics` APIs.

**Spec:** `docs/superpowers/specs/2026-08-20-docprune-paper-fidelity-correction-design.md`

## Global Constraints

- Accept exactly all-kept and DocPrune at top-1, top-2, and top-4.
- Require exact source-order 2,441 QIDs and shared corpus/runtime/upstream/model/processor/generation/hardware/precision/backend/warmup/measurement identity within each pair.
- Validate every retrieved document ID against the pinned corpus and every page index against that PDF's actual page count.
- Emit signed canonical JSON and deterministic Markdown; never overwrite existing outputs without an explicit safe policy.
- Report A100 values as reconstruction measurements, not RTX A6000 parity.
- Prefix Python/test commands with `env PYTHONPATH=/home/lmalveau/DocPrune-benchmark/src` and use `/home/lmalveau/mamba-envs/docprune-sol/bin/python`.

### Task 1: RED matrix and report tests

**Files:** create `tests/test_comparison.py`; modify CLI/evaluation tests only if boundary coverage requires it.

- Add a small explicitly fixture-marked six-run corpus with valid PDF bytes and two source QIDs.
- Assert failures for incomplete/duplicate/extra cells, reordered QIDs, shared identity mismatch, fabricated document, out-of-range page, and altered summary.
- Assert a complete matrix yields six absolute cells, three paired deltas, canonical signed JSON, deterministic Markdown, and no overwrite.
- Run focused tests and record literal RED output before implementing production code.

### Task 2: GREEN comparison implementation

**Files:** create `src/docprune/comparison.py`; modify `src/docprune/cli.py`; add focused CLI tests.

- Normalize path sequences/mappings to the six expected `(mode, page_count)` cells and reject missing/duplicate/extra cells.
- Run each input through the existing single-run validator, then independently read pinned corpus IDs/PDFs and validate every result page.
- Compare exact shared manifest identities per page-count pair and exact result QID order.
- Build canonical finite JSON payload, SHA-256 digest, deterministic Markdown, and atomic no-overwrite output pair.
- Add `compare`/`compare-runs` CLI aliases with six explicit paths, corpus root, and JSON/Markdown outputs.

### Task 3: Verification and report

- Run focused comparison/CLI/evaluation tests, full CPU suite, Ruff, and `git diff --check`.
- Self-review fail-closed behavior, profiling omission rules, A100 classification, and atomic writes.
- Write `.superpowers/sdd/2026-08-20-docprune-paper-fidelity-correction/task-3-report.md` with RED/GREEN evidence and concerns.
