# Task 3 report: independent six-cell comparison validation and reporting

## Changes

Commits `f4b3dbae52d9ab4c6a81492543147630a813a7dc`, `90d63f465d9112e612f6ab98d44e55e1526afc4d`, and `300897cdef0b14c24731aa0a2f655364f6c42f15` add:

- `src/docprune/comparison.py`: fail-closed six-cell normalization, independent single-run validation, pinned-corpus document/page validation using actual PDF page counts, exact paired identity/QID checks, canonical signed JSON, deterministic Markdown, and atomic no-overwrite publication.
- `src/docprune/cli.py`: `compare`/`compare-runs` boundary with explicit all-kept/DocPrune top-1/top-2/top-4 paths, corpus root, and JSON/Markdown outputs.
- `tests/test_comparison.py`: RED/GREEN coverage for incomplete/duplicate/extra matrices, source-order drift, shared identity mismatch, fabricated documents, out-of-range pages, altered summaries, six absolute cells, three deltas, and profiler omission.
- `tests/test_cli.py`: CLI report creation and no-overwrite coverage.
- `docs/superpowers/plans/2026-08-20-six-cell-comparison.md`: implementation plan.

The follow-up commit reuses actual PDF page counts across all six runs during independent validation.
The final API commit also accepts the pinned corpus as a positional argument for the public validator/writer boundary.

The existing `validate_benchmark_run` remains the independent per-run gate; comparison code does not treat signed run claims as proof that retrieved IDs belong to the pinned corpus. The report preserves validated hardware classification and explicitly labels A100 values as reconstruction measurements, not RTX A6000 parity. TFLOPs are omitted unless every cell has explicit valid profiler definition, positive FLOPs, and positive measured total time.

## RED evidence

Command:

```text
env PYTHONPATH=/home/lmalveau/DocPrune-benchmark/src /home/lmalveau/mamba-envs/docprune-sol/bin/python -m pytest tests/test_comparison.py tests/test_cli.py tests/test_evaluation.py -v
```

Before production edits, the command collected 76 tests; the seven new comparison tests failed with `ModuleNotFoundError: No module named 'docprune.comparison'`, while the existing 69 tests passed. This was the intended missing-production-module failure.

## GREEN evidence

Focused command:

```text
env PYTHONPATH=/home/lmalveau/DocPrune-benchmark/src /home/lmalveau/mamba-envs/docprune-sol/bin/python -m pytest tests/test_comparison.py tests/test_cli.py tests/test_evaluation.py -q
```

Result: `78 passed in 4.45s`.

Full CPU suite:

```text
env PYTHONPATH=/home/lmalveau/DocPrune-benchmark/src /home/lmalveau/mamba-envs/docprune-sol/bin/python -m pytest -q
```

Result: `360 passed, 1 skipped in 8.66s`; the skip is the pre-existing opt-in real-model probe (`DOCPRUNE_REAL_MODEL_TEST=1`).

## Static checks

- `env PYTHONPATH=/home/lmalveau/DocPrune-benchmark/src /home/lmalveau/mamba-envs/docprune-sol/bin/python -m ruff check src tests` — `All checks passed!`.
- `git diff --check HEAD^ HEAD` — passed with no whitespace errors.

## Self-review

- Matrix normalization is fail-closed for missing, duplicate, unsupported, or extra cells and derives/compares each cell's manifest mode/page count.
- Every run passes the existing independent validator; every result page is separately checked against the pinned `dev_doc_ids.json`, PDF regular-file identity, and actual page count (`pdfinfo` first, byte-level page-object fallback for minimal fixtures).
- Pair comparison includes corpus, runtime/upstream, model resources, processor identity, generation, and the full measurement identity while intentionally excluding mode/page/index artifact identities.
- Results retain source-order QIDs, quality modality/hop/retrieval metrics, arithmetic-mean stage drop plus retention, timing, encoder/decoder throughput, GPU peak memory, classification, and paired DocPrune-minus-all-kept deltas.
- JSON digest is computed from canonical sorted/separator-minimized unsigned JSON. JSON and Markdown are staged through temporary files and refuse existing destinations.

## Concerns

- No fresh GPU runs or Slurm jobs were started, as required; production validation remains pending the six new sealed evaluation directories.
- The full suite's one real-model test remains skipped unless explicitly enabled in the configured model environment.
- PDF page-count fallback is intentionally conservative for fixture/minimal PDFs; production pinned PDFs should resolve through the pinned `pdfinfo` toolchain.

## Fix Round 1

### RED

Added literal tests for mixed/partial/nonpositive profiler payloads, strict shared identity fields, exact digest/Markdown/value assertions, and transactional publication races. The required focused command:

```text
env PYTHONPATH=/home/lmalveau/DocPrune-benchmark/src /home/lmalveau/mamba-envs/docprune-sol/bin/python -m pytest tests/test_comparison.py tests/test_cli.py tests/test_evaluation.py -v
```

collected 90 tests and produced the intended four failures: three new profiler-invalid regressions and the injected second-publication failure due to the missing `_publish_noreplace` boundary. The other 86 tests passed.

### GREEN

After strict profiler identity validation and no-replace publication/rollback implementation:

```text
env PYTHONPATH=/home/lmalveau/DocPrune-benchmark/src /home/lmalveau/mamba-envs/docprune-sol/bin/python -m pytest tests/test_comparison.py tests/test_cli.py tests/test_evaluation.py -q
```

Result: `91 passed in 4.20s`.

The exact success regressions now recalculate the canonical SHA-256 from unsigned sorted JSON, assert stable Markdown bytes (with only the path-dependent digest placeholder normalized), verify representative quality/retrieval/token/timing/throughput/memory values and paired DocPrune-minus-all-kept deltas, and assert six identical profiler definitions yield six exact `0.5` TFLOPs values. Mixed definitions, partial profiler state, and nonpositive FLOPs fail closed without a payload claim.

Publication stages and fsyncs both complete files, obtains exclusive lock sentinels, publishes with hard-link no-replace semantics, fsyncs the parent directory, and rolls back only destination inodes created by the transaction. Preexisting or concurrently appearing destination bytes remain untouched.

### Fix Round 1 verification and self-review

- `env PYTHONPATH=/home/lmalveau/DocPrune-benchmark/src /home/lmalveau/mamba-envs/docprune-sol/bin/python -m ruff check src tests` — `All checks passed!`.
- `git diff --check` — passed.
- Full CPU suite: `env PYTHONPATH=/home/lmalveau/DocPrune-benchmark/src /home/lmalveau/mamba-envs/docprune-sol/bin/python -m pytest -q` — `373 passed, 1 skipped in 11.88s`; the skip is the pre-existing opt-in real-model probe.
- Final Fix Round 1 commit SHA is recorded below after staging the implementation, tests, and this report.
- No Slurm/GPU work was started. The original A100 reconstruction classification and independent run/corpus/page/matrix gates remain unchanged.

## Fix Round 2

### RED

Added literal overflow/underflow profiler cases (`1e308` and `1e-300` FLOPs with finite positive timing), plus a publication helper regression that links/unlinks the second destination and then raises. The comparison-only RED command:

```text
env PYTHONPATH=/home/lmalveau/DocPrune-benchmark/src /home/lmalveau/mamba-envs/docprune-sol/bin/python -m pytest tests/test_comparison.py -v
```

collected 23 tests and showed the intended three failures: both derived-TFLOPs edge cases and the link-then-raise rollback test; 20 tests passed.

### GREEN

Derived TFLOPs are now computed per cell and rejected unless finite and strictly positive before report construction. Publication records every staged temporary's `(st_dev, st_ino)` before any publish; exception cleanup inspects every destination against those staged identities, so a helper that links/unlinks and raises cannot escape rollback while preexisting/concurrent inode identities remain untouched.

Focused required command:

```text
env PYTHONPATH=/home/lmalveau/DocPrune-benchmark/src /home/lmalveau/mamba-envs/docprune-sol/bin/python -m pytest tests/test_comparison.py tests/test_cli.py tests/test_evaluation.py -q
```

Result: `93 passed in 4.34s`.

Ruff/diff checks:

```text
env PYTHONPATH=/home/lmalveau/DocPrune-benchmark/src /home/lmalveau/mamba-envs/docprune-sol/bin/python -m ruff check src tests
git diff --check
```

Both passed (`All checks passed!`, no diff-check output).

Full CPU command:

```text
env PYTHONPATH=/home/lmalveau/DocPrune-benchmark/src /home/lmalveau/mamba-envs/docprune-sol/bin/python -m pytest -q
```

Result: `375 passed, 1 skipped in 11.31s`; the skip is the pre-existing opt-in cached real-model probe.

Final Fix Round 2 commit is recorded after the final code/report staging.

### Fix Round 2 self-review

- Overflow and underflow are rejected before JSON serialization, so no `inf`, `nan`, or zero TFLOPs claim can be emitted.
- Rollback removes only paths whose current inode equals a staged transaction inode; destination races and preexisting bytes remain protected.
- Existing strict six-cell, independent run, corpus/page, identity, classification, and CLI no-overwrite gates remain covered by the focused suite.
