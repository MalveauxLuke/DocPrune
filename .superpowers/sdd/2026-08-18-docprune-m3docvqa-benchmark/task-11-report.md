# Task 11 report: fix absolute-ancestor mode collision in index manifests

## RED test and observed failure

Added `test_manifest_accepts_all_kept_root_below_unrelated_docprune_ancestor`,
which constructs an `all-kept` artifact root below an unrelated directory named
`docprune`:

```text
PYTHONPATH=src /home/lmalveau/mamba-envs/docprune-sol/bin/python -m pytest -q tests/test_artifacts.py::test_manifest_accepts_all_kept_root_below_unrelated_docprune_ancestor
F                                                                        [100%]
ValueError: manifest mode/path mismatch for all-kept
1 failed
```

The symmetric `docprune`-below-`all-kept` layout also failed against the
unchanged implementation. The failure occurred while constructing
`IndexManifest`, before file validation.

## Root cause

`_require_mode_artifact_paths()` searched every component of the resolved
absolute `artifact_root` and every derived artifact path. Consequently, a
global scratch ancestor named `docprune` was treated as a cross-mode artifact
when validating an `all-kept` root whose leaf was correctly named `all-kept`.

## Production change

The invariant now requires the resolved `artifact_root` leaf to equal the
declared mode. Each resolved derived path must remain under that root, and
opposite-mode checks inspect only path components relative to the root. This
keeps unrelated ancestors out of mode validation while retaining construction-
time traversal rejection and validation-time symlink escape rejection.

Focused tests cover both unrelated-ancestor layouts, wrong-mode root leaves,
nested cross-mode artifact paths, traversal outside the root, and validation-
time symlink escapes.

## Verification

Focused artifact/index/CLI tests:

```text
PYTHONPATH=src /home/lmalveau/mamba-envs/docprune-sol/bin/python -m pytest -q tests/test_artifacts.py tests/test_indexing.py tests/test_cli.py
75 passed in 3.61s
```

Complete CPU test suite:

```text
PYTHONPATH=src /home/lmalveau/mamba-envs/docprune-sol/bin/python -m pytest -q
318 passed, 1 skipped in 10.44s
```

The one skipped test is the opt-in cached real-model probe:
`tests/qwen2vl/test_model.py:149` (`DOCPRUNE_REAL_MODEL_TEST=1`).

Ruff:

```text
PYTHONPATH=src /home/lmalveau/mamba-envs/docprune-sol/bin/ruff check .
All checks passed!
```

`git diff --check` also passed with no output.

## Files changed

- `src/docprune/artifacts.py`
- `tests/test_artifacts.py`
- `.superpowers/sdd/2026-08-18-docprune-m3docvqa-benchmark/task-11-report.md`

## Commit

Implementation commit SHA: `6c19bfc`.

## Concerns

No known concerns. Runtime/control-plane pins were not modified. The opt-in
real-model test remains skipped as expected.
