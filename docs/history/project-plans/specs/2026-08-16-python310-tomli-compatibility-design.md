# Python 3.10 TOML Compatibility Design

## Problem

DocPrune declares support for Python 3.10 and the pinned SOL environment uses
Python 3.10, but `src/docprune/config.py` imports the Python 3.11-only standard
library module `tomllib` unconditionally. The structural SOL smoke therefore
fails during pytest collection before Ruff or the inspection check runs.

## Decision

Keep the existing Python, PyTorch, CUDA, Transformers, and FlashAttention
versions. Use Python's `tomllib` on Python 3.11 and newer, and use the official
Tomli backport under the same local name on Python 3.10.

Declare Tomli as a conditional runtime dependency in `pyproject.toml` and pin
Tomli 2.4.1 in the SOL environment. The explicit SOL pin avoids relying on
pytest's transitive dependency while preserving the already-recorded package
version.

## Changes

1. Add a focused regression test that imports DocPrune configuration code and
   parses the shipped TOML configuration under the supported interpreter.
2. Select `tomllib` or `tomli` according to the interpreter version in
   `src/docprune/config.py`.
3. Add `tomli>=1.1.0; python_version < '3.11'` to project dependencies.
4. Add `tomli==2.4.1` to the pinned SOL pip dependencies.

## Verification

- Demonstrate that the focused test fails on the unmodified Python 3.10 code.
- Run the focused test after the compatibility change.
- Run the complete pytest suite and Ruff in the pinned SOL environment.
- Confirm Python 3.10 imports and parses `legacy/configs/docprune-m3docvqa.toml`.
- Confirm the worktree contains only the intended source, dependency, test, and
  design changes.

## Provenance and boundaries

The existing detached runtime worktree at commit
`99dbece9f7cd09abdfe35c1ba6b61020218e6f1e`, its environment freeze, wheel,
and failed smoke logs remain unchanged. This change does not authorize a GPU
smoke retry, processor probe, dataset acquisition, index construction, or
benchmark execution. Any renewed SOL execution must use a revised handoff with
an immutable replacement runtime commit.
