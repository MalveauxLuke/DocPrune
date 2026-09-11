# Python 3.10 TOML Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make DocPrune's declared and pinned Python 3.10 runtime load TOML configuration without changing the scientific dependency stack.

**Architecture:** Keep the public configuration API unchanged and select the standard-library parser or its official backport at import time. Declare the conditional project dependency and an exact SOL environment pin so the runtime does not depend on pytest installing Tomli transitively.

**Tech Stack:** Python 3.10, `tomllib`, Tomli 2.4.1, pytest 8.4.2, Ruff 0.12.12, setuptools project metadata, mamba environment YAML.

## Global Constraints

- Preserve Python 3.10, PyTorch 2.4.1, CUDA 12.1, Transformers 4.46.3, and FlashAttention 2.5.8.
- Preserve the existing detached runtime worktree and all original SOL evidence unchanged.
- Do not run a GPU smoke, processor probe, dataset operation, index construction, or benchmark.
- Use the isolated `fix/python310-tomli` worktree for every source change.

---

### Task 1: Reproduce Python 3.10 configuration import failure

**Files:**
- Create: `tests/test_python310_compatibility.py`
- Test: `tests/test_python310_compatibility.py`

**Interfaces:**
- Consumes: `docprune.config.load_config(path: pathlib.Path) -> DocPruneConfig`
- Produces: a regression test that exercises the minimum supported interpreter in a subprocess without importing DocPrune during pytest collection

- [ ] **Step 1: Write the failing regression test**

```python
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(
    sys.version_info[:2] != (3, 10),
    reason="regression targets the minimum supported Python 3.10 runtime",
)
def test_python310_imports_and_parses_repository_config() -> None:
    project_root = Path(__file__).resolve().parents[1]
    script = """
from pathlib import Path
from docprune.config import load_config

config = load_config(Path("legacy/configs/docprune-m3docvqa.toml"))
assert tuple(sorted(config.page_settings)) == (1, 2, 4)
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
```

- [ ] **Step 2: Run the focused test in a lightwork allocation and verify RED**

Run:

```bash
srun -p lightwork -q public -t 00:05:00 -c 1 --mem=2G /bin/bash -lc '
  set -euo pipefail
  export PATH=/home/lmalveau/mamba-envs/docprune-sol/bin:$PATH
  export PYTHONNOUSERSITE=1
  cd /home/lmalveau/DocPrune-fix-python310-tomli
  export PYTHONPATH=$PWD/src
  python -m pytest -q tests/test_python310_compatibility.py
'
```

Expected: FAIL because the subprocess reports `ModuleNotFoundError: No module named 'tomllib'`.

- [ ] **Step 3: Commit the failing regression test**

```bash
git add tests/test_python310_compatibility.py
git commit -m "test: cover Python 3.10 TOML compatibility"
```

### Task 2: Add the parser compatibility layer and direct dependencies

**Files:**
- Modify: `src/docprune/config.py:11`
- Modify: `pyproject.toml:14-18`
- Modify: `legacy/environments/docprune-sol.yml:17-41`
- Test: `tests/test_python310_compatibility.py`

**Interfaces:**
- Consumes: Tomli's `load()` interface, which matches `tomllib.load()` for the existing binary file handle
- Produces: the unchanged local name `tomllib`, backed by the standard library on Python 3.11+ and Tomli on Python 3.10

- [ ] **Step 1: Implement the version-gated import**

Add `import sys` with the standard-library imports, then replace the unconditional parser import with:

```python
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib
```

- [ ] **Step 2: Declare the conditional project dependency**

Add to `[project].dependencies` in `pyproject.toml`:

```toml
"tomli>=1.1.0; python_version < '3.11'",
```

- [ ] **Step 3: Pin the SOL runtime dependency**

Add to the pip subsection of `legacy/environments/docprune-sol.yml`:

```yaml
- tomli==2.4.1
```

- [ ] **Step 4: Run the focused test and verify GREEN**

Run the same lightwork command from Task 1, Step 2.

Expected: `1 passed`.

- [ ] **Step 5: Run focused regression and formatting checks in lightwork**

Run:

```bash
srun -p lightwork -q public -t 00:05:00 -c 1 --mem=2G /bin/bash -lc '
  set -euo pipefail
  export PATH=/home/lmalveau/mamba-envs/docprune-sol/bin:$PATH
  export PYTHONNOUSERSITE=1
  cd /home/lmalveau/DocPrune-fix-python310-tomli
  export PYTHONPATH=$PWD/src
  python -m pytest -q tests/test_config.py tests/test_python310_compatibility.py
  ruff check src/docprune/config.py tests/test_python310_compatibility.py
'
```

Expected: all selected tests pass and Ruff reports no errors.

- [ ] **Step 6: Commit the compatibility implementation**

```bash
git add src/docprune/config.py pyproject.toml legacy/environments/docprune-sol.yml
git commit -m "fix: support TOML parsing on Python 3.10"
```

### Task 3: Verify the complete non-GPU revision

**Files:**
- Verify: `src/docprune/config.py`
- Verify: `pyproject.toml`
- Verify: `legacy/environments/docprune-sol.yml`
- Verify: `tests/test_python310_compatibility.py`

**Interfaces:**
- Consumes: the complete repository test and lint entry points
- Produces: evidence that the compatibility change fixes collection without altering unrelated behavior

- [ ] **Step 1: Run the full pytest suite in a lightwork allocation**

```bash
srun -p lightwork -q public -t 00:15:00 -c 4 --mem=16G /bin/bash -lc '
  set -euo pipefail
  export PATH=/home/lmalveau/mamba-envs/docprune-sol/bin:$PATH
  export PYTHONNOUSERSITE=1
  cd /home/lmalveau/DocPrune-fix-python310-tomli
  export PYTHONPATH=$PWD/src
  python -m pytest -q
  ruff check .
  python - <<"PY"
from pathlib import Path

from docprune.config import load_config, tomllib

config = load_config(Path("legacy/configs/docprune-m3docvqa.toml"))
print(type(tomllib).__name__, tomllib.__name__)
print(tuple(sorted(config.page_settings)))
PY
'
```

Expected: all tests pass, Ruff reports no errors, and Python 3.10 reports parser
module `tomli` with page keys `(1, 2, 4)`.

- [ ] **Step 2: Review the complete verification output**

Confirm the single lightwork command completed with exit code zero and contains
all three expected results before checking Git state.

- [ ] **Step 3: Verify scope and cleanliness**

```bash
git diff --check HEAD~2..HEAD
git status --short
git log -3 --oneline
git -C /home/lmalveau/DocPrune-runtime-99dbece status --short
git -C /home/lmalveau/DocPrune-runtime-99dbece rev-parse HEAD
```

Expected: no whitespace errors; the isolated branch is clean; the original runtime is clean at `99dbece9f7cd09abdfe35c1ba6b61020218e6f1e`.

- [ ] **Step 4: Stop before SOL smoke or handoff revision**

Report the branch, commits, test results, and the fact that a separately approved revised handoff is required before another GPU smoke.
