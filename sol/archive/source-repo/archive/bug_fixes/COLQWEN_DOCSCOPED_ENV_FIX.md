# ColQwen Doc-Scoped Environment Fix

## Error
Job `57086115` (`colqwen-doc-top5`) failed after 10 seconds with:

```text
ModuleNotFoundError: No module named 'torch'
```

The traceback occurred at the first `import torch` in
`scripts/run_colqwen_topk.py`, before any retrieval work started. Slurm reported
state `FAILED`, exit code `1:0`.

## Root Cause
The batch job did not launch the Python interpreter from the intended ColQwen
environment. The live task names the expected environment as:

```text
/home/lmalveau/mamba-envs/colqwen25
```

That interpreter can import Torch (`2.8.0+cu128`). The failed job reached a
Python environment where Torch was unavailable, so the failure was an
environment activation/resolution problem rather than a model or document-scope
retrieval problem.

## Fix
Updated `sol/run_colqwen_top5.sbatch` to:

- default `ENV_NAME` to the absolute environment path
  `/home/lmalveau/mamba-envs/colqwen25`
- derive `ENV_PREFIX` from `CONDA_PREFIX` after activation
- print `ENV_NAME`, `ENV_PREFIX`, and `PYTHON_BIN` at job start
- run a fast preflight import check:

```bash
"$PYTHON_BIN" -c 'import sys; print(sys.executable); import torch; print(torch.__version__)'
```

This makes future environment failures fail fast with the resolved interpreter
visible in the Slurm log.
