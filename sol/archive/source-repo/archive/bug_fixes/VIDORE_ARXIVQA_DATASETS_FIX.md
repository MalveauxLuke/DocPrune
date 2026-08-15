# ViDoRe ArxivQA Datasets Dependency Fix

## Error
Job `57128552` (`vidore-arxivqa-colqwen`) failed after 28 seconds with:

```text
ModuleNotFoundError: No module named 'datasets'
```

The traceback occurred in the submit-script preflight before any benchmark
inference started.

## Root Cause
The submit script sets:

```bash
export PYTHONNOUSERSITE=1
```

That is intentional because it prevents user-site packages from leaking into
the benchmark environment. The `datasets` package was available only from the
user site:

```text
/home/lmalveau/.local/lib/python3.12/site-packages
```

but it was missing from the intended ColQwen environment:

```text
/home/lmalveau/mamba-envs/colqwen25
```

With user-site packages disabled, `scripts/run_vidore_arxivqa_colqwen.py` could
not import `from datasets import load_dataset`.

## Fix
Updated `sol/colqwen_environment.yml` to include the Hugging Face `datasets`
package in the environment's pip dependencies. The runtime environment also
needs `datasets` installed inside `/home/lmalveau/mamba-envs/colqwen25` before
resubmitting `sol/run_vidore_arxivqa_colqwen.sbatch`.

Installed `datasets` into the runtime environment with user-site packages
disabled so the Slurm preflight can import it from the environment itself.

## Verification
With the same environment isolation and library path used by the Slurm script:

```bash
ENV_PREFIX=/home/lmalveau/mamba-envs/colqwen25
LD_LIBRARY_PATH="$ENV_PREFIX/lib:$ENV_PREFIX/lib64" \
PYTHONNOUSERSITE=1 "$ENV_PREFIX/bin/python" -m unittest tests/test_vidore_arxivqa_metrics.py
```

Result:

```text
Ran 2 tests in 0.000s
OK
```

The Slurm-style preflight also imports `torch`, `datasets`, and
`ColQwen2_5` successfully:

```text
torch 2.11.0+cu130
datasets 5.0.0
colqwen imports ok
```
