# Task 9 H200 storage ledger

Task 9 keeps inputs and generated artifacts inside
`/mnt/data1/eunwooim/DocPrune/task9-h200-local-data/`. Environments, caches,
downloaded model bytes, and temporary files live under `/mnt/data2/eunwooim`.
Nothing belonging to this task may be written to `/`, `/shared`, `/micron`, or
the other lab's storage.

Run `record_storage_usage.sh` after every setup, transfer, preprocessing,
smoke, production, retry, and aggregation stage. It appends a UTC timestamped
size snapshot to the untracked local file
`task9-h200-local-data/manifests/storage-usage.tsv`. The report deliberately
records missing paths as well as populated paths so the lifecycle is auditable.

## Locations and cleanup meaning

| Label | Location | Contents | Cleanup class |
|---|---|---|---|
| `artifact_root` | `/mnt/data1/eunwooim/DocPrune/task9-h200-local-data` | All Task 9 transferred inputs, relocated inputs, tools, manifests, preprocessing output, run output, and aggregation output | Preserve until results are archived; project-owned |
| `docprune_env` | `/mnt/data2/eunwooim/.conda/envs/docprune-h200` | Frozen DocPrune runtime | Reproducible from tracked environment file |
| `mineru_env` | `/mnt/data2/eunwooim/.conda/envs/mineru-h200` | Isolated pinned MinerU runtime | Reproducible from tracked environment file and pinned source |
| `mineru_model_materialized` | `/mnt/data2/eunwooim/mineru-models/task9-mineru2.5-pro-2604-d3f5e08` | Real-file hard-linked view of the pinned Hugging Face snapshot required by no-symlink authentication | Reproducible; hard links share underlying bytes with the cache |
| `conda_packages` | `/mnt/data2/eunwooim/.conda/pkgs` | Conda download/extraction cache, potentially shared with this user's other environments | Cache; inspect ownership/use before cleanup |
| `hf_cache` | `/mnt/data2/eunwooim/hf-cache` | Hugging Face model cache, including the pinned MinerU snapshot | Re-downloadable; may be shared by other projects |
| `torch_cache` | `/mnt/data2/eunwooim/torch-cache` | Torch cache | Re-downloadable; may be shared |
| `pip_cache` | `/mnt/data2/eunwooim/pip-cache` | Python package cache | Re-downloadable; may be shared |
| `xdg_cache` | `/mnt/data2/eunwooim/xdg-cache` | Tool caches | Re-downloadable; may be shared |
| `task_tmp` | `/mnt/data2/eunwooim/tmp/docprune-task9` | Task-specific temporary files | Disposable only when no Task 9 process is running |

The aggregate paths overlap their children. Do not add the reported byte
values together. This ledger identifies cleanup candidates but does not
authorize deletion; verify the exact target and request explicit approval
before removing material data.
