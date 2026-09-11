# DocPrune Task 6 portability smoke handoff — 2026-08-27

## Authority and scope

Status: **FAILED CONSISTENTLY — jobs `62277597` (A30), `62277598`
(A100-40GB), `62277599` (H100), and `62277600` (L40S) all exited `1:0`
before any answer; no retry authorized by this handoff.** This handoff permits exactly four
independent one-QID Task 6 smoke jobs: one A30, one A100-40GB, one H100, and
one L40S. Each job runs the same sealed nine policy cells over QID
`e1e6ed53f9ad11813845088f4cf2f6b1`. These are portability and numerical-drift
checks only. Their quality and timing must not be pooled.

This handoff does not authorize the 64-QID developmental matrices, retrieval,
index loading/rebuilding, feature building, holdout sealing, or any fresh page
selection. The launcher uses the cached ordered top-4 pages and selected
per-document feature shards only.

## Immutable pins

| Item | Exact value |
|---|---|
| Runtime | `/home/lmalveau/DocPrune-task6-runtime-20260827` |
| Runtime commit | `b0c8742d358319b7b617b9b1d36ba1f5d1ea6c86` |
| Smoke launcher | `/home/lmalveau/DocPrune-task6-runtime-20260827/examples/sbatch/34_docprune_task6_smoke.sbatch` |
| Launcher SHA-256 | `6897cf065ccf206d157316e921fbf83cac3f8111d8efc6d85ac4c3207831db01` |
| Matrix runner SHA-256 | `54a6199a3d702032cf396a1bf85d257843ba8c3102d10ba280cc87c5459a53ae` |
| Four-way analyzer SHA-256 | `840e48d9c45205061770c24ed75c66ffff36e9cf17c03c477742b3ee5c6e03fc` |
| Environment | `/home/lmalveau/mamba-envs/docprune-sol` |
| M3DocRAG checkout | `/home/lmalveau/src/m3docrag-benchmark-29e6ac2` |
| M3DocRAG commit | `29e6ac2294d6b87075a1d45b8a8df175b214248a` |
| Fixed-page fixture | `/scratch/lmalveau/docprune/task6-fixed-page-gate-v1/fixture-stage64-top4-v2.json` |
| Fixture SHA-256 | `32b3ddd6a1f608db509f002f9541dbc92317b59ac769f0b41fcb20bcc536652b` |
| Development gate | `/scratch/lmalveau/docprune/task6-fixed-page-gate-v1/gate-manifest-v4.json` |
| Gate SHA-256 | `4d44e297081b152c2e492ec68be58445c7ac17c8d106ceb2ddb1f6331fd57761` |
| Historical run config | `/scratch/lmalveau/docprune/benchmark-4e2473b/attempt-2/run-configs/docprune-top4.json` |
| Feature manifest | `/scratch/lmalveau/docprune/benchmark-4e2473b/attempt-2/indexes/docprune/top4/docprune/manifest.json` |
| Feature manifest SHA-256 | `ffa5979b3bf157adefcc132b0438af295ddafb2243377db4cdb8fcb8eaafe5da` |
| Parent output root | `/scratch/lmalveau/docprune/task6-smoke-b0c8742-v1` |
| Resources per job | one GPU, 4 CPUs, 96 GiB host memory, 20 minutes |

The runtime is a clean detached checkout at the exact commit above. The
fixture and gate are immutable no-replace artifacts; earlier versions remain
preserved. The runner records historical feature-build provenance separately
from the live runtime commit.

## Exact submissions

Create the absent parent and four absent job roots, then submit exactly these
commands. Record all returned job IDs before inspecting any outcome.

The four commands below were submitted once on 2026-08-27 and returned the
job IDs listed above. Do not submit them again.

```bash
ROOT=/scratch/lmalveau/docprune/task6-smoke-b0c8742-v1; test ! -e "$ROOT" && mkdir -- "$ROOT" && for ROLE in a30 a100-40 h100 l40s; do mkdir -- "$ROOT/$ROLE"; done

sbatch --parsable --chdir=/scratch/lmalveau/docprune/task6-smoke-b0c8742-v1/a30 --constraint=a30 --export=ALL,RUNTIME_DIR=/home/lmalveau/DocPrune-task6-runtime-20260827,RUNTIME_COMMIT=b0c8742d358319b7b617b9b1d36ba1f5d1ea6c86,JOB_ROOT=/scratch/lmalveau/docprune/task6-smoke-b0c8742-v1/a30,GPU_EXPECTED=a30 /home/lmalveau/DocPrune-task6-runtime-20260827/examples/sbatch/34_docprune_task6_smoke.sbatch

sbatch --parsable --chdir=/scratch/lmalveau/docprune/task6-smoke-b0c8742-v1/a100-40 --constraint=a100_40 --export=ALL,RUNTIME_DIR=/home/lmalveau/DocPrune-task6-runtime-20260827,RUNTIME_COMMIT=b0c8742d358319b7b617b9b1d36ba1f5d1ea6c86,JOB_ROOT=/scratch/lmalveau/docprune/task6-smoke-b0c8742-v1/a100-40,GPU_EXPECTED=a100_40 /home/lmalveau/DocPrune-task6-runtime-20260827/examples/sbatch/34_docprune_task6_smoke.sbatch

sbatch --parsable --chdir=/scratch/lmalveau/docprune/task6-smoke-b0c8742-v1/h100 --constraint=h100 --export=ALL,RUNTIME_DIR=/home/lmalveau/DocPrune-task6-runtime-20260827,RUNTIME_COMMIT=b0c8742d358319b7b617b9b1d36ba1f5d1ea6c86,JOB_ROOT=/scratch/lmalveau/docprune/task6-smoke-b0c8742-v1/h100,GPU_EXPECTED=h100 /home/lmalveau/DocPrune-task6-runtime-20260827/examples/sbatch/34_docprune_task6_smoke.sbatch

sbatch --parsable --chdir=/scratch/lmalveau/docprune/task6-smoke-b0c8742-v1/l40s --constraint=l40s --export=ALL,RUNTIME_DIR=/home/lmalveau/DocPrune-task6-runtime-20260827,RUNTIME_COMMIT=b0c8742d358319b7b617b9b1d36ba1f5d1ea6c86,JOB_ROOT=/scratch/lmalveau/docprune/task6-smoke-b0c8742-v1/l40s,GPU_EXPECTED=l40s /home/lmalveau/DocPrune-task6-runtime-20260827/examples/sbatch/34_docprune_task6_smoke.sbatch
```

Do not alter a constraint, role, runtime, commit, input, root, resource, or
launcher. In particular, the A100-40 job must reject A100-80 hardware.

## Admission and recovery

Each job must exit `0`, produce exactly nine ordered result rows, retain the
sealed page identities, and record fixed-page/no-global-index, policy,
geometry, cache-length, and M-RoPE evidence. After all four succeed, run the
committed analyzer once into a fresh file under the parent root. Admission
requires equality to L40S for inputs, policy budgets/selection identity,
geometry, cache topology, M-RoPE identity, and pruning trace. Answer equality
is reported but is not required for numerical-drift admission. No quality or
timing is pooled across GPUs.

If any job fails, preserve every root unchanged, record the job ID, exit state,
and available hashes in the experiment log, and stop before submitting any
development matrix. This handoff grants no retry or replacement root. A retry
requires a successor handoff. If all four pass, this handoff authorizes only
the four-way analysis; the 64-QID matrix still requires the implementation
plan and SOL authority to be advanced after the smoke decision is logged.

All four jobs failed in 24–33 seconds after loading ColPali but before loading
Qwen or writing a result row. The identical traceback ends in
`AuthenticatedFixedPageRetriever.retrieve`: `_ColPaliQueryAdapter` correctly
returns a one-element list of variable-length query tensors, while the fixed
retriever incorrectly passed the list itself to `torch.as_tensor`. Every root
is preserved. This is a runtime interface bug, not a GPU-specific numerical
result; a corrected clean commit and successor handoff are required.
