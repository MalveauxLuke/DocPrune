# DocPrune SOL Recovery Design

## Objective

Resume the bounded DocPrune SOL validation after the original structural smoke
stopped on Python 3.10's missing `tomllib`, without rebuilding the validated
scientific stack or authorizing the benchmark.

## Recorded state

- The original runtime remains preserved at
  `99dbece9f7cd09abdfe35c1ba6b61020218e6f1e` with smoke job `61567743` and
  its failure evidence.
- The Python 3.10 compatibility correction is committed at
  `64ea70c66a8f9e3dbce804d1fde265a4a2b8b09d` and passed 69 tests and Ruff in
  the pinned Python 3.10 environment.
- The existing environment contains PyTorch 2.4.1, CUDA 12.1, Transformers
  4.46.3, Tokenizers 0.20.3, Tomli 2.4.1, and FlashAttention 2.5.8.
- The retained FlashAttention wheel has SHA-256
  `5f5d5a9b4a7a4bd8c2cdd7c58a2a58067d8249c2a14fc68f67f5ab334e1d0394`.
- M3DocVQA acquisition and the processor-contract probe have not run.

## Authority structure

Recovery is split into two handoffs.

1. The active smoke-recovery handoff creates fresh clean runtime worktrees,
   reuses the verified environment and FlashAttention wheel, reinstalls the
   corrected runtime, runs one structural GPU smoke, reports, and stops.
2. The staged acquisition/probe handoff remains inactive until the smoke is
   reported as passing. It acquires and validates the complete M3DocVQA dev
   corpus, renders one deterministic probe page, generates the processor
   contract, reports, and stops.

Index construction, embedding, answer generation, evaluation, training, and
benchmark execution remain unauthorized.

## Smoke recovery

The corrected runtime uses a new detached worktree named for commit `64ea70c`.
The old `99dbece` worktree and scratch evidence are immutable. A new clean
M3DocRAG worktree is created from commit `29e6ac2`; the existing checkout with
generated `egg-info` changes is preserved rather than reset.

The existing environment is updated from the corrected YAML. The saved wheel
and source archive must pass their recorded SHA-256 checks before the exact
wheel is reinstalled. M3DocRAG is installed from a `git archive` so its clean
worktree is not modified by packaging metadata. DocPrune is editable-installed
from the corrected runtime. A new environment freeze is recorded before the
existing GPU smoke wrapper is submitted.

Only scheduler or preemption failures may be retried with identical inputs.
Any repository, hash, environment, import, test, lint, inspection, or
cleanliness failure stops the handoff.

## Acquisition and processor probe

The complete dev split is stored below
`/scratch/$USER/docprune/datasets/m3docvqa`. Metadata preparation and integrity
validation run in compute allocations. PDF acquisition uses a Slurm array and
the pinned builder's `--proc_id` and `--n_proc` interface.

Completion requires 2,441 dev questions, 3,368 expected document IDs, exactly
3,368 corresponding PDFs, and zero missing, extra, or corrupt PDFs. The
observed page count is recorded and compared with the approximate published
count of 41,005. Failed or corrupt PDFs are moved into attempt-specific scratch
quarantine directories before retrying only missing outputs.

After integrity passes, the lexicographically first validated PDF is selected
and only its first page is rendered into a probe-only scratch directory. The
PDF and PNG hashes are recorded. The processor probe uses immutable Qwen and
resolved ColPali revisions and validates the output JSON before returning.

## Outputs and stopping boundary

The smoke handoff returns the runtime and upstream commits, environment freeze,
wheel hash, Slurm job ID, smoke outputs, and any errors. The acquisition/probe
handoff returns the corpus integrity report, attempt logs, selected PDF/image
hashes, resolved model revisions, and processor-contract paths. Each handoff
stops at its own reporting boundary.
