# H200 Task 9 agent instructions

## Authority

Read, in order:

1. [`../../AGENTS.md`](../../AGENTS.md)
2. [`../../agent-context/CURRENT_TASK.md`](../../agent-context/CURRENT_TASK.md)
3. [`../../docs/experiments/task9-contextcite/EXPERIMENT_PLAN.md`](../../docs/experiments/task9-contextcite/EXPERIMENT_PLAN.md)
4. [`../../docs/experiments/task9-contextcite/IMPLEMENTATION_PLAN.md`](../../docs/experiments/task9-contextcite/IMPLEMENTATION_PLAN.md)
5. [`../../docs/experiments/task9-contextcite/EXPERIMENT_LOG.md`](../../docs/experiments/task9-contextcite/EXPERIMENT_LOG.md)
6. [`HANDOFF.md`](HANDOFF.md) and [`CORAL_POLICY.md`](CORAL_POLICY.md)

The CoRAL H200 rules here are runtime ground truth. SOL/HTC files are retained
only for provenance and implementation history. Do not submit an SOL job or
copy SOL scheduler assumptions onto this server.

## First action: observation only

The sparse Git checkout at `/mnt/data1/eunwooim/DocPrune` is the sole permitted
setup action before the survey. After that checkout, run the read-only commands
in [`SURVEY_COMMANDS.sh`](SURVEY_COMMANDS.sh) and record the results outside the
clean execution checkout or in [`ENVIRONMENT_SURVEY.md`](ENVIRONMENT_SURVEY.md).
After the survey is recorded, environment and model setup may proceed while the
sealed input bundle is still transferring. No preprocessing or GPU work may
start until that transfer is complete and checksum-verified.

## Scope

The experiment and launch code are already implemented. Do not redesign the
cohort, mask count, arms, model, retrieval, or statistics. Small path or
environment compatibility fixes are allowed only after recording the survey;
document them before running the smoke.

- Use only the sealed 100 QIDs.
- Use only their exact cached ordered top-4 pages and persisted features.
- Never run retrieval or load the global retrieval index.
- Use 256 fit masks and zero holdout masks.
- Keep model, prompt, decoding, Lasso, mask distribution, dynamic DocPrune
  policy, and whole-region deletion frozen.
- Run one smoke question before production.
- Large/generated artifacts stay outside Git on CoRAL storage.

## Source/H200 work boundary

The source computer sealed the cohort and 100-QID/400-page retrieval-free fixed
inputs and packaged the exact required PDF, page, feature, and provenance bytes.
H200 authenticates and relocates those bytes without retrieval, then owns the
pinned MinerU run, post-BTP/QTP geometry capture, region-mapping construction,
CPU validation, experiment smoke, production, retry, and aggregation. Never
reselect questions/pages, load the global retrieval index, or alter source
bytes. Use separate DocPrune and MinerU environments because their validated
Python/PyTorch/Transformers stacks conflict.

## Storage and GPU safety

Never write caches, temporary files, environments, checkpoints, or outputs to
`/`. Keep experiment artifacts under
`/mnt/data1/eunwooim/DocPrune/task9-h200-local-data/` and environments/caches
under `/mnt/data2/eunwooim` unless the survey proves those paths differ. Do not
use ARC-only or cross-lab storage. Use one explicitly approved CoRAL GPU at a
time by default; additional GPUs require separate explicit approval.
Record a timestamped storage snapshot with `record_storage_usage.sh` after
every setup, transfer, preprocessing, smoke, production, retry, and aggregation
stage. Use `STORAGE_LEDGER.md` to distinguish project-owned artifacts from
shared/re-downloadable caches. The reporter is observational and never
authorizes cleanup or deletion.

CoRAL GPUs are physical IDs 4–7. Check current processes and memory before every
launch because dynamically allocated work may not be obvious. Default to one
explicitly approved physical GPU per run. Never infer multi-GPU permission from
availability; additional GPUs require separate approval for their exact count
and IDs.

Normal use of an assigned CoRAL GPU does not require a per-job announcement.
Anyone running work must be present in the shared channel or have a relay. Do
not use GPUs 0–3 normally. Before borrowing another lab.s idle GPU, check first;
if borrowing will exceed 15 minutes, post in the channel. Remain reachable and
ready to vacate a borrowed GPU within 15 minutes of a request.
