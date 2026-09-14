# H200 operations and CoRAL rules

Read this file and [CORAL_POLICY.md](CORAL_POLICY.md) before any H200 work,
including connecting, inspecting GPUs, transferring files, preparing an
environment, changing code, or running an experiment.

This is the repository-wide operations entry point, consolidated on 2026-09-13.
The supplied CoRAL policy was moved here verbatim from the archived Task 9
folder. The connection and transfer guide was moved here from the DocPrune
root. Other operating rules below were collected from that guide and the
H200 instructions listed under historical sources.

## Authority and scope

- Apply the current user request and root `AGENTS.md`, then these operating
  rules and the supplied CoRAL policy.
- Read a task's scoped `AGENTS.md` and handoff when that task is selected.
  Historical task documents do not authorize a new experiment, transfer,
  environment installation, or GPU allocation. Preserve their restrictions
  when working on their original task.
- Verify current machine identity, mounts, available storage and GPU state
  before relying on recorded paths or allocations. A saved GPU snapshot is
  not a reservation. Resolve a conflict with the supplied lab policy before
  starting affected work.
- SOL scheduler and login-node instructions apply to SOL. Do not copy SOL
  allocation assumptions onto the H200 server.

## GPU use

- CoRAL owns physical GPUs **4–7**; ARC owns **0–3**. Inspect current memory,
  utilization and compute processes immediately before choosing a device and
  launching. Set `CUDA_VISIBLE_DEVICES` explicitly and record the physical ID;
  the process-local CUDA index may differ.
- Normal use of an assigned CoRAL GPU needs **no per-job announcement**.
  Anyone running jobs must be in the shared channel or have an active relay.
- Check before borrowing outside CoRAL. If borrowing will exceed 15 minutes,
  post in the shared channel. Remain reachable and ready to vacate a borrowed
  GPU within 15 minutes of a request. A task may prohibit borrowing entirely.
- Preserve existing workloads. Follow the active task's allocation and launch
  authority; an idle GPU does not extend that authority. Exact-ID approval,
  single-GPU limits and a historical assignment such as GPU 5 belong to the
  handoffs that imposed them, not to a new default allocation.

## Storage and environments

- Verify that `/mnt/data1` and `/mnt/data2` are the CoRAL mounts. `/shared` is
  shared between labs; `/micron` is ARC storage. Do not place project artifacts
  on the other lab's drives. The old 100-question task additionally prohibited
  writing its artifacts to `/shared`; shared access is not blanket permission
  to create or change shared files.
- Use the existing `/mnt/data1/eunwooim/DocPrune` checkout. Keep task inputs
  and outputs in a designated ignored project-local directory. Keep new
  environments, models, caches and temporary files under `/mnt/data2/eunwooim`.
  Do not write these artifacts to the root filesystem.
- Pin Hugging Face, pip/uv, Torch, XDG, Conda/package and temporary directories
  to the intended CoRAL storage before setup or model work. Check free space
  first. Reuse authenticated existing assets and validated environments.
- Keep incompatible Qwen and MinerU stacks separate. Inspect actual installed
  versions and the active model/processor contract; an old environment template
  is not proof of the installed runtime or the correct stack for a new task.
- Inventory storage and distinguish project artifacts from shared or
  reproducible caches before proposing cleanup. Overlapping directory totals
  must not be added together. A storage report does not authorize deletion.
- If material is on the wrong lab's drive, stop and contact an administrator;
  do not silently move or delete another user's data. Recorded policy contacts
  are in `CORAL_POLICY.md`.

## Shared-account and checkout safety

- Multiple people use the remote Unix account. Account ownership alone does
  not establish that a terminal, job or session belongs to this user. Before
  stopping anything, correlate connection origin, start time, TTY, working
  directory, command line and known project activity, within authorized scope.
- Preserve other projects, cohorts, running jobs and unrelated local changes.
  Reuse the existing checkout. Before an authorized update, check whether an
  active job is reading the code; update by fast-forward and extend existing
  sparse-checkout patterns without hiding another track's files.
- Use the separate personal Codex profile at `$HOME/.codex-personal` on the
  shared account. Keep the shared/default profile untouched. The local runbook
  contains the recorded session and connection procedures.

## Connections and transfers

Read [H200_SOL_MAC_RUNBOOK.md](H200_SOL_MAC_RUNBOOK.md) for Mac SSH/VPN,
SOL `soldtn`, transfer examples, host identity checks and personal-session
procedures. **That file is local and Git-ignored** because it includes
site-specific details. If it is absent in another checkout, obtain the private
runbook from the owner; do not invent credentials or connection settings.

- Connect the Mac to the ASU network/VPN before using the documented H200 route.
  Independently verify any changed SSH host identity before continuing.
- Label commands by their execution machine. Use SOL's data-transfer node for
  large or resumable SOL-to-H200 transfers. SOL login nodes remain for light
  work; SOL compute or package builds require the appropriate allocation.
- Transfer only the requested files. Use the documented resumable transfer
  commands; do not add deletion flags or disturb unrelated destination data.
  A same-host copy on H200 uses local paths rather than SSH back to itself.
- Preserve sealed input bytes. Verify supplied manifests/checksums at the
  destination before extraction or use, reject incomplete or mismatched
  transfers, and run the applicable task's input validator. A recipe or a
  successful connection is not proof of a completed transfer.
- Record relevant source revision, artifact identities, environment versions,
  output location and actual execution measurements in the task's own log.
  Resume only matching completed units; retry missing or failed work without
  repeating valid results.

## Historical sources and task limits

These documents were read during consolidation. They remain with their
experiment assets so their scripts, recipes and historical contracts retain
context. They are reference material unless the user activates that task.

| Task | Instructions and supporting records | Scope that remains historical |
| --- | --- | --- |
| Correction-depth | [AGENTS](../h200/correction-depth/AGENTS.md), [handoff](../h200/correction-depth/HANDOFF.md), [execution prompt](../h200/correction-depth/H200_PROMPT.md) | 40 candidates, input/dynamic deletion, 256 masks per boundary, its model setup and cohort isolation |
| Baseline-wrong confirmation | [AGENTS](../archive/experiments/task6_9_2026_09_10/h200/task9-baseline-wrong-100/AGENTS.md), [handoff](../archive/experiments/task6_9_2026_09_10/h200/task9-baseline-wrong-100/HANDOFF.md) | 100 sealed questions, staged survey/setup/smoke/production admissions, exact approved GPU/count and storage snapshots after every stage |
| Shared causal-deletion probe | [AGENTS](../archive/experiments/task6_9_2026_09_10/h200/task9-shared-probe/AGENTS.md), [handoff](../archive/experiments/task6_9_2026_09_10/h200/task9-shared-probe/HANDOFF.md) | 600-question delivery, two-phase SOL handoff, B13 teacher boundary, 32-mask schedule and scoped GPU approvals |

Supporting confirmation records: [environment survey template](../archive/experiments/task6_9_2026_09_10/h200/task9-baseline-wrong-100/ENVIRONMENT_SURVEY.md),
[storage ledger](../archive/experiments/task6_9_2026_09_10/h200/task9-baseline-wrong-100/STORAGE_LEDGER.md),
[transfer manifest](../archive/experiments/task6_9_2026_09_10/h200/task9-baseline-wrong-100/TRANSFER_MANIFEST.md),
and [source preparation record](../archive/experiments/task6_9_2026_09_10/h200/task9-baseline-wrong-100/SOURCE_PREPARATION_TODO.md).
Their recorded status and dependency versions are historical, not a current
inventory. The equivalent Markdown files in the older local research checkout
were compared and were identical at consolidation.

Current research navigation is in [agent-context/INDEX.md](../agent-context/INDEX.md).
