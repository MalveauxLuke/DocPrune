# Current SOL Task

## State

The original environment was constructed, but structural smoke job `61567743`
stopped on the `99dbece` runtime's Python 3.10 `tomllib` incompatibility. The
correction is committed and non-GPU verified at `64ea70c`.

Only the bounded smoke recovery is active under
[`handoffs/DOCPRUNE_SOL_SMOKE_RECOVERY_HANDOFF.md`](handoffs/DOCPRUNE_SOL_SMOKE_RECOVERY_HANDOFF.md).
The complete dev acquisition and processor probe are staged in
[`handoffs/M3DOCVQA_DEV_ACQUISITION_PROBE_HANDOFF.md`](handoffs/M3DOCVQA_DEV_ACQUISITION_PROBE_HANDOFF.md)
but are not active.

## Next action

Execute the smoke-recovery handoff and stop with its report. Do not acquire the
dataset or run the processor probe unless a reviewed passing smoke is recorded
and this file is committed again to activate the staged handoff. The benchmark
SBATCH file remains prepared but unauthorized.
