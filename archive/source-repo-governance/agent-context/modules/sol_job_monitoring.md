# SOL Job Monitoring

Use only when the user explicitly asks to monitor a SOL/Slurm job.

Cadence:
1. Check once immediately.
2. If still active or pending, sleep 2.5 minutes, then check once.
3. If still active or pending, sleep 5 minutes, then check once.
4. If still active or pending, sleep 15 minutes, then check once.
5. If still active or pending, sleep 30 minutes, then check once and stop.

For replacement runs after recent fast failures, add early checks before step 2:
- sleep 1 minute, then check once;
- if still active or pending, sleep 2 minutes, then check once.

Reset the cadence to step 1 whenever a bug surfaces, a job fails, code/config changes are made, or a replacement job is submitted.

Do not poll continuously. Prefer `sacct` for completed/failed jobs and `squeue` for pending/running jobs. Report the exact job id, state, elapsed time, exit code when available, and output/log path if known.
