# CoRAL H200 policy

- CoRAL GPUs: physical IDs 4–7. ARC GPUs: 0–3.
- Check current GPU processes and memory before selecting a device.
- Normal use of assigned CoRAL GPUs does not require a per-job announcement.
- Before borrowing an idle GPU outside the CoRAL allocation, check first. If
  the borrowing will exceed 15 minutes, post in the shared channel.
- Borrowing the other lab's GPU requires being reachable and ready to vacate
  within 15 minutes of a request.
- Anyone running jobs must be in the channel or have an active relay.
- `/shared` is shared between labs. `/mnt/data1` and `/mnt/data2` are CoRAL
  storage on this host; confirm mount names during the survey.
- Do not place artifacts on cross-lab drives.
- Never write to the root filesystem. Keep Task 9 inputs and outputs in the
  ignored project-local `DocPrune/task9-h200-local-data/` tree. Pin Hugging
  Face, pip/uv, Torch, temporary files, and environments to CoRAL project
  storage under `/mnt/data2/eunwooim`.
- If storage was accidentally placed on the wrong drive, stop and contact a
  lab administrator rather than silently moving or deleting another user's
  data.

Administrators named in the supplied policy: ARC `zhaonan2`; CoRAL `achakr40`
and `tanvekar`.
