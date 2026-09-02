# CoRAL H200 policy

- CoRAL GPUs: physical IDs 4–7. ARC GPUs: 0–3.
- Check current GPU processes and memory before selecting a device.
- Short use of an idle GPU is acceptable. If use will exceed 15 minutes, post
  in the shared channel before starting.
- Borrowing the other lab's GPU requires being reachable and able to vacate
  within 15 minutes.
- Anyone running jobs must be in the channel or have an active relay.
- `/shared` is shared between labs. `/mnt/data1` and `/mnt/data2` are CoRAL
  storage on this host; confirm mount names during the survey.
- Do not place artifacts on cross-lab drives.
- Never write to the root filesystem. Pin Hugging Face, pip/uv, Torch, temp,
  checkpoints, environments, and experiment outputs to project storage.
- If storage was accidentally placed on the wrong drive, stop and contact a
  lab administrator rather than silently moving or deleting another user's
  data.

Administrators named in the supplied policy: ARC `zhaonan2`; CoRAL `achakr40`
and `tanvekar`.
