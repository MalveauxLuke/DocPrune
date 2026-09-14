# SOL Agent Guidance

This repository concerns token compression for document QA.

Read `CURRENT_SOL_TASK.md` before any cluster action. There is no executable
task unless that file names an approved handoff with exact checkout, commit,
environment, inputs, scratch root, resources, command, outputs, and recovery
authority.

- Use login nodes only for light inspection, editing, and submission.
- Use a compute allocation for installation, downloads, hashing, extraction,
  preprocessing, inference, and training.
- Request the minimum necessary GPUs, GPU memory, CPU cores, host RAM, and
  wall time. Prefer any compatible GPU, not a specific model. Verify scheduler
  minima; use measured peaks/failures to justify increases. Do not maximize
  parallel GPU reservations merely to reduce elapsed time unless the owner
  explicitly prioritizes that tradeoff.
- Keep code and small provenance in Git; keep large runtime material on
  `/scratch/$USER`.
- For replays or controlled diagnostics with sealed retrieval artifacts, use
  cached ordered page identities and persisted page features. Do not execute
  global retrieval or load/rebuild its index unless the user explicitly asks
  for fresh retrieval. Confirm the submitted launcher selects the fixed-page
  path and enforces exact page identity; quarantine any accidental
  fresh-retrieval output as non-canonical and never score or merge it.
- Treat `archive/source-repo/` as historical reference, never as current
  submission authority.

- For the owner-approved Colfeatures17 task only, the two transfer directories
  named in root `AGENTS.md` may contain packaged evidence and feature tensors.
  Runtime files/cache still stay on scratch. Publish a complete hashed transfer;
  do not untrack until the receiving local agent verifies receipt.
