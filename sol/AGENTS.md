# SOL Agent Guidance

This repository concerns token compression for document QA.

Read `CURRENT_SOL_TASK.md` before any cluster action. There is no executable
task unless that file names an approved handoff with exact checkout, commit,
environment, inputs, scratch root, resources, command, outputs, and recovery
authority.

- Use login nodes only for light inspection, editing, and submission.
- Use a compute allocation for installation, downloads, hashing, extraction,
  preprocessing, inference, and training.
- Keep code and small provenance in Git; keep large runtime material on
  `/scratch/$USER`.
- Treat `archive/source-repo/` as historical reference, never as current
  submission authority.
