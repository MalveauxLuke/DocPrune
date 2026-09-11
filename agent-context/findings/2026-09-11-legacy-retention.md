# Legacy retention decision — 2026-09-11

The owner chose to retain all 69 files from the preparation inventory and move older
material out of active browsing paths. [Resolved inventory](../../legacy/RETENTION.md).

Older task/segmentation/Qwen2 code now lives under `src/docprune/_legacy/`; the
explicit package path preserves its public import names. Older task tests live in
`tests/legacy/` and remain collected. Baseline helpers, environment/config and the
complete curated evidence packet live under `legacy/`. Plan and package metadata
remain in their required active locations. Navigation and live filesystem accesses
follow the new paths. Prior archive/source snapshots retain historical content.

No experimental algorithm, dependency version, model or data selection changed.
No remote jobs, commits or pushes were requested. Verification is recorded in
the [legacy index](../../legacy/README.md) after the matched checks finish.
