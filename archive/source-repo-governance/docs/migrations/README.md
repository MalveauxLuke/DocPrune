# Central-Command Migration Debt

These manifests record legacy tracked artifacts that violate the new
central-command policy. They are migration debt, not approved examples for new
work.

- `tracked_artifact_debt.tsv` records every tracked prohibited binary or file
  larger than 5 MiB, excluding the self-authored archived MinerU fixture.
- `legacy_tracked_roots.tsv` freezes file count, byte total, and a path/size
  digest for `pilot_data/`, `viewer/`, and `sol_results/`.
- New debt is forbidden.
- Removing debt requires a project-specific migration plan, verified
  destination, checksums, recovery instructions, and manifest updates in the
  same commit.
- A changed legacy artifact is new debt unless its migration plan explicitly
  explains and verifies the change.

These manifests do not authorize deleting local or tracked material.
