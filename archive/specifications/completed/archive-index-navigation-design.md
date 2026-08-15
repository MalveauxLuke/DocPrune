# Archive Index Navigation Design

## Goal

Make historical repository material discoverable from `agent-context/INDEX.md`
without encouraging agents to load archived context during ordinary work.

## Design

Add a compact `Archive navigation` section to `agent-context/INDEX.md` that:

- links to `archive/README.md` as the human-readable archive policy and map;
- identifies `archive/code/`, `archive/tests/`, `archive/apps/`, and
  `archive/reports/` by purpose; and
- points cluster-job searches to `sol/archive/jobs/`.

The index will describe categories rather than enumerate files. Detailed
archive structure remains authoritative in `archive/README.md`, preventing the
short agent index from becoming another large inventory that can drift.

## Verification

- Confirm every named path exists.
- Run the repository-structure tests.
- Check the Markdown edit for broken relative links.
