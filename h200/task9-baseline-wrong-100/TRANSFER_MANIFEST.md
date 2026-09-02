# Transfer manifest and boundary

The source computer prepares one finalized, content-addressed bundle before
transfer. The H200 survey confirms only its destination paths and whether the
pinned Qwen snapshot must be copied; it does not reopen the bundle design.

## Required Git content

- Prepared clean branch/commit containing the paths in `SPARSE_CHECKOUT_PATHS.txt`.
- Root and scoped `AGENTS.md` files.
- Canonical experiment plan, implementation plan, and log.
- Task 9 source/examples/config/environment files named in `HANDOFF.md`.
- DocPrune paper markdown and ContextCite provenance already recorded in the
  canonical experiment documents.

## Required sealed/non-Git inputs

- Exact cohort bytes:
  - local path:
    `/home/lmalveau/task9-h200-artifacts/task9-baseline-wrong100-confirmation-v1/cohort.json`
  - file SHA-256:
    `afead001a93666628126260deccfd1493ba5b1f2071e646da6d220a8969ab9fb`
  - internal SHA-256:
    `0bdfdde29b568f54ea541453abeca196cfb49f4b6630f084f80e6c21f158514e`
- The exact 100 source result rows selected by that cohort.
- Exact ordered top-4 cached page identities, their PDF/page bytes, and their
  persisted feature files plus a rewritten/authenticated H200-local manifest.
- Fixed run config and index metadata needed to load those persisted features;
  do not transfer or load the global searchable retrieval index.
- The sealed fixed-page fixture, 100 per-QID input directories, selected source
  rows, and preprocessing manifest built on the source computer.
- All completed authenticated MinerU outputs/completion manifests, geometry
  captures, and exactly 100 final mapping files built and validated on the
  source computer.
- Qwen/Qwen2-VL-7B-Instruct model/processor files at the pinned revision only
  if the H200 survey shows that exact snapshot is absent.

Do not transfer MinerU executables, configuration, or model weights for H200
execution. H200 must not regenerate fixed inputs, MinerU output, geometry, or
mappings.

## Required packaging

Build one content-addressed subset bundle containing the 400 cached pages,
persisted features, fixed inputs, selected source rows, completed MinerU and
geometry artifacts, final mappings, fixed runtime metadata, and model files
absent from the H200 cache. Exclude the global retrieval index and unrelated
experiment scratch trees.

Record source/destination paths, byte counts, and SHA-256 manifests. Preserve
the sealed cohort bytes. Any H200-local path relocation must be represented by
an authenticated relocation manifest; it must not reconstruct scientific
inputs or change their identities.
