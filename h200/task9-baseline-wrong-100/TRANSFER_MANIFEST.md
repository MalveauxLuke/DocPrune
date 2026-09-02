# Transfer manifest and boundary

The survey happens first. Transfer design is finalized only afterward, but the
required logical inputs are already fixed.

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
- Qwen/Qwen2-VL-7B-Instruct model/processor files at the pinned revision.
- MinerU tool/config/model artifacts required by the existing region mapping
  pipeline, unless all 100 authenticated mappings are built before transfer.

## Preferred packaging choices after survey

1. Best: build a content-addressed subset bundle containing only the 400 cached
   pages, their persisted features, selected source rows, mapping prerequisites,
   and model files absent from the H200 cache.
2. Also acceptable: build all 100 region mappings on the source machine and
   transfer the authenticated mapping artifacts plus only runtime page/features.
3. Avoid copying the full global retrieval index or full experiment scratch
   tree. They are unnecessary and create both storage and provenance risk.

Whichever route is chosen, record source/destination paths, byte counts, and
SHA-256 manifests. Do not rewrite the sealed cohort itself; H200-local paths
belong in the fixed-page fixture and transfer manifest.
