# Transfer manifest and boundary

The source computer prepared one content-addressed CPU-input bundle. H200 may
prepare its isolated environments while it transfers, but may not preprocess
or use a GPU until every transferred byte verifies.

## Required Git content

- Prepared clean branch/commit containing the paths in `SPARSE_CHECKOUT_PATHS.txt`.
- Root and scoped `AGENTS.md` files.
- Canonical experiment plan, implementation plan, and log.
- Task 9 source/examples/config/environment files named in `HANDOFF.md`.
- DocPrune paper markdown and ContextCite provenance already recorded in the
  canonical experiment documents.
- Exact small runtime templates under `runtime-metadata/`: source run config
  SHA-256 `2233621303ccdf531267bb5bd2a7670775e54f90d04dede4fb19f264e8cad502`,
  source index manifest SHA-256
  `ffa5979b3bf157adefcc132b0438af295ddafb2243377db4cdb8fcb8eaafe5da`, and
  processor contract SHA-256
  `0ca949740960a9835e3a76c17e01c6f5d9b36b85eb908cdec2a4002304fb315b`.

## Required sealed/non-Git inputs

- Exact cohort bytes:
  - local path:
    `/home/lmalveau/docprune-data/h200-artifacts/task9-baseline-wrong100-confirmation-v1/cohort.json`
  - file SHA-256:
    `afead001a93666628126260deccfd1493ba5b1f2071e646da6d220a8969ab9fb`
  - internal SHA-256:
    `0bdfdde29b568f54ea541453abeca196cfb49f4b6630f084f80e6c21f158514e`
- The exact 100 source result rows selected by that cohort.
- Exact ordered top-4 cached page identities, their PDF/page bytes, and their
  persisted feature files.
- Fixed run config and index metadata needed to load those persisted features;
  do not transfer or load the global searchable retrieval index.
- The sealed fixed-page fixture, 100 per-QID input directories, selected source
  rows, and preprocessing manifest built on the source computer. Their SOL
  absolute paths are relocated on H200 by the tracked authenticated relocation
  utility; source bytes remain unchanged.
- Qwen/Qwen2-VL-7B-Instruct model/processor files at the pinned revision only
  if the H200 survey shows that exact snapshot is absent.

The bundle intentionally does not contain MinerU output, geometry captures,
final mappings, the global retrieval index, or model snapshots. H200 creates
the first three from the sealed transferred inputs using pinned tools and
downloads model snapshots into `/mnt/data2/eunwooim`.

## Required packaging

The sealed source bundle is:

```text
/scratch/lmalveau/docprune/task9-baseline-wrong100-transfer-c0c9bee-v1
files in raw/: 1,014
raw byte count: 2,826,385,204
MANIFEST.sha256 SHA-256: d5eb76ff0e0382d487387e31d9a28cde37ae1de8020644a111ec659d2f1f5bb5
```

It contains 400 cached PNG inputs, 253 unique required PDFs, 253 unique feature
shards, fixed inputs, selected source rows, the cohort, and provenance
metadata. It excludes the global retrieval index and unrelated scratch trees.

Transfer through the Mac bridge into
`/mnt/data1/eunwooim/DocPrune/task9-h200-local-data/inputs/transferred/` (an
already completed upload elsewhere under `/mnt/data1/eunwooim` may be moved
there atomically). Reject `.safetensors.*` partial files. Verify the manifest
and every raw file before running `relocate_task9_h200_inputs.py`. Preserve the
sealed cohort bytes; relocation changes only authenticated absolute paths and
must publish its relocation manifest.
