# Additional SciEGQA Selection Instructions

Build Option 4 from the same pinned SciEGQA source: select every eligible,
deduplicated query whose gold page is in either the original 3,711-page pool or
the approved 2,919-page expansion pool. The exact target is 10,431 queries on
6,630 pages.

- Follow the original 4K eligibility, deduplication, deterministic selection,
  provenance, and audit rules.
- Use only pages from the original and approved expansion manifests; do not add
  another page.
- Do not apply a per-page query cap or domain quota.
- Preserve the exact decomposition: 5,822 queries on original pages and 4,609
  queries on expansion pages.
- Prepare the 2,919 expansion-page PNGs in an isolated Option 4 input tree and
  enrich their page manifest with validated image metadata and hashes.
- Do **not** run OCR, extract semantic sections, label evidence, create splits,
  or train/evaluate models.

Use the dataset-creation module supplied by the user and stop on any eligibility,
count, provenance, determinism, or fixed-page-universe failure.
