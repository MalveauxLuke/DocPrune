# SciEGQA Option 4 New-Page Dataset State

This handoff records dataset state only. It intentionally contains no OCR
procedure, submission commands, runtime parameters, or retry instructions.

## Option 4 Dataset

- Dataset: `Yuwh07/SciEGQA-Train`
- Pinned revision: `4ffb867c88e3264161920b4b2446d5ac6352269e`
- Source JSONL:
  `/scratch/lmalveau/sciegqa_train_4k/raw/SciEGQA-Train.jsonl`
- Source JSONL SHA-256:
  `7eb895fb913ffb607e3b41b01ec66e651685ad9404d230972fd987ba7f005e84`
- Final Option 4 selection root:
  `/scratch/lmalveau/sciegqa_train_4k/evidence_option4_10431/selection`
- Final size: 10,431 queries, 6,630 unique gold pages, and 2,824
  documents.
- Query decomposition: 5,822 queries on the original page pool and 4,609
  queries on the new page pool.
- Every query remains strict single-page/single-region eligible and
  deduplicated.
- No per-page query cap or domain quota was applied to Option 4.

## The 2,919 New Pages

The page IDs in the following file are the complete and exclusive new-page
universe for this handoff:

`/scratch/lmalveau/sciegqa_train_4k/evidence_expansion_3700/selection/pages.jsonl`

- Page-manifest SHA-256:
  `baa1c81df778ce1b2ef613d47312d61866c86bfab155d835871d62fa2b505b7c`
- Unique pages: 2,919
- Documents represented: 1,666
- Queries on these pages in the final Option 4 query manifest: 4,609
- Page-ID overlap with the original 3,711-page pool: 0
- Each page row records its stable page ID, document ID, domain, document
  name, source page number, and expected archive member.

New pages by domain:

| Domain | Pages | Option 4 queries |
|---|---:|---:|
| cs | 432 | 660 |
| econ | 376 | 493 |
| eess | 429 | 777 |
| physics | 380 | 642 |
| q-bio | 515 | 774 |
| q-fin | 524 | 897 |
| stat | 263 | 366 |
| math | 0 | 0 |

Queries per new page:

| Queries on page | Page count |
|---:|---:|
| 1 | 1,803 |
| 2 | 710 |
| 3 | 285 |
| 4 | 84 |
| 5 | 29 |
| 6 | 7 |
| 8 | 1 |

The authoritative query records are in the final Option 4 query manifest, not
the earlier 3,700-query expansion manifest. A query belongs to this new-page
group when its `page_id` appears in the 2,919-page manifest above.

## Original Page Pool Boundary

- Original page manifest:
  `/scratch/lmalveau/sciegqa_train_4k/selection/pages.jsonl`
- Original page-manifest SHA-256:
  `9c5fb94dc3f9b78000b094e5a21b3c9844143038e286fcd5be6688a99d8db5ba`
- Original unique pages: 3,711
- Original-page queries in Option 4: 5,822
- Existing original-page evidence run:
  `/scratch/lmalveau/sciegqa_train_4k/evidence_poc/20260703T090303Z`

The original 3,711 page IDs are outside the new-page handoff scope.

## Authoritative Option 4 Manifests

- Queries:
  `/scratch/lmalveau/sciegqa_train_4k/evidence_option4_10431/selection/queries.jsonl`
  - SHA-256:
    `f072fde385ac91acb38631b29ad396a476df40fb952093244184ed8a0b2ec217`
- Pages:
  `/scratch/lmalveau/sciegqa_train_4k/evidence_option4_10431/selection/pages.jsonl`
  - SHA-256:
    `d47bdc0efb31b2dee18bf5ad86091d773d7129eb3d3c5aac46f4561e67b96724`
- Selection audit:
  `/scratch/lmalveau/sciegqa_train_4k/evidence_option4_10431/selection/selection_audit.json`
  - SHA-256:
    `29d60950f1cd62c44bb52242cefb2197e50d5c2777570cd9414c73ab44ad7e8c`

An independent rebuild produced byte-identical query, page, document, and
audit manifests.

## Source Image Archive State

- Archive:
  `/scratch/lmalveau/sciegqa_train_4k/raw/images.tar`
- Recorded bytes: 77,113,231,360
- Recorded SHA-256:
  `14c0fe4ab007d86a6ade60093c03cd6f0b5e25d5affba3f5b8e410cf9178f8c9`
- The 2,919-page selection manifest records the corresponding expected
  archive member for every new page.

## Prepared New-Page State

- Prepared root:
  `/scratch/lmalveau/sciegqa_train_4k/evidence_option4_10431/ocr_input`
- Extracted page root:
  `/scratch/lmalveau/sciegqa_train_4k/evidence_option4_10431/ocr_input/pages`
- Enriched selection root:
  `/scratch/lmalveau/sciegqa_train_4k/evidence_option4_10431/ocr_input/selection`
- Extraction audit:
  `/scratch/lmalveau/sciegqa_train_4k/evidence_option4_10431/ocr_input/extraction_audit.json`
- Extraction job: `58428426`, completed with exit code `0:0` in `00:17:52`.
- Extracted and validated PNGs: 2,919 of 2,919 requested.
- Queries validated against actual image dimensions: 4,609.
- Documents: 1,666.
- Archive members scanned: 137,834.
- Extraction-audit SHA-256:
  `911d1f4d6a5938f485f963ec8b948d4d247b42c14db1c4f47e7bff9d0ab9dcb1`
- Prepared query-manifest SHA-256:
  `9f23c80adfbd11189d122a2bf47e5de7a05ff434497ad7a7362f410fa488ac0f`
- Prepared page-manifest SHA-256:
  `0665073403996b631d51310c6169f1a89777952fd4691611b1e1d84a12745eb9`
- Prepared document-manifest SHA-256:
  `22f111d5ee08a00d1d776b344d362ff8737254be2d1fbe5e1c3543af64fc1aeb`

Every prepared page row includes its final image path, width, height, byte
size, and image SHA-256. Manifest joins and recorded manifest hashes pass.

## Current Boundary

Selection and new-page preparation are complete and verified. No OCR work has
been run or submitted.

The repository selection changes are currently uncommitted. The selection
audit records base repository revision
`26abfdeb7dce10f5f455e72b02b3bb905d31a7dd`.
