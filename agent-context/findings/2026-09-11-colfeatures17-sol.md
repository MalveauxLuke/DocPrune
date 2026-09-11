# Colfeatures17 SOL preparation — 2026-09-11

Owner requested tests, analysis/gathering/recomputation of Col-style features on
the existing 17 cases, complete image/evidence/feature packaging, and temporary
Git-based transfer followed by untracking after receipt. Interpreted 17 pages as
17 cases / 68 saved pages; recorded explicitly in the handoff.

Verified: 829 existing packet files, 962,374,181 bytes; lossless archive
243,517,726 bytes in six <=40 MiB chunks. All 68 RGB pixel hashes and document/page
identities match recorded mapping provenance. Package round-trip verifies exact
file set, size and SHA256. No new remote experiment has run locally.

Chosen ColQwen2.5 checkpoint and base are pinned independently; source inspection
found query-prefix drift across library versions, so the extractor pins an older
compatible stack and sets the historical prefix explicitly. Retriever alternatives
and source links are in [task specification](../../docs/experiments/corrective-selection/COLFEATURES17.md).

New tests pass for transport and geometric overlap. Legacy regression outcome is
639 passed, 22 skipped, 34 failed; exact failure identities match before/after
retention relocation. Runtime GPU compatibility is deliberately an unverified
SOL smoke gate, not a completed measurement. The extractor has no answerer or
retrieval-index path.

Transfer keeps the original packet, dense projected embeddings, final
pre-projection states, merged vision features, processor tensors, spatial
correspondence, similarities, provenance, available referenced PDFs/OCR, and
missing-artifact inventory. This does not validate corrective learning or the
other stages in the unchanged canonical plan.

## Remote integration before publication

Remote main advanced to c903b28 with already-published H200 correction-depth work.
Merged it while preserving all runtime fixes. Six relocated legacy modules and
all 20 inherited H200 helper/recipe/correction-code/test files match the remote
blobs exactly. Kept Colfeatures17 as current authority and archived the older
H200 task snapshot. Combined task and correction CPU tests: 23 passed. One
initial macOS temporary-directory path-alias assertion passed with canonical
`TMPDIR=/private/tmp`; no production-code change was needed. The 639/22/34
characterization above predates this integration.
