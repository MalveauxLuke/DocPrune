# Colfeatures17 local receipt — 2026-09-11

The complete SOL package was published in consecutive commits 165a944, 9720e3d,
and 089fff32fa2fae2be85f349885d3e366d4be380b (manifest last), then fetched locally.
The unpublished SOL browser startup guide was rebased onto the received history.

Verified local destination:
`/Users/god/DocPrune/outputs/colfeatures17-received-2026-09-11`.
The standard-library unpacker verified all 19 chunks, the archive SHA256, and
all 1,164 restored file sizes/hashes. Archive: 761,073,274 bytes;
SHA256 `7f1dc8f339b643123b858d58cdca5c6fb08759d59c14f8012c521aaabea8308e`.
The CPU feature verifier independently passed all 17 cases / 68 pages, including
completion inventories, finite tensors, similarity reconstruction and region
missingness/dimensions. Original input matches all 829 manifest members exactly.
All 44 gathered files were rehashed against recorded and available expected hashes.
The archived adapter verification records 506 exact tensor matches on SOL; no
model or GPU was loaded during local receipt.

Reviewed `ANALYSIS.md` and the missing-artifact ledger. Three historical PDF
hash mismatches were excluded; unavailable references remain documented. Saved
input images are preserved. Features are ColQwen2.5 retriever representations,
not answerer vision features; the cohort is answer-conditioned and excludes
18 originally-correct cases. This completes the bounded extraction/transfer task,
not all Stage 0 experiments.

After successful receipt, both temporary input/result transfer directories were
removed from Git tracking and exactly ignored. Their local archive bytes and
unpacked outputs remain present. Historical Git objects are unchanged. The
machine-readable local receipt is `outputs/colfeatures17-local-receipt-2026-09-11.json`.
See [execution findings](2026-09-11-colfeatures17-execution.md) for resource,
checkpoint repair, test limitations and SOL provenance.
