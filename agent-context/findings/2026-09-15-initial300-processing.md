# Initial 300 processing on SOL — 2026-09-15

Status: implementation prepared and local mechanics checks passed; transfer and
GPU execution not yet complete. No model has been run on this cohort yet.

Owner asked for ColQwen and MinerU on all 300 frozen questions, efficient
batching/sharding, any compatible GPU, minimal resources, browser-shell
operation, rsync transfer, and no login-node processing.

Binding plan: [SOL handoff](../../sol/initial300/HANDOFF.md).

## Verified preparation

- Manifest SHA256 `f1942130e3b585ede4474515089dda2e3df67bb78a53f8f3acf9cd8b5322a495`.
- Selected transfer: 3,209 files, 546.3 MiB; 3,743 logical pages in the 300
  document families. Preserve all pages and original question identities.
- Browser shell allocation 63321654 is on sc001 (lightwork, 1 CPU, 8 GiB,
  two hours). This was explicitly requested and verified, not inferred from
  the browser tab title, which still says login02.
- Existing ColQwen17 and MinerU 3.0.9 environments/model snapshots are available.
  pypdfium2, Pillow and NumPy are available in the MinerU environment.
- ColQwen batches four similar-shaped pages, stores projected vectors and native
  grids once per unique image, and reuses the exact 506-tensor adapter check.
- MinerU layout-only API is present in the installed pinned library. Full
  transcription is unnecessary for the requested region boxes; raw layouts
  and an explicit layout-only provenance label are retained.
- No new environment or model downloads are planned. GPU smoke requests one
  generic compatible GPU, 2 CPUs, 24,000 MiB RAM, 20 minutes. Production array
  counts/time remain provisional until measured.
- Three targeted CPU unit tests pass (partitioning, immutable writes and
  identity stability); Python compile and shell syntax checks pass. These
  checks do not establish GPU/runtime or segmentation quality.

## Transfer access

Direct Mac SSH passed Duo but still required password authentication. The
verified existing H200 key works from the SOL compute allocation. Proposed
relay uses temporary data2 space (340 GB free observed) rather than data1
(19 GB free). Automatic approval review blocked the relay copy pending explicit
permission for that shared intermediate destination. No relay data copied at
this checkpoint; user approval requested. No credentials transferred.

## Remaining

Transfer and destination SHA verification; CPU catalog; sequential GPU smoke;
visual layout and measured memory/throughput checks; production shard sizing;
arrays; CPU scoped rankings/region profiles; final completeness receipt.
Teacher/answerer calls and training remain outside this preprocessing task.
