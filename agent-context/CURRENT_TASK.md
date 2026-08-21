# Current Task

## Scope

Implement a faithful, training-free reproduction of DocPrune for document QA,
grounded in the CVPR 2026 paper and supplement.

## State

The corrected benchmark runtime is sealed at
`a8d8ca6e32178a2468d729670d7219e3177a9c8b`, with the pinned upstream
M3DocRAG commit `29e6ac2294d6b87075a1d45b8a8df175b214248a`. The gate now
requires real CUDA/FlashAttention-2 execution, complete unpadded ColPali
equivalence and raster maps, exact upstream retrieval order, no QA-time
ColPali/QTP re-encoding, positive 1/2/4 traces, and positive stage timings.
The index surface requires schema 5; evaluation completion independently
validates every cell and then publishes the signed six-cell comparison.

Attempt 2 is diagnostic-only: evaluation array `61830411` was canceled before
work, and six schema-4 indexes `61830405`–`61830410` completed `0:0` but
cannot be promoted. Preserve all scratch artifacts. The next active root is
the fresh `/scratch/lmalveau/docprune/benchmark-a8d8ca6/attempt-1`, using the
immutable runtime checkout `/home/lmalveau/DocPrune-runtime-a8d8ca6` and the
reviewed control checkout/handoff below.

## Binding design

[`../docs/superpowers/specs/2026-08-20-docprune-paper-fidelity-correction-design.md`](../docs/superpowers/specs/2026-08-20-docprune-paper-fidelity-correction-design.md)

## Next action

After independent review, read and execute the active benchmark handoff in
[`../sol/handoffs/DOCPRUNE_M3DOCVQA_BENCHMARK_HANDOFF.md`](../sol/handoffs/DOCPRUNE_M3DOCVQA_BENCHMARK_HANDOFF.md)
from a compute allocation using the fresh `benchmark-a8d8ca6/attempt-1` root.
Submit gate → six indexes → six-cell evaluation → post-array comparator only
after every dependency and manifest check passes. Do not modify or delete
diagnostic attempt-2 artifacts.
