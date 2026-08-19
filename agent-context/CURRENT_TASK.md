# Current Task

## Scope

Implement a faithful, training-free reproduction of DocPrune for document QA,
grounded in the CVPR 2026 paper and supplement.

## State

The benchmark implementation is complete through the hardened Task-6 runtime
commit `d5cefb33f7ca97ce0ef2104fa5e63bd3ad8a5761`. Task 7 supplies the active
SOL gate, six page-specific index launchers, six evaluation cells, and the
commit-pinned benchmark handoff. No benchmark job has been submitted and no
benchmark result is claimed.

## Binding design

[`../docs/superpowers/specs/2026-08-15-docprune-reproduction-design.md`](../docs/superpowers/specs/2026-08-15-docprune-reproduction-design.md)

## Next action

Read and execute the active benchmark handoff in
[`../sol/handoffs/DOCPRUNE_M3DOCVQA_BENCHMARK_HANDOFF.md`](../sol/handoffs/DOCPRUNE_M3DOCVQA_BENCHMARK_HANDOFF.md)
from a compute allocation. Submit the gate first, then the six indexes, then
the six-cell array only after every dependency and manifest check passes.
