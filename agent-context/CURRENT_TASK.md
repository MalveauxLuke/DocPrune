# Current Task

## Scope

Implement a faithful, training-free reproduction of DocPrune for document QA,
grounded in the CVPR 2026 paper and supplement.

## State

The benchmark implementation is complete through the review-clean Task-11
runtime commit `6c19bfcb4fcb73685af5b16ad493097ed6c609a5`. Task 12 supplies
the active SOL gate, six page-specific index launchers, six evaluation cells,
and the commit-pinned benchmark handoff. The first `6c19bfc` attempt is a
failed overall attempt: gate `61820163` and indexes `61820164`-`61820169`
passed, but evaluation array `61820170` tasks 0-2 failed with the Slurm spool
sibling error, tasks 3-5 were canceled, and no evaluation artifacts or
results exist. It is not resumable or a complete benchmark result. The next
active root is the fresh `benchmark-6c19bfc/attempt-2`; runtime, model,
corpus, and Poppler pins remain unchanged, and the reviewed control commit is
sealed per attempt.

## Binding design

[`../docs/superpowers/specs/2026-08-15-docprune-reproduction-design.md`](../docs/superpowers/specs/2026-08-15-docprune-reproduction-design.md)

## Next action

After independent review, read and execute the active benchmark handoff in
[`../sol/handoffs/DOCPRUNE_M3DOCVQA_BENCHMARK_HANDOFF.md`](../sol/handoffs/DOCPRUNE_M3DOCVQA_BENCHMARK_HANDOFF.md)
from a compute allocation using the fresh `benchmark-6c19bfc/attempt-2`
root. Submit the gate first, then the six indexes, then the six-cell array
only after every dependency and manifest check passes.
