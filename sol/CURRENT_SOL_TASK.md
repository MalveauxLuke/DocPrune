# Current SOL Task

## State

The active benchmark runtime is
`6c19bfcb4fcb73685af5b16ad493097ed6c609a5`, with M3DocRAG pinned to
`29e6ac2294d6b87075a1d45b8a8df175b214248a`. The Task-12 handoff below is the
only authority for the processor gate, six page-specific indexes, and six
evaluation cells. `benchmark-6c19bfc/attempt-1` is a failed overall attempt:
gate `61820163` and indexes `61820164`-`61820169` passed, but evaluation array
`61820170` tasks 0-2 failed with the Slurm spool sibling error, tasks 3-5
were canceled, and no evaluation artifacts or results exist. It is not
resumable or a complete benchmark result. The next active root is the fresh
`benchmark-6c19bfc/attempt-2`; runtime, model, corpus, and Poppler pins remain
unchanged, and the reviewed control commit is sealed per attempt.

## Next action

After independent review, execute
[`handoffs/DOCPRUNE_M3DOCVQA_BENCHMARK_HANDOFF.md`](handoffs/DOCPRUNE_M3DOCVQA_BENCHMARK_HANDOFF.md)
from a compute allocation using the fresh `benchmark-6c19bfc/attempt-2`
root. Submit gate → six indexes → six-cell evaluation in that order,
preserving every failed attempt.
