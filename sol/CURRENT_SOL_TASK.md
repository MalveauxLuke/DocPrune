# Current SOL Task

## State

The active runtime is `4e2473bdbbc2e4eca0e92c30d4a0633044501ccf`, with M3DocRAG pinned to
`29e6ac2294d6b87075a1d45b8a8df175b214248a`. The user approved durable short HTC
evaluation shards after a monolithic 256-question task timed out with 92 rows. The active handoff
authorizes the first reusable paired top-4 checkpoint only: shards 0–3 in each mode, followed by a
CPU merge/report job. No historical job may be canceled or modified.

## Historical execution records (immutable, non-promotable)

Historical diagnostic attempt-2 at `/scratch/lmalveau/docprune/benchmark-6c19bfc/attempt-2/`: evaluation `61830411` was canceled before work;
indexes `61830405`–`61830410` completed `0:0` under schema 4 and cannot be
promoted. Preserve those scratch artifacts unchanged. Scheduling-only attempt-1
(gate `61883512`, indexes `61883881`–`61883886`, eval `61883887`, compare
`61883888`) was canceled before work at `00:00:00`; preserve it unchanged. The
failed `benchmark-02385b3/attempt-2` gate `61943239` ran 29s on `scg011` and
failed before GPU/model work because the runtime validator hard-coded
`attempt-1`; downstream jobs `61943240`–`61943247` were auto-canceled. Preserve
that root and every ID unchanged. The old-runtime production graph
`61968793`, `61968794`–`61968797`, `61968799`–`61968800`, `61968821`, and
`61968823` was canceled at `2026-08-21 17:50:21` with no nodes/elapsed 0;
probes `61969352` and `61969614` were canceled with no node/elapsed 0. L40
attempts `61970394`, `61972695`, `61973090`, and `61974092`, plus the A100-40GB
hedge `61974173`, are immutable failed/canceled history and are not promotable.
The active root is the fresh `benchmark-4e2473b/attempt-1` with runtime checkout
`/home/lmalveau/DocPrune-runtime-4e2473b`.

## Exact environment and authority

```text
environment: /home/lmalveau/mamba-envs/docprune-sol
PDF tools: /home/lmalveau/mamba-envs/m3docvqa-acquisition
corpus: /scratch/lmalveau/docprune/datasets/m3docvqa
HF cache: /scratch/lmalveau/hf_cache; Hub cache: /scratch/lmalveau/hf_cache/hub
Qwen: Qwen/Qwen2-VL-7B-Instruct@eed13092ef92e448dd6875b2a00151bd3f7db0ac
ColPali: vidore/colpali-v1.2@961b51745de3e9adb3468ac5c9ccca0ac626c217
ColPali backbone: vidore/colpaligemma-3b-pt-448-base@30ab955d073de4a91dc5a288e8c97226647e3e5a
upstream: /home/lmalveau/src/m3docrag-benchmark-29e6ac2 @ 29e6ac2294d6b87075a1d45b8a8df175b214248a
runtime: /home/lmalveau/DocPrune-runtime-4e2473b @ 4e2473bdbbc2e4eca0e92c30d4a0633044501ccf
control: sealed full SHA in /scratch/lmalveau/docprune/benchmark-4e2473b/attempt-1/control.json
fresh attempt root: /scratch/lmalveau/docprune/benchmark-4e2473b/attempt-1
```

## Next action

Execute [`handoffs/DOCPRUNE_M3DOCVQA_SHARDED_BENCHMARK_HANDOFF.md`](handoffs/DOCPRUNE_M3DOCVQA_SHARDED_BENCHMARK_HANDOFF.md)
using `/scratch/lmalveau/docprune/benchmark-4e2473b/attempt-1`. Submit the two top-4 arrays
`0-3` and their `afterok` checkpoint publisher. Inspect its paired report before the remainder.
