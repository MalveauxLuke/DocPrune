# Current Task

## Scope

Implement a faithful, training-free reproduction of DocPrune for document QA,
grounded in the CVPR 2026 paper and supplement.

## State

The evaluation measurement fix and durable HTC shard pipeline are sealed at runtime
`4e2473bdbbc2e4eca0e92c30d4a0633044501ccf`. The validated schema-5 indexes are promoted without
altering their bytes into fresh attempt `/scratch/lmalveau/docprune/benchmark-4e2473b/attempt-2`.
The approved first action is the reusable paired top-4 256-question checkpoint: four 64-question
shards per mode on A100 80GB. These are final top-4 inputs, not disposable pilot work.

## Historical execution records (immutable, non-promotable)

Historical diagnostic attempt-2 at `/scratch/lmalveau/docprune/benchmark-6c19bfc/attempt-2/`: evaluation array `61830411` was canceled before
work, and six schema-4 indexes `61830405`–`61830410` completed `0:0` but
cannot be promoted. Preserve all scratch artifacts. Scheduling-only attempt-1
(gate `61883512`, indexes `61883881`–`61883886`, eval `61883887`, compare
`61883888`) was canceled before work at `00:00:00`; preserve it unchanged. The
old-runtime production graph `61968793`, `61968794`–`61968797`, `61968799`–`61968800`,
`61968821`, and `61968823` was canceled at `2026-08-21 17:50:21` with no nodes
and elapsed 0. Scheduling probes `61969352` and `61969614` were also canceled
with no node/elapsed 0. Attempts 3–7 are immutable failed or canceled L40/A100
probes: `61970394`, `61972695`, `61973090`, `61974092`, and `61974173`.
Preserve every listed root and ID unchanged. The active root is the fresh
`/scratch/lmalveau/docprune/benchmark-4e2473b/attempt-2`, using immutable runtime checkout
`/home/lmalveau/DocPrune-runtime-4e2473b` and the sharded control checkout/handoff below.

## Exact authority pins

```text
environment: /home/lmalveau/mamba-envs/docprune-sol
PDF tools: /home/lmalveau/mamba-envs/m3docvqa-acquisition
corpus: /scratch/lmalveau/docprune/datasets/m3docvqa
HF cache: /scratch/lmalveau/hf_cache; Hub cache: /scratch/lmalveau/hf_cache/hub
Qwen: Qwen/Qwen2-VL-7B-Instruct@eed13092ef92e448dd6875b2a00151bd3f7db0ac
ColPali: vidore/colpali-v1.2@961b51745de3e9adb3468ac5c9ccca0ac626c217
ColPali backbone: vidore/colpaligemma-3b-pt-448-base@30ab955d073de4a91dc5a288e8c97226647e3e5a
M3DocRAG: /home/lmalveau/src/m3docrag-benchmark-29e6ac2 @ 29e6ac2294d6b87075a1d45b8a8df175b214248a
runtime: /home/lmalveau/DocPrune-runtime-4e2473b @ 4e2473bdbbc2e4eca0e92c30d4a0633044501ccf
control: sealed full SHA in /scratch/lmalveau/docprune/benchmark-4e2473b/attempt-2/control.json
attempt root: /scratch/lmalveau/docprune/benchmark-4e2473b/attempt-2
```

## Binding design

[`../docs/superpowers/specs/2026-08-20-docprune-paper-fidelity-correction-design.md`](../docs/superpowers/specs/2026-08-20-docprune-paper-fidelity-correction-design.md)

## Next action

Execute the active sharded benchmark handoff in
[`../sol/handoffs/DOCPRUNE_M3DOCVQA_SHARDED_BENCHMARK_HANDOFF.md`](../sol/handoffs/DOCPRUNE_M3DOCVQA_SHARDED_BENCHMARK_HANDOFF.md).
Submit only the two top-4 arrays `0-3` and their dependent checkpoint publication first. Inspect
the paired report before submitting the remainder. Preserve every historical root and job.
