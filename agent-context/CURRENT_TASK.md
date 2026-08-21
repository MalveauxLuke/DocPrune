# Current Task

## Scope

Implement a faithful, training-free reproduction of DocPrune for document QA,
grounded in the CVPR 2026 paper and supplement.

## State

The corrected benchmark runtime is sealed at
`02385b3a6fc939f23a8632a7ce58b4cac8bff263`, with the pinned upstream
M3DocRAG commit `29e6ac2294d6b87075a1d45b8a8df175b214248a`. The gate now
requires real CUDA/FlashAttention-2 execution, complete unpadded ColPali
equivalence and raster maps, exact upstream retrieval order, no QA-time
ColPali/QTP re-encoding, positive 1/2/4 traces, and positive stage timings.
The index surface requires schema 5; evaluation completion independently
validates every cell and then publishes the signed six-cell comparison.

Historical diagnostic attempt-2 at `/scratch/lmalveau/docprune/benchmark-6c19bfc/attempt-2/`: evaluation array `61830411` was canceled before
work, and six schema-4 indexes `61830405`–`61830410` completed `0:0` but
cannot be promoted. Preserve all scratch artifacts. Scheduling-only attempt-1
(gate `61883512`, indexes `61883881`–`61883886`, eval `61883887`, compare
`61883888`) was canceled before work at `00:00:00`; preserve it unchanged. The
next active root is the fresh `/scratch/lmalveau/docprune/benchmark-02385b3/attempt-2`, using the
immutable runtime checkout `/home/lmalveau/DocPrune-runtime-02385b3` and the
reviewed control checkout/handoff below.

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
runtime: /home/lmalveau/DocPrune-runtime-02385b3 @ 02385b3a6fc939f23a8632a7ce58b4cac8bff263
control: sealed full SHA in /scratch/lmalveau/docprune/benchmark-02385b3/attempt-2/control.json
attempt root: /scratch/lmalveau/docprune/benchmark-02385b3/attempt-2
```

## Binding design

[`../docs/superpowers/specs/2026-08-20-docprune-paper-fidelity-correction-design.md`](../docs/superpowers/specs/2026-08-20-docprune-paper-fidelity-correction-design.md)

## Next action

After independent review, read and execute the active benchmark handoff in
[`../sol/handoffs/DOCPRUNE_M3DOCVQA_BENCHMARK_HANDOFF.md`](../sol/handoffs/DOCPRUNE_M3DOCVQA_BENCHMARK_HANDOFF.md)
from a compute allocation using the fresh `benchmark-02385b3/attempt-2` root.
Submit gate → six indexes → six-cell evaluation → post-array comparator only
after every dependency and manifest check passes. Do not modify or delete the
scheduling-only attempt-1 root or historical diagnostic attempt-2 artifacts.
