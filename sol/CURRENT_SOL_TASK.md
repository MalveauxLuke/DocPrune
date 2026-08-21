# Current SOL Task

## State

The active corrected benchmark runtime is
`02385b3a6fc939f23a8632a7ce58b4cac8bff263`, with M3DocRAG pinned to
`29e6ac2294d6b87075a1d45b8a8df175b214248a`. The handoff below is the only
authority for the real CUDA/FlashAttention-2 semantic gate, six schema-5
page-specific indexes, six evaluation cells, and the post-array signed
comparison. It also requires complete unpadded ColPali/raster equivalence,
exact upstream retrieval order, no QA-time ColPali, exact measurement identity,
and positive stage timing.

Attempt 2 is diagnostic-only: evaluation `61830411` was canceled before work;
indexes `61830405`–`61830410` completed `0:0` under schema 4 and cannot be
promoted. Preserve those scratch artifacts unchanged. Scheduling-only attempt-1
(gate `61883512`, indexes `61883881`–`61883886`, eval `61883887`, compare
`61883888`) was canceled before work at `00:00:00`; preserve it unchanged. The
next active root is the fresh `benchmark-02385b3/attempt-2` with runtime checkout
`/home/lmalveau/DocPrune-runtime-02385b3`.

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
runtime: /home/lmalveau/DocPrune-runtime-02385b3 @ 02385b3a6fc939f23a8632a7ce58b4cac8bff263
control: sealed full SHA in /scratch/lmalveau/docprune/benchmark-02385b3/attempt-2/control.json
fresh attempt root: /scratch/lmalveau/docprune/benchmark-02385b3/attempt-2
```

## Next action

After independent review, execute
[`handoffs/DOCPRUNE_M3DOCVQA_BENCHMARK_HANDOFF.md`](handoffs/DOCPRUNE_M3DOCVQA_BENCHMARK_HANDOFF.md)
from a compute allocation using the fresh `benchmark-02385b3/attempt-2` root.
Submit gate → six indexes → six-cell evaluation → post-array comparator in
that order, preserving every failed attempt.
