# Active — initial 300 Qwen3-VL-8B unpruned baseline

Owner selected the 8B reader and authorized baseline execution.
Binding handoff: [initial300-baseline/HANDOFF.md](initial300-baseline/HANDOFF.md).
Exact checkout `/home/lmalveau/DocPrune`, commit supplied as
`DP_BASELINE_COMMIT`; runtime root `/scratch/lmalveau/docprune-initial300/20260915-v1`.
Setup only on CPU allocation; smoke only on GPU allocation. No computation on
login. Read handoff for pinned checkpoint, environment, cached page admission,
resource limits, outputs, smoke gate and resume policy. No masking or training.

Prior preprocessing [handoff](initial300/HANDOFF.md) is completed, including
combine 63333385; all 300 questions/3,710 unique images covered.
