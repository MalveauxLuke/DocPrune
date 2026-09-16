# Frozen 471-question Qwen3 reader baseline

Owner authorizes adaptation, smoke, measured production shards and correctness
assessment for all 471 admitted questions. No masking or training.
Checkout /home/lmalveau/DocPrune. Exact submitted Git revision is supplied as
DP_BASELINE_COMMIT and recorded in the launch log; launcher rejects drift.
Inputs /scratch/lmalveau/docprune-m3doc600/20260915-v2/admitted471-v2/stage,
verified catalog bf220ddb7c0b67b58a992292108e9ec386b8694d3b33f542c71dd5d35f0b4d08.
Outputs stage/baseline-qwen3-8b-admitted-v2, immutable per-question resume.
Reuse initial300-baseline environment and pinned Qwen3-VL-8B snapshot;
add torchvision0.23.0 for torch2.8.0 and word2number1.1 on a compute node.

Preserve all admitted pages in frozen presentation_order. Do not use ColQwen
reranking for admission. Original-four vs supplemented tag is evaluation-only.
Native BF16 SDPA, processor256–2560 merged tokens/page, greedy256-token output,
question first and complete-answer instruction inherited from initial300.
No dynamic resolution/precision/page reduction. Batch1 avoids padding and keeps
VRAM low. Generic available GPUs; record GPU per answer, reject incompatible
hardware or surface OOM rather than change the scientific contract.

Smoke: longest context in each of six modality/coverage strata, chosen without
answer outcomes. GenericGPU1,2CPU,24000M (last measured scheduler minimum),10min.
Use sol/m3doc471-baseline/run.sbatch, DP_BASELINE_SMOKE=1. Inspect resource peaks,
answers and truncation before production. Production uses same script,
DP_BASELINE_SMOKE=0, DP_BASELINE_SHARDS=N, bounded contiguous array; size/time
from smoke. Prefer tight resumable jobs over speculative resource padding.

Afterok CPU score.sbatch:1CPU1GB3min, official repository EM/F1 normalization,
JSON-array parsing, complete reference spans, all471 plus stratum summaries.
Non-exact matches are not asserted to be semantic errors; retain review status.
Truncated output is explicitly flagged and smoke truncation blocks production.

Recovery: identical missing-question resume, retain successes; retry timed-out
shards. OOM may justify compatible higher-memory GPU for affected questions,
not automatic context/precision changes. No unrelated jobs may be changed.
