# Initial 300 native Qwen3 baseline

Owner explicitly chose Qwen3-VL-8B-Instruct and authorized initial baseline.
This supersedes the prior no-answerer restriction for this baseline only.
Preprocessing completed successfully, including combine 63333385.

Checkout: `/home/lmalveau/DocPrune`, exact tested commit in
`DP_BASELINE_COMMIT`; launcher rejects drift. Code:
`experiments/initial300/baseline.py` and `common.py`.
Root: `/scratch/lmalveau/docprune-initial300/20260915-v1`.
Inputs: existing verified catalog and `combined/rankings.json`; manifest SHA
`f1942130e3b585ede4474515089dda2e3df67bb78a53f8f3acf9cd8b5322a495`.
No fresh retrieval, segmentation, masks, teacher banks, or training.

## Frozen baseline

- Qwen/Qwen3-VL-8B-Instruct revision
  `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`.
- Native Transformers generation, BF16 SDPA, all visual tokens retained.
- Top four pages from cached within-document ranking, fewer if necessary;
  retrieval order, no annotated-evidence injection.
- Question first; explicit complete-answer-only instruction, JSON list when
  multiple items, units when needed, `Not answerable` abstention string.
- Processor 256–2560 merged visual tokens per page (32x32 pixel units),
  maximum about 10,240 visual tokens for four pages. Actual grids recorded.
- Greedy generation, 256 output tokens, repetition penalty 1.0; EOS recorded.
- Immutable per-question answers, page/image identity, prompt token IDs,
  visual grids, truncation flag, timing and memory. Correctness remains pending
  answer-contract evaluation; generation is not a correctness adjudicator.
- This new reader/rendering/prompt contract has fresh outputs. No old Qwen2.5
  scores reused. Subsequent mask scoring must retain this contract or regenerate.

## Environment and execution

Isolated `$ROOT/envs/qwen3-baseline`: Python 3.12 inherited from existing
Colfeatures environment; torch 2.8.0, transformers 4.57.3, accelerate 1.10.1,
Pillow 11.3.0, huggingface-hub 0.36.0. All caches on scratch.
CPU setup: `sbatch --export=ALL,DP_BASELINE_COMMIT=SHA sol/initial300-baseline/setup.sbatch`
uses lightwork, 2 CPUs, 8 GiB, 45 min; no GPU reserved during download/install.
GPU smoke: `sbatch --dependency=afterok:SETUP --export=ALL,DP_BASELINE_COMMIT=SHA,DP_BASELINE_SMOKE=1 sol/initial300-baseline/run.sbatch`.
Use any compatible GPU, 2 CPUs, 24,000 MiB host RAM, 20 min. Exclude 20GB MIG
nodes initially: about 16–17 GB model weights plus visual/prefill/cache workspaces
make 20GB a tight unmeasured target. A30 24GB and larger remain eligible.
Smoke selects one long input per source without looking at answers. Inspect
output length, decoding, native image grids and memory before production.
Full run uses bounded array shards, same script with `DP_BASELINE_SHARDS=N`,
no smoke flag, and `--array=0-(N-1)%2`. Choose time/shards from measured smoke.
Existing completed smoke answers are reused under the identical contract.

Recovery: resume identical missing questions, preserve existing outputs and
contracts; diagnose failures before changing resources. No silent CPU offload,
quantization, reduced resolution or smaller page count. Update this handoff
and output identity before any scientific contract change.
