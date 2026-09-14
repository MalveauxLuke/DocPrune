# Stage 0 adaptive acquisition preparation

2026-09-14. The owner authorized continuing through steps 1–3: freeze the
controller, implement/test it locally, and connect the frozen-reader scorer.
Remote mutation and GPU execution remain outside this completed preparation.

## Implemented

- A frozen three-arm comparison: original R and A banks, plus reversible grouped
  adaptive acquisition. Exactly 32 distinct logical contexts per arm/question,
  always 5,016 retained original visual tokens. The existing banks are unchanged.
- Eight A seed observations, then three rounds of broad/focus/audit/focus. Exact
  rules govern scale, raw G/S responsiveness, gold-aware ranking, edge priority,
  group opening, balanced complementary exchanges, quiet audits, discounting and
  explicit capped fallbacks. No cross-question budget allocation in v1.
- A separate revealed history per arm, shared physical-mask cache, deterministic
  proposal journals, checksummed immutable results and resumable replay.
- A persistent B_input reader adapter built from existing Qwen vision, deletion
  and teacher-forced-scoring primitives. Exact frozen targets, original wrong
  answer, processor/prompt/token identities, reference parity, masked adapter
  parity, full-context generated answer and forced-deletion trace are checked.
- A portable sealed input/code bundle, read-only environment doctor, process-local
  cache settings, compatibility constraints and separately gated future smoke and
  scoring commands. No H200 environment was changed or freshly surveyed.

## Verified evidence

All 17 authenticated cases and 68 original page PNGs were prepared without
historical masked scores in controller inputs. The 1,659 original actions become
1,065 initial groups (48–80 groups/question). Every grouped space admits exactly
5,016 tokens. These are grouping feasibility results, not region-importance labels.

Twenty-two focused local tests pass using Python 3.12 and NumPy 1.26.4. Tests cover
exact-cost sampling, cancellation, G/S co-movement with flat C, reversed preferences,
soft discounting, reversible groups, all 17 real geometries, failed contracts,
separate histories, explicit fallbacks, single-writer publication and interrupted
replay without repeating completed physical scores. The reader boundary is tested
with a mock backend; real Torch/CUDA execution is not claimed.

An all-17-question synthetic run completes 1,632 logical observations and 1,351
unique synthetic masks. Its adaptive requests comprise 136 seeds, 102 broad,
204 focused, 100 audit and 2 explicit fallback observations. These artificial
scores validate execution branches only. They do not estimate pilot answer
improvement, adaptive superiority or H200 runtime.

## Next gate and limitations

Step 4 is a current H200 survey followed by an authorized one-question smoke.
The pinned remote stack was recorded previously, not reverified during this work.
No transfer, remote installation, model load, new teacher scores or decoded
experimental answers have been produced. The full experiment still needs actual
GPU likelihoods and subsequent decoded-candidate review. Adaptive versus A tests
the combined grouping/acquisition policy, not grouping in isolation.

## Navigation

- [Frozen specification](../../docs/experiments/corrective-selection/STAGE0_ADAPTIVE_ACQUISITION.md)
- [H200 handoff and environment](../../h200/adaptive-acquisition/README.md)
- [Runner command](../../scripts/stage0_acquisition.py)
- [Original prepared banks](../../docs/experiments/corrective-selection/STAGE0_MASKING_PREPARATION.md)
- Local ignored delivery: `outputs/stage0-adaptive-acquisition-2026-09-14/`.
  It contains the sealed inputs, portable bundle/archive, verification records and
  synthetic output journals. The archive contains no weights or Python environment.


## SOL smoke memory diagnosis — 2026-09-14

This later update supersedes the earlier "Next gate" execution status above.
SOL job 63235689 at code b60e78b2c334a6e1088c7d0d91d8905ce81c8df6
started 14:08:03 MST and failed after 2m53s on an A100 80GB. Host MaxRSS
was 18,139,692 K (about 17.3 GiB). The vision SDPA operation failed requesting
47.99 GiB, with process GPU memory already 70.04 GiB. Log:
`/scratch/lmalveau/docprune-adaptive-acquisition/20260914-smoke01/smoke-63235689.log`.
No successful teacher measurements or parity receipt were produced.

Read-only diagnosis verified the installed Transformers 4.49.0
Qwen2_5_VLVisionSdpaAttention.forward constructs a boolean
[1, seq_length, seq_length] mask, marks cu_seqlens blocks, and calls SDPA with
rank-three [heads, seq_length, head_dim] q/k/v tensors. The installed Torch
header sdp_utils_cpp.h rejects non-rank-four inputs for fused kernels.
Thus the frozen vision SDPA path falls back to dense math attention; selecting
"sdpa" alone does not guarantee a memory-efficient kernel. See also the exact
[PyTorch 2.4.1 source](https://github.com/pytorch/pytorch/blob/v2.4.1/aten/src/ATen/native/transformers/sdp_utils_cpp.h).

The local compact vision adapter passes all four pages through each block
together. Q12 has 4 x 10,032 = 40,128 fine patches before vision merging,
not merely the 10,032 merged decoder visual tokens. Its dense mask has
1,610,256,384 elements (about 1.50 GiB as bool); a hypothetical BF16 attention
matrix across its 16 heads is 47.99 GiB. This size matches the failed allocation,
but the exact internal temporary responsible was not profiled. Window/page
masking prevents cross-boundary information flow but does not remove the dense
allocation. Even window-attention layers receive the full concatenated sequence.

Historical evidence supports smaller-GPU feasibility for related work:
archived SOL input diagnostic job 62424211 completed 320 masks on an L40S
in 4m23s with 24 GiB host RAM. That older experiment used a 3,586-token
post-BTP/QTP mapping and a different historical runtime; it is not proof that
the current Q12 full-vision configuration already passed on L40S. The historical
fair CTP baseline also specifies FlashAttention-2, unlike the current frozen SDPA
runtime. See the archived INPUT_256_DIAGNOSTIC and FAIR_CTP_BASELINE documents.

Proposed repair: execute attention independently within the existing
cu_seqlens windows/pages, using an explicit batch dimension for efficient SDPA.
Preserve the exact attention boundaries, patch order, rotary positions, weights,
precision, images and masks. Mathematical independence supports this change,
but floating-point parity still requires testing against the original path on
small tensors and the unchanged Q12 1e-4/generation gates. Full-page splitting
alone reduces a dense per-call square by 16x for four equal pages; splitting
actual windows avoids still more wasted work. This is an analytical allocation
comparison, not a measured whole-model speedup or memory-fit guarantee.
No implementation change, package change, or new GPU submission was made during
this diagnosis. Do not infer that requesting a larger GPU is necessary.
