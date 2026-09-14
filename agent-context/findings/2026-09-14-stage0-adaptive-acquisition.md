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
