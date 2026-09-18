# Approved 64-question selector training

Owner resumed on September17: finish bank audit and trainer checks, then submit
training; a gpt-5.6-sol subagent independently reviews the audit. No new teacher
collection or API calls. Stop after successful training submission.

## Inputs and gates

Completed collection63544061 (38m38s, exit0) in
`/scratch/lmalveau/docprune-m3doc600/20260915-v2/admitted471-v2/stage/training-pilot-quality-v1`.
64training/24dev,86new banks plus2immutable smoke banks. Run
`sol/training-pilot/audit.sbatch` first: CPU2,6000M,8min,htc/public, no GPU. First audit took5m02s and
peaked4093684KiB at the4000M limit; this is a measured adjustment.
Its immutable report is `stage/training-pilot-v1/bank-audit-<job>.json`.
Require passed complete identity/channel/mask validation and usable strict pairs.
Do not decrease margins because yield is low. Epsilon0.1/margin0.05 remain fixed.
Zero-pair training questions remain in the frozen split but contribute no updates.

Checkout `/home/lmalveau/DocPrune`; exact tested/pushed revision passed as
DP_PILOT_COMMIT and checked before executing. Collection is complete, so the
checkout may now advance. Code via Git; all inputs/caches already on scratch.
No fresh/global retrieval and no MinerU/ColQwen model runs.

## Training launch

`sbatch --export=ALL,DP_PILOT_COMMIT=<tested-pin>,DP_BANK_AUDIT=<passed-report>
--output=<stage>/training-pilot-v1/train-%j.out
--error=<stage>/training-pilot-v1/train-%j.err sol/training-pilot/train.sbatch`

One generic GPU (including20GB slices),2CPU,24000M hostRAM,20min initially.
The native2B smoke peak was6.97GB device/approximately6GiB host; five-page examples
have9500 reader tokens versus7600 for the four-page smoke, not a64-fold memory
increase because only one question is resident.20GB is a conservative compatible
class given the measured7GB peak; a longer-context OOM remains possible and must
be reported, not silently reduced. No A100-only restriction. Host24000M is the
previously enforced GPU-job minimum; recheck before submission.

20min is a bounded estimate: the2question smoke's entire training was35.89s;
scaling by32 gives19.14min before disk-cache savings but omits additional dev/I/O
and potentially slower hardware. Persistent frozen caches avoid repeated vision
and frozen language passes. Accept a bounded timeout rather than reserve a much
longer job; resume completed epochs. Compare20/15min scheduler estimates; do not
add shards or reserve multiple GPUs. Monitor assigned-GPU utilization eachsecond
and record actual per-epoch memory/runtime and billed TRES after completion.

## Environment and execution

Existing `/scratch/lmalveau/docprune-initial300/20260915-v1/envs/qwen3-baseline`
Python: torch2.8.0+cu128,transformers4.57.3,PEFT0.18 overlay from
`/scratch/lmalveau/docprune-m3doc600/20260915-v2/admitted471-v2/selector-smoke-env/peft-0.18.0`.
Native selector snapshot
`/scratch/lmalveau/hf_cache/hub/models--Qwen--Qwen3-VL-Reranker-2B/snapshots/4bd860ac4f15ad1897a214615cccc700f8f71818`.
BF16/SDPA, no install/download. Reader never loaded in this job.
Output `stage/training-pilot-quality-v1-train-seed0` with exclusive writer lock.

Native Rich, ColQwen129feature schema, Head1+Head2, capacity1.0, microbatch1,
accumulation4, fixed family-balanced G/S or S-only labels. Seed0, warm-up2epochs
at3e-4; independent frozen/LoRA branches at1e-4 head LR and2e-5 LoRA LR, up to4
further epochs each, patience2. Same warm-up weights/RNG and fresh optimizers.
Dev24 never enters optimization. Fixed dev ranking chooses branch checkpoints.
Save untrained and shortlisted Head1 selections at75%primary/50%secondary.

One question streamed at a time; disk caches sealed by model/runtime/input
identity. Native vision reusable throughout; frozen language memories only while
LoRA is identity. Cached Rich outputs are forbidden while its weights learn.
Prepared pixels load only before the first vision encoding. RAM caches bound to2
questions. Atomic epoch checkpoints include trainable heads, identity/adapted
LoRA, optimizer/scheduler and RNG. A timeout loses at most the current epoch;
re-run the same pinned launcher/report/output to resume, never overwrite banks.

## Development reader evaluation, later separate process

`experiments/training_pilot/evaluate64.py` consumes the passed audit and completed
training directory. It compares untrained/frozen/LoRA75%/50%, deterministic
ColQwen75%/50% and3fixed Bernoulli.5random masks. Exact duplicate masks reuse
measurements. S-only for correct cases; G/S for incorrect. Optional `--decode`
stores full new answers and checks all-keep regeneration; those answers require
fresh grading, not reuse of original baseline labels. This launcher submits
training only. Ranking improvements are not decoded-answer improvements.
