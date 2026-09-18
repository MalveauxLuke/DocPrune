# Authorized native 2B + Head 2 wiring smoke

Owner approved two of the ten verified-incorrect questions retained in training463,
32 masks each, real teacher measurement and real selector optimization on SOL.
This is an exposed diagnostic, not train/dev evidence or a 463-question launch.

Checkout /home/lmalveau/DocPrune; exact DP_SELECTOR_COMMIT must match HEAD and
scoped clean files. Runtime inputs/output under
/scratch/lmalveau/docprune-m3doc600/20260915-v2/admitted471-v2/stage.
463 input: training463-evidence-filtered-v1; original471 is cache source only.
Verified review: omp10-20260917-v1/review/frozen.json.
QIDs d579841142ec52ca5ed6b765eaad8f7c (Ethan Suplee),
1cf687db5007935b41eaacbf2332e27f (Pam Shriver); chosen for distinct modalities,
verified wrong status and smaller four-page contexts, not mask outcomes.

Canonical frozen reader8B revision0c351dd01ed87e9c1b53cbc748cba10e6187ff3b,
existing qwen3-baseline environment, BF16/SDPA, independent full-prefix scores.
LoRA dependency is isolated in
/scratch/lmalveau/docprune-m3doc600/20260915-v2/admitted471-v2/selector-smoke-env/peft-0.18.0
(PEFT0.18.0, installed with no dependencies on compute); prepend via PYTHONPATH.
Native selector Qwen/Qwen3-VL-Reranker-2B snapshot
4bd860ac4f15ad1897a214615cccc700f8f71818 from existing hf_cache.
No shared vision adapter, downloads, retrieval or segmentation reruns.
Frozen vision; LoRA rank8/alpha16 language attention + rich readout + both heads.
Feature-disabled readout (optional retrieval-fusion ablation remains separate).

32 seeded random distinct exact-achievable-half-budget masks per question;
mask bank sealed before scores. This exercises the current equal-cost training
contract, unlike the older Bernoulli OMP discovery probes. G and fixed original S;
gold-aware preferences epsilon0.1/margin0.05 are smoke settings, not calibrated
research hyperparameters. Head2 linked loss weight1. Four optimization steps,
one question/microbatch, all32 masks from one encoding, gradient checkpointing.
No frozen-backbone warmup in this gradient-path smoke.

Phases: teacher -> train -> reader-evaluate, separate processes so8B and2B never
occupy GPU together. Check finite scores/loss, every trainable component's
nonzero gradients/updates, frozen parameter discipline, stable native actions,
checkpoint restoration, direct-mask allocation and final reader G/S.
Save resource peaks and1s GPU utilization. Full production readiness not claimed.

Initial GPU request1 generic compatible GPU,2CPU,24000M hostRAM,10min htc/public;
exclude known20GBMIG nodes sg048/sg049/sg050 for the reader phase. No upward
resource change absent observed failure. Resume immutable teacher measurements;
failed training attempts use fresh attempt outputs. Queue estimates before submit.
CPU setup uses lightwork allocation. Code Git; cohort rsync; no data in Git.
Launcher sol/selector-smoke/run.sbatch; implementation experiments/selector_smoke/run.py.
Exact tested commit and job receipt will be recorded in findings after submission.

## Verified receipt

Bounded smoke passed via teacher63539268 + train/evaluate63539612. Successful
training code ae8260339ccf5fa0ee91b3f392c661e25f78fd3d. Source teacher directory
selector-smoke-463-v2; training/evaluation selector-smoke-463-v3. Recovery used
DP_TEACHER_SOURCE pointing to v2 and a5min time override; other resources unchanged.
The source contract and hashed examples were verified before reuse.
See agent-context/findings/2026-09-17-selector-smoke.md for failures, measurements
and limits. Existing v3 output is immutable; do not rerun into it.
