# Native selector wiring smoke

Owner authorized two previously verified wrong questions from training463,32 masks each, native 2B vision plus frozen reader8B, rich readout, LoRA and both heads. Full-cohort training is not authorized by this smoke.

## Preparation verified

- Frozen QIDs: `d579841142ec52ca5ed6b765eaad8f7c` and `1cf687db5007935b41eaacbf2332e27f`; text and image cases with four admitted pages, chosen before teacher results.
- Cohort transferred by owner rsync. SOL compute verification confirmed463 and pool hash; selection SHA256 `7944de8221c9da7973a9d94cd2ce4eabb3c30230ceb4dcec98251c511f029533` matches Mac.
- Tested commit `c9276654f5e77a66a2c861e4aaa71a679ee869ed`, pushed and pulled on SOL. 33 targeted CPU tests passed against an isolated copy of the exact staged code; random bank sanity check also passed. This is not GPU validation.
- PEFT0.18.0 installed in isolated scratch overlay on lightwork63538064; imports verified with existing accelerate1.10.1. Reader environment unchanged. Initial unbounded-thread import was slow; bounded-thread import succeeded.
-32 unique seeded random-utility allocations at the achievable half-token budget. This follows the current equal-cost training contract; differs from Bernoulli OMP discovery masks.
-4 optimizer steps, one question at a time, all32 mask preferences per encoding; checkpoint gradient/update/reload checks and final reader scores. No held-out performance claim.
-Resources: one generic GPU excluding previously insufficient20GBMIG nodes;2CPU,24000M host RAM,10min htc/public. Teacher/train/evaluation use separate processes, each loading only its required model.

See [binding handoff](../../sol/selector-smoke/HANDOFF.md). Job **63539041**, exact commit above, submitted17:43:33MST and running17:44:06 on sg027 (33-second queue). Log: `/scratch/lmalveau/docprune-m3doc600/20260915-v2/admitted471-v2/stage/selector-smoke-63539041.log`. Measured results pending.

## First GPU attempt and correction

63539041 loaded the reader but failed before any mask scores: cached MinerU page identity was compared with an already-qualified prompt occurrence ID. Corrected validation order: validate immutable cached page first, then assign occurrence ID for action partition. No source data changed. Corrected commit `7395bd2f695db1b5e3c9f4c307099d14cde7379c`; retry63539268 uses fresh output `selector-smoke-463-v2` and unchanged resources. GPU UUID monitor now supplies the required GPU prefix. Initial retry state pending Priority; no successful GPU completion claimed.

## Teacher completed; CUDA validation fix

63539268 completed64 scores and both repeatability checks: text129 regions/490 preference pairs; image194 regions/494 pairs;7600 visual tokens each. Training failed before its first optimization step because CUDA does not support the integer matrix-vector operation in TeacherBank validation. Replaced it with exact integer elementwise multiply plus sum (same arithmetic result), retaining unrelated local correct-case work unstaged.33 targeted tests pass.

Commit `ae8260339ccf5fa0ee91b3f392c661e25f78fd3d` adds verified teacher-source reuse: source cohort, questions, model revisions, mask policy and teacher contract must match, and serialized examples remain hash-verified. The recovery runs only train/evaluate in fresh `selector-smoke-463-v3`, using v2 teacher artifacts; reduced wall request to5min, other resources unchanged.

## Passed end-to-end smoke

Recovery **63539612 COMPLETED**,1m44s, A10080GB on sg012. Code `ae8260339ccf5fa0ee91b3f392c661e25f78fd3d`. All four optimizer steps had finite nonzero gradients through LoRA, rich reader, direct head and correction head; all four groups changed weights.4,334,466 trainable parameters; frozen base/vision received no gradients. Checkpoint reload reproduced both direct and total mask scores exactly after deliberately perturbing trainable weights. Checkpoint SHA256 `802c7b5a56465f35fb39fce6ccfe27ace782ecfe758b04e596b1dd3a4ce72979`.

Training/checkpoint checks20.65s excluding loading/preparation; GPU allocated peak6.740GB, reserved6.971GB. Final reader evaluation9.76s excluding loading/preparation; GPU allocated19.301GB, reserved19.489GB. Scheduler job maxRSS14,525,640KiB (~13.85GiB), higher than per-phase Python RSS, so use scheduler peak for future host RAM planning.24000M request remains the existing GPU scheduler minimum. Billing35core-equivalents ×104seconds ≈1.01core-hours for successful recovery.

32-mask training-bank preference accuracy (in-sample, not generalization): text54.49%, image74.29%; Head1 and total Head2 rankings had the same accuracy after four steps. Both selected masks retain3800/7600 tokens. Loss was not consistently decreasing: text1.449→1.992, image1.282→1.179. This establishes working gradients and updates, not useful learning or optimized hyperparameters.

Final reader G/S, direct selection: text -4.0115/-0.0113; image -15.9952/-0.0401. Head2-best within training bank: text -4.9071/-0.2283; image -20.2890/-0.0130. These likelihoods do not establish corrected decoded answers; decoded answers were not generated in this smoke.

Artifacts on SOL: stage/selector-smoke-463-v2/{contract.json,teacher-complete.json,QID/example,QID/measurements}; stage/selector-smoke-463-v3/{contract.json,training-complete.json,trained-masks.json,checkpoint.pt,complete.json,utilization-63539612.csv}. Both under the handoff scratch root. Final logs/receipts inspected through SOL browser shell; large artifacts not copied back yet.

Next decision is training/acquisition design and stability, not another identical wiring smoke or automatic full463 launch. Correct-case API and shared-vision adapter remain outside the validated scope.
