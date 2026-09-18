# Active — frozen train64/dev24 collection

Owner approved quality-first splits and next collection/training launch; stop after
submission. Binding [collection handoff](training-pilot/COLLECTION_HANDOFF.md).
This submission is collection only; no production training queued yet.

# Active — API-correct variable-retention pilot smoke

Owner approved the revised plan and two-correct-case smoke. Binding handoff:
[training-pilot/HANDOFF.md](training-pilot/HANDOFF.md). Exact tested revision and
submission receipt must be recorded. Full64 collection/training is not launched.

# Completed — native 2B selector end-to-end smoke

Owner approved two verified-incorrect questions from filtered training463,32 masks
each, native selector vision, language LoRA, rich readout, Head1 and Head2.
Binding handoff: [selector-smoke/HANDOFF.md](selector-smoke/HANDOFF.md).
This supersedes earlier prohibitions on training only for this bounded smoke.
No full-cohort training. Browser shell; Git code and rsync inputs; minimum resources.
Passed via teacher63539268 and train/evaluate63539612; see the handoff receipt.

# Completed — ten-error Qwen3 masking and OMP audit

Owner explicitly approved steps 1–3: semantic review/freeze ten actual errors, two-case masking/scoring smoke, 22 independent masks per question, G/S OMP and visual inspection. Use the SOL browser shell, allocations for computation, and minimum measured resources. Binding handoff: [omp10/HANDOFF.md](omp10/HANDOFF.md). The reviewed metadata freeze and two-case smoke are active once the exact tested Git pin and launch receipt are recorded. Production is gated on the matching passed smoke; the newly authorized combined full-prefix/batching run performs both smoke cases then all ten in the same process to amortize loading. See the superseding execution policy in the handoff. No later follow-up masks or selector training.

## Previous completed baseline authority

# Active — admitted 471 Qwen3 reader baseline

Owner approved reader baseline smoke followed by efficient production for all471,
with complete answers and correctness assessment, separate original-four and
supplemented results. Binding handoff: [m3doc471-baseline/HANDOFF.md](m3doc471-baseline/HANDOFF.md).
Use minimum measured resources, any compatible GPU, tight resumable jobs.
Prior ColQwen/MinerU preprocessing is complete; preserve its outputs. No mask
teacher collection or training is authorized by this baseline stage.

Current additional approved diagnostic: one five-minute padding/batching profile via sol/omp10/profile.sbatch; exact scope and output isolation in omp10/HANDOFF.md final section. Ten-question production is complete. No production rerun.

Owner-approved follow-up after padding diagnosis: one bounded variable-length attention diagnostic via sol/omp10/varlen.sbatch, as specified in omp10/HANDOFF.md. No dependency installation or production-backend change.
