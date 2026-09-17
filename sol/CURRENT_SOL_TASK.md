# Active — ten-error Qwen3 masking and OMP audit

Owner explicitly approved steps 1–3: semantic review/freeze ten actual errors, two-case masking/scoring smoke, 22 independent masks per question, G/S OMP and visual inspection. Use the SOL browser shell, allocations for computation, and minimum measured resources. Binding handoff: [omp10/HANDOFF.md](omp10/HANDOFF.md). The reviewed metadata freeze and two-case smoke are active once the exact tested Git pin and launch receipt are recorded. Production is gated on the matching passed smoke; the newly authorized combined full-prefix/batching run performs both smoke cases then all ten in the same process to amortize loading. See the superseding execution policy in the handoff. No later follow-up masks or selector training.

## Previous completed baseline authority

# Active — admitted 471 Qwen3 reader baseline

Owner approved reader baseline smoke followed by efficient production for all471,
with complete answers and correctness assessment, separate original-four and
supplemented results. Binding handoff: [m3doc471-baseline/HANDOFF.md](m3doc471-baseline/HANDOFF.md).
Use minimum measured resources, any compatible GPU, tight resumable jobs.
Prior ColQwen/MinerU preprocessing is complete; preserve its outputs. No mask
teacher collection or training is authorized by this baseline stage.
