# Current Task

## Scope

Implement a faithful, training-free reproduction of DocPrune for document QA,
grounded in the CVPR 2026 paper and supplement.

## State

Local implementation is complete through the Python 3.10 compatibility fix at
`64ea70c66a8f9e3dbce804d1fde265a4a2b8b09d`. The pinned SOL environment and a
checksum-recorded FlashAttention 2.5.8 wheel were built successfully. Structural
smoke job `61567743` stopped during test collection on the older `99dbece`
runtime because it imported Python 3.11-only `tomllib`; the corrected runtime
subsequently passed 69 non-GPU tests and Ruff under Python 3.10.

The corrected runtime has not yet passed the structural GPU smoke. M3DocVQA
acquisition, the processor-contract probe, benchmark runs, and performance
claims remain unvalidated.

## Binding design

[`../docs/superpowers/specs/2026-08-15-docprune-reproduction-design.md`](../docs/superpowers/specs/2026-08-15-docprune-reproduction-design.md)

## Next action

Execute only the active smoke-recovery handoff in
[`../sol/handoffs/DOCPRUNE_SOL_SMOKE_RECOVERY_HANDOFF.md`](../sol/handoffs/DOCPRUNE_SOL_SMOKE_RECOVERY_HANDOFF.md).
Stop and review its report. The separate M3DocVQA acquisition/processor-probe
handoff remains staged and inactive until that smoke passes. Do not submit the
benchmark runner.
