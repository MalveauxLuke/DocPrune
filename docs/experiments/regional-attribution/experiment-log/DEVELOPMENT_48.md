# 48-question development log

## 2026-09-01 — Canonical pilot result

- Cohort: 24 baseline-correct and 24 baseline-wrong questions.
- Baseline-correct: ContextCite and native DocPrune both preserved `24/24`.
- Baseline-wrong: ContextCite mean token F1 `0.38125` versus native DocPrune
  `0.13750`; wins/ties/losses `9/15/0`; exact rescues `4` versus `0`.
- Balanced ContextCite-minus-DocPrune F1 difference: `+0.121875`, descriptive
  95% question-bootstrap interval `[0.051042, 0.203125]`.
- Against matched region-size-aware random, balanced paired F1 difference:
  `+0.155208`.
- Global LDS mean/min: `0.775738/0.308651`; native-budget-local LDS mean/min:
  `0.622393/-0.103006`. Direct generated answers remain primary.

Canonical artifact:
`/scratch/lmalveau/docprune/task9-preliminary-dynamic48-90f27d7-unified-v1/analysis.json`

Internal SHA-256:
`985a837b2094b5a925730eeb4b9bf07c594344f9021c4a718c49c11aa655a8c5`

## 2026-09-02 — Mask-count decision

- Five 192-mask repeats preserved all `120/120` baseline-correct evaluations
  but only `16/20` canonical rescue opportunities.
- Wrong-stratum mean F1 fell from `0.38125` at 256 masks to `0.32333` at 192.
- The frozen functional gate failed; 256 fitting masks remain supported.
- No strict outcome-blind diagnostic reliably identified every question that
  required escalation from 192 to 256.

Artifacts:

- `/scratch/lmalveau/docprune/task9-mask-count-ablation-d6de9d8-v1/analysis.json`
- `/scratch/lmalveau/docprune/task9-mask192-generation-356f118-v1`

Focused analysis: [mask count and nesting](../analysis/mask-count-and-nesting-2026-09-02.md).
