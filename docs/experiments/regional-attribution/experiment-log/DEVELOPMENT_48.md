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

## 2026-09-06 — CPU visual audit of development failures

Completed [visual and quantitative audit](../analysis/VISUAL_AUDIT_24_2026-09-06.md)
of all 24 baseline-wrong cases and six controls chosen before pruning outcomes.
The gallery includes all 120 supplied page instances and 576 token-footprint
illustrations reconstructed from existing selected-arm IDs, mappings and MinerU
PDFs. Initial question/page observations precede case-specific outcome reads;
native-detail and overlay review is explicitly unblinded.

- Reproduced support/native mean F1 `0.38125/0.13750`, exact `4/0`, and
  wins/ties/losses `9/15/0`; support versus unpruned is `7/17/0`.
- Of four nominal exact rescues, Q29 is supported by the retrieved table,
  Q36/Q37 lack required retrieved evidence, and Q43 matches stale gold
  contradicted by its supplied pages. Historical scores were not changed.
- Actual wrong-case boundaries are `B_14` (16) and `B_16` (8), with `B_K`
  defined after zero-based block K. Multi-item gold-target issues and
  retrieved-document/cohort overlaps are recorded in the report.
- All 30 selected-arm/mapping hashes and compact-token ownership/budget checks
  passed. No new model run, GPU job, training, or mask sweep was performed;
  the 600 preparation was not changed and confirmation outcomes were excluded.

Artifacts: `/home/lmalveau/DocPrune/outputs/regional-audit-2026-09-06/`
(`index.html`, `cases.csv`, `cases.json`, `arm-measurements.csv`,
`cohort-audit.json`, `validation.json`). Source runtime is
`90f27d7ed8b99ad10f1a5fe405c131127456ae5d`; audit checkout started at
`3c25f805863ece09f7d5bea82311baa9b1bd9742`.
