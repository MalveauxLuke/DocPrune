# Task 9: regional ContextCite

Task 9 tests whether an adapted, answer-conditioned regional surrogate can choose a better budget-matched visual context than native DocPrune attention pruning.

## Read in this order

1. [Consolidated results](../../results/README.md)
2. [Scientific plan](EXPERIMENT_PLAN.md)
3. [Implementation plan](IMPLEMENTATION_PLAN.md)
4. [Experiment log](EXPERIMENT_LOG.md)

Supporting material:

- [Mask-count and adaptive-diagnostic analysis](analysis/mask-count-and-nesting-2026-09-02.md)
- [Shared-probe layer-trajectory design](research/shared-probe-layer-trajectory-2026-09-03.md)
- [Shared-probe research review](research/shared-probe-review-2026-09-03.md)

## Cohorts

| Cohort | Role | State |
| --- | --- | --- |
| 48 questions: 24 baseline-correct, 24 baseline-wrong | Development and mask-count ablation | Complete |
| 100 new baseline-wrong questions | Independent downstream confirmation | Prepared for H200 |
| 600 document-disjoint questions | Shared-probe feasibility and diagnostic learning | Prepared as a separate H200 track |

The cohorts must never be mixed. The 48-question result is developmental; the 100-question run tests rescue frequency among ordinary baseline failures; the 600-question study asks whether cheaper pre-answer diagnostics can predict a fixed `B13` intervention target.

## Current decisions

- Native DocPrune determines each question's comparison layer and token budget in the downstream pilot.
- Regional ContextCite uses whole MinerU regions and physical deletion.
- The supported fit uses 256 masks. The 192-mask alternative failed the frozen functional gate.
- The extra 64-mask surrogate-fidelity holdout is omitted from the independent 100-question confirmation because downstream answer quality is the primary endpoint and the fitting procedure is frozen.
- FastV and redundant random/coverage arms are not rerun where existing evidence already answers those comparisons.
