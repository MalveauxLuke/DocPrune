# Consolidated results

This page is the human-facing results index. The Task 9 [experiment log](../experiments/task9-contextcite/EXPERIMENT_LOG.md) remains the authoritative job, hash, failure, and decision ledger.

## Main findings

### Full top-4 M3DocVQA comparison

| Arm | Questions | Mean token F1 |
| --- | ---: | ---: |
| All visual tokens kept | 2,441 | 37.7911 |
| Full local DocPrune reconstruction | 2,441 | 36.7603 |

On the 2,126 questions with identical ordered retrieved pages, DocPrune minus all-kept was `-1.1980` F1 with paired 95% interval `[-2.3471, -0.0640]`. Full DocPrune removed 81.72% of original visual tokens after CTP. This is evidence about the local reconstruction, not unpublished author code.

Artifact: `/scratch/lmalveau/docprune/benchmark-4e2473b/attempt-2/analysis/top4-paired-quality.json`

### Controlled 245-question stage localization

| Stage | Mean token F1 |
| --- | ---: |
| All-kept | 44.8612 |
| BTP only | 43.1143 |
| BTP + QTP | 43.6327 |
| BTP + QTP + literal/current CTP | 41.3347 |

The incremental CTP change was `-2.2980` F1 with paired 95% interval `[-4.2980, -0.6286]`; BTP and QTP intervals included zero.

Artifact: `/scratch/lmalveau/docprune/benchmark-4e2473b/attempt-2/diagnostics/stage245-v1-incremental/stage245-analysis.json`

### CTP reconstruction studies

- Aggregate-logit CTP scored 42.4163 F1 on the same 245 questions: `+1.0816` over literal CTP but `-1.2163` below BTP+QTP. It is the best-supported local reconstruction, not confirmed author behavior.
- At exactly matched per-question token counts on 64 development questions, aggregate top-M exceeded literal top-M by only about 0.11 F1.
- On the 64-question random-control development study, aggregate CTP scored 39.9219 F1 versus 39.4273 for matched global random and 39.5820 for coverage-matched random. The superiority interval against global random was `[-1.9008, 3.2469]`, so the result was unresolved.
- A preliminary, not yet canonically authenticated, 1,213-question pass reported aggregate CTP 45.9744 F1 versus global random 44.6437, delta `+1.3307`. Do not treat its quick IID interval as the prespecified clustered analysis.

### Task 9 one-question 256-mask diagnostic

For the accepted-answer target at B13, held-out LDS was `0.91484`; RMSE was `0.19368` versus `0.99428` for the constant baseline. Coefficient-refit Spearman mean/min was `0.66975/0.57412`; selected-set Jaccard mean/min was `0.70779/0.65`. This supported proceeding despite failure of the earlier exact-set-identity gate.

Artifact: `/scratch/lmalveau/docprune/task9-paired-256-accepted-answer-0d40fad-v1/analysis.json`

### Task 9 regional ContextCite development pilot

| Stratum | ContextCite F1 | Native DocPrune F1 | EM | Wins/ties/losses |
| --- | ---: | ---: | --- | --- |
| Baseline-correct, n=24 | 1.0000 | 1.0000 | 24/24 vs 24/24 | 0/24/0 |
| Baseline-wrong, n=24 | 0.38125 | 0.13750 | 4/24 vs 0/24 | 9/15/0 |

ContextCite produced four exact rescues and no harms. The balanced F1 delta was `+0.121875`, 95% interval `[0.051042, 0.203125]`; the natural-pool reweighted delta was `+0.154209`, interval `[0.064583, 0.257015]`. Against matched region-size-aware random, the balanced paired F1 delta was `+0.155208`.

Surrogate fidelity was heterogeneous: global LDS mean/min `0.775738/0.308651`, budget-local LDS mean/min `0.622393/-0.103006`. Downstream generated answers, not LDS alone, are the primary endpoint.

Artifact: `/scratch/lmalveau/docprune/task9-preliminary-dynamic48-90f27d7-unified-v1/analysis.json`

Internal digest: `985a837b2094b5a925730eeb4b9bf07c594344f9021c4a718c49c11aa655a8c5`

### Mask-count ablation

Across five 192-mask repeats, baseline-correct preservation was 120/120, but only 16/20 canonical rescues were retained. Wrong-stratum mean F1 fell from 0.38125 at 256 masks to 0.32333; mean accepted-answer likelihood changed by `-0.01391` nats/token. The frozen gate therefore retained 256 masks.

No strict outcome-blind diagnostic reliably identified the rescue-sensitive failure. See the [full analysis](../experiments/task9-contextcite/analysis/mask-count-and-nesting-2026-09-02.md).

Artifacts:

- `/scratch/lmalveau/docprune/task9-mask-count-ablation-d6de9d8-v1/analysis.json`
- `/scratch/lmalveau/docprune/task9-mask192-generation-356f118-v1`

## Artifact catalog

### Local CPU analyses

- `/home/lmalveau/docprune-data/cpu-artifacts/ctp-scoring-fingerprint-v1/analysis.json`
- `/home/lmalveau/docprune-data/cpu-artifacts/ctp-scoring-fingerprint-kv-group-v3/analysis.json`
- `/home/lmalveau/docprune-data/cpu-artifacts/ctp-post245-residual-audit-v1/analysis.json`
- `/home/lmalveau/docprune-data/cpu-artifacts/ctp-query-timing-audit-v1/analysis.json`

### SOL analyses

- `/scratch/lmalveau/docprune/ctp-aggregate-logit-stage64-fixed-v1/analysis.json`
- `/scratch/lmalveau/docprune/ctp-aggregate-logit-stage245-fixed-v1/analysis-final.json`
- `/scratch/lmalveau/docprune/ctp-aggregate-logit-threshold-v1/l40-recovery-3f93787/analysis.json`
- `/scratch/lmalveau/docprune/ctp-kv-group-probe-v1/l40/analysis.json`
- `/scratch/lmalveau/docprune/ctp-generated-query-capture-v2/l40s/analysis.json`
- `/scratch/lmalveau/docprune/ctp-generated-query-qa-v1/l40s/analysis.json`
- `/scratch/lmalveau/docprune/pruning-semantics-v1/ctp-original-relative-v1/analysis.json`
- `/scratch/lmalveau/docprune/pruning-semantics-v1/qtp-mean-stage245-v1/analysis.json`
- `/scratch/lmalveau/docprune/task6-smoke-8b02837-v2/analysis.json`
- `/scratch/lmalveau/docprune/task9-regional-development-7616b29-v1/analysis.json`
- `/scratch/lmalveau/docprune/task9-paired-256-diagnostics-c49abb5-v2/analysis.json`

## Interpretation boundaries

- “DocPrune” above means the tested local reconstruction.
- Task 9 uses an adapted answer-conditioned regional surrogate, not vanilla ContextCite.
- The 48-question pilot and mask-count study are development evidence.
- The 1,213-question random comparison remains preliminary until its exact union and prespecified clustered bootstrap are authenticated.
- The 100-question and 600-question H200 cohorts are separate future evidence and must not be pooled with development outcomes.
