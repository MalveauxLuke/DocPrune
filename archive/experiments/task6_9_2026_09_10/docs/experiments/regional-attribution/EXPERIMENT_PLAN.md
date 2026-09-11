# Regional attribution experiments: canonical plan

Status: approved active design  
Documentation consolidation: 2026-09-05

This is the scientific source of truth. It preserves the approved direction
while separating completed development evidence from the two active studies.
Operational commands belong in machine handoffs, results belong in the
experiment log, and superseded detail remains in the archive.

## Purpose

The project asks two connected questions:

1. Can privileged, accepted-answer-conditioned regional interventions choose a
   better budget-matched document context than the local DocPrune attention
   policy?
2. Can pre-answer frozen model states predict those intervention effects well
   enough to avoid hundreds of ContextCite-style interventions at test time?

The supported method is an adapted regional diagnostic. MinerU supplies whole
semantic regions; physical deletion occurs inside the frozen document VLM.
Neither component makes this vanilla ContextCite, a token oracle, a grounding
method, or a deployable selector.

## Frozen contract shared by all tasks

- Model: `Qwen/Qwen2-VL-7B-Instruct`, revision
  `eed13092ef92e448dd6875b2a00151bd3f7db0ac`.
- Runtime: the corrected local DocPrune decoder at
  `dd5f000a909a718826541df816a4b65396764e1c` or a reviewed descendant.
- Inputs: each question's authenticated cached ordered top-four pages and
  persisted page features.
- Retrieval: never run fresh/global retrieval, load the global index, or rebuild
  retrieval features for these studies.
- Preprocessing and decoding: preserve the fixed raster bytes, processor, prompt,
  chat template, BTP/QTP semantics, BF16 execution, greedy decoding, 128-token
  cap, and EOS IDs `[151645, 151643]`.
- Intervention: physically delete whole mapped regions while preserving retained
  token order and original Qwen M-RoPE positional identity.
- Region tool: MinerU `mineru-3.1.0-released`, commit
  `d9cd58add047c2364c1198eefcb1ee9cd63a971a`, with
  MinerU2.5-Pro-2604-1.2B revision
  `d3f5e08d073c21466bbabe21c71bb1e9c2e595da`.
- ContextCite reference: MadryLab/context-cite commit
  `c11f8ace6e68ba0121b2e2f1f5c896da9e4156f4`.
- Cohorts: the 48-question development set, 100-question confirmation set, and
  600-question probe set remain disjoint and keep their declared roles.
- Information control: freeze cohort membership, masks, model selection, and
  decision rules before inspecting the affected outcomes.

For one region mask `v`, score both fixed teacher-forced targets from the same
masked computation:

- `G(v)`: accepted-answer normalized log likelihood;
- `S(v)`: original generated-answer normalized log likelihood; and
- `C(v) = G(v) - S(v)`: accepted-versus-generated margin.

Generated downstream answers, not surrogate correlation alone, determine
whether a selected context actually helps.

## Task 1 — Completed foundations

The local decoder, physical-deletion path, fixed-page provenance checks,
regional mapping, ContextCite-style surrogate, native DocPrune comparator,
matched regional random comparator, aggregation, and prefix/boundary-state
caching are implemented.

Earlier native-policy, random, coverage, and explicit-visual-state studies are
complete development evidence. Do not rerun them unless code or inputs that
support a current comparison materially change. Their exact specifications are
preserved in the [legacy scientific plan](archive/legacy-governance-2026-09-05/EXPERIMENT_PLAN.md),
and their results are summarized in [the results index](../../results/README.md).

The conditional Wang standalone-contribution study is not an automatic fallback
and is not part of either active H200 track.

## Task 2 — Completed 48-question development study

### Cohort and comparison

- Stratified random sample: 24 questions where the unpruned model was correct
  and 24 where it was wrong.
- Native DocPrune chose each question's dynamic comparison boundary and retained
  token count before regional outcomes were inspected.
- Arms: unpruned, native dynamic DocPrune, accepted-answer support ContextCite,
  accepted-answer margin ContextCite when available, and matched
  region-size-aware random pruning.
- Every pruning arm used the same per-question retained-token budget.
- The canonical surrogate used 256 fitting masks. Separate global and
  native-budget-local held-out masks were used for development diagnostics.

### Established result and decisions

- ContextCite preserved all 24 baseline-correct answers.
- It produced four exact rescues among 24 baseline-wrong questions; native
  DocPrune produced none.
- Baseline-wrong mean token F1 was `0.38125` for ContextCite and `0.13750` for
  native DocPrune.
- The 192-mask ablation preserved only 16 of 20 repeated rescue opportunities
  and reduced wrong-stratum mean F1 to `0.32333`; therefore 256 remains the
  supported full regional estimator.
- No strict outcome-blind diagnostic reliably identified every question harmed
  by the 192-mask reduction.
- Moving the one-question intervention from `B13` to model input did not improve
  reliability. Fixed-target probe work therefore uses `B13`.

This cohort remains development data. It cannot support population prevalence,
confirmatory, deployability, or token-level-oracle claims.

## Task 3 — Baseline-wrong confirmation

### Question

How often does the frozen 256-mask regional diagnostic improve downstream
answers among ordinary new baseline failures?

### Cohort

- Exactly 100 new baseline-wrong questions sampled randomly from the eligible
  authenticated pool.
- Baseline wrong means that the fixed unpruned model, retrieval, prompt,
  decoding, and model revision did not produce an exact-match answer.
- One question per support-document component; the sealed cohort achieved this
  without fallback.
- Exclude every question in the 48-question development study.
- Do not enrich the primary sample for likely distractors. Mechanistic labels
  may be added only after selection.

### Frozen evaluation

- Use each question's exact cached top-four pages and persisted features.
- Native dynamic DocPrune selects the comparison layer and token budget first.
- Compare unpruned, native DocPrune, accepted-answer support ContextCite,
  accepted-answer margin ContextCite when available, and the existing matched
  region-size-aware random control.
- Use 256 fitting masks and zero surrogate holdout masks. The fitting method,
  Lasso regularization, mask distribution, selection rule, model, prompt, and
  decoding are frozen.
- Primary endpoint: the generated downstream answer after budget-matched
  physical pruning.
- Report normalized token F1, exact match, paired win/tie/loss, rescue rate,
  gold likelihood, and gold-versus-generated margin. Cluster uncertainty by
  support-document component if required.

This confirms headroom among ordinary baseline failures. It does not estimate
performance on the natural mixture of correct and incorrect questions and does
not make the privileged diagnostic deployable.

Execution authority: [H200 confirmation handoff](../../../h200/task9-baseline-wrong-100/HANDOFF.md).

## Task 4 — Shared-probe feasibility study

### Question

Can compact pre-answer frozen-state features predict gold- and self-conditioned
post-boundary regional deletion responses on unseen documents, and can those
predictions support useful pruning without test-time interventions?

### Cohort and labels

- Exactly 600 new document-disjoint questions, excluding the 48 and locked 100.
- Splits: 360 balanced train, 120 balanced validation, 60 balanced primary test,
  and a separate 60-question natural-prevalence secondary test.
- Keep one question per support-document component and preserve available
  template/vendor metadata.
- Each training question uses 32 deterministic masks: 8 singleton, 16
  native-budget-local, 4 less-aggressive, and 4 pair/small-set interventions.
- Train directly on centered observed `G(v)` and `S(v)` outcomes. Do not use
  per-question Lasso coefficients as labels.
- Every teacher label uses physical deletion at fixed boundary `B13`.

### Representation task

1. Reuse the 48-question data for the frozen additivity/interaction Gate 0.
2. At every zero-based read block `B0..B13`, train matched independent
   low-capacity probes against the same B13 teacher target.
3. Evaluate isolated geometry/region-metadata-only, native-DocPrune-only, and
   question-only controls, plus pooled region hidden state, compact pre-answer
   QK features, and hidden-plus-QK features.
4. Every feature must precede accepted- or generated-answer teacher forcing.
5. Select the best snapshot using balanced validation only. The primary metric
   is question-equal gold budget-local mask-response R². Differences within
   `0.005` are ties, broken by baseline-wrong gold R², safe-deletion AUPRC,
   earlier read block, then simpler family.
6. Compare the selected snapshot with a small local-trajectory linear model at
   endpoints `B3..B13`. It may use standardized current compact QK or
   hidden-plus-QK features, their adjacent difference, and the latest-four-layer
   mean. It may not use raw cross-layer hidden-state differences.
7. Compare with current-only, mean-only, delta-only, parameter-matched capacity,
   shuffled-history, and shuffled-order controls. If ordering controls do not
   degrade performance, report multi-layer aggregation rather than trajectory
   information.

Gold and self heads remain separate. At epsilon `0.05` nat/token, report their
predicted deletion deltas, signed difference, and fraction with absolute
difference at least epsilon on baseline-wrong and oracle-rescuable slices. This
is descriptive and cannot select a model or establish causality.

### Conditional escalation

- Skip the trajectory comparison if neither QK nor hidden-plus-QK has positive
  validation gold budget-local R².
- Permit full-prefix low-rank aggregation only if the local trajectory improves
  gold budget-local R² by at least `0.02`, its support-component-bootstrap 95%
  lower bound is above zero, baseline-wrong R² degrades by no more than `0.02`,
  and it beats shuffled-history and shuffled-order controls.
- Permit rank-4 question-region bilinear probing only if Gate 0 passes and the
  best QK/combined linear model fails either positive baseline-wrong gold R² or
  Spearman `>= 0.30`.
- Models 0 and 1 from the external review are skipped in the initial
  implementation. An RNN or transformer across layers is not authorized.
- A multi-boundary causal study is a separate future experiment and is not
  authorized by this plan.

### Execution split

- SOL: cohort sealing, implementation, CPU Gate 0 analysis, focused validation,
  packaging, and final CPU result analysis.
- CoRAL H200: MinerU/geometry preparation when required, teacher-data model
  inference, feature extraction, probe training, and GPU evaluation.
- Phase 1 may generate only authenticated teacher data. Phase 2 begins after SOL
  pushes the finalized probe implementation and exact execution handoff.

Execution authority: [shared-probe H200 handoff](../../../h200/task9-shared-probe/HANDOFF.md).
Detailed rationale is available on demand in the [research review](research/shared-probe-review-2026-09-03.md)
and [layer-trajectory amendment](research/shared-probe-layer-trajectory-2026-09-03.md).

## Task 5 — Analysis and claim boundaries

- Keep all uncertainty question-equal and cluster by support-document component
  where multiple observations share a component.
- Keep training/model selection isolated from primary-test results.
- Report direct downstream behavior separately from surrogate fidelity.
- A regional advantage shows privileged selection headroom in this local setup.
- Probe success shows decodability under the frozen probe and distribution. It
  does not establish that the backbone normally uses the decoded property.
- The local implementation is not the unpublished DocPrune author code.
- No result establishes grounding, multi-hop generalization, another model,
  another dataset, or natural-population performance unless separately tested.

## Change control

Scientific changes require explicit user approval before affected outcomes are
inspected. Record the dated amendment in this file. Status changes go in the
implementation plan; jobs, artifacts, failures, and results go in the relevant
experiment-log file. Do not redefine the experiment from chat history,
archived plans, implementation notes, or machine-specific workarounds.
