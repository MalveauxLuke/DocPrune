# Task 9 nested-mask diagnostic analysis

Date: 2026-09-02

Status: CPU evidence package for review; no canonical plan/log changes and no launch authorization

## Executive summary

The present 48-question development sample does **not** support adopting a strict 192-stage adaptive escalation rule. The 192-mask result remains scientifically useful, but the evidence is too sparse and the observed rescue failure is not reliably identifiable from diagnostics available after 192 masks.

The formal reconstruction agrees with the completed-run summary:

- Repeat 0 preserves 24/24 baseline-correct answers, but only 3/4 canonical 256-mask rescues. Wrong-stratum mean token F1 is 0.295 versus 0.38125 at 256, a loss of 0.08625.
- Across all five development repeats, baseline-correct preservation is 120/120, rescue preservation is 16/20, wrong-stratum mean token F1 is 0.32333, and the mean accepted-answer likelihood change relative to 256 is -0.01391 nats/token.
- The four exact-match rescue losses are four repeats of **one question**, `e5d9ec29613dce539c3c381ff17bb77a`. Repeat 2 of the same question succeeds at 192. Consequently, a leave-one-question-out model cannot estimate recall for the critical rescue-loss endpoint: holding out that question removes every positive example from training.
- A broader functional label, any token-F1 loss relative to 256, supplies only 11 positive question/repeat rows from three distinct questions. Grouped leave-one-question-out screening is possible for that endpoint, but the rules have low precision and weak compute savings.
- The rescue-sensitive question does not look unreliable by ordinary 192-stage fidelity or stability measures. On its four failing repeats, global LDS is 0.887--0.908, budget-local LDS is 0.951--0.967, global RMSE/constant-RMSE is 0.427--0.449, and the true nested 128-to-192 token-weighted selection Jaccard is 0.950--0.966. Even the unavailable 192-to-256 Jaccard is 0.950--0.981. The observed failure therefore occurs despite apparently strong fidelity and stability.
- The least expensive strict grouped-CV rule that detects all four observed critical-loss repeats is high source count. It escalates 28/48 repeat-0 questions (58.3%), has only 2/3 repeat-0 any-F1-loss recall and 7.1% precision, and projects to 355,813 evaluations. Low Lasso nonzero fraction catches more of the broader losses across repeats (9/11 rather than 6/11), but escalates 31/48 repeat-0 questions and still has only 6.5% precision. Rules with 3/3 repeat-0 any-F1-loss recall escalate at least 34/48 (70.8%) and do not retain stable critical recall across the five repeats.
- Two outcome-aware rules reproduce 256 performance in this balanced pilot while saving compute: escalate all baseline-wrong questions (50% escalated; 349,344 projected evaluations) or escalate when the generated 192 answer has EM=0 (43.75%; 344,492). Both use accepted-answer outcomes and are privileged. The latter is especially circular as evidence for a generally usable uncertainty diagnostic. Neither should be presented as a strict 192-fit reliability rule.

The appropriate current decision is therefore: retain 256 as the supported configuration for the rescue-sensitive experiment; do not claim that the development data validate adaptive escalation. If an answer-privileged research-only allocation is scientifically acceptable, baseline-wrong escalation is a transparent contingency worth separate review, but it changes the claim boundary.

## Scope and estimand

This analysis concerns the adapted, accepted-answer-conditioned, region-level ContextCite oracle at each question's independently selected decoder layer and token budget. It is not vanilla ContextCite, a deployable selector, or a token-level oracle.

The proposed nested estimator has the following intended evaluation schedule:

1. Evaluate a pre-specified 192-row fitting prefix.
2. Fit and calculate a reliability diagnostic using only information then available.
3. Evaluate 64 additional fitting masks only for escalated questions.
4. Refit on all 256 rows, reusing the original 192 rows exactly.
5. Retain the existing 64 held-out masks for fidelity measurement.

Thus a non-escalated question costs 192 fitting + 64 held-out = 256 intervention evaluations, while an escalated question costs 320. For 1,213 questions,

`total evaluations = 1,213 * 256 + 64 * number_escalated`.

The existing implementation already caches the vision pass and decoder prefix through the intervention boundary once per question, then branches masks from the cached boundary. Accepted-answer and generated-answer targets for a mask are obtained together. Branch batching may improve wall-clock utilization, but there is no unimplemented prefix-caching saving to substitute for the statistical question here; see `src/docprune/qwen2vl/model.py` and `src/docprune/qwen2vl/decoder.py`.

## Provenance and artifact validation

Analysis was performed in the requested clean worktree at commit `356f11842916243308f67c0134254f988041f5e8`. No GPU job was launched.

### Authoritative inputs

| Artifact | Whole-file SHA-256 | Internal sealed digest | Validation |
|---|---:|---:|---|
| `/scratch/lmalveau/docprune/task9-preliminary-random48-v1/cohort.json` | `123607a6a1226b4e3436f43cb82d45e64e8a3008e9ab6deefd7526efeabd0273` | `465fcf6e8e0adee6e79845db8cb1d6f1fbc97e01d02b7cf5c1e28c7c33c3c5f9` | Internal digest recomputed; 48 unique QIDs |
| `/scratch/lmalveau/docprune/task9-preliminary-dynamic48-90f27d7-unified-v1/analysis.json` | `541327e8fed25be50d00e6ab0716697c3a39611ae286389b21da8a8c681cc526` | `985a837b2094b5a925730eeb4b9bf07c594344f9021c4a718c49c11aa655a8c5` | Internal digest recomputed; 48 questions |
| `/scratch/lmalveau/docprune/task9-mask-count-ablation-d6de9d8-v1/analysis.json` | `a03a7d4c8db697486d4c16bc6c8540d185c861dd2a97d096f5909d6003d5746a` | `9d6a9ac6ece6712ca9cdad33bba0bc61caf122121435c3d99a6fb40e460edc51` | Internal digest recomputed; 48 questions |
| `/scratch/lmalveau/docprune/task9-mask192-generation-356f118-v1` | Directory | Per-file/per-result digests | 12/12 completion manifests, 48/48 question artifacts, and 192 newly generated unique selected-arm results validated |

The QID sets and ordinals match exactly across cohort, canonical pilot, ablation, and generated-response artifacts. Every one of the 48 ablation question artifacts was read. All 21 fits per question (64/96/128/192, five repeats each, plus canonical 256) passed internal digest validation. Every generated question artifact, completion manifest, selected-arm binding, and result digest was independently checked. The generated artifacts report runtime commit `356f11842916243308f67c0134254f988041f5e8` and completed status. Reused selections and duplicate selections across repeats were resolved through their sealed selection bindings rather than counted as additional GPU generations.

The joined analysis contains 240 rows: 48 questions x five 192-mask development repeats. A CPU replay of the documented `StandardScaler` + `Lasso(alpha=0.01, random_state=0)` fit and canonical lexical-region knapsack reproduced the sealed 192 selection exactly on 240/240 rows. This is an important independent check that the joined diagnostics refer to the same fitted selections whose generated outcomes were scored.

Five global LDS values, five budget-local LDS values, and five coefficient-rank correlations are mathematically undefined because a vector is constant. They are treated as explicit reliability warnings for low-correlation screens, not silently dropped or imputed as ordinary correlations.

### Nested-prefix caveat

The sealed ablation's 64/96/128/192 samples are independent, SHA-256-seeded subsets without replacement. They are development repeats, not a nested full-scale schedule. Therefore, the pre-existing 64/96/128 fits cannot all be described as prefixes of a given 192 fit.

For this analysis only, each sealed 192 subset was sorted by canonical mask index and its first 64, 96, and 128 rows were refit on CPU. This produced 720 genuinely nested exploratory fits (48 x five repeats x three prefix sizes). The convention was chosen after outcomes existed, so these nested stability features are **exploratory**, not a pre-registered validation of a production mask order. A future experiment must freeze the 256-row order and prefix rule before outcomes.

## Metrics and failure labels

All functional comparisons hold question, decoder layer, token budget, prompt, generation settings, and canonical 256 arm fixed. Only the fitting-mask sample changes.

### Outcome and fidelity metrics

- **EM and token F1:** normalized generated-answer exact match and token F1 already used by the Task 9 pilot.
- **Gold likelihood:** mean accepted-answer log likelihood in nats/token. Deltas below are 192/policy minus canonical 256, so negative is worse.
- **Global LDS:** Spearman correlation between held-out intervention targets and surrogate predictions across all 64 holdouts.
- **Budget-local LDS:** the corresponding correlation in the native-budget-local held-out slice.
- **RMSE ratio:** held-out surrogate RMSE divided by the RMSE of a constant-mean predictor. Values below 1 beat the constant baseline.
- **Selection stability:** token-cost-weighted Jaccard of retained region sets. This measures selection overlap, not answer equivalence.
- **Coefficient stability:** Spearman rank correlation of region coefficients.
- **Lasso behavior:** nonzero/positive/negative fractions, L1 mass, top-five absolute-coefficient concentration, maximum coefficient, and solver iterations.
- **Boundary/geometry:** selected-versus-unselected coefficient gap, region/source count, selected fraction, region-cost CV, maximum region cost share, retention fraction, and budget gap.

### Failure hierarchy

The following labels must not be conflated:

1. **Selection change:** the 192 and 256 retained region sets differ. This occurs on 208/240 rows (86.7%) and is common; it is not itself a functional failure.
2. **Any F1 loss:** 192 token F1 is strictly below canonical 256 token F1. This is the primary broader functional screening label: 11/240 rows from three QIDs. All 11 losses are at least 0.05 F1.
3. **Critical EM/rescue loss:** canonical 256 has EM=1 but 192 has EM=0. This is the rescue-sensitive label: 4/240 rows, all from one QID.
4. **Aggregate frozen-gate failure:** the five-repeat summaries fail the pre-specified rescue-preservation and wrong-stratum F1 gates. This is a study-level decision, not a row-level classification label.
5. **Gold drop >=0.05 nats/token:** 27/240 rows. This is a continuous-likelihood warning, not equivalent to generated-answer failure.

The critical label is the endpoint that matters most for the scientific motivation, but it is not question-level cross-validatable here. Treating four repeats of one question as four independent positives would be pseudo-replication and would substantially overstate evidence.

## Reconstructed 192-versus-256 results

### Frozen functional gate

| Fit | Baseline-correct preserved | Canonical rescues preserved | Wrong-stratum mean F1 | F1 delta vs 256 | Mean gold-likelihood delta vs 256 |
|---|---:|---:|---:|---:|---:|
| 192 repeat 0 | 24/24 | 3/4 | 0.29500 | -0.08625 | -0.01209 |
| 192 repeat 1 | 24/24 | 3/4 | 0.32833 | -0.05292 | -0.00996 |
| 192 repeat 2 | 24/24 | 4/4 | 0.33667 | -0.04458 | -0.00978 |
| 192 repeat 3 | 24/24 | 3/4 | 0.32833 | -0.05292 | -0.01850 |
| 192 repeat 4 | 24/24 | 3/4 | 0.32833 | -0.05292 | -0.01923 |
| 192, pooled five repeats | 120/120 | 16/20 | 0.32333 | -0.05792 | -0.01391 |
| Canonical 256 | 120/120 | 20/20 | 0.38125 | 0 | 0 |

The independent reconstruction therefore confirms that 192 passes the baseline-correct and gold-likelihood portions but fails rescue preservation (16/20 versus required 18/20) and wrong-stratum F1. The repeat-0 result also fails by losing one of the four exact rescues.

### Failure concentration

| Ordinal / QID | Canonical 256 F1/EM | 192 repeats with F1 loss | 192 F1 on losing repeats | Interpretation |
|---|---:|---:|---:|---|
| 34 / `2f34ec15d30206535baa99d709903307` | 0.27 / 0 | 0,1,2,3,4 | 0.00 each | Partial-answer gain from 256 lost in all repeats |
| 36 / `e5d9ec29613dce539c3c381ff17bb77a` | 1.00 / 1 | 0,1,3,4 | 0.00 each | The only critical rescue-loss QID; repeat 2 remains 1.00/1 |
| 44 / `eea285074866dd36b5f5a7333b74d09a` | 0.80 / 0 | 0,2 | 0.00 each | Large partial-answer gain lost in two repeats |

The other canonical rescues are stable at 192 on all five repeats:

| Ordinal / QID | 192 EM across repeats 0--4 |
|---|---|
| 29 / `d82d381af11d76d8a70601251ecb8ca5` | 1/1/1/1/1 |
| 37 / `1db882f16ee4721cc4c3fe9cf55e07e0` | 1/1/1/1/1 |
| 43 / `77240911edd7c3cc1d48a0623e54d32d` | 1/1/1/1/1 |

Raw per-question evidence is available in the matching `batch-NN/OO-QID.json` files under both the ablation and generated-response roots above. The QID and ordinal table is sufficient to resolve every affected artifact without relying on aggregate row order.

## Diagnostic availability and leakage classes

The adapted Task 9 oracle is already answer-conditioned. “Strict” below therefore means outcome-blind **within this oracle experiment**: it uses no 256 fit, no selected-arm generated answer, and no reference-scored outcome beyond the 192 fitting intervention targets. It does not mean deployable in a setting where accepted answers are unknown.

| Diagnostic family | Available after 192 fitting masks? | Class / caution |
|---|---|---|
| Global/local held-out LDS and RMSE ratios | Yes, if the fixed 64 holdouts are evaluated | Strict within Task 9; consumes the already-budgeted holdouts |
| 192 Lasso sparsity, concentration, solver behavior | Yes | Strict within Task 9 |
| Region count, size/cost distribution, retention and budget pressure | Yes, mostly before fitting | Strict within Task 9 |
| True nested 64/96/128-to-192 fit, rank, and selection stability | Yes under a pre-specified prefix order | Strict in principle; exploratory here because the nested order was reconstructed post hoc |
| Existing independent 64/96/128-to-192 disagreement | Already available in development artifacts | Not a future nested-stage diagnostic because those smaller samples are not prefixes of the 192 set |
| Accepted-answer likelihood on the selected 192 arm | Available only after evaluating that selected arm | Outcome-aware/privileged; additional inference semantics beyond fit diagnostics |
| Generated 192 answer, EM, F1, or comparison with native DocPrune | Available after generation and reference scoring | Privileged outcome-aware rule; not an internal reliability diagnostic |
| Unpruned baseline-correct/wrong stratum | Known in this benchmark from accepted answers | Privileged, though potentially usable for research allocation if declared |
| 192-to-256 coefficient/selection stability | No | Hindsight only; cannot decide whether to acquire the missing 64 rows |

No analysis below quietly combines these classes. Hindsight and outcome-aware results are included as ceilings or contingencies, not evidence for a strict estimator.

## Cross-validated diagnostic results

### Procedure

For each scalar candidate and direction, a threshold was fitted separately in each leave-one-question-out fold. All five repeats of the held-out question were excluded from threshold selection. On the other 47 questions, the threshold was chosen to minimize escalation while targeting 100% training recall for the broader any-F1-loss endpoint. Reported repeat-0 metrics then use exactly one pre-specified 192 result per question; the five-repeat results are a robustness view, not 240 independent questions.

This procedure is deliberately simple because 48 questions and only three positive QIDs do not justify a multivariable classifier. A multivariable fit would add researcher degrees of freedom and severe separation/overfitting risk. There is no valid analogous procedure for critical EM loss because its only positive QID disappears in the held-out fold.

### Strict and exploratory screens

The table gives the most informative grouped-CV rules. Recall and precision are for repeat-0 any-F1 loss (three positive questions). “Critical caught” refers only to the one repeat-0 critical loss and must not be interpreted as an estimated population recall.

| Escalation signal | Class | Escalated | Any-F1 recall | Precision | Critical caught | Rescues | Wrong F1 | Evals/q | Projected evals |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Coefficient boundary gap low | Strict | 48/48 (100%) | 3/3 | 6.3% | 1/1 | 4/4 | 0.38125 | 320.0 | 388,160 |
| Nested 128-to-192 coefficient Spearman low | Exploratory strict | 42/48 (87.5%) | 3/3 | 7.1% | 1/1 | 4/4 | 0.38125 | 312.0 | 378,456 |
| 192 top-five absolute-coefficient share low | Strict | 34/48 (70.8%) | 3/3 | 8.8% | 1/1 | 4/4 | 0.38125 | 301.3 | 365,517 |
| 192 nonzero-coefficient fraction low | Strict | 31/48 (64.6%) | 2/3 | 6.5% | 1/1 | 4/4 | 0.34792 | 297.3 | 360,665 |
| Source count high | Strict | 28/48 (58.3%) | 2/3 | 7.1% | 1/1 | 4/4 | 0.37000 | 293.3 | 355,813 |
| Maximum region-cost share high | Strict | 21/48 (43.8%) | 2/3 | 9.5% | 0/1 | 3/4 | 0.33958 | 284.0 | 344,492 |
| Global RMSE ratio high | Strict | 38/48 (79.2%) | 2/3 | 5.3% | 1/1 | 4/4 | 0.34792 | 306.7 | 371,987 |
| Budget-local RMSE ratio high | Strict | 37/48 (77.1%) | 2/3 | 5.4% | 1/1 | 4/4 | 0.34792 | 305.3 | 370,369 |
| Global LDS low | Strict | 41/48 (85.4%) | 2/3 | 4.9% | 1/1 | 4/4 | 0.34792 | 310.7 | 376,839 |
| Budget-local LDS low | Strict | 46/48 (95.8%) | 2/3 | 4.3% | 0/1 | 3/4 | 0.33958 | 317.3 | 384,925 |
| Nested 128-to-192 selection Jaccard low | Exploratory strict | 35/48 (72.9%) | 2/3 | 5.7% | 0/1 | 3/4 | 0.33958 | 302.7 | 367,135 |

Several patterns rule out a positive claim:

- The nominal boundary-gap rule degenerates to always-256 because the Lasso/knapsack boundary is zero for almost every row. It is not a useful diagnostic.
- Rules achieving 3/3 repeat-0 any-F1 recall require 70.8%--100% escalation. The cheapest of these saves only 22,643 of 388,160 evaluations (5.8%). Across all five repeats, that top-five-concentration rule catches 9/11 F1 losses and only 2/4 critical losses, preserving 18/20 rescues.
- The lower-cost strict nonzero-fraction screen catches all four observed critical-loss rows across repeats but only 9/11 broader losses, at 145/240 escalations (60.4%) and 6.2% precision. Because all four critical rows are one question, this apparent 4/4 recall is not independent evidence.
- Standard fidelity alarms are inefficient: global/local LDS or RMSE requires 77%--96% escalation while still missing one repeat-0 F1-loss question. Budget-local LDS actually misses the critical question under its grouped-CV threshold despite escalating 46/48.
- Nested selection instability is not a rescue detector here. Its rule misses the critical question while escalating 35/48.

### Why the critical failure evades ordinary uncertainty measures

The four critical-loss rows look **more**, not less, stable than typical noncritical rows:

| Feature | Critical-loss range/mean | Noncritical median | Expected alarm direction |
|---|---:|---:|---|
| Global LDS | 0.887--0.908; mean 0.893 | 0.823 | Low |
| Budget-local LDS | 0.951--0.967; mean 0.958 | 0.664 | Low |
| Global RMSE ratio | 0.427--0.449; mean 0.440 | 0.537 | High |
| Budget-local RMSE ratio | 0.392--0.473; mean 0.418 | 0.460 | High |
| Nested 128-to-192 selection Jaccard | 0.950--0.966; mean 0.957 | 0.892 | Low |
| Nested 128-to-192 coefficient Spearman | 0.835--0.909; mean 0.862 | 0.814 | Low |
| Hindsight 192-to-256 selection Jaccard | 0.950--0.981; mean 0.958 | 0.951 | Low |

The only conspicuous scalar is a low 192 Lasso nonzero fraction (0.123--0.169, versus noncritical median 0.413). That feature is question/fit-specific rather than a direct uncertainty estimate, and its threshold is extremely sensitive in this sample. It may be a useful hypothesis, but it is not validated.

### Threshold sensitivity

The following is an intentionally **in-sample descriptive** sweep of `nonzero_fraction <= threshold`, included to expose the cliff rather than to select a threshold. The critical rows all belong to one QID.

| Threshold | Escalated, all repeats | Any-F1 recall | Precision | Critical rows caught | Repeat-0 escalated | Repeat-0 rescues | Repeat-0 wrong F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.125 | 26/240 (10.8%) | 1/11 | 3.8% | 1/4 | 5/48 | 3/4 | 0.29500 |
| 0.150 | 33/240 (13.8%) | 2/11 | 6.1% | 2/4 | 7/48 | 3/4 | 0.29500 |
| 0.175 | 41/240 (17.1%) | 4/11 | 9.8% | 4/4 | 9/48 | 4/4 | 0.33667 |
| 0.200 | 52/240 (21.7%) | 4/11 | 7.7% | 4/4 | 11/48 | 4/4 | 0.33667 |
| 0.300 | 98/240 (40.8%) | 4/11 | 4.1% | 4/4 | 20/48 | 4/4 | 0.33667 |
| 0.400 | 118/240 (49.2%) | 7/11 | 5.9% | 4/4 | 23/48 | 4/4 | 0.34792 |

A change from 0.150 to 0.175 appears to restore the rescue, but that is simply the threshold crossing the same question's coefficient fraction. It provides no evidence that 0.175 will catch a different rescue-sensitive question. The grouped-CV 100%-training-recall threshold varies from 0.431 to 0.462 and escalates 31/48 repeat-0 questions, illustrating how conservative threshold selection becomes with only two positive training QIDs in some folds.

## Compute/performance Pareto view

The primary table uses repeat 0 because it is the pre-specified 192 subset and gives one observation per question. Projected totals apply the observed balanced-pilot escalation fraction to 1,213 questions and round to the nearest evaluation. Expected fitting masks exclude the fixed 64 holdouts; evaluations include them.

| Policy | Leakage class | Escalated | Fit masks/q | Evals/q | Projected total | Rescues | Baseline-correct | Wrong F1 | Gold delta vs 256 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Always 192 | Fixed | 0/48 | 192.0 | 256.0 | 310,528 | 3/4 | 24/24 | 0.29500 | -0.01209 |
| Max region-cost share high | Strict grouped CV | 21/48 | 220.0 | 284.0 | 344,492 | 3/4 | 24/24 | 0.33958 | -0.00159 |
| Source count high | Strict grouped CV | 28/48 | 229.3 | 293.3 | 355,813 | 4/4 | 24/24 | 0.37000 | -0.00530 |
| Nonzero fraction low | Strict grouped CV | 31/48 | 233.3 | 297.3 | 360,665 | 4/4 | 24/24 | 0.34792 | -0.00627 |
| Top-five coefficient share low | Strict grouped CV | 34/48 | 237.3 | 301.3 | 365,517 | 4/4 | 24/24 | 0.38125 | -0.00037 |
| Always 256 | Fixed | 48/48 | 256.0 | 320.0 | 388,160 | 4/4 | 24/24 | 0.38125 | 0 |
| Escalate baseline-wrong | Privileged stratum | 24/48 | 224.0 | 288.0 | 349,344 | 4/4 | 24/24 | 0.38125 | +0.00102 |
| Escalate if generated 192 EM=0 | Privileged outcome | 21/48 | 220.0 | 284.0 | 344,492 | 4/4 | 24/24 | 0.38125 | -0.00080 |

The strict rows are not a validated Pareto frontier: they are grouped-CV screens on the same small development cohort and differ in which failures they miss across repeats. They are shown to quantify the trade-off, not to nominate a winner.

Baseline-wrong escalation exactly restores canonical wrong-stratum behavior by construction while retaining the empirically safe 192 arm on baseline-correct questions. Its 50% rate reflects the deliberately balanced 24/24 pilot. If the eligible-pool baseline-wrong fraction 155/245 (63.3%) were used instead, the rough projection would be 232.49 fit masks and 296.49 evaluations/question, or about 359,642 evaluations total. This is a prevalence sensitivity calculation, not a full-cohort validation.

Generated-192-EM escalation is even more privileged: it requires generation and accepted-answer scoring at the 192 arm, and the development policy escalates precisely the observed incorrect answers. Its apparent perfect rescue preservation is therefore not evidence of calibrated, outcome-blind uncertainty. It is a research-only upper-bound contingency.

## Leakage and claim boundaries

Four boundaries should remain explicit in any follow-up reasoning:

1. **Oracle boundary.** Every fit target is accepted-answer likelihood. Even strict diagnostics are only strict relative to the 192-stage oracle and do not make ContextCite deployable.
2. **Outcome boundary.** Baseline strata, selected-arm EM/F1, and selected-arm accepted likelihood are privileged. They can support a clearly labeled research allocation but cannot be mixed into an outcome-blind reliability claim.
3. **Future-data boundary.** Any comparison against 256 coefficients or selection is hindsight. It diagnoses failure mechanisms but cannot trigger acquisition of the missing 64 rows.
4. **Design boundary.** The post-hoc sorted-index nested prefixes were not the sealed ablation design. Their stability results may generate a hypothesis only. A future nested estimator must pre-specify mask order, prefix sizes, rule, threshold, and tie handling.

The five 192 repeats estimate sensitivity to the random fitting subset; they are not five independent full-scale fits and would not all be run for each of 1,213 questions. Performance projection must use a single frozen repeat/prefix policy.

## Limitations

- There are only 48 questions, intentionally balanced 24 baseline-correct/24 baseline-wrong. This is not the natural full-scale prevalence.
- There is one critical-failure QID and three any-F1-loss QIDs. Critical recall, precision, and threshold calibration are not estimable with question-level holdout.
- The same rescue-sensitive QID both motivates and visually separates the low-nonzero-fraction hypothesis. Any fixed threshold chosen now is outcome-adaptive.
- The development repeats share question, document, regions, layer, and budget. Treating them as independent narrows uncertainty spuriously.
- Generated-answer metrics are discontinuous. Very high coefficient/selection overlap can still cross an answer-generation boundary, as the critical QID demonstrates.
- The 64 holdouts provide surrogate-fidelity evidence, not direct uncertainty about the discrete generated answer after knapsack selection.
- Scalar screens were favored to limit overfitting. The sample cannot support a credible multivariable learner, interactions, or subgroup-specific thresholds.
- The analysis does not estimate GPU wall-clock savings. It addresses intervention counts; caching is already present, and branch batching is orthogonal.
- No confidence interval can rescue the identification problem: with one critical QID, uncertainty is structural, not merely a matter of using a different interval formula.

## Recommendation and bounded next experiment

### Current recommendation

Do not replace canonical 256 with a claimed strict adaptive 192-to-256 estimator on the present evidence. Use always-256 for the rescue-sensitive enriched pilot unless a reviewer explicitly accepts a privileged research-only allocation. Do not launch the full-scale experiment, method holdout, or enriched pilot from this analysis alone.

If reducing cost is mandatory before further evidence, the most transparent contingency is not a fitted “uncertainty” model. It is the declared privileged rule “256 for baseline-wrong; 192 for baseline-correct,” because it targets the stratum containing all observed rescues and because 192 preserved 120/120 baseline-correct outcomes across development repeats. This remains provisional: the zero observed harms do not prove zero future harm, and natural-prevalence projection must replace the balanced 50% rate.

### Bounded diagnostic-validation experiment

A useful next experiment should be separate from the method holdout and pre-registered before any outcomes:

1. Freeze one canonical 256-row mask order per question and define the 192 stage as its first 192 rows. Keep the existing 64 holdouts disjoint.
2. Freeze no more than two strict candidate rules. Based on this analysis, a low-nonzero-fraction alarm may be carried only as a hypothesis; pair it with a fidelity/stability veto or comparator rather than tuning many alternatives.
3. Acquire all remaining 64 fitting rows for every question in a new development diagnostic cohort, not only predicted escalations, so failure labels are observable without verification bias. Generate and score both 192 and 256 selected arms.
4. Hold questions out as groups. Report repeat-0/single-prefix performance as primary and any extra random prefixes as sensitivity only.
5. Pre-specify a stop rule: do not advance an adaptive policy unless there are enough distinct failure QIDs to estimate rescue-sensitive recall. A minimum of 10 distinct critical failures would still give wide uncertainty; at the observed one-in-48 rate, roughly 480 unselected development questions would be needed in expectation, which is likely too expensive. A smaller 48--64-question baseline-wrong diagnostic cohort can test mechanism but is unlikely to validate population recall.
6. If event accrual remains sparse, stop searching for a learned diagnostic and report that rescue failures are not predictably separable at this sample/cost. Retain 256 or use only a transparent privileged allocation.

The key design trade-off is unavoidable: evaluating the missing 64 rows for all questions is necessary in the bounded validation cohort to learn whether a proposed escalation rule missed failures. Running them only when the rule escalates would make false negatives unobservable.

## Reproduction notes and relevant code

The analysis joined sealed records by QID, repeat, fit digest, and selection digest rather than positional assumptions. It recomputed internal JSON digests using the repository's canonical serialization convention, replayed the exact sklearn fit/knapsack selection, and derived outcome deltas from the sealed generated selected-arm results.

Relevant repository code:

- `examples/analyze_task9_mask_count_ablation.py`
- `examples/aggregate_task9_mask_count_ablation.py`
- `src/docprune/task9_attribution.py`
- `examples/aggregate_task9_preliminary_pilot.py`
- `src/docprune/qwen2vl/model.py`
- `src/docprune/qwen2vl/decoder.py`

Authoritative artifact roots and the QIDs listed above are the durable evidence trail for independent follow-up. No canonical experiment plan or experiment log was modified by this analysis.
