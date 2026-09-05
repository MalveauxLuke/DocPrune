# Task 9 H200 baseline-wrong confirmation design

## Goal

Prepare, but do not launch, a confirmation experiment on 100 new ordinary baseline-wrong questions. The H200 operator should only need to survey the machine, install/validate the environment, transfer the sealed inputs, run a smoke, and launch prepared jobs.

## Cohort

- Source the candidates from the authenticated 1,213-question Task 6 holdout, joining its cached unpruned result rows to its sealed eligibility records.
- Baseline wrong means canonical list exact match is zero for the fixed unpruned model output.
- Exclude every QID in the completed 48-question Task 9 pilot.
- Prefer one question per supporting document: hash-order support-document components, then hash-order one eligible question in each component. If fewer than 100 independent components exist, take one per component first and hash-order the remaining eligible questions; record the fallback and cluster inference by support component.
- Select 100 using a frozen string seed. Do not enrich for likely distractors.
- Reuse the exact four cached pages in their original order and with their original scores. Never invoke retrieval.

## Experiment

For every selected question, use the completed dynamic-DocPrune pilot procedure:

- unpruned answer;
- native dynamic threshold DocPrune at its native layer and budget;
- 256-mask gold-support regional ContextCite;
- 256-mask gold-margin ContextCite when the generated answer is a distinct non-gold alternative;
- deterministic region-size-aware random control;
- whole-region physical deletion at the same achieved token budget.

One intervention supplies both gold and generated-answer supervision. There are no global or budget-local holdout masks. Surrogate LDS/RMSE is not an endpoint in this confirmation cohort; downstream generated-answer outcomes are.

## H200 boundary

The local implementation creates code, sealed cohort/fixture tooling, launch templates, and handoff documents. It does not inspect or run the H200.

On the H200, the first agent action is a read-only environment survey followed by writing its findings into the provided notes template. CoRAL H200 rules are runtime authority; SOL instructions are historical context only.

## Verification scope

Run targeted tests only: deterministic cohort selection and exclusions, document/component preference and fallback, exact cached-page preservation, 256 fit masks with zero holdouts, fit-only arm selection, and launcher validation. Do not rerun the full project suite.
