# Task 9 Answer-Conditioned Oracle Pilot Preparation Handoff

Date: 2026-08-31
Status: CPU implementation and sealing authority only. This is not an `sbatch`
handoff and authorizes no pilot outcome.

## Objective and claim boundary

Prepare a controlled developmental comparison of an accepted-answer-
conditioned ContextCite whole-region selector against query-only and privilege-
matched gold-answer-conditioned attention at `B_13`. The experiment estimates
potential selection headroom and distractor-removal behavior. It is not vanilla
ContextCite, a token oracle, a fair comparison of deployable methods, evidence
ground truth, a population estimate, or a method-holdout experiment.

The original one-question Jaccard gate is superseded only as specified in the
canonical experiment plan. Exact support identity remains descriptive; global
and budget-local surrogate fidelity plus actual generated outcomes from the
budgeted selected sets are the relevant stability evidence.

## Evidence authorizing preparation

- B13 256+64 job `62423463`: completed and admitted all 320 masks; completion-
  manifest SHA-256
  `db46773f64c45fdbcbc9109af6f92458a96378ec177e6a3c61c978b61d0dec86`.
- Input 256+64 corrected job `62424211`: completed and admitted all 320 masks;
  completion-manifest SHA-256
  `8c2051e559064fcbc47ae317e123aaccb2cb685a3930c39c61b6ea9a6efe752e`.
- Accepted-answer B13: LDS `0.9148352`; RMSE `0.193681` versus constant
  `0.994279`; coefficient-refit Spearman mean/minimum
  `0.669753`/`0.574119`; selection Jaccard mean/minimum
  `0.707792`/`0.650000`; no solver warning.
- Moving deletion to `B_input` did not improve fidelity or selection stability,
  so B13 is frozen for the pilot.
- Review inputs have SHA-256
  `ffeaa63b8dbb0f989c31862beec838376176254c438d4d8f77631c16d990141b`
  and
  `64fac4a15f971390c64b337144a2453be9cfd5efe330eca582056d686c855699`.

## Preliminary stratified-random cohort contract

Run this pilot before the enriched cohort below. The eligible pool is the 245
authenticated single-hop questions with completed BTP+QTP/no-CTP outputs in
the sealed stage-localization analysis. Canonical list EM classifies 90 as
baseline correct and 155 as baseline wrong. With a recorded seed, sample 24
without replacement from each stratum. Seal source paths and hashes, pool
counts, ordered selections, and the 90/245 versus 155/245 natural-pool weights.
Selection must not read attention, ContextCite, or subsequent arm outcomes.

This cohort is now sealed at
`/scratch/lmalveau/docprune/task9-preliminary-random48-v1/cohort.json`. Seed is
`docprune-task9-preliminary-random48-v1`; file SHA-256 is
`123607a6a1226b4e3436f43cb82d45e64e8a3008e9ab6deefd7526efeabd0273`;
internal cohort SHA-256 is
`465fcf6e8e0adee6e79845db8cb1d6f1fbc97e01d02b7cf5c1e28c7c33c3c5f9`.
The selected set contains 48 unique QIDs, exactly 24 per stratum; natural-pool
weights are 90/245 correct and 155/245 wrong.

At identical achieved whole-region costs and 55/65/80% retention, run unpruned,
query-only DocPrune attention-region, gold-support ContextCite, conditional
gold-margin ContextCite using a distinct non-gold unpruned response, and one
deterministic region-size-aware random comparator. FastV remains excluded.
Report correct and wrong strata separately, paired normalized token-F1/EM,
win/tie/loss, rescue, preservation, gold-likelihood change, gold-margin change,
and question-clustered uncertainty. Any combined descriptive result uses the
frozen eligible-pool weights and is not labeled a dataset-wide population
estimate.

## Subsequent enriched cohort contract to implement

Only after the preliminary pilot, seal 48 unique questions from
the authenticated fixed-page eligible pool using a recorded seed:

1. 16 uniform-anchor questions sampled first;
2. 16 traceable distractor errors;
3. 8 high-ambiguity baseline-correct questions; and
4. 8 clean baseline-correct controls.

The latter three exclude the uniform-anchor members. Eligibility may use
accepted answers, already-observed unpruned answers, fixed retrieved pages,
document/region text and structure, and pre-ContextCite human audit. It may not
use attention scores, ContextCite coefficients/outcomes, or method contrasts.
Freeze semantic-type matching, ambiguity tier, clean-confidence threshold,
stratum precedence, insufficient-pool behavior, annotation schema, input file
hashes, ordered QIDs, and a balanced 16-question mask-count calibration subset.

This sampling design deliberately enriches mechanism-sensitive questions. All
results are reported by stratum. The 16-question uniform anchor is descriptive;
no pooled panel value is labeled a natural-distribution effect.

## Per-question perturbation contract

- Candidate universe: cached post-BTP/QTP visual tokens mapped 100% to whole
  MinerU regions or bounded residual cells.
- Boundary/operator: hard physical deletion at `B_13`; retain nonvisual states,
  surviving state order, and original Qwen M-RoPE positions.
- Target: maximum accepted-reference normalized full-sequence log-likelihood.
  Preserve per-reference rows. Generated-response likelihood remains secondary.
- Fit: 256 unique deterministic Bernoulli-0.5 masks using the pinned
  StandardScaler/Lasso interface.
- Validation: 32 unique global holdouts plus 32 unique validation-only masks
  within the primary 65% achieved budget `M′ ± 5%`.
- Robust score: elementwise median signed raw-space coefficient from five
  deterministic 80%-without-replacement refits of the 256 rows.
- Sensitivities: canonical all-256 coefficients, five individual refit sets,
  and nested 64/128/192/256 fits on the sealed calibration subset.
- Budgets: deterministic half-up 55%, 65% primary, and 80% requested retention;
  exact knapsack first maximizes common attainable whole-region cost `M′ <= M`
  and then maximizes each arm's score.

## Controlled arms

Run on identical pages, mappings, region costs, boundary, achieved budget,
positions, decoder, generation, and evaluator:

1. unpruned reference;
2. query-only aggregate-logit DocPrune attention aggregated by region;
3. gold-answer-conditioned answer-token attention aggregated by region;
4. robust-median accepted-answer ContextCite;
5. canonical all-256 accepted-answer ContextCite sensitivity;
6. reverse robust ContextCite manipulation check;
7. five refit-selected ContextCite stability sets; and
8. traceable-only audited gold-in/distractor-out constraint.

Literal query attention, mean-over-member-token gold attention, and mask-count
fits are named sensitivities. Do not add FastV or a new random/coverage family
to the enriched panel. The preliminary pilot's single matched regional random
comparator is separate. The existing 1,213-question Task 6 random-versus-
DocPrune evidence is not recomputed for Task 9.

## Required analysis

The unified analysis JSON must include all question/stratum/arm/budget rows,
source and artifact hashes, metric definitions in plain language, and these
contrasts at 65% with 55%/80% sensitivities:

- generated-answer F1/EM and ContextCite minus query-attention headroom;
- gold-attention minus query-attention privilege contribution;
- ContextCite minus gold-attention intervention contribution;
- accepted-answer likelihood, retained tokens, decoder time, and peak memory;
- global and primary-budget-local LDS/Spearman and RMSE versus constant;
- robust/canonical/refit set overlap and direct budgeted selection regret;
- traceable-case rescue, normalized gold-vs-wrong likelihood-margin change,
  and gold/distractor retained/removed state; and
- baseline-correct harm rate.

Use paired support-document-component bootstrap intervals and the existing
`±1.0 F1` practical margin descriptively. Never convert this enriched
developmental result into a confirmatory or population claim.

## Implementation and execution boundary

Authorized now:

- write CPU code/tests, cohort and mask seals, schemas, validators, and analysis;
- use only fixed cached pages/features; and
- run a bounded real-input correctness/cost smoke after CPU verification.

Git consolidation is complete on `codex/task9-analysis-cpu-20260828`: Task 6
merge `689432e`, Task 7 merge `c5b212f`, and Task 8 merge `c38cd9c` contain all
unique accepted Task 6–9 work. No pilot job was submitted.

Not authorized now:

- either 48-question pilot submission;
- a method holdout or larger cohort;
- retrieval, global-index loading, or feature rebuilding;
- inspecting partial pilot outcomes;
- FastV, random/coverage reruns, `B_input`, or Task 10; or
- reusing any completed Task 9 launcher as submission authority.

The later executable handoff must bind a clean consolidated commit, sealed
cohort/masks/mappings, launcher and input hashes, fresh no-replace roots, and a
terminal validator. Planned HTC shape is four-question batch jobs on first-
available suitable CUDA GPUs with recorded GPU identity and per-question
isolated outputs, 24 GiB host RAM, no mask-level sharding, no requeue, no
cross-family absolute timing comparison, and short independently
recoverable jobs. Exact wall time, questions per shard, concurrency, and job
IDs remain unset until the bounded smoke measures the final implementation.

## Immediate next steps

1. Implement the sealed preliminary cohort's mapping/mask inputs and matched
   selectors test-first.
2. Implement the direct selected-set runner, validator, and
   unified analysis JSON test-first.
3. Run CPU verification and one bounded first-available-GPU smoke.
4. Write and review a new executable preliminary-pilot handoff; only then request
   submission.
