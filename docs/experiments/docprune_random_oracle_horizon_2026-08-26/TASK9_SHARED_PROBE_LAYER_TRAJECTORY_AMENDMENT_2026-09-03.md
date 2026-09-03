# Task 9 shared-probe layer/trajectory amendment

Status: approved before cohort sealing, teacher generation, feature extraction,
probe training, or new-cohort outcome inspection

Source review SHA-256:
`c3b592268fd61b7d27591a371a64051a64b3c34bea0d08b0c652fa6870c55bc4`

## Decision

The shared-probe experiment keeps its 600-question cohort, 32-mask teacher
schedule, direct centered gold/self targets, and fixed physical-deletion
boundary `B13`. It changes the initial representation experiment from one
primary QK layer plus a rank-4 escalation to a controlled layerwise design.

The initial representation study now has this order:

1. Run the existing 48-question Gate 0 additivity/interaction analysis
   unchanged.
2. For each zero-based decoder block `r` in `0..13`, train the same
   low-capacity probe independently to predict the same B13 deletion outcomes.
   A read layer is where pre-answer features are obtained; it never changes the
   deletion boundary or label.
3. At every read layer evaluate four prespecified feature families:
   metadata plus native DocPrune controls, pooled region hidden state,
   compact final-prompt-query-to-region QK summaries, and hidden plus QK.
   Every feature is computed before any accepted or generated answer token is
   teacher-forced.
4. Select the best single-layer family using validation data only, with the
   selection rule frozen before primary-test access.
5. Compare that snapshot with a small local-trajectory linear model. For each
   eligible endpoint `r` in `3..13`, the local model receives standardized
   compact features at `r`, the adjacent difference `z_r-z_{r-1}`, and the
   mean of `z_{r-3}..z_r`. Select its endpoint on validation only.
6. A trajectory claim requires improvement over the validation-selected best
   snapshot and matched controls: current-layer only, recent-layer mean,
   current plus adjacent delta, parameter-matched single-layer capacity,
   shuffled earlier-layer identity within question, and shuffled layer order.
   If ordering controls do not degrade performance, describe the result as
   multi-layer aggregation rather than trajectory information.
7. A full-prefix low-rank layer aggregation model is conditional. It may be
   implemented only if the local-trajectory model passes its frozen validation
   gate. Do not begin with an RNN, transformer over layers, or raw cross-layer
   hidden-state differencing.
8. Rank-4 question-region bilinear probing remains a conditional within-layer
   expressivity diagnostic; it is no longer the sole or automatic first
   escalation.

Gold and self heads remain separate at every layer. Baseline-wrong analysis
must directly test whether their trajectories diverge; predictive trajectory
performance alone does not establish that the model balances correct and
incorrect interpretations.

## Later causal-boundary diagnostic

A small multi-boundary singleton/pair intervention study is scientifically
separate from the fixed-target read-layer sweep. It may test whether corrective
deletion effects diminish as distractor influence propagates, but it requires
a later outcome-blind cohort, exact boundary grid, compute envelope, and
explicit execution handoff. It is not authorized by this amendment and cannot
delay the initial fixed-B13 study.

## Unchanged constraints

- The preliminary 48 remains Gate 0 development data only.
- The locked confirmation 100 remains excluded from training and selection.
- The new cohort remains exactly 600 document-disjoint questions with the
  previously approved split counts.
- Retrieval, global-index access, feature rebuilding, SOL GPU work, and
  changing the B13 teacher target remain forbidden.
- The first small push still seals the cohort and admits only resumable H200
  preprocessing and teacher-data generation. Layerwise feature extraction,
  training, selection, and evaluation remain Phase 2 work.
