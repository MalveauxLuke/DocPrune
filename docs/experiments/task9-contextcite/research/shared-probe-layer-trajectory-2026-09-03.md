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
3. At every read layer evaluate the isolated geometry/region-metadata-only,
   native-DocPrune-only, and question-only controls, in addition to pooled
   region hidden state, compact final-prompt-query-to-region QK summaries, and
   hidden plus QK. Do not collapse geometry and native DocPrune into one
   combined shortcut control. Every feature is computed before any accepted or
   generated answer token is teacher-forced.
4. Select the best single-layer family using validation data only, with the
   selection rule frozen before primary-test access.
5. Compare that snapshot with a small local-trajectory linear model. For each
   eligible endpoint `r` in `3..13`, the local model receives standardized
   compact QK or hidden-plus-QK features at `r`, the adjacent difference
   `z_r-z_{r-1}`, and the mean of `z_{r-3}..z_r`. It never receives raw
   cross-layer hidden-state differences. Select its endpoint on validation
   only. If neither QK nor hidden-plus-QK has positive validation gold
   budget-local R², skip the trajectory comparison.
6. A trajectory claim requires improvement over the validation-selected best
   snapshot and matched controls: current-layer only, recent-layer mean,
   current plus adjacent delta, parameter-matched single-layer capacity,
   shuffled earlier-layer identity within question, and shuffled layer order.
   If ordering controls do not degrade performance, describe the result as
   multi-layer aggregation rather than trajectory information. If an isolated
   control wins snapshot selection, retain it as the selected baseline and do
   not claim a richer representation or escalate on that basis.
7. A full-prefix low-rank layer aggregation model is conditional. It may be
   implemented only if the local-trajectory model passes the frozen gate below.
   Do not begin with an RNN or transformer over layers.
8. Rank-4 question-region bilinear probing remains a conditional within-layer
   expressivity diagnostic; it is no longer the sole or automatic first
   escalation.

Gold and self heads remain separate at every layer. Baseline-wrong analysis
must directly test whether their predicted gold/self deletion deltas diverge.
Use ε=`0.05` nat/token and, at each layer, report the separate predicted gold
and self deltas, their signed difference, and the fraction with absolute
difference at least ε, separately on baseline-wrong and oracle-rescuable
slices. This is a descriptive diagnostic only; it cannot establish causality
or serve as a selection gate. Predictive trajectory performance alone does not
establish that the model balances correct and incorrect interpretations.

## Frozen metrics, controls, and escalation gates

The primary snapshot-selection metric is question-equal gold budget-local
mask-response R² on balanced validation. Differences within `0.005` are
tie-breaks, resolved in this order: baseline-wrong gold budget-local R², then
safe-deletion AUPRC, then earlier read block, then simpler feature family. The
same rule is frozen before primary-test access and is not changed for controls
or trajectory endpoints.

The full-prefix trigger is fixed: the local trajectory must improve gold
budget-local R² by at least `0.02` over the best overall snapshot, have a
support-component-bootstrap 95% lower bound above zero, degrade baseline-wrong
R² by no more than `0.02`, and beat both shuffled-history and shuffled-order
controls. If any condition fails, do not implement or interpret full-prefix
aggregation as warranted.

The rank-4 bilinear trigger is also fixed: Gate 0 must pass, and the best QK or
combined linear model must fail at least one of positive baseline-wrong gold
budget-local R² or Spearman `>= 0.30`. Otherwise keep the linear model and do
not escalate. The later deletion-boundary sweep remains unapproved.

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
