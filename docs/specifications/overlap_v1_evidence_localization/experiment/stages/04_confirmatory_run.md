# Stage 04: Three-Seed Confirmatory POC

## Activation scope

Stage 04 trains promoted M0/M1 on all eligible training questions for three
registered seeds and opens internal test exactly once after every POC decision
is frozen.

## Inputs

- passed Stage 03 completion report;
- immutable confirmatory registration hash;
- full eligible train/validation and sealed internal-test manifests;
- locked R0/R1 controls;
- selected MiniVGent depth and exact three seeds;
- frozen model/input/loss/optimizer/schedule/threshold/checkpoint rules; and
- approved compute/run roots.

## Procedure

1. Verify every registration/predecessor hash.
2. Train M0/M1 for each seed from matched per-seed initialization.
3. Select each checkpoint under the frozen validation rule.
4. Lock checkpoints and any score-to-set threshold.
5. Record internal-test opening, then create the first R0/R1/M0/M1 test
   predictions on the byte-identical view.
6. Compute per-seed, aggregate, paired, sliced, cost, and document-clustered
   results.
7. Apply the registered interaction and R1 noninferiority gates.
8. Publish deployment recommendation and negative/inconclusive results without
   changing the experiment.

## Outputs

- per-seed locks, logs, checkpoints, predictions, and metrics;
- registration and internal-test-open audit;
- `delta_pairwise`, `delta_shared_memory`, and `delta_interaction` results;
- MiniVGent promotion decision;
- selected deployment system;
- Git-safe primary report; and
- Stage 04 completion report.

## Completion

Integrity pass means the result is reportable even if MiniVGent loses. Any
post-opening change is a new exploratory experiment, not a confirmatory rerun.
The control plane may separately activate optional Stage 05 after choices are
immutable.
