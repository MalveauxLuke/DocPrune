# Stage 03: One-Seed MiniVGent Screen

## Activation scope

Stage 03 compares two-block M0 and M1 with one registered seed on a
deterministic source/document-balanced sample of at most 10,000 eligible
training questions. Internal test and holdout remain sealed.

## Inputs

- passed Stage 02 completion report;
- frozen screen-sample and validation manifests;
- locked R0/R1 controls;
- byte-identical M0/M1 added-module initialization;
- registered optimizer, schedule, loss, threshold, and checkpoint selection;
  and
- exact compute budget.

## Procedure

1. Train two-block M0/M1 on identical rows, order, objective, optimizer, and
   schedule.
2. Evaluate both on the same validation candidate view.
3. Compare R1, M0, and M1 ranking metrics, anchor-set analysis, slices, and
   costs.
4. Interpret M0-R1 as the shared-page operational contrast and M1-M0 as the
   interaction ablation.
5. Decide two-versus-four-block confirmatory MiniVGent using only the frozen
   validation rule and resource fit.

## Outputs

- frozen screen sample;
- M0/M1 configs, checkpoints, predictions, metrics, and costs;
- paired M0/M1 and operational R1/M0 validation comparisons;
- selected confirmatory depth/config; and
- Stage 03 completion report.

## Pass and stop

Promote four blocks only under `gates.md`; otherwise retain two. Before Stage
04, freeze the exact three seeds, noninferiority/promotion rules, architecture,
model/input hashes, thresholds, optimizer/schedule, and checkpoint selection.

## Next-stage handoff

Stage 03 recommends but cannot launch confirmatory work. The control plane
reviews the registration and activates Stage 04 separately.
