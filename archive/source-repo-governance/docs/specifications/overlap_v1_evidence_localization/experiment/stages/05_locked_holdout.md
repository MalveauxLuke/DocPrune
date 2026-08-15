# Stage 05: Locked InfographicsVQA Robustness Evaluation

## Activation scope

Stage 05 opens `infographicsvqa_holdout` once for already selected
R0/R1/M0/M1 systems. It performs no training, selection, threshold fitting,
prompt repair, depth change, or architecture promotion.

## Inputs

- passed Stage 04 report and final model choice;
- exact Stage 04 model/checkpoint/config/prompt/processor/threshold hashes;
- immutable holdout source;
- frozen candidate/view construction rules; and
- approved create-once output root.

## Procedure

1. Verify all decision hashes equal Stage 04.
2. Build the holdout candidate/eligibility view under the same candidate
   definition and report its distinct inputs/revision/oracle.
3. Evaluate frozen selected system and required R0/R1/M0/M1 controls once.
4. Report eligibility, oracle, rankings, slices, representation diagnostics,
   failures, and costs.
5. Make no model or POC decision from holdout results.

## Outputs

- holdout candidate/view/oracle artifacts;
- frozen-system predictions and metrics;
- separately labeled robustness report; and
- Stage 05 completion report.

## Completion

Stage 05 ends the active answer-anchor POC. HierDoc/A1 answer-sufficiency work
requires the later experiment's own specification and authorization; it is not
Stage 06.
