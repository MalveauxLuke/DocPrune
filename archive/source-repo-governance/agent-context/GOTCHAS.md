# Gotchas

- Do not confuse the control checkout, immutable V1 package, and future POC
  experiment worktree.
- Do not reuse `~/Evidence-DINO-Units` or its scratch root for the POC.
- Do not compute, install, download, hash, or extract on a login node.
- Do not treat scratch as permanent or Git-safe.
- Never rewrite V1 when deriving reranker eligibility or exclusions.
- Preserve V1 document-grouped splits and keep InfographicsVQA sealed during
  primary model development.
- Measure candidate-oracle recall before attributing misses to the reranker.
- Do not call answer-anchor labels complete-evidence labels.
- Do not mine hard negatives from validation, test, or holdout records.
- Do not change candidate revision, model input, or scoring between the stock
  and tuned benchmark.
- Do not let HierDoc H0/H1 or A1 enter the active R0/R1/M0/M1 POC. They belong
  to the later answer-sufficiency experiment.
- Do not inspect the internal test before Stage 04 or the InfographicsVQA
  holdout before optional Stage 05.
- Treat answer-string or plausible-context non-overlaps as possible false
  negatives until audited.
- Stop on nondeterminism, failed gates, insufficient scratch, or provenance
  conflicts rather than lowering requirements.
- Confirm submitted jobs start, then stop polling unless monitoring was
  explicitly requested.
