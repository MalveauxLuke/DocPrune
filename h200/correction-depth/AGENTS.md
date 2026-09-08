# Correction-depth package

Read `HANDOFF.md` in this directory. This is an exploratory 40-candidate
baseline and depth-specific regional-oracle comparison, not the 600-question
shared-probe study or the 100-question confirmation.

Reuse existing authenticated assets and environments. Do not modify those
cohorts, their preparation, manifests, or running jobs. Update the existing
DocPrune checkout without creating another source directory. No fresh
retrieval, global-index loading, feature rebuilding, training, or model changes.
Unknown answer scores require adjudication; they are not incorrect labels.
Keep generated inputs/results in the existing checkout’s ignored
`task9-h200-local-data/correction-depth40/` folder. Resume completed
units only when their recorded identities agree. Record admissions, failures,
and measured results in the new output directory's experiment log.
