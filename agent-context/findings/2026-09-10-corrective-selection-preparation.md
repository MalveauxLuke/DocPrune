# Corrective-selection repository preparation — 2026-09-10

Owner selected `/Users/god/DocPrune/docs/ExperimentPlan.md` as the next design and
authorized archiving old experiments. Work is in DocPrune, not the COLPALI repository.

## Completed changes

Old Task 6–9 wrappers, scheduler scripts, regional-attribution plans/logs and
H200/SOL handoffs were moved to `archive/experiments/task6_9_2026_09_10`. Replaced navigation/current-task files
have original snapshots there. SHA-256 provenance records preserve all moved bytes.
The owner's plan is unchanged and remains at its supplied path.

The current navigation now selects corrective region selection, records staged
readiness and Stage 0 inputs, and collects retained ambiguous files in one report.
Library implementations and baseline dependencies are unchanged; tests that read
old fixtures now address the archive. No new model architecture was implemented.

## Evidence and limitations

Task-numbered runtime modules have baseline/segmentation or regression consumers;
removing them blindly would break reuse. Historical 256-mask/B13/100/600 settings
are not new-study defaults. Explicit rich-2B sections take precedence in the new
navigation while the plan's wording remains untouched.

The initial full test attempt in the available Python 3.12 environment could not
collect 11 modules because colpali_engine, faiss, word2number and pdf2image were absent.
This is an environment limitation, not an observed implementation regression.
See [verification](../../docs/experiments/corrective-selection/VERIFICATION.md) for
the final checks. No remote state or job was changed. No commit or push was made.
