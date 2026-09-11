# Regional attribution experiments

This folder contains the active scientific authority for DocPrune's regional
attribution and learned-probe experiments. The procedure is an adapted,
answer-conditioned, region-level use of ContextCite machinery. It is not
vanilla ContextCite or a token-level oracle.

## Choose what to read

| Need | Read |
| --- | --- |
| Scientific direction or frozen design | [Experiment plan](EXPERIMENT_PLAN.md) |
| Current stage and next action | [Implementation plan](IMPLEMENTATION_PLAN.md) |
| Results, jobs, hashes, or failures | [Experiment-log index](experiment-log/INDEX.md) |
| Human-facing results | [Consolidated results](../../results/README.md) |
| Visual audit of 24 development failures + six controls | [Completed visual and quantitative audit](analysis/VISUAL_AUDIT_24_2026-09-06.md) |
| New evidence-verified correction corpus and scorer | [Collection, answer contracts and depth-study design](research/CORRECTION_CORPUS_2026-09-06.md) |
| Curated 40-candidate baseline/depth comparison package | [H200 assembly and execution handoff](../../../h200/correction-depth/HANDOFF.md) |
| Presentation storyline and experimental evidence | [Presentation evidence packet](analysis/PRESENTATION_EVIDENCE_2026-09-06.md) |
| Exact H200 execution | The applicable handoff under [`h200/`](../../../h200/) |

Do not read the full research review or archived monoliths by default. Open
them only when the active plan links them for a specific question.

## Authority

1. [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md) is the scientific source of truth.
2. A scoped machine handoff supplies exact execution commands and machine rules;
   it cannot change the science.
3. [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) records only high-level
   progress and the next action.
4. [`experiment-log/`](experiment-log/INDEX.md) records evidence and provenance;
   it does not redefine the experiment.
5. [`archive/`](archive/legacy-governance-2026-09-05/README.md) is nonbinding
   history.

Chat history and historical handoffs are not experimental authority.

## Active tracks

| Track | Cohort | State |
| --- | ---: | --- |
| Regional-surrogate development | 48 | Complete |
| Baseline-wrong confirmation | 100 | Prepared for H200 execution |
| Shared-probe feasibility | 600 | Staged SOL/H200 implementation |
| Evidence-reviewed correction depth comparison | 40 candidates | Packaged and CPU checked; 36 asset-complete, H200 smoke pending |

The cohorts are separate and must never be pooled or reused across their
declared roles.

## Folder map

```text
EXPERIMENT_PLAN.md       Canonical design, organized by task
IMPLEMENTATION_PLAN.md   High-level status and next action
experiment-log/          Indexed history for each active track
analysis/                Focused completed analyses
research/                Detailed reviews opened only when needed
archive/                 Superseded governing documents
development-qid-registry.json
                         Frozen development-question registry
```
