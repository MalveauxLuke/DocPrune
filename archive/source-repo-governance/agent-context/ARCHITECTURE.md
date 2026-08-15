# Architecture

## Repository roles

`main` is the research central-command line. It records current authority,
canonical specifications, experiment locations, compact provenance, reports,
and archived task state. Model experiments do not execute from this checkout.

The immutable dataset is external:

```text
/home/lmalveau/overlap_first_document_corpus/v1
```

The overlap-first V1 answer-anchor POC needs a dedicated execution checkout, branch,
scratch root, and SOL handoff. They are intentionally unresolved rather than
being inferred from older Evidence-DINO or MMLongBench workspaces.

## Active project surface

- `docs/specifications/overlap_v1_evidence_localization/README.md`: binding POC
  design and later answer-sufficiency boundary.
- `docs/specifications/overlap_v1_evidence_localization/architectures/`:
  separate Qwen, MiniVGent, and deferred HierDoc/A1 architecture contracts.
- `docs/specifications/overlap_v1_evidence_localization/experiment/`: staged
  POC plan, gates, metrics, and artifact contracts.
- `agent-context/CURRENT_TASK.md`: concise state and next action.
- `agent-context/modules/overlap_v1_evidence_localization.md`: durable POC
  summary.
- `src/candidates/`: planned frozen candidate and eligibility code.
- `src/training/`: planned scoring, mining, fine-tuning, and metric code.
- `tests/reranker/`: planned deterministic unit and integration contracts.

## Data flow through the current boundary

```text
immutable overlap-first V1
  -> frozen DeepSeek-OCR-2 semantic candidate revision
  -> answer-anchor coverage and exclusion audit
  -> candidate-oracle report
  -> stock Qwen R0 scores over (segment, query)
  -> training-only hard non-anchor mining and audit
  -> tuned Qwen R1 scores over the frozen pool
  -> MiniVGent M0/M1 systems preflight and one-seed screen
  -> frozen three-seed R0/R1/M0/M1 confirmatory comparison
  -> optional locked holdout
```

No candidate generation, inference, mining, or training begins until the
workspace registry and SOL handoff resolve the execution gate.

## Scientific boundary

The V1 boxes identify answer-bearing locations. They do not establish complete
evidence sufficiency. This experiment measures answer-anchor segment ranking,
not multi-hop reasoning or complete-evidence set selection.

HierDoc H0/H1, A1, answer-sufficiency supervision, and page routing remain a
separate later experiment. The proposed residual-image candidate remains
nonbinding future architecture work.
