# Overlap-First V1 Segment-Reranker Baseline

## Record

- Experiment ID: `overlap_v1_segment_reranker_baseline`
- Owner approval: 2026-08-11
- Status: `subsumed`
- Execution: not scheduled or started
- Binding specification:
  `../../../../docs/specifications/overlap_v1_segment_reranker_baseline.md`
- Implementation plan:
  `../../../../docs/superpowers/plans/2026-08-11-overlap-v1-segment-reranker-baseline.md`
- Source proposal:
  `../proposals/document_specific_multimodal_reranker.md`
- Successor program:
  `overlap_v1_evidence_localization.md`

The complete baseline survives as R0/R1 and Stages 00-01 of the successor. It
is no longer scheduled as a separate experiment or SOL handoff.

## Hypothesis

A stock multimodal reranker can rank answer-anchor-containing semantic
segments above same-page alternatives, and training-only audited hard
non-anchor fine-tuning improves held-out Recall@1 over the same stock model
without changing candidates, input formatting, or score extraction.

## Frozen experiment shape

- Dataset: immutable overlap-first document corpus V1.
- Candidates: one frozen DeepSeek-OCR-2 semantic-segment revision.
- Input: one rendered segment and the original question.
- Candidate models: `Qwen/Qwen3-VL-Reranker-2B` or
  `jinaai/jina-reranker-m0`.
- Control: released pretrained checkpoint without project fine-tuning.
- Treatment: the same checkpoint after training-only audited hard non-anchor
  fine-tuning.
- Primary metric: query-macro answer-anchor Recall@1.
- Primary effect: tuned minus stock Recall@1 with document-clustered bootstrap
  95% confidence interval.

## Interpretation boundary

The labels identify answer-bearing locations. The result cannot establish
complete-evidence selection, multi-hop reasoning, or MiniVGent-style set
reasoning. Candidate-oracle misses are reported separately from reranker
misses.

## Current blockers

- No execution checkout, branch, scratch root, or SOL handoff is approved.
- No candidate revision is built.
- No Qwen/Jina model choice or immutable model revision is locked.
- No inference, mining, fine-tuning, or benchmark run has occurred.
