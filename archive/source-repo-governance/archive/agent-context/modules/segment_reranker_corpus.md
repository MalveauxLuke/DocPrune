# Segment Reranker Corpus

Status: approved architecture; implementation not started.

Complete specification:
[`docs/specifications/segment_reranker_corpus.md`](../../../docs/specifications/segment_reranker_corpus.md).

## Boundary

- Development sources: DUDE, SlideVQA, and TAT-DQA.
- MMLongBench-Doc remains held out for evaluation under the leakage contract.
- Frozen source revisions, stable IDs, document hashes, and page-level
  deduplication are mandatory.
- This module does not authorize new acquisition or training work by itself.

Read the full specification only when building or reviewing this corpus.
