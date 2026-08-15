# Completed Task State: Overlap-First Document Corpus V1

Archived: 2026-08-11
Outcome: complete and verified

## Goal

The overlap-first Visual-CoT plus BoundingDocs document corpus V1 was created,
verified, and frozen.

## Final dataset

- Package: `/home/lmalveau/overlap_first_document_corpus/v1`
- Canonical documents: 12,856
- Canonical questions preserved: 66,551
- Usable questions with resolved OCR-backed pages: 65,787
- Quarantined questions without a resolved OCR page: 764
- Canonical page positions: 23,043
- Unique OCR/image contents: 23,006
- Added BoundingDocs-only questions: 8,085 across 6,191 selected pages

The package contains `documents.jsonl`, `questions.jsonl`, `pages.jsonl`,
`ocr.jsonl`, `excluded_questions.jsonl`, `manifest.json`, and `README.md`.
Images and OCR payloads are referenced by absolute path and SHA-256 rather
than copied again.

## Splits

- Split grouping key: `canonical_document_id`; no document crosses splits.
- DocVQA and DUDE: deterministic SHA-256 80/10/10 train/validation/test.
- InfographicsVQA: separate `infographicsvqa_holdout`; it is absent from the
  primary three splits.
- TextVQA, TextCaps, and SROIE are not members of this merged document-QA V1.
- Conflict annotations are preserved, not resolved.
- Consumers must require `usable_in_v1=true` in `questions.jsonl`.

## Verification

All 23,043 page paths existed, every page had a completed DeepSeek-OCR-2
record, all OCR payload hashes were recomputed successfully, and every usable
question mapped to at least one packaged page. A random eight-page overlay
review included ordinary, repetition-flagged, truncation-flagged, dense, and
rotated pages and found aligned OCR and gold-answer boxes.

## Transition

The owner approved the overlap-first V1 segment-reranker baseline as the next
experiment on 2026-08-11. Its authority moved to
`docs/specifications/overlap_v1_segment_reranker_baseline.md` and
`agent-context/CURRENT_TASK.md`.
