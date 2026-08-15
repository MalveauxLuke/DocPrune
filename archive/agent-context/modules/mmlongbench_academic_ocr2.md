# MMLongBench Academic Gold-Page OCR2 Corpus

## Purpose

Prepare and run a frozen OCR-only corpus before deciding semantic segmentation
labels. This module records the local selection boundary and points to the
active SOL contract.

## Frozen local corpus

- Root: `pilot_data/mmlongbench_ocr2_academic_gold/`
- Builder: `scripts/mmlongbench/academic_gold.py`
- CLI: `scripts/mmlongbench/prepare_academic_gold.py`
- Tests: `tests/test_mmlongbench_academic_gold.py`
- MMLongBench-Doc revision:
  `2ff6aa9237fc777b6627dc57a486e9225ac5fb86`
- Source parquet SHA-256:
  `bcdac3c96669634c34184814cede4fe57cf7ac0f98dde0e85936394f6a56a02d`

The selection keeps rows whose `doc_type` is exactly `Academic paper`, whose
`answer_format` is present, and whose official evidence-page list is nonempty
and wholly valid against the source PDF. It does not infer or repair page
numbers.

Frozen result: 26 documents, 150 answerable questions, 193 unique gold pages,
260 question-page links, and 54 audited exclusions (50 unanswerable plus four
invalid evidence-page records).

Each tracked PDF contains only that source document's selected gold pages.
`page_plan.jsonl` maps each original 1-based source page to the corresponding
1-based page in the compact PDF. Questions, answers, evidence sources,
exclusions, source hashes, PDF hashes, and full-corpus checksums are retained.

## Boundary

The active SOL task runs only the pinned full-page DeepSeek-OCR-2 Markdown OCR
pipeline and returns raw/Markdown/atomic-box/diagnostic/provenance artifacts.
Semantic segmentation, correct-segment selection, Qwen VQA, and website work
are intentionally deferred until the returned OCR is inspected locally.

Binding execution contract:
`sol/task_spec/mmlongbench_academic_gold_deepseek_ocr2.md`.
