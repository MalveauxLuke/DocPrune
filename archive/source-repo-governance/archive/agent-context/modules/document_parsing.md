# Document Parsing

## Scope

Shared grounded-document primitives used by every active MMLongBench workflow.

## Authoritative files

- `scripts/document_parsing/schema.py`: boxes, IDs, hashes, JSONL helpers.
- `scripts/document_parsing/grounding.py`: grounded-output parsing.
- `scripts/document_parsing/deepseek_runner.py`: pinned model configuration,
  sharding, inference, and attempt validation.
- `scripts/document_parsing/semantic_sections.py`: text and visual candidate
  construction, geometry, and labels.
- `scripts/document_parsing/run_deepseek_ocr.py`: OCR CLI.
- `scripts/document_parsing/build_semantic_sections.py`: manifest enrichment
  CLI.

## Invariants

- Markdown ATX headings `#` through `######` are explicit text boundaries.
- Consecutive headings merge as title/subtitle members.
- Bullets attach to the preceding section and never create boundaries.
- Raw OCR is preserved even when atomic parsing fails.

## Tests

- `tests/test_document_parsing_schema.py`
- `tests/test_deepseek_grounding.py`
- `tests/test_semantic_sections.py`

Full segmentation contract:
`docs/specifications/deepseek_semantic_segmentation.md`.
