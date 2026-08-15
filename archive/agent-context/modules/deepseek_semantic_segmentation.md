# DeepSeek Semantic Segmentation

Use this page for routing only. The complete contract is
[`docs/specifications/deepseek_semantic_segmentation.md`](../../../docs/specifications/deepseek_semantic_segmentation.md).

## Source

- `scripts/document_parsing/semantic_sections.py`
- `scripts/document_parsing/grounding.py`
- `scripts/mmlongbench/viewer/viewer_bundle.py`

## Key rules

- Explicit text sections split only on Markdown ATX headings.
- Consecutive headings merge; bullets remain with their section.
- Visual/title association first follows native reading order, then uses the
  documented geometry fallback.
- MinerU is historical comparison evidence, not an active label source.

## Tests

- `tests/test_semantic_sections.py`
- `tests/test_mmlongbench_ocr2_viewer_bundle.py`
