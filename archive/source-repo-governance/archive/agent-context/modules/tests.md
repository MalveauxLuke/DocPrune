# Tests

The active suite is restricted to `tests/` by `pytest.ini`.

## Coverage

- `test_mmlongbench_ocr2_pilot.py`: 313-page preparation, rendering, packaging.
- `test_mmlongbench_ocr2_deep_parse_*`: prompt-arm preparation and viewer.
- `test_mmlongbench_ocr2_viewer_*`: static and bundle behavior.
- `test_mmlongbench_ocr2_sol.py`: current/retained MMLongBench SOL contracts.
- `test_document_parsing_schema.py`, `test_deepseek_grounding.py`, and
  `test_semantic_sections.py`: shared parsing behavior.

## Commands

```bash
python -m pytest -q --tb=short --disable-warnings --show-capture=no
python -m pytest -q tests/test_semantic_sections.py::test_name --tb=short
```

Historical tests live under `archive/tests/` and are not maintained against the
active import layout. The retired academic-corpus preparation test is archived
there with its implementation.
