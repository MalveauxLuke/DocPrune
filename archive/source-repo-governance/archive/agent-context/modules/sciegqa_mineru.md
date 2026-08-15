# SciEGQA MinerU

## Files
- `scripts/document_parsing/schema.py` - shared active dependency retained outside the archive.
- `archive/code/sciegqa/sciegqa_mineru/source_pages.py` - source PDF download, rendering, and validation.
- `archive/code/sciegqa/sciegqa_mineru/selection.py` - SPSR row filtering and selection.
- `archive/code/sciegqa/sciegqa_mineru/canonicalize.py` - MinerU content canonicalization.
- `archive/code/sciegqa/sciegqa_mineru/mineru_runner.py` - local/API execution helpers.
- `archive/code/sciegqa/sciegqa_mineru/viewer_bundle.py` - generated viewer manifest/site builder.
- `archive/code/sciegqa/run_sciegqa_mineru.py`, `archive/code/sciegqa/run_mineru_smoke.py`, and `archive/code/sciegqa/build_sciegqa_mineru_*` - entry points.
- `sol/archive/jobs/sciegqa/` - historical SOL wrappers.

## Tests
- `archive/tests/sciegqa/test_sciegqa_mineru_*.py`
- `archive/tests/sciegqa/test_run_mineru_smoke.py`
- `archive/tests/sciegqa/test_mineru_*`

## Long-test map
- `archive/tests/sciegqa/test_sciegqa_mineru_end_to_end.py` covers cross-module flow. Search for the failing behavior before reading.
- `archive/tests/sciegqa/test_sciegqa_mineru_source_pages.py` covers PDF download/rendering and source-page validation. Search by helper or error message first.

## Useful searches
```bash
rg --no-ignore -n "BBox|make_stable_id|run_page|canonicalize|validate_smoke_segments" archive/code/sciegqa archive/tests/sciegqa scripts/document_parsing
rg --no-ignore -n "MINERU_|mineru34|run_mineru" sol/archive/jobs/sciegqa archive/code/sciegqa archive/tests/sciegqa
```
