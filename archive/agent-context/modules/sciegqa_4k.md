# SciEGQA 4K

## Files
- `scripts/sciegqa_4k/selection.py` - query normalization, classification, deduplication, allocation, and row ordering.
- `scripts/sciegqa_4k/pipeline.py` - profile, review sample, query selection, and manifest writing.
- `scripts/sciegqa_4k/extract.py` - archive member validation and page extraction.
- `scripts/build_sciegqa_4k.py` - command-line selection/profile builder.
- `scripts/extract_sciegqa_4k_pages.py` - extraction command-line entry.
- `sol/download_sciegqa_train_images.sbatch`, `sol/extract_sciegqa_4k_pages.sbatch` - SOL data setup wrappers.

## Tests
- `tests/test_sciegqa_4k_selection.py`
- `tests/test_sciegqa_4k_pipeline.py`
- `tests/test_sciegqa_4k_extract.py`
- `tests/test_sciegqa_4k_sol.py`

## Useful searches
```bash
rg -n "select_queries|build_profile|extract_selected_pages|DOMAIN_QUOTAS" scripts tests
rg -n "sciegqa_train_4k|images.tar|extract_sciegqa" sol scripts tests
```
