# Evidence Dataset

## Files
- `scripts/sciegqa_evidence_poc.py` - evidence row selection, OCR attempt validation, section labeling, split assignment, audits, and output writing.
- `scripts/build_sciegqa_evidence_poc.py` - command-line builder.
- `scripts/build_sciegqa_final_labeling.py` - final labeling entry point.
- `sol/prepare_sciegqa_option4_pages.sbatch` - Option 4 page preparation.
- `sol/finalize_sciegqa_4k_evidence.sbatch` - combined/final evidence job.
- `sol/CURRENT_SOL_TASK.md` - current evidence dataset state when active.

## Final Combined Run
- Combined Option 4 dataset: `/scratch/$USER/sciegqa_train_4k/evidence_option4_10431/combined_evidence/20260707T210746Z`.
- Source counts: 6,630 pages, 10,431 queries, 6,591 OCR-valid pages, 39 terminal OCR failures.
- Final labeling retained 4,463 pages and 6,918 queries; quarantined 2,167 pages and 3,513 queries.
- Fresh page-grouped split seed `20260707`: train 4,842 queries, validation 1,038 queries, test 1,038 queries.
- Use this run as the dataset source for the segment evidence classifier spec; do not rebuild it unless explicitly instructed.

## Tests
- `tests/test_sciegqa_evidence_poc.py`
- `tests/test_sciegqa_final_labeling.py`

## Long-file map
- `scripts/sciegqa_evidence_poc.py` is large. Search the exact function family first:
  - selection: `select_expansion_rows`, `select_additional_rows`, `create_option4_selection`
  - OCR attempts: `validate_ocr_attempt`, `select_valid_attempts`, `merge_attempts`
  - labels/splits: `label_page_sections`, `assign_page_splits`, `build_evidence_records`
  - run outputs: `prepare_run`, `prepare_combined_run`, `finalize_run`
- `tests/test_sciegqa_evidence_poc.py` is large. Search for the target behavior before reading slices.

## Useful searches
```bash
rg -n "create_option4_selection|write_evidence_records|assign_page_splits|validate_ocr_attempt" scripts tests
rg -n "evidence_option4|combined_evidence|finalize_sciegqa" sol scripts tests
```
