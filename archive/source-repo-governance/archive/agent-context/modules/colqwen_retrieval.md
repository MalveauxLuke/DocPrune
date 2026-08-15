# ColQwen Retrieval

## Files
- `scripts/run_colqwen_topk.py` - document-scoped candidate construction, model loading, encoding, scoring, and result writing.
- `scripts/run_vidore_arxivqa_colqwen.py` - Vidore/ArxivQA evaluation path.
- `scripts/materialize_mmdocir_documents.py` - materializes selected documents/pages.
- `scripts/select_mmdocir_subset.py` and `scripts/select_mmdocir_diverse_longdoc_subset.py` - subset selection.
- `sol/run_colqwen_top5.sbatch`, `sol/run_colqwen2_doc_top5.sbatch`, `sol/run_vidore_arxivqa_*.sbatch` - SOL wrappers.

## Tests
- `tests/test_run_colqwen_topk.py`
- `tests/test_mmdocir_subset.py`
- `tests/test_diverse_longdoc_subset.py`
- `tests/test_vidore_arxivqa_metrics.py`
- `tests/test_mmdocir_colqwen2_sbatch.py`
- `tests/test_vidore_*_sbatch.py`

## Useful searches
```bash
rg -n "score_topk|build_document_scoped_candidate_indices|infer_model_arch|write_results" scripts tests
rg -n "run_colqwen|vidore|HF_HOME|TRANSFORMERS_CACHE" sol scripts tests
```
