# ColQwen Verifier

## Files
- `archive/code/colqwen/build_colqwen_verifier_dataset.py` - builds split examples and summaries from retrieval runs.
- `archive/code/colqwen/colqwen_verifier_model.py` - verifier model and alignment features.
- `archive/code/colqwen/train_colqwen_verifier.py` - training/evaluation loop.
- `archive/code/colqwen/evaluate_colqwen_verifier.py` - result summarization.
- `archive/code/colqwen/build_colqwen_failure_casebook.py` - HTML failure-case review artifact.
- `sol/archive/jobs/colqwen/run_colqwen_verifier_v1.sbatch` - historical SOL training/evaluation wrapper.

## Tests
- `archive/tests/colqwen/test_build_colqwen_verifier_dataset.py`
- `archive/tests/colqwen/test_colqwen_verifier.py`
- `archive/tests/colqwen/test_colqwen_failure_casebook.py`

## Useful searches
```bash
rg --no-ignore -n "ColQwenEvidenceVerifier|AlignmentFeature|train_one_seed|validate_dataset_integrity" archive/code/colqwen archive/tests/colqwen
rg --no-ignore -n "CONTENT_ONLY|NO_INPUT_LAYERNORM|colqwen_verifier" sol/archive archive/code/colqwen archive/tests/colqwen
```
