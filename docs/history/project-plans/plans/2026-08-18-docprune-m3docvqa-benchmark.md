# DocPrune M3DocVQA Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the missing production M3DocRAG integration, pass the ordered SOL gates, and execute baseline and DocPrune M3DocVQA top-1/top-2/top-4 benchmarks.

**Architecture:** Add fail-closed benchmark configuration and artifact manifests, a sparse ColPali retrieval path for BTP, exact processor mappings for QTP/Qwen, baseline and DocPrune answerers, and deterministic resumable evaluation. Activate a commit-pinned SOL handoff only after local verification, then build separate baseline/DocPrune indexes and execute the six evaluation cells.

**Tech Stack:** Python 3.10, PyTorch 2.4.1, Transformers 4.46.3, ColPali Engine 0.3.1, M3DocRAG, Qwen2-VL, FAISS, safetensors, pytest, Ruff, Slurm.

**Spec:** `docs/superpowers/specs/2026-08-18-docprune-m3docvqa-benchmark-design.md`

## Global Constraints

- Use DocPrune runtime lineage beginning at `64ea70c66a8f9e3dbce804d1fde265a4a2b8b09d` and pin the final runtime commit in the activated handoff.
- Use M3DocRAG commit `29e6ac2294d6b87075a1d45b8a8df175b214248a` without modifying its checkout.
- Use Qwen `Qwen/Qwen2-VL-7B-Instruct@eed13092ef92e448dd6875b2a00151bd3f7db0ac`.
- Use ColPali adapter `vidore/colpali-v1.2@961b51745de3e9adb3468ac5c9ccca0ac626c217` and backbone `vidore/colpaligemma-3b-pt-448-base@30ab955d073de4a91dc5a288e8c97226647e3e5a`.
- Identify the dataset by the official MMQA archive checksums, pinned M3DocRAG transformation, and final integrity report; do not require a nonexistent Hugging Face dataset revision.
- Preserve all nonvisual Qwen tokens and original visual positional order.
- Build separate manifest-bound baseline and DocPrune indexes; never mutate or silently share an index.
- Use all 2,441 dev questions in deterministic source order for every full evaluation cell.
- Never track credentials, weights, caches, datasets, page images, indexes, predictions, or raw profiles.
- No training or fine-tuning is authorized.

---

### Task 1: Correct processor resources and prove raster mappings

**Files:**
- Modify: `src/docprune/processor_probe.py`
- Modify: `tests/test_processor_probe.py`
- Modify: `tests/test_cli.py`
- Modify: `docs/reproduction/RECONSTRUCTION_GAPS.md`

**Interfaces:**
- Produces: `ColPaliVisualMapping(image_token_id: int, visual_start: int, visual_stop: int, grid_hw: tuple[int, int], raster_indices: tuple[int, ...])`
- Produces: `collect_processor_contract(...)` with `raster_order_verified=True` only after exact pinned-processor invariants pass.
- Consumes: ColPali v1.2 `image_seq_length`, input IDs, attention mask, and PaliGemma processor image placeholders.

- [ ] **Step 1: Write failing mapping tests**

Add literal fake-processor cases that require a contiguous visual span of exactly
`image_seq_length`, a square grid, unique row-major raster indices, and rejection
of missing, repeated, noncontiguous, or padded visual positions.

```python
def test_colpali_mapping_proves_contiguous_row_major_visual_span():
    mapping = resolve_colpali_visual_mapping(
        input_ids=torch.tensor([[7, 42, 42, 42, 42, 9]]),
        attention_mask=torch.ones(1, 6),
        image_token_id=42,
        image_seq_length=4,
    )
    assert mapping.grid_hw == (2, 2)
    assert mapping.raster_indices == (0, 1, 2, 3)
```

- [ ] **Step 2: Run `python -m pytest tests/test_processor_probe.py tests/test_cli.py -v` and confirm the new tests fail for missing mapping behavior.**
- [ ] **Step 3: Implement immutable mapping extraction and make `validate_processor_contract` require all three mapping checks, including `raster_order_verified`.**
- [ ] **Step 4: Replace every runnable/test resource reference to `vidore/colpali-v1` with the corrected v1.2 adapter and add the backbone resource/revision to schema version 2.**
- [ ] **Step 5: Run the targeted tests and `ruff check src/docprune/processor_probe.py tests/test_processor_probe.py tests/test_cli.py`.**
- [ ] **Step 6: Commit with `fix: pin and validate the official ColPali contract`.**

### Task 2: Benchmark configuration, corpus adapter, and immutable manifests

**Files:**
- Create: `src/docprune/benchmark_config.py`
- Create: `src/docprune/artifacts.py`
- Create: `src/docprune/m3docvqa_dataset.py`
- Create: `tests/test_benchmark_config.py`
- Create: `tests/test_artifacts.py`
- Create: `tests/test_m3docvqa_dataset.py`
- Modify: `legacy/configs/docprune-m3docvqa.toml`

**Interfaces:**
- Produces: `BenchmarkRunConfig.from_env(mode: str, page_count: int) -> BenchmarkRunConfig`
- Produces: `CorpusIdentity.validate() -> None`
- Produces: `IndexManifest.to_dict() -> dict[str, object]` and `IndexManifest.validate_files() -> None`
- Produces: `M3DocVQADevDataset` yielding `SampleInput` in exact JSONL order and loading pages at 144 DPI with the pinned upstream modal-size rule.

- [ ] **Step 1: Write failing configuration tests for every exact commit/revision, required corpus path, mode, page count, generation setting, and missing-file/hash mismatch.**
- [ ] **Step 2: Write failing corpus tests using two tiny PDFs/JSONL records that prove source order, answer extraction from `[{"answer": ...}]`, unique qids, doc-ID membership, and modal-size page normalization.**
- [ ] **Step 3: Write failing manifest tests that bind mode, corpus integrity SHA, source PDF/doc ordering digest, model revisions, processor contract SHA, pruning configuration, embedding shapes/dtype, index checksum, and runtime/M3DocRAG commits.**
- [ ] **Step 4: Run `python -m pytest tests/test_benchmark_config.py tests/test_artifacts.py tests/test_m3docvqa_dataset.py -v` and confirm missing APIs fail.**
- [ ] **Step 5: Implement frozen dataclasses, canonical JSON hashing, path/hash validation, and direct corpus loading without mutating or duplicating the acquired dataset.**
- [ ] **Step 6: Extend the TOML with immutable model resources and generation policy (`max_new_tokens=128`, greedy decoding, official short-answer prompt) while keeping paper values and reconstruction defaults separate.**
- [ ] **Step 7: Run the targeted tests, `git diff --check`, and Ruff.**
- [ ] **Step 8: Commit with `feat: add benchmark inputs and artifact manifests`.**

### Task 3: Retrieval-stage BTP and manifest-bound ColPali indexes

**Files:**
- Create: `src/docprune/colpali/__init__.py`
- Create: `src/docprune/colpali/compat.py`
- Create: `src/docprune/colpali/vision.py`
- Create: `src/docprune/colpali/embedding.py`
- Create: `src/docprune/indexing.py`
- Create: `tests/colpali/test_compat.py`
- Create: `tests/colpali/test_vision.py`
- Create: `tests/colpali/test_embedding.py`
- Create: `tests/test_indexing.py`

**Interfaces:**
- Produces: `assert_supported_colpali(model, processor) -> ColPaliCompatibility`
- Produces: `sparse_siglip_features(vision_tower, pixel_values, patch_keep_mask) -> Tensor`
- Produces: `encode_colpali_page(model, batch, mapping, patch_keep_mask) -> ColPaliPageEmbedding`
- Produces: `build_index(config, mode, output) -> IndexBuildResult`

- [ ] **Step 1: Write failing compatibility tests for ColPali Engine 0.3.1, Transformers 4.46.3, 448-by-448 input, 14-pixel SigLIP patches, 32-by-32 raster positions, 128-dimensional projected output, and exact PaliGemma component paths.**
- [ ] **Step 2: Write a failing sparse-SigLIP test that compares an all-true mask to the stock vision tower, then proves selected patch embeddings retain original positional embeddings and order.**
- [ ] **Step 3: Write a failing ColPali sequence test that removes rejected image placeholders only, preserves every text token, runs the language model on compacted image features, and returns visual embeddings plus original raster indices.**
- [ ] **Step 4: Run `python -m pytest tests/colpali -v` and confirm the missing sparse path fails.**
- [ ] **Step 5: Implement version-gated sparse SigLIP/PaliGemma execution without modifying installed packages or the pinned upstream checkout. Baseline/all-kept must call the same adapter with an all-true mask and compare numerically to stock ColPali.**
- [ ] **Step 6: Write failing index tests for deterministic document/page order, safetensors shape checks, baseline-versus-pruned manifest separation, FAISS width 128, token-to-page mapping persistence, checksum validation, and resume of only manifest-identical completed documents.**
- [ ] **Step 7: Implement per-document atomic safetensors writes, persisted `token2pageuid`, `IndexFlatIP(128)`, index checksum, and an immutable completion ledger. BTP uses the exact post-ColPali-resize uint8 raster and the page-count retrieval threshold.**
- [ ] **Step 8: Run `python -m pytest tests/colpali tests/test_indexing.py -v` and Ruff.**
- [ ] **Step 9: Commit with `feat: add sparse ColPali indexing`.**

### Task 4: QTP mapping, baseline Qwen, and real DocPrune answerers

**Files:**
- Modify: `src/docprune/pipeline.py`
- Modify: `src/docprune/qtp.py`
- Create: `src/docprune/_legacy/qwen2vl/preprocessing.py`
- Create: `src/docprune/answerers.py`
- Modify: `src/docprune/_legacy/qwen2vl/model.py`
- Modify: `src/docprune/_legacy/qwen2vl/decoder.py`
- Create: `tests/qwen2vl/test_preprocessing.py`
- Create: `tests/test_answerers.py`
- Modify: `tests/test_pipeline.py`
- Modify: `tests/qwen2vl/test_model.py`

**Interfaces:**
- Produces: `dense_relevance_from_sparse_tokens(tokens, raster_indices, source_hw, question_tokens) -> Tensor`
- Produces: `prepare_qwen_page(processor, image) -> PreparedQwenPage`
- Produces: `AllKeptQwenAnswerer.answer(images, question) -> AnswerOutput`
- Produces: `DocPruneQwenAnswerer.answer(images, question) -> AnswerOutput`

- [ ] **Step 1: Write failing QTP tests proving post-BTP ColPali tokens are scattered back to their original raster cells before resize/smoothing and rejected background cells cannot be resurrected by QTP.**
- [ ] **Step 2: Write failing preprocessing tests proving the retained uint8 raster uses the pinned Qwen resize dimensions, produces the processor's exact `image_grid_thw`, and maps one placeholder per 2-by-2 merge group in page order.**
- [ ] **Step 3: Write failing answerer tests proving identical short-answer prompts and greedy settings, all-kept traces with four equal counts, decoded generated-token trimming, nonempty page masks, and monotonic DocPrune traces.**
- [ ] **Step 4: Run `python -m pytest tests/test_pipeline.py tests/test_answerers.py tests/qwen2vl -v` and confirm the new behavior is absent.**
- [ ] **Step 5: Implement sparse-to-dense QTP mapping, exact Qwen preprocessing, stock baseline generation, and DocPrune generation using retrieval document/query embeddings from the same pinned ColPali instance.**
- [ ] **Step 6: Correct real-model cache/mask/position behavior exposed by the tests while preserving the existing tiny-model contract. Add an opt-in `DOCPRUNE_REAL_MODEL_TEST=1` test that compares all-kept adapter output/logits with stock Qwen on the fixed probe page.**
- [ ] **Step 7: Run targeted tests, the complete CPU suite, and Ruff.**
- [ ] **Step 8: Commit with `feat: add production Qwen benchmark answerers`.**

### Task 5: Concrete factory, safe resume, and benchmark CLI

**Files:**
- Create: `src/docprune/m3docvqa_factory.py`
- Modify: `src/docprune/m3docrag.py`
- Modify: `src/docprune/cli.py`
- Create: `tests/test_m3docvqa_factory.py`
- Modify: `tests/test_m3docrag.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Produces: `build_workload(*, operation, config, page_count, output, mode, run_config, index_manifest) -> EvaluationWorkload | IndexBuildResult`
- Produces CLI options: `--mode {all-kept,docprune}`, `--run-config`, `--index-manifest`, `--limit`, and `--sample-ids`.
- Produces: qid-aware resume that verifies existing records and skips exactly completed source-order entries.

- [ ] **Step 1: Write failing factory preflight tests for dirty/wrong M3DocRAG checkout, model revision mismatch, invalid processor contract, wrong index mode/page count, missing cache artifacts, and output collisions.**
- [ ] **Step 2: Write failing retrieval tests that require unique `(doc_id, page_index)` outputs and exactly 1/2/4 pages. Over-fetch token neighbors deterministically until enough unique pages exist while preserving upstream MaxSim scoring semantics.**
- [ ] **Step 3: Write failing CLI tests for concrete-factory defaults, complete manifest fields, embed result validation, deterministic sample limits, duplicate-qid rejection, and interruption/resume without duplicate JSONL records.**
- [ ] **Step 4: Run `python -m pytest tests/test_m3docvqa_factory.py tests/test_m3docrag.py tests/test_cli.py -v` and confirm the failures.**
- [ ] **Step 5: Implement lazy model loading, official retrieval/page boundaries, mode-specific answerers and indexes, complete run manifests, atomic result appends, and qid-aware resume.**
- [ ] **Step 6: Run the targeted tests, complete suite, three CLI dry runs for each mode/page count, and Ruff.**
- [ ] **Step 7: Commit with `feat: add the M3DocVQA benchmark factory`.**

### Task 6: Official quality metrics and independent result validation

**Files:**
- Create: `src/docprune/evaluation.py`
- Modify: `src/docprune/metrics.py`
- Modify: `src/docprune/cli.py`
- Create: `tests/test_evaluation.py`
- Modify: `tests/test_metrics.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Produces: `evaluate_m3docvqa(results, source_rows) -> M3DocVQAMetrics`
- Produces: `validate_benchmark_run(run_dir, expected_questions=2441) -> ValidationReport`
- Produces CLI subcommand: `validate-run --run DIR --expected-questions 2441`.

- [ ] **Step 1: Write literal-fixture tests for list EM/F1, modality F1, single/multi-hop F1, question-type metrics, and document recall at 1/2/4/5/10. Match the correct upstream formulas but do not copy its stale hop/question-type dictionary bug.**
- [ ] **Step 2: Write failing validator tests for wrong count, duplicate/missing/unexpected qids, source-order drift, manifest/index digest mismatch, non-monotonic traces, unrecorded warmup, and summary reproduction mismatch.**
- [ ] **Step 3: Extend measurement fixtures to require peak allocated GPU bytes and an explicit warmup-excluded flag; record profiler definition and FLOPs only when profiling was enabled.**
- [ ] **Step 4: Run `python -m pytest tests/test_evaluation.py tests/test_metrics.py tests/test_cli.py -v` and confirm failures.**
- [ ] **Step 5: Implement the evaluator, measurement aggregates, deterministic summary serialization, and independent run validation.**
- [ ] **Step 6: Run the targeted tests, complete suite, Ruff, and a sealed 2-record validator fixture.**
- [ ] **Step 7: Commit with `feat: validate M3DocVQA benchmark results`.**

### Task 7: Activate exact SOL gates and benchmark launchers

**Files:**
- Modify: `examples/sbatch/11_docprune_m3docvqa.sbatch`
- Create: `examples/sbatch/12_docprune_m3docvqa_gate.sbatch`
- Create: `examples/sbatch/13_docprune_m3docvqa_index.sbatch`
- Create: `examples/sbatch/14_docprune_m3docvqa_eval_array.sbatch`
- Create: `sol/handoffs/DOCPRUNE_M3DOCVQA_BENCHMARK_HANDOFF.md`
- Modify: `agent-context/CURRENT_TASK.md`
- Modify: `sol/CURRENT_SOL_TASK.md`
- Modify: `sol/README.md`
- Modify: `docs/NAVIGATION.md`
- Modify: `docs/reproduction/DOCPRUNE.md`

**Interfaces:**
- Produces one active handoff pinned to the post-Task-6 runtime commit and one control commit.
- Produces gate, two-index, and six-cell evaluation submissions with explicit Slurm dependencies.

- [ ] **Step 1: Update launchers to require the corrected Qwen/ColPali/backbone revisions, corpus-integrity digest, runtime/M3DocRAG commits, processor contract, mode, page count, index manifest, attempt root, and exact factory. Remove `DATASET_REVISION`.**
- [ ] **Step 2: Make the gate launcher run processor probe, mapping inspection, deterministic official-versus-all-kept samples, and one DocPrune sample for pages 1/2/4. It writes one machine-readable gate JSON and exits nonzero on any failed invariant.**
- [ ] **Step 3: Make the index launcher select `all-kept` or `docprune`, write to a new attempt directory, resume only manifest-identical documents, and validate the final index checksum/manifest.**
- [ ] **Step 4: Make the evaluation array map six immutable cells (`all-kept|docprune` crossed with `1|2|4`), require the correct index dependency, validate 2,441 results, and write independently reproduced summaries.**
- [ ] **Step 5: Write the handoff with exact clean worktrees, environment, corpus hashes, model revisions, A100 80 GB/8 CPU/128 GB resources, commands, outputs, dependency graph, pass conditions, and recovery rules. Pin the exact implementation commit produced by Task 6; then commit the handoff/state changes and use that commit as the control commit.**
- [ ] **Step 6: Run `bash -n` on all launchers, all tests, Ruff, CLI inspection/dry runs, Markdown link checks, `git diff --check`, tracked-artifact scans, and clean-checkout assertions.**
- [ ] **Step 7: Commit with `docs: activate the M3DocVQA benchmark handoff`.**

### Task 8: Execute gates, indexes, six benchmark cells, and final audit

**Files:**
- Create only small Git-safe provenance/report files if all runs validate.
- Keep all generated runtime artifacts under `/scratch/lmalveau/docprune/benchmark-<runtime-short-commit>/`.

**Interfaces:**
- Consumes the active handoff from Task 7.
- Produces validated gate evidence, two indexes, six 2,441-record result sets, six summaries, and a final comparison report.

- [ ] **Step 1: Create/verify the detached runtime worktree at the handoff commit, verify the clean pinned M3DocRAG checkout and environment freeze, and download model snapshots by immutable revision into the shared Hugging Face cache from a compute allocation.**
- [ ] **Step 2: Submit the gate job and require processor mapping, all-kept equivalence, and pages-1/2/4 one-sample DocPrune checks to pass before continuing. Preserve any failed attempt unchanged.**
- [ ] **Step 3: Submit baseline and DocPrune index jobs with dependency on the gate; validate document/page counts, embedding shapes, checksums, and manifests.**
- [ ] **Step 4: Submit the six-cell evaluation array with dependencies on the matching validated indexes. Resume only scheduler-interrupted manifest-identical cells.**
- [ ] **Step 5: Run `docprune-m3docvqa validate-run` independently for every cell and require exactly 2,441 unique source-ordered qids and reproducible summaries.**
- [ ] **Step 6: Compare baseline and DocPrune EM/F1, modality/hop F1, retrieval recall, stage token counts/drop rates, warmup-excluded throughput, and peak allocated GPU memory. Report FLOPs only for cells with a recorded profiler definition.**
- [ ] **Step 7: Run a final requirement-by-requirement audit against the design, verify all source worktrees clean, and commit only the small provenance/report update with `docs: report the M3DocVQA benchmark`.**
