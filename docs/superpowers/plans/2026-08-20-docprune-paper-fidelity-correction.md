# DocPrune Paper-Fidelity Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct retrieval/QTP dataflow, ColPali index scope, Qwen backend, and stage timing before rerunning the six-cell benchmark.

**Architecture:** Persist complete ColPali page sequences with an explicit visual-row map, return immutable retrieval context to QTP without re-encoding, match the pinned upstream top-K and FlashAttention paths, and record separate synchronized Qwen encoder/decoder timings while preserving end-to-end measurements.

**Tech Stack:** Python 3.10, PyTorch 2.4.1, Transformers 4.46.3, FlashAttention 2, ColPali Engine 0.3.1, FAISS, safetensors, pytest, Ruff, Slurm.

**Spec:** `docs/superpowers/specs/2026-08-20-docprune-paper-fidelity-correction-design.md`

## Global Constraints

- Keep M3DocRAG, Qwen, ColPali adapter, and ColPali backbone at the immutable revisions in `configs/docprune-m3docvqa.toml`.
- Preserve every nonvisual ColPali and Qwen token and every retained visual token's original raster/rotary identity.
- QTP consumes retrieval-produced document and question embeddings; it never runs ColPali again.
- Indexed retrieval uses exactly `k=top_k`, matching pinned M3DocRAG MaxSim aggregation.
- Qwen uses `flash_attention_2`; CTP recomputes only the final query attention row.
- Build fresh schema-versioned indexes and a fresh attempt; never mutate or promote attempt-2 artifacts.
- Use all 2,441 dev questions in source order and the same A100 class, prompt, greedy generation, and warmup for paired modes.
- Use TDD: each behavior test is observed failing for the intended reason before production changes.
- Do not implement paper baselines or ablations outside the requested all-kept versus DocPrune six-cell benchmark.

---

### Task 1: Full-sequence ColPali indexes and retrieval context

**Files:**
- Modify: `src/docprune/indexing.py`
- Modify: `src/docprune/artifacts.py`
- Modify: `src/docprune/m3docrag.py`
- Modify: `src/docprune/m3docvqa_factory.py`
- Modify: `src/docprune/answerers.py`
- Modify: `tests/colpali/test_embedding.py`
- Modify: `tests/test_indexing.py`
- Modify: `tests/test_artifacts.py`
- Modify: `tests/test_m3docrag.py`
- Modify: `tests/test_answerers.py`
- Modify: `tests/test_m3docvqa_factory.py`

**Interfaces:**
- Produce `RetrievedPageFeatures(doc_id, page_index, visual_embeddings, raster_indices, source_hw)`.
- Produce `RetrievalOutput(pages, query_embeddings, page_features)` from indexed retrieval.
- `DocPruneQwenAnswerer.answer(..., retrieval_output=...)` requires aligned persisted features and performs no ColPali forward.
- Index schema 5 stores complete ColPali sequence embeddings and row-aligned raster indices using `-1` for nonvisual rows.

- [ ] **Step 1: Add literal failing tests.** Prove all-kept persistence equals the complete stock ColPali output; every page retains nonvisual rows; visual rasters are increasing and nonvisual rows use `-1`; schema-4 manifests fail closed; indexed retrieval searches exactly `top_k`; returned page features align with selected identities; and the DocPrune answerer cannot call ColPali during QA.
- [ ] **Step 2: Run `python -m pytest tests/colpali/test_embedding.py tests/test_indexing.py tests/test_artifacts.py tests/test_m3docrag.py tests/test_answerers.py tests/test_m3docvqa_factory.py -v` and record the expected failures.**
- [ ] **Step 3: Persist `encoded.embeddings[0]` for each page and construct a row-aligned raster tensor.** Use `-1` outside the compact visual span and `encoded.raster_indices` inside it. Update per-document offsets, ledgers, final safetensors, metadata, and validators as one schema-5 contract.
- [ ] **Step 4: Replace over-fetch retrieval with exact pinned `k=top_k` MaxSim and return `RetrievalOutput`.** Extract each selected page's visual rows from the signed index tensors and validate identity, count, raster range/order, width 128, and 32-by-32 source grid.
- [ ] **Step 5: Pass `RetrievalOutput` through `DocPruneM3DocRAG` and make QTP consume it directly.** Remove page/query ColPali recomputation from `DocPruneQwenAnswerer`; keep result JSON serialization limited to page identities/scores.
- [ ] **Step 6: Run the focused suite, the complete CPU suite, Ruff, and `git diff --check`; commit `fix: reuse faithful ColPali retrieval features`.**

### Task 2: FlashAttention CTP and paper-aligned stage timing

**Files:**
- Modify: `src/docprune/m3docvqa_factory.py`
- Modify: `src/docprune/qwen2vl/decoder.py`
- Modify: `src/docprune/qwen2vl/model.py`
- Modify: `src/docprune/answerers.py`
- Modify: `src/docprune/m3docrag.py`
- Modify: `src/docprune/metrics.py`
- Modify: `src/docprune/evaluation.py`
- Modify: `tests/qwen2vl/test_decoder.py`
- Modify: `tests/qwen2vl/test_model.py`
- Modify: `tests/test_answerers.py`
- Modify: `tests/test_metrics.py`
- Modify: `tests/test_evaluation.py`
- Modify: `tests/test_m3docvqa_factory.py`

**Interfaces:**
- `_load_qwen` supplies `attn_implementation="flash_attention_2"` and bfloat16 vision configuration.
- `GenerationResult`, `AnswerOutput`, and `SampleTiming` carry positive `encoder_seconds` and `decoder_seconds` for production rows.
- Summary timing includes `encoder`, `decoder`, `qa`, and `retrieval`, plus `encoder_samples_per_second` and `decoder_samples_per_second`.

- [ ] **Step 1: Add failing tests.** A fake loader must observe the exact FlashAttention arguments; CTP final-query attention must project one query row, never allocate an `N x N` causal mask for recomputation, and remain numerically equivalent to the existing literal fixture; both answerers must emit separate stage timings; production validation must reject missing, nonfinite, or nonpositive stage times.
- [ ] **Step 2: Run `python -m pytest tests/qwen2vl/test_decoder.py tests/qwen2vl/test_model.py tests/test_answerers.py tests/test_metrics.py tests/test_evaluation.py tests/test_m3docvqa_factory.py -v` and record the expected failures.**
- [ ] **Step 3: Load the pinned Qwen backend exactly and implement reduced-query CTP attention.** Project only `hidden[:, -1:, :]` for Q, project the full sequence for K, apply the corresponding final rotary row to Q, and return only `[batch, heads, 1, keys]` attention.
- [ ] **Step 4: Add synchronized stage timers.** Time only the Qwen vision encoder as encoder time and language-model prefill/decode as decoder time; retain complete QA wall time and whole-QA peak allocated memory. Ensure the stock all-kept and sparse DocPrune paths use the same definitions.
- [ ] **Step 5: Extend canonical records, summaries, manifests, and independent validation with exact stage-timing fields and derived samples/second.** Keep profiler fields absent when disabled and label A100 results as reconstruction measurements.
- [ ] **Step 6: Run focused tests, full CPU tests, the cached real-model all-kept probe, Ruff, and `git diff --check`; commit `fix: align Qwen backend and efficiency timing`.**

### Task 3: Re-seal and execute the corrected benchmark

**Files:**
- Modify: `docs/reproduction/DOCPRUNE.md`
- Modify: `docs/reproduction/RECONSTRUCTION_GAPS.md`
- Modify: `sol/handoffs/DOCPRUNE_M3DOCVQA_BENCHMARK_HANDOFF.md`
- Modify: `agent-context/CURRENT_TASK.md`
- Modify: `sol/CURRENT_SOL_TASK.md`
- Modify launchers/tests only if the new schemas require it.

**Interfaces:**
- Produce a new runtime commit, reviewed control commit, fresh attempt root, semantic gate, six schema-5 indexes, six validated 2,441-row evaluations, and final comparison report.

- [ ] **Step 1: Update documentation and provenance to record the paper audit, superseded attempt-2 evaluation, corrected index scope/dataflow/backend/timing definitions, and the new immutable runtime/control commits.**
- [ ] **Step 2: Run focused launcher tests, full CPU tests, Ruff, shell syntax, Markdown links, artifact scans, and clean-checkout checks; obtain independent Luna-high spec and quality review.**
- [ ] **Step 3: Submit a fresh SOL semantic gate.** Require exact runtime/upstream/model/backend identity, complete ColPali all-kept equivalence, exact upstream fixed-sample retrieval ordering, QTP no-recompute evidence, baseline Qwen equivalence, positive 1/2/4 pruning traces, and positive encoder/decoder timing probes.
- [ ] **Step 4: Build and independently validate six fresh schema-5 indexes, then submit the six-cell evaluation array with strict `afterok` dependencies.**
- [ ] **Step 5: Validate all six 2,441-row runs and report EM/F1, modality/hop F1, retrieval recall, stage token reductions, end-to-end timing, encoder/decoder throughput, and peak allocated GPU memory.** Report TFLOPs only if profiling was explicitly enabled, and do not claim A6000 hardware parity.
- [ ] **Step 6: Complete a requirement-by-requirement final audit and commit only small provenance/report files.**
