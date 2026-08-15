# DocPrune Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a locally tested, runnable reproduction of DocPrune's BTP, QTP, and CTP stages for the official M3DocRAG and Qwen2-VL pipeline, then prepare a SOL handoff for GPU parity evaluation.

**Architecture:** Keep paper-equation logic in framework-light tensor modules, then integrate it through a version-gated Qwen2-VL adapter and a thin M3DocRAG runner. Every run writes a provenance manifest that separates paper values from reconstruction defaults. Local tests use synthetic tensors and a toy decoder; model accuracy and throughput remain SOL-only gates.

**Tech Stack:** Python 3.10-3.12, PyTorch, Pillow, Transformers/Qwen2-VL, ColPali/M3DocRAG, pytest, Ruff, TOML, Slurm/SBATCH.

## Global Constraints

- The CVPR 2026 paper and supplement are the method authority.
- Official M3DocRAG baseline contract: commit `29e6ac2294d6b87075a1d45b8a8df175b214248a`.
- No training or fine-tuning is part of DocPrune.
- Never track model weights, datasets, caches, predictions, or generated page images.
- Label unreported choices `reconstruction_default`; do not attribute them to the authors.
- Preserve every nonvisual token during QTP and CTP.
- Prune Qwen2-VL vision inputs only in complete 2 by 2 spatial-merge groups.
- A clean local test suite authorizes handoff, not paper-parity claims.

---

### Task 1: Package, configuration, and provenance contract

**Files:**
- Create: `pyproject.toml`
- Create: `src/docprune/__init__.py`
- Create: `src/docprune/config.py`
- Create: `src/docprune/provenance.py`
- Create: `configs/docprune-m3docvqa.toml`
- Create: `tests/test_config.py`
- Create: `tests/test_provenance.py`

**Interfaces:**
- Produces: `DocPruneConfig.for_pages(page_count: int) -> PagePruningConfig`
- Produces: `RunProvenance.to_dict() -> dict[str, object]`
- Produces: `load_config(path: Path) -> DocPruneConfig`

- [ ] **Step 1: Write failing configuration tests**

```python
def test_paper_page_settings_are_exact():
    cfg = DocPruneConfig.paper_defaults()
    assert asdict(cfg.for_pages(4)) == {
        "retrieval_background_threshold": 1.0,
        "qa_background_threshold": 0.8,
        "background_error_tolerance": 1.0,
        "question_threshold": 0.4,
        "comprehension_threshold": 45.0,
        "attention_threshold": 0.075,
    }

def test_unsupported_page_count_is_rejected():
    with pytest.raises(ValueError, match="1, 2, or 4"):
        DocPruneConfig.paper_defaults().for_pages(3)
```

- [ ] **Step 2: Run `python -m pytest tests/test_config.py -v` and confirm import failure**
- [ ] **Step 3: Implement frozen validated dataclasses and TOML loading**
- [ ] **Step 4: Run the configuration tests and confirm they pass**
- [ ] **Step 5: Write a failing provenance test asserting separate `paper_values` and `reconstruction_defaults` objects**
- [ ] **Step 6: Implement deterministic JSON-serializable provenance with model, dataset, Git, dependency, and hardware fields**
- [ ] **Step 7: Run `python -m pytest tests/test_config.py tests/test_provenance.py -v`**
- [ ] **Step 8: Commit with `feat: add DocPrune configuration and provenance`**

### Task 2: Visual layout and Background Token Pruning

**Files:**
- Create: `src/docprune/layout.py`
- Create: `src/docprune/btp.py`
- Create: `tests/test_layout.py`
- Create: `tests/test_btp.py`

**Interfaces:**
- Produces: `VisualLayout(grid_thw, spatial_merge_size)` with group offsets and fine-token indices
- Produces: `background_scores(images, patch_size, error_tolerance) -> Tensor`
- Produces: `background_keep_mask(scores, threshold, layout) -> Tensor`

- [ ] **Step 1: Write a failing layout test with a hand-enumerated 4 by 4 grid grouped into four 2 by 2 merge units**
- [ ] **Step 2: Run `python -m pytest tests/test_layout.py -v` and confirm the missing API failure**
- [ ] **Step 3: Implement immutable layout validation, group indexing, and per-page offsets**
- [ ] **Step 4: Run the layout test and confirm it passes**
- [ ] **Step 5: Write failing BTP tests using literal grayscale fixtures**

```python
def test_background_ratio_uses_strict_error_boundary():
    image = torch.tensor([[[[100, 100], [101, 130]]]], dtype=torch.uint8)
    got = background_scores(image, patch_size=2, error_tolerance=1)
    assert got.tolist() == [[0.5]]

def test_threshold_keeps_equal_ratio_and_prunes_greater_ratio():
    scores = torch.tensor([[0.8, 0.81]])
    assert threshold_keep_mask(scores, 0.8).tolist() == [[True, False]]
```

- [ ] **Step 6: Run the BTP tests and verify they fail because behavior is absent**
- [ ] **Step 7: Implement BT.601 grayscale conversion, modal intensity, paper equation 3, paper equation 4, and complete-group retention**
- [ ] **Step 8: Run layout and BTP tests; refactor only while green**
- [ ] **Step 9: Commit with `feat: implement background token pruning`**

### Task 3: Question-aware Token Pruning

**Files:**
- Create: `src/docprune/qtp.py`
- Create: `tests/test_qtp.py`

**Interfaces:**
- Produces: `cosine_sum_relevance(document_tokens, question_tokens) -> Tensor`
- Produces: `gaussian_smooth_2d(scores, sigma) -> Tensor`
- Produces: `question_keep_mask(..., layout, threshold) -> Tensor`

- [ ] **Step 1: Write a failing hand-derived cosine-sum test**

```python
def test_cosine_sum_aggregates_every_question_token():
    docs = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    query = torch.tensor([[[1.0, 0.0], [1.0, 1.0]]])
    got = cosine_sum_relevance(docs, query)
    assert torch.allclose(got, torch.tensor([[1.7071068, 0.7071068]]), atol=1e-6)
```

- [ ] **Step 2: Run the test and verify the missing API failure**
- [ ] **Step 3: Implement normalized matrix multiplication and sum over query tokens**
- [ ] **Step 4: Run the cosine test and confirm it passes**
- [ ] **Step 5: Write failing tests for bilinear resize dimensions, a literal Gaussian impulse response, inclusive relevance thresholding, and any-member 2 by 2 group retention**
- [ ] **Step 6: Implement separable Gaussian convolution without SciPy, bilinear resize, thresholding, and group-safe masks**
- [ ] **Step 7: Run `python -m pytest tests/test_qtp.py tests/test_layout.py -v`**
- [ ] **Step 8: Commit with `feat: implement question-aware token pruning`**

### Task 4: Comprehension-aware Token Pruning

**Files:**
- Create: `src/docprune/ctp.py`
- Create: `tests/test_ctp.py`

**Interfaces:**
- Produces: `ComprehensionController.observe(layer_index, last_hidden_state) -> bool`
- Produces: `visual_attention_scores(attention, visual_indices) -> Tensor`
- Produces: `ctp_keep_indices(token_count, visual_indices, scores, threshold) -> Tensor`
- Produces: `CTPDecision(layer_index, norm, retained_visual_indices)`

- [ ] **Step 1: Write failing tests for first crossing and no crossing**

```python
def test_trigger_selects_only_the_first_crossing():
    controller = ComprehensionController(threshold=5.0)
    assert controller.observe(3, torch.tensor([[3.0, 0.0]])) is False
    assert controller.observe(4, torch.tensor([[3.0, 4.0]])) is True
    assert controller.observe(5, torch.tensor([[6.0, 8.0]])) is False
    assert controller.selected_layer == 4
```

- [ ] **Step 2: Run the test and verify the missing API failure**
- [ ] **Step 3: Implement paper equation 8 with explicit batch-size-one validation**
- [ ] **Step 4: Run first-crossing tests and confirm they pass**
- [ ] **Step 5: Write failing tests for mean head aggregation, inclusive attention thresholding, original token order, and unconditional nonvisual-token retention**
- [ ] **Step 6: Implement paper equation 9 and typed decision traces**
- [ ] **Step 7: Run `python -m pytest tests/test_ctp.py -v` and the entire suite**
- [ ] **Step 8: Commit with `feat: implement comprehension-aware token pruning`**

### Task 5: Qwen2-VL sparse vision and decoder integration

**Files:**
- Create: `src/docprune/qwen2vl/__init__.py`
- Create: `src/docprune/qwen2vl/compat.py`
- Create: `src/docprune/qwen2vl/vision.py`
- Create: `src/docprune/qwen2vl/sequence.py`
- Create: `src/docprune/qwen2vl/decoder.py`
- Create: `src/docprune/qwen2vl/model.py`
- Create: `tests/qwen2vl/test_compat.py`
- Create: `tests/qwen2vl/test_vision.py`
- Create: `tests/qwen2vl/test_sequence.py`
- Create: `tests/qwen2vl/test_decoder.py`

**Interfaces:**
- Produces: `assert_supported_qwen2vl(model) -> QwenCompatibility`
- Produces: `compact_vision_batch(pixel_values, position_ids, page_offsets, group_keep_mask) -> CompactVisionBatch`
- Produces: `compact_multimodal_sequence(input_ids, embeds, position_ids, attention_mask, image_token_id, keep_mask) -> CompactSequence`
- Produces: `DocPruneDecoder.forward_prefill(...) -> DecoderOutput`
- Produces: `DocPruneQwen2VL.generate_with_trace(...) -> GenerationResult`

- [ ] **Step 1: Pin a tested Transformers compatibility range and write failing structural-contract tests using minimal fake Qwen modules**
- [ ] **Step 2: Verify unsupported models fail before generation with the missing or changed attribute named in the error**
- [ ] **Step 3: Implement the compatibility inspector without importing private classes at module import time**
- [ ] **Step 4: Write failing sparse-vision tests proving selected 2 by 2 groups stay contiguous and retain original rotary positions**
- [ ] **Step 5: Implement sparse vision input compaction and per-page cumulative sequence lengths**
- [ ] **Step 6: Write failing multimodal-sequence tests proving only rejected image placeholders disappear and all text positions remain ordered**
- [ ] **Step 7: Implement sequence, mask, and M-ROPE position compaction**
- [ ] **Step 8: Write a failing toy-decoder test where CTP triggers at layer 2, later layers receive fewer visual tokens, and generated text tokens remain present**
- [ ] **Step 9: Implement a no-cache prefill decoder loop, last-token attention recomputation, heterogeneous per-layer KV state, and one-time CTP compaction**
- [ ] **Step 10: Add generation-step tests for per-layer cache lengths and causal-mask dimensions after CTP**
- [ ] **Step 11: Implement greedy generation with reduced deeper-layer caches and per-sample traces**
- [ ] **Step 12: Run `python -m pytest tests/qwen2vl -v` and the entire suite**
- [ ] **Step 13: Commit with `feat: integrate DocPrune with Qwen2-VL`**

### Task 6: M3DocRAG adapter, metrics, and command-line runner

**Files:**
- Create: `src/docprune/m3docrag.py`
- Create: `src/docprune/metrics.py`
- Create: `src/docprune/cli.py`
- Create: `tests/test_m3docrag.py`
- Create: `tests/test_metrics.py`
- Create: `tests/test_cli.py`

**Interfaces:**
- Produces: `DocPruneM3DocRAG.run_sample(sample) -> SampleResult`
- Produces: `StageMetrics.update(trace, timings) -> None`
- Produces console script: `docprune-m3docvqa`

- [ ] **Step 1: Write a failing adapter-contract test using real core pruning modules and deterministic fake retrieval/QA boundary objects**
- [ ] **Step 2: Implement dependency injection around the official M3DocRAG retrieval and VQA APIs**
- [ ] **Step 3: Write failing metric tests with literal token-count, drop-rate, and throughput expectations**
- [ ] **Step 4: Implement immutable per-sample JSONL results and aggregate metric serialization**
- [ ] **Step 5: Write failing CLI dry-run tests that validate paths, revisions, page count, and output collision behavior without loading models**
- [ ] **Step 6: Implement `inspect`, `embed`, `evaluate`, and `summarize` subcommands; refuse overwrite unless an explicit resume manifest matches**
- [ ] **Step 7: Run the entire test suite and `docprune-m3docvqa inspect --config configs/docprune-m3docvqa.toml`**
- [ ] **Step 8: Commit with `feat: add M3DocRAG reproduction runner`**

### Task 7: Reproduction documentation and SOL handoff

**Files:**
- Modify: `README.md`
- Modify: `docs/NAVIGATION.md`
- Modify: `src/README.md`
- Modify: `experiments/README.md`
- Create: `docs/reproduction/DOCPRUNE.md`
- Create: `docs/reproduction/RECONSTRUCTION_GAPS.md`
- Create: `environments/docprune-sol.yml`
- Create: `examples/sbatch/10_docprune_smoke.sbatch`
- Create: `examples/sbatch/11_docprune_m3docvqa.sbatch`
- Create: `sol/handoffs/DOCPRUNE_SOL_HANDOFF.md`
- Modify: `sol/CURRENT_SOL_TASK.md`

**Interfaces:**
- Produces: a copy-pasteable smoke command and gated benchmark command
- Produces: an exact SOL handoff tied to the final Git commit

- [ ] **Step 1: Document paper facts, reconstruction defaults, component equations, trace schema, and non-parity status**
- [ ] **Step 2: Add a pinned environment whose CUDA/PyTorch/FlashAttention combination is internally compatible and record upstream Git/Hugging Face revisions**
- [ ] **Step 3: Add an SBATCH smoke job for one fixed sample and a benchmark job parameterized by top-1/top-2/top-4**
- [ ] **Step 4: Write a handoff with checkout, environment, inputs, scratch root, resources, exact commands, expected artifacts, smoke gate, parity gates, and recovery authority**
- [ ] **Step 5: Update navigation and state files without asserting that SOL execution occurred**
- [ ] **Step 6: Run `git diff --check`, all tests, Ruff, CLI inspection, tracked-artifact scans, and link/path checks**
- [ ] **Step 7: Commit with `docs: add DocPrune reproduction and SOL handoff`**

### Task 8: Final local audit

**Files:**
- Modify only files required by audit findings

**Interfaces:**
- Produces: a clean commit with locally verified code and an explicitly unexecuted SOL task

- [ ] **Step 1: Create a requirement checklist from the design and map every item to code, tests, documentation, or the SOL gate**
- [ ] **Step 2: Run every local verification command from a fresh virtual environment**
- [ ] **Step 3: Confirm no tracked file exceeds 10 MiB and no generated/model/data artifact is tracked**
- [ ] **Step 4: Confirm every paper-underspecified choice appears in provenance and `RECONSTRUCTION_GAPS.md`**
- [ ] **Step 5: Confirm the worktree is clean and report the exact final commit**
