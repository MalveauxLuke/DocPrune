# MiniVGent Qwen Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and verify a compact MiniVGent answer-anchor selector over frozen Qwen3-VL-Reranker-2B page memory, then make it ready for a matched M0-versus-M1 single-hop experiment over the frozen overlap-first V1 candidate universe.

**Architecture:** The official Qwen reranker prompt encodes the original question before one full page image. A frozen Qwen adapter returns selected question-conditioned language memories and a layer-28 visual grid. Actual candidate member boxes, OCR text, type, geometry, and reading order initialize 1,024-wide candidate queries. Two decoder blocks cross-attend to Qwen layers 14 and 28; M0 masks all off-diagonal candidate interaction and M1 permits bidirectional interaction. Independent logits are trained with OR-positive, within-query ranking, and verified-negative losses.

**Tech Stack:** Python 3.11, PyTorch 2.8, Torchvision 0.23, Transformers 4.57.3, qwen-vl-utils 0.0.14, Qwen3-VL-Reranker-2B, JSON/JSONL locks, pytest, Slurm on SOL.

## Global Constraints

- This plan is nonbinding until the owner explicitly authorizes MiniVGent implementation and names an approved checkout, branch, scratch root, and SOL handoff.
- Do not implement this plan in the current control checkout under the active segment-reranker task.
- Do not download model weights, create an environment, run Qwen, or submit a job before the execution gate closes.
- Treat `/home/lmalveau/overlap_first_document_corpus/v1` as immutable.
- Consume, but never regenerate or mutate, the candidate revision frozen by the active V1 segment-reranker baseline.
- Use original single-hop questions and answer-anchor labels only. Do not add synthetic multi-hop, answer-feedback RL, backbone LoRA, answer generation, box regression, or complete-evidence claims.
- Keep `infographicsvqa_holdout` sealed from training, model selection, threshold fitting, and early stopping.
- Resolve every Hub reference to an immutable commit and record exact package versions before model loading.
- Store model weights, Hugging Face caches, page tensors, predictions, logs, and checkpoints outside Git.
- Run image processing, model inference, environment creation, and training only in an approved compute allocation.
- Use the exact architecture and tensor semantics in `agent-context/research/architecture_registry/proposals/minivgent_qwen_implementation_readiness.md`. A change to widths, taps, prompt, ROI rule, loss, or label semantics creates a new config hash and requires owner review.

---

### Task 1: Add the Frozen Configuration and Typed Tensor Contracts

**Files:**
- Create: `src/models/minivgent/__init__.py`
- Create: `src/models/minivgent/config.py`
- Create: `src/data/minivgent_records.py`
- Create: `configs/minivgent/qwen3_vl_reranker_2b_two_block.json`
- Create: `environments/minivgent-sol.yml`
- Create: `tests/minivgent/test_config_and_records.py`
- Modify: `src/models/README.md`

**Interfaces:**
- Produces: `MiniVGentConfig.from_json(path: Path) -> MiniVGentConfig`.
- Produces: `MiniVGentConfig.sha256() -> str` over canonical sorted JSON.
- Produces: immutable `CandidateTensorBatch` and `AnchorTargetBatch` dataclasses.
- Rejects unknown config keys, mutable model revisions, invalid tap sequences, and inconsistent mask shapes.

- [ ] **Step 1: Write failing config and shape-contract tests**

```python
def test_two_block_config_is_exact() -> None:
    cfg = MiniVGentConfig.from_json(CONFIG_PATH)
    assert cfg.model_id == "Qwen/Qwen3-VL-Reranker-2B"
    assert cfg.model_revision == "4bd860ac4f15ad1897a214615cccc700f8f71818"
    assert cfg.memory_layers == (14, 28)
    assert cfg.decoder_width == 1024
    assert cfg.decoder_blocks == 2
    assert cfg.attention_heads == 16
    assert cfg.swiglu_hidden_width == 2736
    assert cfg.ocr_max_tokens == 128
    assert cfg.image_max_tokens == 1800


def test_candidate_batch_rejects_member_mask_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="member_mask"):
        CandidateTensorBatch(
            member_boxes_1000=torch.zeros(2, 3, 4, 4),
            member_mask=torch.ones(2, 3, 3, dtype=torch.bool),
            ocr_token_ids=torch.zeros(2, 3, 128, dtype=torch.long),
            ocr_token_mask=torch.ones(2, 3, 128, dtype=torch.bool),
            candidate_type_ids=torch.ones(2, 3, dtype=torch.long),
            reading_orders=torch.zeros(2, 3, 4, dtype=torch.long),
            candidate_mask=torch.ones(2, 3, dtype=torch.bool),
        )
```

- [ ] **Step 2: Run the focused test and confirm the missing-module failure**

Run: `python -m pytest -q tests/minivgent/test_config_and_records.py --tb=short`  
Expected: FAIL because `src.models.minivgent.config` and `src.data.minivgent_records` do not exist.

- [ ] **Step 3: Implement strict frozen dataclasses**

```python
@dataclass(frozen=True)
class CandidateTensorBatch:
    member_boxes_1000: torch.Tensor   # [B,C,M,4], float32
    member_mask: torch.Tensor         # [B,C,M], bool
    ocr_token_ids: torch.Tensor       # [B,C,128], long
    ocr_token_mask: torch.Tensor      # [B,C,128], bool
    candidate_type_ids: torch.Tensor  # [B,C], long in [0,5]
    reading_orders: torch.Tensor      # [B,C,M], long
    candidate_mask: torch.Tensor      # [B,C], bool


@dataclass(frozen=True)
class AnchorTargetBatch:
    alternative_positive_mask: torch.Tensor  # [B,A,C], bool
    alternative_mask: torch.Tensor           # [B,A], bool
    verified_negative_mask: torch.Tensor     # [B,C], bool
    candidate_mask: torch.Tensor             # [B,C], bool
```

Validate dtype, rank, shared `B/C/M`, box order, `[0,1000]` bounds, and that
padding positions contain no active members or targets. Require at least one
active positive alternative and one verified negative per training row.

- [ ] **Step 4: Write the exact config and environment files**

The JSON config must include the model and official-source revisions, prompt,
package assumptions, BF16 dtype, SDPA backend, `use_cache=false`, two memory
taps, all dimensions, dropout, loss weights, seed `1729`, and the six-entry
candidate-type vocabulary. The environment must pin:

```yaml
name: minivgent-sol
channels:
  - conda-forge
dependencies:
  - python=3.11
  - pip
  - pip:
      - torch==2.8.0
      - torchvision==0.23.0
      - transformers==4.57.3
      - accelerate==1.12.0
      - qwen-vl-utils==0.0.14
      - scipy==1.16.3
```

Do not add FlashAttention to this environment.

- [ ] **Step 5: Run focused tests**

Run: `python -m pytest -q tests/minivgent/test_config_and_records.py --tb=short`  
Expected: PASS.

- [ ] **Step 6: Commit the contracts on the approved implementation branch**

```bash
git add src/models/minivgent/__init__.py src/models/minivgent/config.py \
  src/data/minivgent_records.py \
  configs/minivgent/qwen3_vl_reranker_2b_two_block.json \
  environments/minivgent-sol.yml tests/minivgent/test_config_and_records.py \
  src/models/README.md
git commit -m "feat: define minivgent tensor contracts"
```

### Task 2: Reproduce the Official Qwen Prompt and Score Path

**Files:**
- Create: `src/models/minivgent/qwen_reranker.py`
- Create: `tests/minivgent/test_qwen_reranker.py`

**Interfaces:**
- Produces: `format_page_instruction(query: str, page_image: Image.Image) -> list[dict[str, object]]`.
- Produces: `build_yes_no_head(lm, tokenizer) -> nn.Linear`.
- Produces: `score_official(inputs: Mapping[str, Tensor]) -> Tensor[B]`.
- Produces: `QwenModelLock` with model SHA, source SHA, prompt hash, processor settings, dtype, attention backend, and package versions.

- [ ] **Step 1: Write failing prompt-order and score-head tests with fakes**

```python
def test_query_precedes_document_image() -> None:
    messages = format_page_instruction("What is the total?", fixture_image())
    content = messages[1]["content"]
    query_i = next(i for i, item in enumerate(content) if item.get("text") == "What is the total?")
    image_i = next(i for i, item in enumerate(content) if item.get("type") == "image")
    assert query_i < image_i


def test_yes_no_head_is_lm_head_difference() -> None:
    head = build_yes_no_head(fake_lm(), fake_tokenizer(yes=7, no=11))
    torch.testing.assert_close(head.weight[0], fake_lm().lm_head.weight[7] - fake_lm().lm_head.weight[11])
```

- [ ] **Step 2: Confirm the test fails**

Run: `python -m pytest -q tests/minivgent/test_qwen_reranker.py --tb=short`  
Expected: FAIL because `qwen_reranker.py` does not exist.

- [ ] **Step 3: Implement the official wrapper contract**

Load `Qwen3VLForConditionalGeneration`, preserve `lm.model`, build the fixed
yes/no head before releasing the outer LM reference, and load the processor
with `padding_side="left"`. Freeze the backbone and call `eval()`.

Use the official system prompt verbatim and freeze this task instruction:

```text
Given a search query, retrieve relevant candidates that answer the query.
```

Set `min_pixels=4*32*32`, `max_pixels=1800*32*32`, maximum sequence length
`10240`, image patch size `16`, no video, and one page image.

- [ ] **Step 4: Add model-lock validation tests**

Reject `main`, `master`, tags, abbreviated SHAs, a source SHA other than the
configured 40-character value, processor/model revision mismatch, or a lock
whose prompt hash does not match the exact formatter.

- [ ] **Step 5: Run focused tests**

Run: `python -m pytest -q tests/minivgent/test_qwen_reranker.py --tb=short`  
Expected: PASS without downloading or loading a real model.

- [ ] **Step 6: Commit the Qwen wrapper**

```bash
git add src/models/minivgent/qwen_reranker.py \
  tests/minivgent/test_qwen_reranker.py
git commit -m "feat: add frozen qwen reranker adapter"
```

### Task 3: Implement Selective Qwen Layer-Memory Capture

**Files:**
- Create: `src/models/minivgent/qwen_memory.py`
- Create: `tests/minivgent/test_qwen_memory.py`

**Interfaces:**
- Produces: immutable `PageMemory`.
- Produces: `capture_page_memory(backbone, inputs, layers: tuple[int, ...]) -> PageMemory`.
- Uses intermediate forward hooks and normalized outer `last_hidden_state` for layer 28.

- [ ] **Step 1: Write failing fake-backbone tests for exact layer mapping**

```python
def test_human_layers_map_to_zero_based_modules() -> None:
    memory = capture_page_memory(fake_28_layer_backbone(), fake_inputs(), layers=(14, 28))
    assert memory.layer_numbers == (14, 28)
    assert torch.all(memory.memories[0] == 14)
    assert torch.all(memory.memories[1] == 2800)  # fake normalized final output


def test_hooks_are_removed_after_forward_exception() -> None:
    model = fake_28_layer_backbone(raise_at=20)
    with pytest.raises(RuntimeError):
        capture_page_memory(model, fake_inputs(), layers=(14, 28))
    assert all(not module._forward_hooks for module in model.language_model.layers)
```

- [ ] **Step 2: Confirm failure**

Run: `python -m pytest -q tests/minivgent/test_qwen_memory.py --tb=short`  
Expected: FAIL because `qwen_memory.py` does not exist.

- [ ] **Step 3: Implement scoped capture**

```python
@dataclass(frozen=True)
class PageMemory:
    layer_numbers: tuple[int, ...]
    memories: tuple[torch.Tensor, ...]  # each [B,T,2048]
    memory_mask: torch.Tensor           # [B,T], bool
    final_hidden: torch.Tensor          # [B,T,2048], normalized
    image_token_mask: torch.Tensor      # [B,T], bool
    image_grid_thw: torch.Tensor        # [B,3]
```

Register hooks only for requested intermediate layers. For requested layer 28,
take the outer normalized `last_hidden_state`; do not use the layer-27 hook.
Run the frozen backbone under `torch.no_grad()`, detach outputs, reject missing
or duplicate hook fires, and remove all handles in `finally`.

- [ ] **Step 4: Add mask and failure tests**

Test left padding, wrong hidden width, non-monotonic/duplicate layer requests,
more or fewer than 28 modules, video inputs, multiple images per sample, and
an image token count inconsistent with `image_grid_thw`.

- [ ] **Step 5: Run focused tests**

Run: `python -m pytest -q tests/minivgent/test_qwen_memory.py --tb=short`  
Expected: PASS.

- [ ] **Step 6: Commit selective extraction**

```bash
git add src/models/minivgent/qwen_memory.py \
  tests/minivgent/test_qwen_memory.py
git commit -m "feat: capture selected qwen page memories"
```

### Task 4: Reconstruct the Qwen Visual Grid and Pool Member Boxes

**Files:**
- Create: `src/models/minivgent/visual_grid.py`
- Create: `tests/minivgent/test_visual_grid.py`

**Interfaces:**
- Produces: `reconstruct_visual_grids(memory: PageMemory) -> list[Tensor[2048,H,W]]`.
- Produces: `pool_candidate_members(grids, boxes, member_mask) -> Tensor[B,C,2048]`.
- Uses only member boxes, never an enclosing crop.

- [ ] **Step 1: Write failing row-major and disconnected-member tests**

```python
def test_visual_tokens_reconstruct_row_major_grid() -> None:
    memory = page_memory_fixture(grid_thw=(1, 4, 6), values=range(6))
    grid = reconstruct_visual_grids(memory)[0]
    assert grid.shape == (2048, 2, 3)
    assert grid[0].tolist() == [[0, 1, 2], [3, 4, 5]]


def test_disconnected_members_do_not_pool_gap() -> None:
    grid = gap_sensitive_grid_fixture()
    feature = pool_candidate_members(
        [grid],
        boxes_fixture(left_box=True, right_box=True, enclosing_box=False),
        member_mask_fixture(2),
    )
    assert feature[0, 0, 0].item() < 1.0
```

- [ ] **Step 2: Confirm failure**

Run: `python -m pytest -q tests/minivgent/test_visual_grid.py --tb=short`  
Expected: FAIL because `visual_grid.py` does not exist.

- [ ] **Step 3: Implement grid reconstruction**

For each sample, assert one image with `t=1`, `h` and `w` divisible by 2, and
`image_token_mask.sum() == h*w/4`. Gather normalized final hidden states and
reshape row-major to `[2048,h/2,w/2]`.

- [ ] **Step 4: Implement member-level ROI Align**

Map normalized-1000 boxes directly to merged-grid coordinates. Deduplicate
exact member boxes, call ROI Align with `output_size=(2,2)`,
`spatial_scale=1.0`, and `aligned=True`, average the four cells, and aggregate
member vectors using original normalized member area. Reject zero-area boxes;
return a learned-empty marker flag when a valid candidate has no poolable
member rather than silently using a page mean.

- [ ] **Step 5: Add geometry edge-case tests**

Test full-page boxes, narrow boxes smaller than one visual cell, exact duplicate
members, non-identical overlaps with an emitted diagnostic, boundary boxes at
0 and 1000, padded candidates, variable per-sample grid sizes, and deterministic
results under member reordering.

- [ ] **Step 6: Run focused tests**

Run: `python -m pytest -q tests/minivgent/test_visual_grid.py --tb=short`  
Expected: PASS.

- [ ] **Step 7: Commit the visual-grid stage**

```bash
git add src/models/minivgent/visual_grid.py \
  tests/minivgent/test_visual_grid.py
git commit -m "feat: pool minivgent qwen visual regions"
```

### Task 5: Implement the Exact Candidate Encoder

**Files:**
- Create: `src/models/minivgent/candidate_encoder.py`
- Create: `tests/minivgent/test_candidate_encoder.py`

**Interfaces:**
- Produces: `build_geometry_features(batch: CandidateTensorBatch) -> Tensor[B,C,14]`.
- Produces: `CandidateEncoder.forward(visual_roi, batch, token_embeddings) -> Tensor[B,C,1024]`.
- Expected trainable parameter count: `4_425_472`.

- [ ] **Step 1: Write failing feature and parameter-count tests**

```python
def test_geometry_uses_member_union_not_sum() -> None:
    features = build_geometry_features(overlapping_member_fixture())
    assert features[0, 0, 9].item() == pytest.approx(0.75)


def test_candidate_encoder_parameter_count_is_frozen() -> None:
    model = CandidateEncoder(candidate_type_count=6)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 4_425_472
```

- [ ] **Step 2: Confirm failure**

Run: `python -m pytest -q tests/minivgent/test_candidate_encoder.py --tb=short`  
Expected: FAIL because `candidate_encoder.py` does not exist.

- [ ] **Step 3: Implement OCR pooling and fixed type IDs**

Use the frozen Qwen input-embedding table under `torch.no_grad()`. Tokenization
happens in the data adapter with no special tokens and first-64/last-64
truncation. Masked-mean non-padding embeddings; use one trainable 2,048-wide
`empty_ocr` vector only for an empty OCR mask.

Implement the six fixed type IDs from the readiness report. Unknown source
types map to `UNKNOWN=5` and are counted in the data audit.

- [ ] **Step 4: Implement geometry and fusion exactly**

Compute the 14 documented features, including exact rectangle-union area and
normalized first/last reading order. Implement the 512-wide visual and OCR
projections, 128-wide geometry MLP, 64-wide type embedding, and 1,216-to-1,024
fusion path exactly as frozen.

- [ ] **Step 5: Add mask, truncation, and invariance tests**

Test empty OCR, 128 tokens, 129 tokens, head/tail preservation, unknown types,
member-order invariance, padding zeroing, and finite BF16 forward/backward.

- [ ] **Step 6: Run focused tests**

Run: `python -m pytest -q tests/minivgent/test_candidate_encoder.py --tb=short`  
Expected: PASS with the exact parameter count.

- [ ] **Step 7: Commit the candidate encoder**

```bash
git add src/models/minivgent/candidate_encoder.py \
  tests/minivgent/test_candidate_encoder.py
git commit -m "feat: encode minivgent document candidates"
```

### Task 6: Implement the M0 and M1 Set Decoders

**Files:**
- Create: `src/models/minivgent/decoder.py`
- Create: `tests/minivgent/test_decoder.py`

**Interfaces:**
- Produces: `MiniVGentDecoder(config, candidate_interaction: bool)`.
- Consumes: candidates `[B,C,1024]`, memories `([B,T,2048], [B,T,2048])`, and masks.
- Produces: candidate logits `[B,C]`.
- Expected one-block parameter count: `18_911_584`.

- [ ] **Step 1: Write failing shape, isolation, and permutation tests**

```python
def test_m0_candidate_isolation() -> None:
    model = decoder_fixture(candidate_interaction=False)
    before = model(candidate_fixture(), memory_fixture()).logits
    changed = candidate_fixture(change_only_candidate=2)
    after = model(changed, memory_fixture()).logits
    torch.testing.assert_close(before[:, 0], after[:, 0])


def test_m1_is_permutation_equivariant() -> None:
    model = decoder_fixture(candidate_interaction=True).eval()
    logits = model(candidate_fixture(), memory_fixture()).logits
    perm = torch.tensor([2, 0, 1])
    permuted = model(candidate_fixture(perm=perm), memory_fixture()).logits
    torch.testing.assert_close(permuted, logits[:, perm])
```

- [ ] **Step 2: Confirm failure**

Run: `python -m pytest -q tests/minivgent/test_decoder.py --tb=short`  
Expected: FAIL because `decoder.py` does not exist.

- [ ] **Step 3: Implement the two-block pre-norm decoder**

Use `nn.MultiheadAttention(batch_first=True)` with `embed_dim=1024`, 16 heads,
and `kdim=vdim=2048` for cross-attention. Use a 1,024-wide 16-head self-
attention and a 2,736-wide SwiGLU feed-forward. Each sublayer has its own
LayerNorm, residual, and 0.1 dropout.

Block 0 consumes layer 14; block 1 consumes normalized layer 28. M1 uses only
the candidate padding mask. M0 additionally uses a Boolean attention mask with
every off-diagonal position forbidden. Zero padded candidate states after each
residual block.

- [ ] **Step 4: Add exact count and masking tests**

Assert `18_911_584` parameters per block and `37_823_168` for two blocks.
Test all-padding rejection, mixed candidate counts, left-padded Qwen memory,
no attention to padded memory, dropout determinism in eval mode, and finite
gradients in train mode.

- [ ] **Step 5: Run focused tests**

Run: `python -m pytest -q tests/minivgent/test_decoder.py --tb=short`  
Expected: PASS.

- [ ] **Step 6: Commit the set decoder**

```bash
git add src/models/minivgent/decoder.py tests/minivgent/test_decoder.py
git commit -m "feat: add minivgent candidate set decoder"
```

### Task 7: Implement Numerically Stable Answer-Anchor Losses

**Files:**
- Create: `src/training/minivgent_losses.py`
- Create: `tests/minivgent/test_losses.py`

**Interfaces:**
- Produces: `minivgent_loss(logits, targets, config) -> LossBreakdown`.
- Implements OR-positive, smooth positive versus top-eight verified-negative ranking, and verified-negative BCE.
- Masks partial/unresolved/unverified candidates by construction.

- [ ] **Step 1: Write failing hand-computable loss tests**

```python
def test_or_positive_does_not_require_every_overlapping_candidate() -> None:
    low_high = torch.tensor([[-10.0, 10.0, 0.0]])
    high_low = torch.tensor([[10.0, -10.0, 0.0]])
    targets = two_or_positive_fixture()
    assert minivgent_loss(low_high, targets, loss_config()).or_loss < 1e-3
    assert minivgent_loss(high_low, targets, loss_config()).or_loss < 1e-3


@pytest.mark.parametrize("value", [-80.0, 0.0, 80.0])
def test_loss_and_gradient_are_finite_at_extreme_logits(value: float) -> None:
    logits = torch.full((1, 3), value, requires_grad=True)
    loss = minivgent_loss(logits, target_fixture(), loss_config()).total
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
```

- [ ] **Step 2: Confirm failure**

Run: `python -m pytest -q tests/minivgent/test_losses.py --tb=short`  
Expected: FAIL because `minivgent_losses.py` does not exist.

- [ ] **Step 3: Implement the frozen objective**

Use stable `logsigmoid`/`log1p` identities for probability-at-least-one. Use
temperature `0.1` smooth maximum over active positives, the eight highest
current-logit verified negatives, margin `0.2`, and weights
`or=1.0/rank=0.5/negative=1.0`. Return scalar components, contributing row
counts, positive and negative counts, and top-negative IDs for diagnostics.

- [ ] **Step 4: Add semantic guard tests**

Reject overlap between positive and verified-negative masks, targets on padded
candidates, empty positive rows, empty negative rows, NaN/Inf logits, and any
candidate label outside the three allowed states: positive, verified negative,
or masked.

- [ ] **Step 5: Run focused tests**

Run: `python -m pytest -q tests/minivgent/test_losses.py --tb=short`  
Expected: PASS.

- [ ] **Step 6: Commit the objective**

```bash
git add src/training/minivgent_losses.py tests/minivgent/test_losses.py
git commit -m "feat: add minivgent answer anchor losses"
```

### Task 8: Compose, Freeze, Count, and Checkpoint the Full Model

**Files:**
- Create: `src/models/minivgent/model.py`
- Create: `src/models/minivgent/checkpoint.py`
- Create: `tests/minivgent/test_model.py`
- Create: `tests/minivgent/test_checkpoint.py`

**Interfaces:**
- Produces: `MiniVGentModel.forward(qwen_inputs, candidate_batch) -> MiniVGentOutput`.
- Produces: `trainable_parameters(model) -> Iterable[nn.Parameter]` excluding Qwen.
- Saves only added-module weights, config, model lock, and provenance.
- Expected two-block trainable count: `42_251_713`.

- [ ] **Step 1: Write failing end-to-end fake-backbone tests**

```python
def test_full_two_block_trainable_count() -> None:
    model = full_model_fixture(candidate_interaction=True)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 42_251_713


def test_backward_updates_added_modules_only() -> None:
    model = full_model_fixture(candidate_interaction=True)
    loss = model(fake_qwen_inputs(), candidate_batch_fixture()).logits.sum()
    loss.backward()
    assert all(p.grad is None for p in model.backbone.parameters())
    assert all(p.grad is not None for p in model.added_parameters())
```

- [ ] **Step 2: Confirm failure**

Run: `python -m pytest -q tests/minivgent/test_model.py tests/minivgent/test_checkpoint.py --tb=short`  
Expected: FAIL because the composition and checkpoint modules do not exist.

- [ ] **Step 3: Compose the exact forward path**

Run frozen Qwen under no-grad, reconstruct the layer-28 visual grid, pool
candidate members, embed OCR through the frozen token embedding, build
candidate states, decode against `(H_14,H_28)`, and return logits plus shape and
mask diagnostics. Do not expose gold fields to `forward`.

- [ ] **Step 4: Implement added-weights-only checkpoints**

Save candidate encoder, decoder, final norm/head, exact config JSON and hash,
model lock, source candidate revision, training manifest hashes, optimizer and
scheduler state, seed, step, and validation metric. Reject restore against a
different Qwen revision, candidate revision, config hash, or type vocabulary.

- [ ] **Step 5: Add round-trip and optimizer-exclusion tests**

Test byte-stable metadata JSON, identical logits after save/load, no Qwen keys
in the added-weight state dict, no Qwen parameter IDs in the optimizer, exact
42,251,713 count, M0/M1 initialization identity before training, and a
16-record synthetic overfit that reduces the loss by at least 90%.

- [ ] **Step 6: Run focused tests**

Run: `python -m pytest -q tests/minivgent/test_model.py tests/minivgent/test_checkpoint.py --tb=short`  
Expected: PASS.

- [ ] **Step 7: Commit model composition**

```bash
git add src/models/minivgent/model.py src/models/minivgent/checkpoint.py \
  tests/minivgent/test_model.py tests/minivgent/test_checkpoint.py
git commit -m "feat: compose frozen qwen minivgent model"
```

### Task 9: Add the Real-Qwen GPU Preflight and Visual Overlay Audit

**Files:**
- Create: `scripts/minivgent/__init__.py`
- Create: `scripts/minivgent/preflight_qwen_memory.py`
- Create: `scripts/minivgent/render_visual_grid_audit.py`
- Create: `tests/minivgent/test_preflight_cli.py`
- Create: `tests/minivgent/test_visual_audit.py`

**Interfaces:**
- Produces: `model_lock.json`, `preflight_report.json`, and PNG visual-grid overlays under an explicit scratch output root.
- Runs no training and reads no V1 labels.
- Refuses a login node and an output path inside Git.

- [ ] **Step 1: Write failing CLI contract tests**

Test refusal of mutable revisions, missing allocation markers, Git-contained
output paths, video input, multiple images, overwritten reports, and a
preflight report missing any required provenance field.

- [ ] **Step 2: Confirm failure**

Run: `python -m pytest -q tests/minivgent/test_preflight_cli.py tests/minivgent/test_visual_audit.py --tb=short`  
Expected: FAIL because the preflight scripts do not exist.

- [ ] **Step 3: Implement an offline synthetic-page generator**

Generate two deterministic 1,024-by-1,408 page images in the scratch output
root with labeled quadrants, a table, and disconnected text boxes. Hash every
image. This avoids making the Qwen interface preflight depend on V1 or a future
candidate manifest.

- [ ] **Step 4: Implement real-model parity checks**

For both pages, compare the local yes/no score with the official wrapper path,
then compare selected taps with `output_hidden_states=True`:

- hook layer 14 versus all-hidden-state entry 14;
- normalized outer layer 28 versus all-hidden-state final entry;
- BF16 score and state tolerances `atol=5e-3, rtol=5e-3`;
- exact image-token count and row-major grid shape; and
- zero Qwen gradients after one decoder backward.

The all-hidden-state path is a parity oracle only and is released before the
normal selected-tap memory measurement.

- [ ] **Step 5: Implement overlay rendering and report output**

Render merged-grid cell centers, the original member boxes, and pooled-member
highlights. The JSON report records revisions, package versions, GPU, dtype,
attention backend, prompt and image hashes, tap shapes, maximum parity deltas,
token counts, parameter counts, peak allocated/reserved CUDA memory, and wall
time. Reports are create-once.

- [ ] **Step 6: Run CPU CLI tests**

Run: `python -m pytest -q tests/minivgent/test_preflight_cli.py tests/minivgent/test_visual_audit.py --tb=short`  
Expected: PASS without loading Qwen.

- [ ] **Step 7: Run the authorized GPU preflight**

From an approved `htc` or `public` GPU allocation:

```bash
python scripts/minivgent/preflight_qwen_memory.py \
  --config configs/minivgent/qwen3_vl_reranker_2b_two_block.json \
  --output-root "$MINIVGENT_RUN_ROOT/preflight"
```

Expected: both score comparisons and both layer comparisons pass, image-token
counts are exact, all overlays are created, the backbone has no gradients, and
the report contains measured memory and latency. Any mismatch or OOM stops the
stage and preserves the failure report.

- [ ] **Step 8: Manually review every generated overlay**

Record `pass` or a precise failure reason for quadrant orientation, table box,
disconnected members, border boxes, and thin boxes. Do not advance on a
transposed/flipped grid or a pooled gap.

- [ ] **Step 9: Commit code and small validated reports only**

```bash
git add scripts/minivgent/__init__.py \
  scripts/minivgent/preflight_qwen_memory.py \
  scripts/minivgent/render_visual_grid_audit.py \
  tests/minivgent/test_preflight_cli.py tests/minivgent/test_visual_audit.py
git commit -m "test: add minivgent qwen memory preflight"
```

Do not commit generated images, model locks containing machine-local paths,
GPU logs, caches, or model weights.

### Task 10: Integrate the Frozen V1 Candidate Revision

**Prerequisite:** The active segment-reranker baseline has emitted and frozen
its candidate, relation, eligibility, exclusion, negative-audit, oracle, and
manifest artifacts. Stop if this prerequisite is absent.

**Files:**
- Create: `src/data/minivgent_manifest.py`
- Create: `scripts/minivgent/build_minivgent_view.py`
- Create: `tests/minivgent/test_manifest.py`
- Create: `configs/minivgent/v1_answer_anchor_view.json`

**Interfaces:**
- Consumes: immutable V1 plus one exact `candidate_revision` and its audited derived manifests.
- Produces: `minivgent_questions.jsonl`, `minivgent_candidates.jsonl`, `minivgent_targets.jsonl`, `excluded_questions.jsonl`, and `minivgent_manifest.json`.
- Mutates neither V1 nor the reranker artifacts.

- [ ] **Step 1: Write failing manifest-join tests**

Test document-group split preservation, one supplied page image per question,
candidate order determinism, member-box preservation, alternative-positive
OR groups, verified negatives only from audited fields, holdout exclusion, and
one exclusion row for every ineligible source question.

- [ ] **Step 2: Confirm failure**

Run: `python -m pytest -q tests/minivgent/test_manifest.py --tb=short`  
Expected: FAIL because `minivgent_manifest.py` does not exist.

- [ ] **Step 3: Implement the derived view**

Join by canonical IDs and exact candidate revision. Preserve candidate ID,
page ID, page-image path and hash, member boxes, member reading orders, text,
kind, split, accepted alternative ID, and audited relation label. Sort
candidates by `(first_reading_order, candidate_id)`. Tokenize OCR with the
Qwen tokenizer only at data-load time; do not serialize model tensors into the
manifest.

- [ ] **Step 4: Enforce label boundaries**

Only `positive_anchor` enters positive masks. Only audited ordinary or hard
`non_anchor` enters verified-negative masks. Partial anchors, unresolved
relations, unverified context, and source conflicts are masked or cause a
recorded exclusion according to the frozen reranker manifest; they never
become negatives by default.

- [ ] **Step 5: Emit an oracle and systems audit**

Report questions and candidates by split/source/type, candidate-count and
member-count quantiles, Qwen token-length estimates, unknown type count,
member-overlap rate, positive/verified-negative/masked counts, strict
candidate-oracle Recall@K, and every exclusion reason. Hash all inputs and
outputs.

- [ ] **Step 6: Run focused tests and a 32-question compute smoke**

Run: `python -m pytest -q tests/minivgent/test_manifest.py --tb=short`  
Expected: PASS.

Run inside the approved allocation:

```bash
python scripts/minivgent/build_minivgent_view.py \
  --config configs/minivgent/v1_answer_anchor_view.json \
  --candidate-root "$RERANKER_RUN_ROOT/frozen_candidates" \
  --output-root "$MINIVGENT_RUN_ROOT/view_smoke" \
  --max-questions 32 \
  --verify-hashes
```

Expected: deterministic derived files, no V1 writes, no holdout rows, and every
training row has at least one positive and one verified negative.

- [ ] **Step 7: Commit the data adapter**

```bash
git add src/data/minivgent_manifest.py \
  scripts/minivgent/build_minivgent_view.py \
  tests/minivgent/test_manifest.py \
  configs/minivgent/v1_answer_anchor_view.json
git commit -m "feat: build minivgent v1 derived view"
```

### Task 11: Add the Online Frozen-Qwen Trainer

**Files:**
- Create: `src/training/minivgent_trainer.py`
- Create: `scripts/minivgent/train_minivgent.py`
- Create: `tests/minivgent/test_trainer.py`
- Create: `configs/minivgent/two_block_screen.json`

**Interfaces:**
- Runs online frozen-Qwen forwards; does not cache full hidden states.
- Trains added modules only.
- Supports identical M0/M1 initialization, row order, optimizer, and schedule.
- Produces create-once checkpoints and metrics under an explicit run root.

- [ ] **Step 1: Write failing optimizer, resume, and matching tests**

```python
def test_optimizer_contains_only_added_parameters() -> None:
    model = full_model_fixture(candidate_interaction=True)
    optimizer = build_optimizer(model, trainer_config())
    optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert optimized == {id(p) for p in model.added_parameters()}


def test_m0_and_m1_start_from_identical_state() -> None:
    m0, m1 = matched_arm_fixture(seed=1729)
    assert m0.added_state_dict().keys() == m1.added_state_dict().keys()
    for key in m0.added_state_dict():
        torch.testing.assert_close(m0.added_state_dict()[key], m1.added_state_dict()[key])
```

- [ ] **Step 2: Confirm failure**

Run: `python -m pytest -q tests/minivgent/test_trainer.py --tb=short`  
Expected: FAIL because the trainer does not exist.

- [ ] **Step 3: Implement the frozen smoke schedule**

Use AdamW with learning rate `1e-4`, betas `(0.9,0.95)`, weight decay `0.05`,
5% linear warmup followed by cosine decay, gradient clip `1.0`, BF16 autocast,
microbatch `1`, gradient accumulation `16`, three epochs, seed `1729`, and
validation at each epoch end. Select the checkpoint by validation Recall@1;
ties prefer the earlier step.

The data loader groups no gold fields into model inputs, preserves the frozen
question and candidate order, and records actual candidate and token counts.

- [ ] **Step 4: Implement safe resume and finite-gradient guards**

Resume only from an exact config/model/candidate/manifest hash match. Before
every optimizer step, reject non-finite loss or gradients. After backward,
assert again that every Qwen gradient is `None`. Record component losses,
learning rate, gradient norm, wall time, Qwen forward time, decoder time, peak
memory, and skipped/failed rows.

- [ ] **Step 5: Add deterministic tiny-run tests**

Using the fake backbone, assert identical two-run metrics under the same seed,
correct accumulation, no validation optimizer update, create-once output
semantics, exact resume, and a failure on changed hashes.

- [ ] **Step 6: Run focused tests**

Run: `python -m pytest -q tests/minivgent/test_trainer.py --tb=short`  
Expected: PASS.

- [ ] **Step 7: Run the authorized 16-question real-Qwen overfit gate**

Run M1 on 16 training questions selected deterministically from the derived
view. Require at least 90% training-loss reduction, Recall@1 at least 0.95 on
that same diagnostic batch, finite gradients in every added module, zero Qwen
gradients, and a successful added-weights checkpoint round trip. This is a
systems gate, not a reported scientific result.

- [ ] **Step 8: Commit the trainer**

```bash
git add src/training/minivgent_trainer.py \
  scripts/minivgent/train_minivgent.py tests/minivgent/test_trainer.py \
  configs/minivgent/two_block_screen.json
git commit -m "feat: train minivgent on frozen qwen memory"
```

### Task 12: Add Matched Ranking Evaluation and the Two-Arm Screen

**Files:**
- Create: `src/evaluation/minivgent_metrics.py`
- Create: `scripts/minivgent/evaluate_minivgent.py`
- Create: `tests/minivgent/test_metrics.py`
- Modify: `src/evaluation/README.md`

**Interfaces:**
- Produces one scalar score and rank for every valid frozen candidate.
- Reuses the active reranker evaluator's query-macro conventions when that implementation exists.
- Compares M0 and M1 on byte-identical validation and internal-test manifests.

- [ ] **Step 1: Write failing ranking and paired-comparison tests**

Test alternative positives, Recall@1/3/5, MRR, nDCG@5, query-macro averaging,
document-clustered paired bootstrap, selected-candidate cost, missing-query
rejection, duplicate-candidate rejection, mixed-revision rejection, and
holdout refusal.

- [ ] **Step 2: Confirm failure**

Run: `python -m pytest -q tests/minivgent/test_metrics.py --tb=short`  
Expected: FAIL because `minivgent_metrics.py` does not exist.

- [ ] **Step 3: Implement matched ranking output**

Write prediction rows with run/config/model/candidate hashes, question,
document, page, candidate, score, rank, label class, and cost fields. Do not
write answer text or gold boxes into the model input log. Fit any binary
threshold on validation only; ranking metrics require no threshold.

- [ ] **Step 4: Implement document-clustered uncertainty**

Use seed `1729` and 10,000 document-cluster bootstrap replicates for M1-minus-M0
Recall@1 and MRR. Keep all questions belonging to each resampled document.
Report point estimates and 95% percentile intervals.

- [ ] **Step 5: Run focused tests**

Run: `python -m pytest -q tests/minivgent/test_metrics.py --tb=short`  
Expected: PASS.

- [ ] **Step 6: Run the one-seed two-block screen only after every gate passes**

Train M0 and M1 from byte-identical added-module initialization with identical
rows, order, loss, optimizer, and schedule. Evaluate both on the same frozen
validation manifest. Promote four-block M1 only if:

- M1 is operationally stable;
- candidate-oracle coverage supports the selector claim;
- M1 improves validation Recall@1 or MRR over M0 without a material regression
  in the other; and
- measured memory and latency fit the approved allocation.

If not promoted, evaluate the already-trained two-block M0/M1 once on the
internal test and report the negative or inconclusive result. Do not tune on
internal test or holdout.

- [ ] **Step 7: Commit evaluation code**

```bash
git add src/evaluation/minivgent_metrics.py \
  scripts/minivgent/evaluate_minivgent.py \
  tests/minivgent/test_metrics.py src/evaluation/README.md
git commit -m "feat: evaluate matched minivgent arms"
```

### Task 13: Verify the Implementation Handoff

**Files:**
- Modify: `agent-context/research/architecture_registry/proposals/minivgent_answer_anchor_poc.md`
- Modify: `agent-context/research/architecture_registry/experiments/README.md`
- Create only after authorization: `agent-context/research/architecture_registry/experiments/minivgent_answer_anchor_poc.md`
- Modify only under a binding handoff: `sol/CURRENT_SOL_TASK.md`

- [ ] **Step 1: Run the complete focused unit suite**

Run:

```bash
python -m pytest -q tests/minivgent --tb=short
```

Expected: PASS with real-model tests skipped unless their explicit GPU marker
and local model lock are supplied.

- [ ] **Step 2: Run repository policy tests**

Run:

```bash
python -m pytest -q \
  tests/test_repository_structure.py \
  tests/test_repository_artifact_policy.py \
  tests/test_documentation_contract.py \
  --tb=short
```

Expected: PASS; no generated artifact, model weight, cache, environment, or
Slurm log is tracked.

- [ ] **Step 3: Audit source and claim boundaries**

Verify that the implementation report distinguishes:

- official Qwen and library facts;
- project architecture choices;
- measured preflight observations;
- answer-anchor versus complete-evidence labels;
- supplied-page versus document-level retrieval; and
- single-hop localization versus multi-hop reasoning.

- [ ] **Step 4: Inspect the final diff and unreachable code**

Run `git diff --check`, inspect `git status --short`, and review every changed
file. Report any newly unreachable path before deleting it. Do not modify or
discard unrelated owner changes.

- [ ] **Step 5: Update authority files only after owner approval**

Create an experiment record only when the owner schedules or runs the
experiment. Update `sol/CURRENT_SOL_TASK.md` only when the handoff pins exact
commits, checkout, scratch root, environment, model lock, candidate revision,
commands, and recovery authority. Planning documents alone do not close these
gates.

- [ ] **Step 6: Commit the reviewed handoff**

Run `git branch --show-current`, then commit only the intentional
implementation and handoff files on the approved branch. Do not add co-author
lines.
