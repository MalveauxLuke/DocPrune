# MiniVGent Architecture

## Role and scientific boundary

MiniVGent is the shared-page parallel candidate selector in the unified
program:

- **M0:** frozen full-page Qwen memory plus candidate cross-attention, with all
  off-diagonal candidate communication masked;
- **M1:** the same architecture, initialization, rows, objective, and schedule,
  with bidirectional candidate self-attention enabled.

The transferred VGent idea is:

```text
frozen question-conditioned multimodal memory
  -> content-rich candidate queries
  -> candidate-to-memory cross-attention
  -> candidate self-attention
  -> parallel candidate logits
```

VGent demonstrated this pattern for natural-image grounding with detector box
proposals and a frozen MLLM, not document answer-anchor selection. Its
QuadThinker reward used known spatial/count/coordinate targets, not downstream
answer correctness. The project architecture below is therefore a documented
adaptation, not a reproduction.

The VGent ablation in which joint encoder/decoder unfreezing reduced F1 from
60.55 to 45.66 supports the first-stage decision to freeze Qwen. Its reported
90.1% average single-target REC result makes a single-anchor systems POC
legitimate, but V1 cannot validate complete multi-evidence complementarity.

## Backbone and source lock

Approved starting pins:

| Component | Pin |
|---|---|
| Qwen model | `Qwen/Qwen3-VL-Reranker-2B@4bd860ac4f15ad1897a214615cccc700f8f71818` |
| Official wrapper | `QwenLM/Qwen3-VL-Embedding@393e2978d27852b0d0230d6994f37f9c15bed73c` |
| Python | `3.11` |
| PyTorch | `2.8.*` |
| Torchvision | `0.23.*` |
| Transformers | `4.57.3` |
| Accelerate | `1.12.*` starting environment |
| `qwen-vl-utils` | `0.0.14` |
| Attention | `sdpa` first; FlashAttention 2 only after parity |
| Backbone dtype | BF16 |
| KV cache | disabled |
| Image budget | at most 1,800 merged visual tokens |

The Qwen language width is 2,048 with 28 decoder layers. Stage 02 records the
actually resolved package and artifact locks. The technical report's
1,280-visual-token training cap and the released wrapper's 1,800 default are
distinct; this program starts at 1,800 for wrapper parity and reports actual
token counts.

Do not use `torch.compile` in the first smoke because selective hooks are part
of the contract. FlashAttention is a later profile, not an environment gate.

## Official prompt and score diagnostic

Use the same official task instruction and sequence order as R0/R1:

```text
Given a search query, retrieve relevant candidates that answer the query.

system instruction
task instruction
original query
Document prefix
one full supplied-page image
assistant-generation prefix
```

The query precedes the image. Qwen's causal language mask therefore permits
page-image token states to depend on the earlier query, and final prompt states
to depend on both query and page.

The MiniVGent adapter retains the official scalar diagnostic:

```text
h_final = lm.model(**inputs).last_hidden_state[:, -1, :]
score = sigmoid((lm_head[yes] - lm_head[no]) dot h_final)
```

Before memory extraction, it must reproduce the official wrapper on identical
prompt and pixels within `atol=5e-3, rtol=5e-3` in BF16 on the same device.

## Frozen Qwen memory extraction

The full model supports `output_hidden_states=True`, but that path retains the
embedding plus all 28 layer outputs: 29 full `[B,T,2048]` tensors. Use it only
as a parity oracle.

Normal execution uses scoped forward hooks for selected intermediate layers,
the normalized final `last_hidden_state` for layer 28, and removes all hooks in
a `finally` block. Human layer `n` maps to zero-based module:

```text
lm.model.language_model.layers[n - 1]
```

The adapter returns:

```text
PageMemory:
  memories: tuple[Tensor[B,T,2048], ...]
  memory_mask: BoolTensor[B,T]
  final_hidden: Tensor[B,T,2048]
  image_token_mask: BoolTensor[B,T]
  image_grid_thw: LongTensor[B,3]
```

Tap semantics:

- `H_7`, `H_14`, and `H_21` are direct outputs of modules 6, 13, and 20 before
  final RMSNorm;
- `H_28` is normalized `last_hidden_state` after final RMSNorm;
- two-block smoke/screen taps are `(H_14, H_28)`;
- four-block treatment taps are `(H_7, H_14, H_21, H_28)`; and
- every block owns its pre-cross-attention LayerNorm and Q/K/V projections, so
  the mixed pre-final/final normalization is explicit.

Run the complete backbone in evaluation mode with every parameter
`requires_grad=False`. Use `torch.no_grad()`, detach the memories, then return
to normal grad mode for added modules. Do not use `torch.inference_mode()` for
tensors that subsequently enter autograd-recorded decoder operations.

Cross-attention consumes the complete non-padding language sequence at each
tap, not only image tokens. Candidate visual initialization alone uses the
image-token subset of normalized H28.

## Visual grid and member ROI

For processor output `image_grid_thw=[t,h,w]` and Qwen
`spatial_merge_size=2`:

```text
language image-token count = t * h * w / 4
merged visual grid          = [t, h/2, w/2, 2048]
nominal resized-image stride = 32 x 32 pixels
```

The 32-pixel nominal stride follows from patch size 16 and spatial merge 2.
The inspected Qwen implementation uses merged-height then merged-width
row-major order.

The POC accepts exactly one image and no video per sample. For each sample:

1. assert `image_token_mask.sum() == t*h*w/4`;
2. gather normalized H28 at image-token positions;
3. require `t == 1`;
4. reshape to `[2048,h/2,w/2]`; and
5. map normalized page coordinates to the merged grid:

```text
x_grid = x_1000 / 1000 * (w/2)
y_grid = y_1000 / 1000 * (h/2)
```

Pool every actual member box with:

```text
torchvision.ops.roi_align(
  output_size=(2,2), spatial_scale=1.0, aligned=True
)
```

Average the four ROI cells, then area-weight the member vectors. Never pool the
candidate envelope in place of disconnected members. Exact duplicate boxes are
deduplicated. Measure non-identical member overlap; retain the area-weighted
member rule for this POC without calling it an exact union integral.

## Candidate tensor contract

There is no second vision encoder. Visual features come from the
question-conditioned Qwen H28 grid. OCR features reuse the frozen Qwen input
embedding table.

For batch `B`, maximum candidates `C`, maximum members `M`, and OCR limit
`L=128`:

```text
member_boxes_1000: [B,C,M,4]
member_mask:       [B,C,M]
ocr_token_ids:     [B,C,128]
ocr_token_mask:    [B,C,128]
candidate_type:    [B,C]
candidate_mask:    [B,C]

visual ROI:        V in R[B,C,2048]
OCR mean:          O in R[B,C,2048]
geometry:          G in R[B,C,14]
type embedding:    E in R[B,C,64]
candidate state:   U_0 in R[B,C,1024]
```

OCR uses `add_special_tokens=false`. Above 128 tokens, retain first 64 and last
64. Mean-pool frozen Qwen embeddings over non-padding tokens. Empty OCR maps to
a learned trainable 2,048-wide `empty_ocr` vector, never to the pad embedding.

Fixed candidate types:

```text
PAD=0
document_preamble=1
headed_text=2
deepseek_paragraph=3
visual_bundle=4
UNKNOWN=5
```

The 14 inference-time geometry values are:

```text
envelope x0, y0, x1, y1
envelope center x, center y
envelope width, height, area
member-box geometric-union area
union/envelope fill ratio
first reading order / page candidate count
last reading order / page candidate count
log1p(member count) / log1p(32)
```

Coordinates/areas are `[0,1]`; member count clips at 1. Union area is geometric
union, not summed box area.

## Exact candidate encoder

```text
v = GELU(Linear(2048,512)(LayerNorm(V)))
o = GELU(Linear(2048,512)(LayerNorm(O)))
g = Linear(128,128)(GELU(Linear(14,128)(G)))
e = Embedding(6,64)(candidate_type)
U_0 = Linear(1024,1024)(Dropout(0.1)(GELU(
        Linear(1216,1024)(LayerNorm([v;o;g;e])))))
```

With biased linear layers and the trainable `empty_ocr`, this encoder has
4,425,472 trainable parameters.

## Exact decoder

Two-block engineering and screen configuration:

```text
width:              1024
blocks:             2
memory taps:        [14,28]
heads:              16
head width:         64
SwiGLU hidden:      2736
dropout:            0.1
relative bias:      none
```

Four-block promoted configuration uses identical blocks and taps
`[7,14,21,28]`.

Each block performs:

```text
U <- U + CrossAttention(
       Q=LN(U), K=H_l, V=H_l,
       q_width=1024, kv_width=2048,
       key_padding_mask=~memory_mask)
U <- U + CandidateSelfAttention(
       LN(U), key_padding_mask=~candidate_mask)
U <- U + SwiGLU(LN(U), hidden_width=2736)
```

M1 permits all valid candidates to attend bidirectionally. M0 uses the same
self-attention module and parameters but masks every off-diagonal candidate
pair. Because one-token attention weakly identifies some Q/K paths, M0 is an
operational interaction control, not a perfect causal capacity match.

No relative structural bias is used initially: the candidate contract has no
trusted hierarchy and U0 already includes geometry and reading order. A bias is
a separately named future ablation only after verified relation types exist.

Output:

```text
candidate_logits = Linear(1024,1)(LayerNorm(U)).squeeze(-1)  # [B,C]
```

Padding is masked in cross/self-attention, target construction, loss, ranking,
and serialization. Do not overwrite padded training logits with infinity.

One decoder block has 18,911,584 trainable parameters. Exact complete counts:

| Configuration | Trainable parameters |
|---|---:|
| two-block M0/M1 | 42,251,713 |
| four-block M0/M1 | 80,074,881 |

The older approximate 75.6M or 77M statements described a rounded four-block
decoder core, not the complete model. Runtime asserts the exact count.

## Target tensors

```text
alternative_positive_mask: Bool[B,A,C]
alternative_mask:          Bool[B,A]
verified_negative_mask:    Bool[B,C]
candidate_mask:            Bool[B,C]
```

Accepted alternatives are OR-equivalent. Partial anchors, unresolved
candidates, and unverified context appear in neither positive nor negative
masks. Every training row must contain at least one active positive alternative
and one verified negative; otherwise manifest validation fails.

## Exact training loss

Collapse active alternatives by OR to positive candidate set `P`:

```text
L_or   = -log(1 - product_{i in P}(1 - sigmoid(z_i)))
s_pos  = 0.1 * logsumexp(z_i / 0.1 for i in P)
N_8    = eight highest-logit verified negatives, or all if fewer than eight
L_rank = mean_{j in N_8} softplus(0.2 - s_pos + z_j)
L_neg  = mean_{j in verified negatives} BCEWithLogits(z_j, 0)

L = L_or + 0.5 * L_rank + 1.0 * L_neg
```

Compute OR stably with `logsigmoid`/`log1p` identities and only
floating-point-epsilon clamping needed to avoid `log(0)`. Test finite forward
and backward behavior at logits `-80`, `0`, and `80`.

These are project smoke defaults, not paper hyperparameters. Do not use a
candidate softmax, literal count target, or make every overlapping alternative
a mandatory positive. Any tuning uses validation only and creates a new config
hash.

## Training schedule and freezing

The first real trainer uses online frozen-Qwen forwards and optimizes added
modules only. The existing implementation plan proposes:

```text
optimizer: AdamW
learning rate: 1e-4
betas: (0.9,0.95)
weight decay: 0.05
warmup: 5% linear, then cosine
gradient clip: 1.0
dtype: BF16 autocast
microbatch: 1
gradient accumulation: 16
epochs: 3 for the one-seed screen
seed: 1729
selection: validation Recall@1, earlier step on ties
```

The POC implementation plan must either retain these values or register
any change before training. It must reject non-finite loss/gradients, assert
zero Qwen gradients after backward, and log component loss, learning rate,
gradient norm, wall time, Qwen time, decoder time, peak memory, and failures.

M0 and M1 begin from byte-identical added-module initialization and use
identical rows, order, loss, optimizer, and schedule. Checkpoints contain only
added-module weights plus immutable model/environment/config/candidate/view
locks. Resume requires exact hash equality.

## Online versus cached execution

The Qwen page memory is question-conditioned. At sequence length 1,800, one
BF16 `[T,2048]` tap is roughly 7.4 MB before container/shard overhead. Roughly
41,000 training questions would require about 600 GB for two taps or 1.2 TB for
four taps. These are order-of-magnitude estimates; measure actual lengths.

The binding POC therefore uses one online frozen page forward per question,
independent of candidate count. Start at microbatch 1 with accumulation and
change batching only from measured candidate count, sequence length, memory,
and latency. Full hidden-state caching is a separate systems experiment.

Any future cache key must include page hash, exact question, prompt hash,
processor, model revision, environment lock, tap semantics, dtype, and image
grid. Hidden states, weights, caches, and checkpoints never enter Git.

## Mandatory pre-training verification

Data training cannot begin until all checks pass:

1. official score-path parity on identical model, prompt, and pixels;
2. all-hidden-state versus selective-tap parity for layers 14 and 28;
3. exact image-token count from `image_grid_thw`;
4. synthetic row-major grid reconstruction;
5. rendered candidate-to-grid overlay audit;
6. full-page, border, thin-box, table, and disconnected-member ROI tests;
7. M0 off-diagonal isolation;
8. M1 permutation equivariance when all candidate fields/outputs permute;
9. zero backbone gradients and optimizer parameters;
10. finite nonzero gradients in every trainable module;
11. exact trainable parameter count;
12. added-weights-only checkpoint round trip with locks; and
13. deterministic 16-question overfit.

The real-Qwen preflight report records exact revisions, packages, GPU, prompt
and image hashes, tap shapes, parity deltas, visual-token counts, trainable and
loaded parameters, peak allocated/reserved memory, wall time, and manual
overlay verdicts. Any mismatch or OOM preserves a failure report and stops the
stage.

The 16-question real overfit gate requires at least 90% training-loss reduction,
Recall@1 at least 0.95 on that diagnostic batch, finite gradients in all added
modules, zero Qwen gradients, and successful checkpoint round trip. It is a
systems gate, not a scientific result.

## Depth promotion

Execution order:

1. build the two-block model and synthetic tests;
2. pass the real-Qwen preflight and overfit gate;
3. run the one-seed two-block M0/M1 screen;
4. promote four-block M1 only if two-block M1 is stable, improves validation
   Recall@1 or MRR over M0 without material regression in the other, and fits
   the approved allocation; and
5. do not build six blocks or copy a 2,048-wide decoder before the compact
   treatment demonstrates value.

## Interpretation

- `M0 > R1` supports a shared, query-conditioned page memory benefit over
  repeated independent candidate crops.
- `M1 > M0` supports an off-diagonal candidate-interaction benefit for
  competition or redundancy on single-hop data.
- `M1` gains only when OCR contains the answer indicate an OCR shortcut, not
  stronger visual understanding.
- `M1 > M0` on V1 does not prove complementary multi-evidence reasoning.
- `M1 <= M0` means shared memory may help while explicit interaction does not.
- all MiniVGent arms `<= R1` means retain the simpler pairwise reranker.

## Excluded features

The authorized architecture has no second vision encoder, Qwen LoRA/backbone
updates, count head, cardinality objective, candidate softmax, box regression,
free-form coordinate generation, answer generation, answer-feedback reward,
synthetic multi-hop, or invented hierarchy. Each addition would be a new
architecture and experiment arm.
