# MiniVGent Qwen Implementation Readiness

## Status and authority

- Status: `implementation_ready_proposal`
- Research date: 2026-08-14
- Execution authorization: none
- Intended backbone: `Qwen/Qwen3-VL-Reranker-2B`
- Intended first implementation: two-block MiniVGent engineering smoke
- Intended data dependency: the frozen candidate revision produced by the
  approved overlap-first V1 segment-reranker baseline

This record resolves the model-interface and tensor-design questions needed
for an implementation agent. It does not authorize model download, source-data
processing, MiniVGent implementation, inference, training, or SOL submission.
The active segment-reranker workspace and SOL handoff still come first.

## Decision

Qwen3-VL-Reranker-2B can expose the language-layer hidden states MiniVGent
needs. The pinned model configuration has language width 2,048 and 28 layers.
The pinned Hugging Face implementation registers
`Qwen3VLTextDecoderLayer` as a hidden-state source, and its decorated forward
path supports `output_hidden_states=True`.
[Model configuration](https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B/blob/main/config.json),
[pinned Qwen3-VL source](https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L497-L510)

Do not use the all-hidden-states result as MiniVGent's normal training path.
At the pinned Transformers revision, hidden-state recording captures the input
embedding and every decoder-layer output, then injects the tuple into the model
output. That is useful as a parity oracle, but it retains 29 full
`[B,T,2048]` tensors when MiniVGent needs only two or four. [Transformers
capture implementation](https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/utils/generic.py#L871-L1012)

The implementation contract is therefore:

1. use `output_hidden_states=True` in one integration test to verify layer
   numbering and values;
2. use scoped forward hooks for intermediate selected layers in the normal
   path;
3. use the model's normalized `last_hidden_state` for layer 28; and
4. remove every hook in a `finally` block after each forward.

For the two-block smoke, capture human-numbered layers `14` and `28`. For the
four-block treatment, capture `7`, `14`, `21`, and `28`. Human layer `n` maps
to zero-based module `lm.model.language_model.layers[n - 1]`.

### Paper, checkpoint, and project-design boundary

The Qwen technical report establishes a pointwise multimodal reranker built on
Qwen3-VL with causal attention and a yes/no relevance decision. It does not
specify a MiniVGent decoder, selected intermediate taps, document ROI pooling,
or this project's answer-anchor loss. Those are project proposals derived from
the released checkpoint and inspected implementation. [Qwen3-VL Embedding and
Reranker technical report](https://arxiv.org/pdf/2601.04720v2)

The report describes dynamic-resolution training with a 1,280-visual-token
cap, while the released official wrapper defaults to 1,800. This plan selects
1,800 for released-wrapper parity, records actual token counts, and does not
present that project choice as a paper hyperparameter.

## Frozen source and environment proposal

These are proposed research pins. The binding model lock must repeat them and
record the actually resolved artifacts after execution is authorized.

| Component | Proposed pin | Reason |
|---|---|---|
| Model | [`Qwen/Qwen3-VL-Reranker-2B@4bd860ac4f15ad1897a214615cccc700f8f71818`](https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B/tree/4bd860ac4f15ad1897a214615cccc700f8f71818) | Immutable released 2B reranker revision observed on 2026-08-14 |
| Official wrapper source | `QwenLM/Qwen3-VL-Embedding@393e2978d27852b0d0230d6994f37f9c15bed73c` | Freezes prompt, preprocessing, and yes/no score extraction |
| Python | `3.11` | The official package requires Python 3.11 or newer |
| PyTorch | `2.8.*` | Official implementation pin |
| Torchvision | `0.23.*` | Matching ROI operator family for PyTorch 2.8 |
| Transformers | `4.57.3` | Minimum official repository version and inspected source baseline |
| `qwen-vl-utils` | `0.0.14` | Official Qwen3-VL preprocessing baseline |
| Attention backend | `sdpa` | Lowest-dependency first smoke; no FlashAttention build gate |
| Backbone dtype | `bfloat16` | Released checkpoint dtype and official usage |
| Cache | `use_cache=false` | No autoregressive decode is needed |
| Image budget | at most 1,800 merged visual tokens | Matches the released wrapper's `MAX_PIXELS` contract |

The official repository currently declares Python `>=3.11`, Torch `2.8.*`,
Torchvision `>=0.23.0`, Transformers `>=4.57.3`, Accelerate `>=1.12.0`, and
`qwen-vl-utils>=0.0.14`. The experiment environment should use exact resolved
versions instead of open-ended lower bounds. [Official package
metadata](https://github.com/QwenLM/Qwen3-VL-Embedding/blob/393e2978d27852b0d0230d6994f37f9c15bed73c/pyproject.toml)

FlashAttention 2 may be profiled after the SDPA path passes. It is not a first
implementation dependency. `torch.compile` is also excluded from the first
smoke because selected forward hooks are part of the contract.

## Official score-path parity

The official wrapper loads `Qwen3VLForConditionalGeneration`, retains
`lm.model`, and constructs a fixed bias-free score head from
`lm_head["yes"] - lm_head["no"]`. It scores the final sequence position and
applies a sigmoid. [Official reranker
implementation](https://github.com/QwenLM/Qwen3-VL-Embedding/blob/393e2978d27852b0d0230d6994f37f9c15bed73c/src/models/qwen3_vl_reranker.py#L91-L125)

The MiniVGent adapter must preserve this path as a diagnostic method:

```text
h_final = lm.model(**inputs).last_hidden_state[:, -1, :]
score = sigmoid((lm_head[yes] - lm_head[no]) dot h_final)
```

Before extracting memories, the adapter must reproduce the official wrapper's
score on the same prompt and pixels within `atol=5e-3, rtol=5e-3` in BF16 on
the same device. A failure stops the preflight; it is not hidden by a looser
tolerance.

## Prompt and causal-order contract

Use the official wrapper's system prompt and formatting logic. Freeze this
instruction for the POC:

```text
Given a search query, retrieve relevant candidates that answer the query.
```

The sequence order is:

```text
system yes/no judgment instruction
task instruction
query text
Document prefix
full page image tokens
assistant-generation prefix
```

The official formatter places query content before document content and puts a
document image after its `Document` prefix. [Official prompt
formatter](https://github.com/QwenLM/Qwen3-VL-Embedding/blob/393e2978d27852b0d0230d6994f37f9c15bed73c/src/models/qwen3_vl_reranker.py#L299-L355)

Qwen3-VL's language model uses a causal mask. Therefore, page-image token
states can attend to the earlier query, while the last prompt positions can
attend to both query and page. [Pinned language-model
forward](https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L707-L791)

MiniVGent cross-attention uses the complete non-padding language sequence at
each selected tap, not only image tokens. Candidate ROI initialization uses
the image-token subset of normalized layer 28.

## Exact Qwen memory extraction

Let `D=2048`. The normal path returns:

```text
PageMemory:
  memories: tuple[Tensor[B,T,D], ...]
  memory_mask: BoolTensor[B,T]
  final_hidden: Tensor[B,T,D]
  image_token_mask: BoolTensor[B,T]
  image_grid_thw: LongTensor[B,3]
```

Semantics:

- `H_7`, `H_14`, and `H_21` are the direct outputs of zero-based decoder
  modules `6`, `13`, and `20`, before the model's final RMSNorm;
- `H_28` is `lm.model(...).last_hidden_state`, after the final RMSNorm;
- the two-block smoke uses `(H_14, H_28)`;
- the four-block treatment uses `(H_7, H_14, H_21, H_28)`; and
- each MiniVGent block has a separate pre-cross-attention LayerNorm and its own
  Q/K/V projections, so mixed pre-final and final normalization is explicit.

The backbone must be in evaluation mode with every parameter
`requires_grad=False`. Run it inside `torch.no_grad()`, detach the captured
memories, and then enter normal grad mode for the trainable decoder. Do not use
`torch.inference_mode()` for tensors that feed trainable layers: PyTorch states
that inference-mode tensors cannot later participate in autograd-recorded
computations, while no-grad outputs can. [PyTorch grad-mode
comparison](https://docs.pytorch.org/docs/stable/notes/autograd.html#grad-modes)

## Visual-token grid and candidate ROI contract

The pinned Qwen model replaces positions where
`input_ids == image_token_id` with vision features. It verifies that the image
feature and placeholder counts agree. [Pinned multimodal
forward](https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L962-L1068)

For one page image with processor output `image_grid_thw=[t,h,w]` and Qwen's
`spatial_merge_size=2`:

```text
number of language image tokens = t * h * w / 4
merged visual grid shape         = [t, h/2, w/2, 2048]
nominal image area per token      = 32 x 32 resized-image pixels
```

The 32-pixel nominal stride follows from vision patch size 16 and spatial merge
size 2. The Qwen source constructs merged height then merged width indices in
row-major order. [Pinned visual position
construction](https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L547-L580)

For the POC, require exactly one image and no video per sample. For each sample:

1. assert `image_token_mask.sum() == t*h*w/4`;
2. gather normalized `H_28` at the image mask;
3. require `t == 1`;
4. reshape to `[2048, h/2, w/2]`; and
5. map every member box from normalized page coordinates to the merged grid:

```text
x_grid = x_1000 / 1000 * (w/2)
y_grid = y_1000 / 1000 * (h/2)
```

Pool every actual member box with
`torchvision.ops.roi_align(output_size=(2,2), spatial_scale=1.0,
aligned=True)`, average the four output cells, then area-weight the member
vectors. Never substitute the candidate's enclosing rectangle; doing so would
inject unrelated content between disconnected members. Torchvision defines
ROI Align input as `[N,C,H,W]`, boxes as `(x1,y1,x2,y2)`, and output as
`[K,C,out_h,out_w]`. [Torchvision ROI Align](https://docs.pytorch.org/vision/stable/generated/torchvision.ops.roi_align.html)

The implementation must audit overlapping member boxes. Exact duplicate boxes
are deduplicated before pooling. If non-identical member overlap exists, record
the overlap rate and keep the existing area-weighted member rule for the POC;
do not silently claim it is an exact union integral.

## Exact candidate tensor design

The first implementation uses no second vision encoder. Candidate visual
features come from the question-conditioned Qwen layer-28 visual grid. OCR
features reuse the frozen Qwen token embedding table.

For maximum candidate count `C`, maximum member count `M`, and OCR limit
`L=128`:

```text
member_boxes_1000: [B,C,M,4]
member_mask:       [B,C,M]
ocr_token_ids:     [B,C,L]
ocr_token_mask:    [B,C,L]
candidate_type:    [B,C]
candidate_mask:    [B,C]

visual ROI:        V in R[B,C,2048]
OCR mean:          O in R[B,C,2048]
geometry:          G in R[B,C,14]
type embedding:    E in R[B,C,64]
candidate state:   U_0 in R[B,C,1024]
```

OCR tokenization uses `add_special_tokens=false`. If a candidate exceeds 128
tokens, retain the first 64 and last 64. Mean-pool frozen Qwen input embeddings
over non-padding tokens. An empty OCR field maps to a learned trainable
`empty_ocr` vector; it does not reuse the pad-token embedding.

The fixed candidate types are:

```text
PAD=0
document_preamble=1
headed_text=2
deepseek_paragraph=3
visual_bundle=4
UNKNOWN=5
```

The 14 geometry values are all inference-time fields:

```text
envelope x0, y0, x1, y1
envelope center x, center y
envelope width, height, area
member-box union area
union/envelope fill ratio
first reading order / page candidate count
last reading order / page candidate count
log1p(member count) / log1p(32)
```

Coordinates and areas are normalized to `[0,1]`; the member-count feature is
clipped to `1`. Member-box union area must use geometric union rather than the
sum of overlapping box areas.

The candidate encoder is fixed as:

```text
v = GELU(Linear(2048,512)(LayerNorm(V)))
o = GELU(Linear(2048,512)(LayerNorm(O)))
g = Linear(128,128)(GELU(Linear(14,128)(G)))
e = Embedding(6,64)(candidate_type)
U_0 = Linear(1024,1024)(Dropout(0.1)(GELU(
        Linear(1216,1024)(LayerNorm([v;o;g;e])))))
```

Including the 2,048-wide trainable `empty_ocr` vector, this encoder has
4,425,472 trainable parameters under biased linear layers.

## Exact decoder design

The engineering smoke uses:

```text
decoder width:       1024
decoder blocks:      2
Qwen memory taps:    [14,28]
attention heads:     16
head width:          64
SwiGLU hidden width: 2736
dropout:             0.1
relative bias:       none
```

The full treatment, if promoted, uses four identical blocks with taps
`[7,14,21,28]`. Each block performs:

```text
U <- U + CrossAttention(
       Q=LN(U), K=H_l, V=H_l,
       q_width=1024, kv_width=2048,
       key_padding_mask=~memory_mask)
U <- U + CandidateSelfAttention(
       LN(U), key_padding_mask=~candidate_mask)
U <- U + SwiGLU(LN(U), hidden_width=2736)
```

M1 permits every valid candidate to attend bidirectionally to every other
valid candidate. M0 uses the identical self-attention module and parameters but
masks every off-diagonal candidate pair. Because one-token attention makes its
Q/K paths weakly identified, M0 remains an operational interaction control,
not a perfect capacity-matched causal control.

No relative document-structure bias is used in the first implementation. The
current candidate contract has no trusted hierarchy, and geometry and reading
order are already present in `U_0`. A relative bias becomes a separate ablation
only after its relation types exist in the frozen candidate manifest.

The output is:

```text
candidate_logits = Linear(1024,1)(LayerNorm(U)).squeeze(-1)  # [B,C]
```

Padding is masked in attention, loss, ranking, and serialization. The model
does not overwrite padded logits with infinity during training.

One decoder block has 18,911,584 trainable parameters. With the fixed candidate
encoder, final LayerNorm, and output head, the expected counts are:

| Configuration | Expected trainable parameters |
|---|---:|
| Two-block engineering smoke | 42,251,713 |
| Four-block treatment | 80,074,881 |

The prior approximately 77M statement described only a rounded four-block
decoder core. It was not a complete model count. The implementation must assert
the exact count for each frozen config.

## Target and loss interface

The model output is a ranking over frozen candidates. The target object is:

```text
alternative_positive_mask: Bool[B,A,C]
alternative_mask:          Bool[B,A]
verified_negative_mask:    Bool[B,C]
candidate_mask:            Bool[B,C]
```

All accepted alternatives are OR-equivalent. `partial_anchor`, unresolved, and
unverified-context candidates appear in neither positive nor verified-negative
masks.

For each row, collapse active alternative masks with OR to the positive set
`P`. Use:

```text
L_or   = -log(1 - product_{i in P}(1 - sigmoid(z_i)))
s_pos  = 0.1 * logsumexp(z_i / 0.1 for i in P)
N_8    = the eight highest-logit verified negatives, or all if fewer than 8
L_rank = mean_{j in N_8} softplus(0.2 - s_pos + z_j)
L_neg  = mean_{j in verified negatives} BCEWithLogits(z_j, 0)

L = L_or + 0.5 * L_rank + 1.0 * L_neg
```

Compute the OR term with `logsigmoid` and `log1p` identities, clamp only at the
floating-point epsilon required to avoid `log(0)`, and test finite forward and
backward behavior at logits `-80`, `0`, and `80`. Rows without an accepted
positive or without a verified negative fail manifest validation instead of
silently changing the objective.

These are smoke defaults, not paper-derived hyperparameters. Any later tuning
uses validation only and records a new configuration hash.

## Online versus cached backbone execution

Use online frozen-Qwen forwards for the POC. Full page memories are
question-conditioned and large. At sequence length 1,800, one BF16
`[T,2048]` tap is about 7.4 MB before container and shard overhead. Caching two
taps for roughly 41,000 training questions is about 600 GB; four taps are about
1.2 TB. Actual storage must be computed from measured sequence lengths, but the
order of magnitude makes full-memory caching a separate systems experiment.

The online path performs one frozen page forward per question, irrespective of
candidate count, then trains only the approximately 42M-parameter two-block
decoder. Start with microbatch 1 and gradient accumulation; change batching
only after measuring the candidate-count, sequence-length, peak-memory, and
latency distributions.

Do not serialize hidden-state caches in Git. If a later cache experiment is
authorized, its key must include the page hash, exact question, prompt hash,
processor config, model revision, package lock, tap semantics, dtype, and image
grid.

## Mandatory tests before data training

An implementation is not ready for dataset training until all of these pass:

1. official score-path parity on the same model, prompt, and pixels;
2. all-hidden-state versus selected-tap parity for layers 14 and 28;
3. exact image-token count from `image_grid_thw`;
4. synthetic row-major visual-grid reconstruction;
5. a rendered overlay audit confirming candidate boxes land on their expected
   grid regions;
6. full-page and disconnected-member ROI tests;
7. M0 off-diagonal isolation;
8. M1 candidate permutation equivariance when every candidate field and output
   are permuted together;
9. zero backbone gradients and zero backbone optimizer parameters;
10. finite nonzero gradients in every trainable module;
11. expected trainable parameter count;
12. exact checkpoint round trip using only added-module weights plus the model
    and environment locks; and
13. deterministic overfit of a 16-question synthetic batch.

The real-Qwen GPU preflight must emit a JSON report containing exact revisions,
package versions, GPU type, prompt and image hashes, tap shapes, parity deltas,
visual-token counts, trainable/total parameter counts, peak allocated and
reserved memory, and wall time.

## What an implementation agent can and cannot start

After explicit implementation authorization, an agent can build the typed
batch contract, pure visual-grid/ROI functions, candidate encoder, decoder,
loss, and synthetic tests without the V1 candidate artifact.

Real data integration and every model-backed test remain gated on:

1. an approved experiment checkout, branch, scratch root, and recovery owner;
2. a binding SOL task;
3. the frozen candidate and eligibility manifests from the active reranker
   baseline;
4. an owner-approved immutable Qwen model revision and environment lock; and
5. candidate-oracle coverage sufficient for the intended claim.

The next executable document is
[`docs/superpowers/plans/2026-08-14-minivgent-qwen-implementation.md`](../../../../docs/superpowers/plans/2026-08-14-minivgent-qwen-implementation.md).
