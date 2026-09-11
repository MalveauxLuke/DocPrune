> Historical baseline evidence. Commands and status below describe prior experiments; current work is defined in `docs/ExperimentPlan.md`.

# DocPrune executable-code versus paper fidelity audit

## Scope and evidence

This audit compares executable behavior in the current working tree at commit
`a53bf95b8e4288f180553f04b4fa1eec8a841df5` with the DocPrune paper and
supplement. It does not derive implementation claims from prior handoffs,
experiment notes, comments, or intended design. Comments and tests are used
only after the corresponding executable path has been traced.

The audited pruning source and configuration have no uncommitted changes. The
working tree's pre-existing modifications are documentation and task-authority
files. The production benchmark runtime `4e2473b` differs from current HEAD in
the pruning path only by later diagnostic-stage switches; the full DocPrune
path audited here retains the production semantics.

Primary paper source: arXiv `2604.22281v1`, 15-page paper plus supplement,
downloaded from the canonical paper identifier. Audited PDF SHA-256:
`351b2479f738934c01e2df83d01362f544db28b974e7b3e86ff9f0717aeae8fb`.
The repository's canonical CVF links are in
`references/papers/docprune-cvpr-2026.md`.

Classifications:

- **EXACT**: executable behavior matches an explicit equation, table value, or
  implementation statement.
- **PAPER-CONSISTENT-BUT-UNDERSPECIFIED**: the high-level paper operation is
  implemented, but the authors do not define the listed mechanics.
- **UNSPECIFIED**: the code makes a choice for which the paper gives no
  direction.
- **CONFLICT**: executable behavior differs from an explicit paper statement
  or changes the supposedly shared baseline computation.
- **UNVERIFIED**: neither source inspection nor the paper establishes that the
  runtime/library behavior matches the authors' implementation.

## Audit conclusion

The high-level three-stage architecture, Equations 3-8, the keep inequality in
Equation 9, all Table B thresholds, training-free execution, and 2x2 merger-safe
pruning are implemented closely to the publication.

The current run is nevertheless not an exact paper implementation. Three
material discrepancies are present in executable code or pinned resources:

1. The paper names `ColPali-v1`; the code loads `vidore/colpali-v1.2`.
2. Equation 9 thresholds attention weights directly; the code thresholds
   mean-head attention multiplied by the current visual-token count.
3. All-kept and DocPrune do not use identical generation termination. Stock
   all-kept generation honors EOS IDs `[151645, 151643]`; manual DocPrune
   generation honors only model-config EOS `151645`.

The first two are paper-facing conflicts. The third is an internal controlled-
baseline conflict: it can change predictions even if every pruning mask keeps
all tokens.

The paper also leaves many important choices unspecified. The highest-impact
ones are 2x2 block aggregation, CTP head aggregation, which token count (if
any) normalizes CTP attention, the meaning of "last/output token," Gaussian
sigma/kernel/border behavior, image rasterization and resize conventions,
checkpoint/runtime revisions, and the decoder token-drop measurement.

## Direct conflicts and incomplete paper-parity claims

| Item | Executable evidence | Paper authority | Classification | Consequence |
|---|---|---|---|---|
| Retriever checkpoint | `legacy/configs/docprune-m3docvqa.toml:7-10`; `src/docprune/m3docvqa_factory.py:751-764` load `vidore/colpali-v1.2` | Main Section 4.1 names ColPali-v1; supplement resource list links `vidore/colpali-v1` | **CONFLICT** | Retrieval and every QTP embedding differ from the named paper resource. |
| CTP score scale | `src/docprune/ctp.py:68-75` averages heads and multiplies by the number of current visual tokens; `ctp.py:100-103` thresholds the result | Main Section 3.4 defines `a` as output-token-to-visual-token attention weights and Equation 9 compares `a_i >= tau_att`; supplement adds only last-query recomputation | **CONFLICT** | Exact Table B `tau_att` values are applied to a different operand. |
| Paired EOS semantics | `src/docprune/answerers.py:250-265` uses stock `generate`; `answerers.py:419-447` passes only integer `model.config.eos_token_id`; `qwen2vl/model.py:235-260` uses a manual loop. Pinned `generation_config.json` has EOS `[151645,151643]`, while `config.json` has `151645` | DocPrune is presented as pruning the same Qwen2-VL QA model, not changing its decoding rule | **CONFLICT** | DocPrune may continue after token `151643` where all-kept stops. All-kept equivalence is not guaranteed. |
| Paper hardware efficiency | `src/docprune/m3docvqa_factory.py:789-794` accepts any CUDA GPU; `metrics.py:26-39` labels non-A6000 data reconstruction-only | Main Section 4.1 uses one RTX A6000 | **CONFLICT** only if non-A6000 timing is presented as paper parity | The code correctly labels the limitation; quality can still be compared. |
| TFLOPs | `src/docprune/metrics.py:54-113,273-288` disables profiling by default | Main Tables 2-4 report encoder and decoder TFLOPs | **UNVERIFIED / NOT IMPLEMENTED BY DEFAULT** | Current default output cannot reproduce the paper's TFLOP columns. |
| Decoder drop definition | `qwen2vl/model.py:264-275` records endpoint counts; `metrics.py:218-255` reports `1 - post_ctp/original` | Section 4.2 describes visual-token removal across layers but gives no formula | **UNVERIFIED** | Local endpoint drop must not be asserted to be numerically identical to Table 2's decoder-drop definition. |
| Retrieval index | `src/docprune/indexing.py:141-150` constructs exact `IndexFlatIP` | DocPrune only says M3DocRAG; the released upstream recipe defaults to IVFFlat | **UNSPECIFIED by DocPrune; inherited-baseline deviation** | Can change absolute retrieved pages and scores. |

## What is directly paper-faithful

| Paper requirement | Executable evidence | Classification |
|---|---|---|
| Training-free inference | Models are `.eval()` in `m3docvqa_factory.py:751-785`; generation paths use `torch.no_grad()` | **EXACT** |
| Qwen2-VL-7B-Instruct family | `legacy/configs/docprune-m3docvqa.toml:5-6`; `m3docvqa_factory.py:767-785` | **EXACT** at model-family level; revision is unspecified |
| Retrieve top-K pages, then answer | `m3docrag.py:177-245` | **EXACT**, main Equations 1-2 |
| Page counts 1, 2, and 4 | `config.py:17,72-82`; `m3docrag.py:188-193` | **EXACT**, Table 2/Table B |
| BTP before retrieval encoder | `indexing.py:445-476` | **EXACT**, Figure 7/Section 3.2 |
| BTP before QA vision encoder | `pipeline.py:75-87`; `qwen2vl/model.py:193-204` | **EXACT** |
| Nonoverlapping BTP patches | `btp.py:41-47` uses window=stride=`patch_size` | **EXACT**, Section 3.2 |
| Image-wide grayscale mode | `btp.py:36-40` | **EXACT** mathematically; grayscale and tie conventions are not specified |
| Strict pixel tolerance | `btp.py:48` uses `abs(pixel-mode) < tau_e` | **EXACT**, Equation 3 |
| Background ratio | `btp.py:41-49` averages the per-pixel indicators | **EXACT**, Equation 3 |
| BTP keep boundary | `btp.py:52-57` retains `R_i <= tau_bg` | **EXACT**, Equation 4 |
| QTP reuses retrieval embeddings | `answerers.py:346-381`; `m3docrag.py:433-466` | **EXACT**, Section 3.3 |
| QTP uses visual document rows | `m3docrag.py:439-464` | **EXACT**, Equation 5's document visual tokens |
| Sum of cosine similarities | `qtp.py:13-32` | **EXACT**, Equation 5 |
| Bilinear resolution bridge | `qtp.py:83-96,174-176` | **EXACT** at method level, Section 3.3 |
| Gaussian smoothing after resize | `qtp.py:99-120,174-177` | **EXACT** at operation/order level, Equation 6 |
| QTP keep boundary | `qtp.py:123-128` retains `S'_i >= tau_qst` | **EXACT**, Equation 7 |
| Complete 2x2 pruning groups | `layout.py:61-78`; `qwen2vl/compat.py:55-57`; `qwen2vl/vision.py:52-68` | **EXACT**, supplement implementation details |
| BTP and QTP before Qwen merger | `qwen2vl/model.py:193-204`; `qwen2vl/vision.py:52-68` | **EXACT**, supplement |
| Progressive BTP/QTP composition | `qwen2vl/model.py:21-26` intersects masks | **EXACT** at set-composition level |
| CTP L2 statistic | `ctp.py:38-49`; `qwen2vl/decoder.py:145` computes last-token vector norm | **EXACT**, Equation 8 |
| First threshold crossing | `ctp.py:42-49`; `decoder.py:123-148` | **EXACT**, Equation 8's minimum layer |
| Inclusive comprehension boundary | `ctp.py:44-49` triggers at `>= tau_comp` | **EXACT**, Equation 8 |
| One-time CTP decision | Controller refuses subsequent selections; decoder prunes once | **EXACT**, method and ablation description |
| FlashAttention-compatible reduced query | `decoder.py:50-89,134-153` recomputes only final-query attention at the selected layer | **EXACT**, supplement implementation details |
| CTP after selected layer | `decoder.py:144-170` filters the selected layer's output before the next layer | **EXACT**, Equation 9 prose/Figure 8 |
| CTP visual keep boundary | `ctp.py:100-103` retains score `>= tau_att` | **EXACT** for the inequality; the score operand is not exact |
| Visual-only CTP removal | `ctp.py:100-103` initializes nonvisual tokens as kept | **PAPER-CONSISTENT-BUT-UNDERSPECIFIED** |
| Published thresholds | `config.py:85-93`; TOML lines 19-41 match all Table B values for 1/2/4 pages | **EXACT** |
| Dataset-wide arithmetic mean EM/F1 | `evaluation.py:330-395` | **EXACT**, Section 4.2 |
| Image/table/text and hop slices | `evaluation.py:217-240,367-395` | **EXACT** at reported-slice level; grouping mechanics are inherited |
| Released M3DocRAG answer metrics | `evaluation.py:47-144` reproduces upstream normalization, list EM, numeric gating, token-set F1, and optimal alignment | **EXACT to pinned upstream**, not independently specified by DocPrune |

## Paper-underspecified BTP choices

| Executable choice | Code | Classification | What the paper omits |
|---|---|---|---|
| Retrieval patch size 14 and 32x32 grid | `indexing.py:27,456-468`; `colpali/compat.py:11-15` | **UNSPECIFIED** | Section 3.2 leaves `P` symbolic. |
| QA patch size from Qwen processor | `pipeline.py:54-78`; `qwen2vl/preprocessing.py:115-117` | **UNSPECIFIED** | Numeric `P` and processor geometry. |
| BTP raster after model-specific resize | `indexing.py:166-185`; `answerers.py:396-407`; `qwen2vl/preprocessing.py:75-148` | **UNSPECIFIED** | Original versus resized pixel domain. |
| BT.601 coefficients | `btp.py:20-21` | **PAPER-CONSISTENT-BUT-UNDERSPECIFIED** | Meaning of grayscale. |
| Float32 conversion, round, uint8 cast | `btp.py:20-21` | **UNSPECIFIED** | Arithmetic and rounding. |
| `torch.mode` tie-breaking | `btp.py:40` | **UNSPECIFIED** | Mode tie rule. |
| Row-major patch flattening | `btp.py:41-47`; `layout.py:65-78` | **PAPER-CONSISTENT-BUT-UNDERSPECIFIED** | Ordering. |
| 2x2 block retention by any member | `btp.py:69-78`; default in `config.py:51` | **PAPER-CONSISTENT-BUT-UNDERSPECIFIED** | Block score/reduction. |
| Callable `mean` and `all` alternatives | `btp.py:72-78` | **UNSPECIFIED** | Repository extensions. |
| Preserve all nonimage ColPali tokens | `colpali/embedding.py:90-126` | **PAPER-CONSISTENT-BUT-UNDERSPECIFIED** | Special/text-token handling. |
| Raise if a retrieval page loses every patch | `indexing.py:474-475`; `colpali/embedding.py:72-73` | **UNSPECIFIED** | Degenerate fallback. |
| Raise if a QA page loses every group | `qwen2vl/vision.py:43-48` | **UNSPECIFIED** | Minimum retention/fallback. |
| Float32 BTP score computation | `btp.py:20-21,40-49` | **UNSPECIFIED** | Numerical precision/device. |
| Temporal pages rejected in QA | `pipeline.py:54-56` | **UNSPECIFIED** | Batch/tensor contract. |

## Paper-underspecified QTP choices

| Executable choice | Code | Classification | What the paper omits |
|---|---|---|---|
| ColPali question processor prefix and suffix | Factory keeps all active query rows at `m3docvqa_factory.py:893-920`; installed processor builds `Question: {query}` plus ten literal `<pad>` tokens | **UNSPECIFIED** | Which tokenizer/prompt/special rows count as `N_qst`. |
| PyTorch normalization epsilon/zero-vector behavior | `qtp.py:29-32` | **UNSPECIFIED** | Numerical cosine convention. |
| Fixed 128-dimensional embeddings | `answerers.py:356-371` | **UNSPECIFIED** | Equation 5 leaves width `C` symbolic. |
| Fixed ColPali 32x32 raster and row-major IDs | `answerers.py:354-365`; `processor_probe.py:188-234` | **UNVERIFIED** against author runtime | Exact token-to-raster contract. |
| Sparse retrieval-map reconstruction with `-inf` | `qtp.py:35-80` | **UNSPECIFIED** | Handling retrieval-BTP holes. |
| Replace nonfinite/missing sparse cells by zero before resize | `qtp.py:190-201` | **UNSPECIFIED** | Missing-value semantics. |
| Nearest-neighbor validity resize | `qtp.py:196-200` | **UNSPECIFIED** | A separate validity map is absent from the paper. |
| Require all four target fine cells valid | `qtp.py:208-213` | **UNSPECIFIED** | Adds a hard rejection beyond Equation 7. |
| Bilinear `align_corners=False` | `qtp.py:91-96` | **PAPER-CONSISTENT-BUT-UNDERSPECIFIED** | Coordinate convention. |
| Gaussian sigma 1.0 | `config.py:50`; TOML line 46 | **UNSPECIFIED** | The paper names sigma but never publishes it. |
| Gaussian truncation 3 sigma, minimum radius 1 | `qtp.py:103-118` | **UNSPECIFIED** | Kernel support. |
| Replicate border padding | `qtp.py:119-120` | **UNSPECIFIED** | Boundary rule. |
| Kernel dtype/device follows map | `qtp.py:111-120` | **UNSPECIFIED** | Numerical precision. |
| 2x2 block retention by any member | `qtp.py:140-150`; `config.py:51` | **PAPER-CONSISTENT-BUT-UNDERSPECIFIED** | Block score/reduction. |
| Callable mean/all block variants | `qtp.py:143-150` | **UNSPECIFIED** | Repository extensions. |
| QTP is scored independently, then intersected with QA BTP | `pipeline.py:75-108`; `qwen2vl/model.py:21-26` | **PAPER-CONSISTENT-BUT-UNDERSPECIFIED** | Whether QTP is computed only on QA-BTP survivors. |
| Preserve original Qwen rotary positions after compaction | `qwen2vl/sequence.py:35-48`; `qwen2vl/vision.py:55-57` | **UNSPECIFIED** | Positional treatment after pruning. |
| Keep retrieved page serialization order | `m3docrag.py:196-230`; `pipeline.py:53-108` | **UNSPECIFIED** | Top-K set ordering/ties. |
| Batch broadcasting and batch-one assumptions | `qtp.py:25-28`; answerer integration | **UNSPECIFIED** | Batch semantics. |
| No explicit empty-question guard | `qtp.py:13-32` | **UNSPECIFIED** | Empty input behavior. |
| No keep-one fallback for zero survivors | Downstream `qwen2vl/vision.py:47-48` raises | **UNSPECIFIED** | Degenerate behavior. |

For top-4 specifically, retrieval `tau_bg=1.0` means Equation 4 retains every
finite background ratio, so sparse retrieval holes are not active in the
canonical top-4 configuration. They remain part of the general 1/2/4-page
implementation audit.

## Paper-underspecified CTP and decoder choices

| Executable choice | Code | Classification | What the paper omits |
|---|---|---|---|
| Trigger observed after a complete decoder block, before final model norm | `qwen2vl/decoder.py:134-145` | **PAPER-CONSISTENT-BUT-UNDERSPECIFIED** | Exact layer boundary and normalization. |
| Python zero-based layer index | `decoder.py:123,163-168` | **UNSPECIFIED** | Index origin. |
| No crossing means no CTP | `decoder.py:174-175`; `qwen2vl/model.py:264-268` | **UNSPECIFIED** | Empty minimum-set behavior. |
| Last prompt token during prefill | `qwen2vl/model.py:226-234`; `decoder.py:63-89` | **PAPER-CONSISTENT-BUT-UNDERSPECIFIED** | Prompt-prefill versus generated output token. |
| Recompute Q/K from selected layer input while trigger uses its output | `decoder.py:61-89,134-148` | **PAPER-CONSISTENT-BUT-UNDERSPECIFIED** | Which representation supplies attention. |
| Full-key softmax, then select visual entries | `decoder.py:86-89`; `ctp.py:68` | **PAPER-CONSISTENT-BUT-UNDERSPECIFIED** | All-token versus visual-only normalization. |
| Arithmetic mean across attention heads | `ctp.py:68-75` | **UNSPECIFIED** | Head aggregation. |
| Optional max-head extension | `ctp.py:71-74` | **UNSPECIFIED** | Repository diagnostic extension. |
| Scale by current post-BTP/QTP visual count | `ctp.py:75`; `decoder.py:110,149-153` | **CONFLICT** for literal Equation 9 | Any score scaling and which token-count basis. |
| Standard Qwen input layer norm, Q/K projections, M-ROPE, GQA key repeat, `1/sqrt(d)` | `decoder.py:61-86` | **PAPER-CONSISTENT-BUT-UNDERSPECIFIED** | Low-level attention formula/runtime. |
| Score matmul in model dtype; softmax in float32; cast back | `decoder.py:86-89` | **UNSPECIFIED** | Numerical accumulation. |
| Causal-mask handling differs by Flash/eager path | `decoder.py:27-47,87-89` | **UNSPECIFIED** | Mask/API convention. |
| Recomputed scores affect only pruning, not selected-layer hidden output | `decoder.py:134-169` | **EXACT** at strategy level | Consistent with supplement recomputation. |
| Nonvisual tokens always retained | `ctp.py:100-103` | **PAPER-CONSISTENT-BUT-UNDERSPECIFIED** | Exact treatment of prompt/special tokens. |
| Independent threshold, no top-k or target ratio | `ctp.py:100-103` | **EXACT**, Equation 9 |
| Allows CTP to remove every visual token | `ctp.py:100-103` | **UNSPECIFIED** | Minimum retention/fallback. |
| Original order is preserved | `ctp.py:100-103`; `decoder.py:169-173` | **UNSPECIFIED** | Set serialization. |
| Selected layer retains full K/V; deeper layers receive compact K/V | `decoder.py:134-175` | **PAPER-CONSISTENT-BUT-UNDERSPECIFIED** | Cache eviction/topology. |
| Generated tokens see heterogeneous per-layer cache lengths | `decoder.py:178-201` | **UNSPECIFIED** | Autoregressive cache behavior. |
| Original M-ROPE positions, including gaps, survive BTP/QTP/CTP | `qwen2vl/sequence.py:42-48`; `decoder.py:169-172` | **UNSPECIFIED** | Positional reindexing. |
| Per-layer compact `cache_position` | `decoder.py:141,189-198` | **UNSPECIFIED** | Physical versus logical cache coordinates. |
| Next generated M-ROPE ID is max retained position plus one | `qwen2vl/model.py:239-259` | **UNSPECIFIED** | Post-pruning generated positions. |
| Batch size exactly one; padding rejected | `qwen2vl/model.py:180-187`; `decoder.py:106-109` | **UNSPECIFIED** | Batch/padding semantics. |
| DynamicCache only | `decoder.py:104-116,186-198` | **UNSPECIFIED** | Cache implementation. |
| Finite/nonnegative comprehension validation | `ctp.py:31-34` | **UNSPECIFIED** | API validation. |
| Norm evaluated in hidden dtype/device, then `.item()` | `ctp.py:38-45` | **UNSPECIFIED** | L2 precision. |
| Finite attention threshold validation | `ctp.py:87-103`; `config.py:29-43` | **UNSPECIFIED** | API validation. |
| Disabled diagnostic uses threshold `1e9` | `answerers.py:431-439` | **UNSPECIFIED** | Ablation bypass implementation. |
| Only integer model-config EOS is honored | `answerers.py:419-447` | **CONFLICT** with stock paired path | Shared generation semantics. |

## Model, data, retrieval, prompt, and evaluation choices absent from DocPrune

| Executable choice | Code | Classification |
|---|---|---|
| Exact Qwen revision `eed130...` | TOML lines 5-6 | **UNSPECIFIED** |
| Exact ColPali/backbone revisions | TOML lines 7-10 | **UNSPECIFIED**, in addition to the v1/v1.2 conflict |
| BF16 weights and vision execution | `m3docvqa_factory.py:751-785` | **UNSPECIFIED** |
| Transformers 4.46.3, ColPali-engine 0.3.1, Torch 2.4.1, CUDA 12.1 | `legacy/environments/docprune-sol.yml`; compatibility modules | **UNSPECIFIED** |
| FlashAttention-2 concrete package/kernel behavior | Factory/decoder integration | **UNVERIFIED** against author runtime |
| One user message, images first, question text last | `answerers.py:104-121` | **UNSPECIFIED** |
| Prompt `question: ...\noutput only answer.` | `benchmark_config.py:28-29`; `answerers.py:104-111` | **UNSPECIFIED by DocPrune**, exact to pinned released M3DocRAG |
| Greedy decoding, one beam, 128-token cap | `answerers.py:260-265`; `qwen2vl/model.py:235-260` | **UNSPECIFIED by DocPrune**, inherited from released M3DocRAG |
| Decode suffix, skip specials, do not clean tokenization spaces, strip | `answerers.py:146-160` | **UNSPECIFIED** |
| Global dev-document search | Dataset/index/factory path | **PAPER-CONSISTENT-BUT-UNDERSPECIFIED** for open-domain M3DocVQA |
| Exact corpus: 2,441 questions, 3,366 PDFs, 44,638 rendered pages and local byte hashes | `benchmark_config.py:87-213`; dataset path | **UNSPECIFIED** |
| Poppler/pdf2image at 144 DPI and RGB | `m3docvqa_dataset.py:113-119` | **UNSPECIFIED**, inherited released M3DocRAG |
| Resize every page to its document's modal page size | `m3docvqa_dataset.py:113-119` | **UNSPECIFIED**, inherited released M3DocRAG |
| Qwen smart-resize min/max pixels and bicubic processor resampling | `qwen2vl/preprocessing.py:83-105` | **UNSPECIFIED** |
| ColPali processor resize/rescale/normalize conventions | `indexing.py:166-185,445-476` | **UNSPECIFIED** |
| Exact FlatIP token index | `indexing.py:141-150` | **UNSPECIFIED by DocPrune** |
| Search only `top_k` token neighbors per query token | `m3docrag.py:371-388` | **UNSPECIFIED by DocPrune**, exact to pinned upstream indexed branch |
| Max per page per query token, then sum and stable sort | `m3docrag.py:375-391` | **UNSPECIFIED by DocPrune**, exact to pinned upstream |
| Unique-page enforcement and tie order | `m3docrag.py:314-335,388-404` | **UNSPECIFIED** |
| Full dev split/source order/sharding mechanics | `m3docvqa_dataset.py:51-85`; `m3docvqa_factory.py:160-189` | **PAPER-CONSISTENT-BUT-UNDERSPECIFIED** only for the complete merge |
| Gold `str(item["answer"])` conversion | `m3docrag.py:62-77` | **UNSPECIFIED by DocPrune**, exact to pinned evaluator |
| Answer normalization details | `evaluation.py:47-87` | **UNSPECIFIED by DocPrune**, exact to pinned evaluator |
| Hungarian-equivalent list alignment and per-question rounding to two decimals | `evaluation.py:105-144` | **UNSPECIFIED by DocPrune**, exact to pinned evaluator |
| Twelve-type multi-hop allowlist | `evaluation.py:27-42` | **UNSPECIFIED by DocPrune**, exact to pinned evaluator |
| Supporting-document rather than evidence-page recall diagnostics | `evaluation.py:244-274` | **UNSPECIFIED** and not a Table 2 metric |
| One unrecorded warmup per execution shard | `m3docrag.py:247-253`; CLI execution | **UNSPECIFIED** |
| CUDA timing boundaries and synchronization | `qwen2vl/model.py:66-156`; `metrics.py:99-105` | **UNSPECIFIED** |
| Peak allocated memory definition | `answerers.py:52-67`; `metrics.py:95-98` | **UNSPECIFIED** |
| Any CUDA GPU accepted for quality | `m3docvqa_factory.py:789-794` | **PAPER-CONSISTENT** for quality only; not efficiency parity |
| Manifest runtime commit is syntax-checked but executing DocPrune HEAD/cleanliness is not verified by the factory | `benchmark_config.py:279-305`; `m3docvqa_factory.py:624-631,676-699` | **UNVERIFIED**; launch wrappers may add a separate preflight |

## Configuration wiring defects

The manifest records reconstruction defaults, but the production factory does
not pass them to the answerer:

- `m3docvqa_factory.py:1251-1256` omits
  `reconstruction=config.reconstruction_defaults`.
- `answerers.py:317-322` silently constructs a new `ReconstructionDefaults()`.
- Consequently configured `gaussian_sigma` and `group_retention` do not control
  production QA unless they happen to equal the dataclass defaults.
- `grayscale` is metadata-only because `btp.py:10-21` always executes BT.601.
- `attention_head_aggregation` is metadata-only because
  `answerers.py:425-447` does not forward `head_aggregation` to the adapter.
- `ctp_timing` is never consulted by executable generation.

The canonical configuration happens to use the same values as the silently
constructed defaults, so this does not change the already-run default
semantics. It does mean the recorded configuration is not a complete executable
control surface and alternative manifests could misdescribe runtime behavior.

Similarly, the workload factory verifies the external M3DocRAG checkout but
only validates the supplied DocPrune `runtime_commit` as a 40-character hex
string. A launcher may separately check the active checkout, but a factory-made
manifest alone does not prove which DocPrune source actually executed.

## What tests establish, and what they cannot establish

The focused tests lock internal behavior for the exact inequalities, local
BT.601 conversion, any-member block reduction, cosine sum, Gaussian kernel,
mask intersection, local token ordering, first CTP crossing, mean-head-times-N
scaling, threshold equality, and heterogeneous cache lengths. They demonstrate
that the repository consistently executes its chosen reconstruction.

They do not establish author parity. In particular, current tests do not prove:

- author grayscale, block reduction, Gaussian sigma/kernel/border, or head
  aggregation;
- a real-checkpoint all-kept equivalence oracle covering both checkpoint EOS
  IDs;
- numeric equality of manual CTP attention under real FlashAttention-2 and the
  attention used by the authors;
- real checkpoint ColPali/Qwen raster ordering against author code;
- author cache, M-ROPE, or generated-position behavior;
- paper Table 2 decoder-drop or TFLOP measurement identity.

Some Flash-oriented unit tests select the local Flash mask branch on tiny test
objects; they do not instantiate a real `Qwen2VLFlashAttention2` module. The
real-model parity test is optional and does not provide an author CTP oracle.

## Confidence statement

We can guarantee that the repository is a careful implementation of the
published high-level algorithm and most literal equations. We cannot guarantee
that it is the authors' implementation. The paper does not specify enough
information to reconstruct all token identities, and the current code adds
material choices in precisely those gaps. In addition, the ColPali resource,
CTP score operand, and paired EOS behavior require correction or explicit
reclassification before claiming exact paper fidelity.
