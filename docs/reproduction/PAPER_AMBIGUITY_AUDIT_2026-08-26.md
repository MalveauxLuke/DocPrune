> Historical baseline evidence. Commands and status below describe prior experiments; current work is defined in `docs/ExperimentPlan.md`.

# DocPrune paper ambiguity audit after the aggregate-logit diagnostics

## Scope and evidence

This is a fresh paper-first audit of the executable repository at
`a53bf95b8e4288f180553f04b4fa1eec8a841df5`. It was performed after the
pre-softmax-head-mean/visual-softmax CTP diagnostics. It does not assume that a
choice is author-faithful merely because it appears in an earlier handoff,
test, configuration, or reconstruction note.

Primary sources were read independently:

- CVF main paper PDF, SHA-256
  `5ae06a592325fa9c28706b784238a39542b4157fdf463900004e5c5ac729618b`;
- CVF supplement PDF, SHA-256
  `b9aadf9db46a9b23c2110c7c65b9151290421abd16c1d9083505c45a4854a6e6`;
- the repository source at the commit above; and
- only after the source comparison, the authenticated recent diagnostic
  analyses recorded in `sol/CURRENT_SOL_TASK.md`.

The canonical paper links are retained in
`references/papers/docprune-cvpr-2026.md`. No retrieval, model inference, or
new experiment was run for this audit. Existing artifacts and uncommitted
changes were preserved.

Classifications used here:

- **Guaranteed**: directly specified by an equation, table, or prose statement
  and implemented by the active executable path.
- **Paper ambiguity**: the paper permits multiple materially different
  implementations and supplies no author code or oracle that selects one.
- **Paper inconsistency**: two statements or figures in the publication cannot
  simultaneously be exact.
- **Direct conflict**: the active executable resource or behavior differs from
  an explicit paper statement.
- **Local control defect**: the repository's baseline/candidate comparison or
  manifest can differ even before deciding what the authors did.
- **Experimentally deprioritized**: still ambiguous in the paper, but recent
  controlled evidence makes it an unlikely cause of the observed quality gap.

## Executive finding

The aggregate-logit work has narrowed the CTP search, but it has **not resolved
the paper's attention definition**. The leading tested reconstruction averages
raw query-head logits, applies a visual-only softmax, and then multiplies by the
post-QTP visual-token count. Only the act of recomputing a reduced last-token
query at the selected layer is explicit in the supplement. The aggregation
order, visual-only normalization, and token-count multiplier remain inferred.

On 245 fixed-page questions this candidate improved F1 over current CTP by
`+1.0816`, but remained `-1.2163` behind BTP+QTP. Its interval against current
included zero, and it did not reproduce the paper's positive CTP stage effect.
The first-generated-token/layer-output variant matched the derived 65% count
fingerprint almost exactly, but its 16-question live QA answers were identical
to BTP+QTP and it failed its predeclared promotion gate. A count match therefore
does not identify the author score or reproduce the claimed quality behavior.

The highest-value unresolved paper ambiguities are now:

1. the exact CTP attention operand, head/GQA aggregation, normalization domain,
   scaling, and layer boundary;
2. whether the “output token” is the last prompt token or an already-generated
   answer token;
3. the definition of encoder/decoder token drop, which means 65% is not an
   explicit paper target;
4. the decoder cache, rotary-position, and generated-position semantics after
   pruning;
5. the 2x2 BTP/QTP block decision and QTP smoothing/raster details;
6. the hyperparameter-selection, prompt, preprocessing, and decoding protocol.

Two direct conflicts and one paired-control defect remain separate from those
ambiguities: the repository loads ColPali-v1.2 instead of the named
ColPali-v1; canonical CTP thresholds a locally transformed score rather than
the literal stated attention weight; and all-kept versus DocPrune generation
uses different EOS termination semantics.

## What the paper and code establish closely

These claims can be treated as paper-faithful at the stated level. They do not
resolve lower-level choices listed later.

| Stage | Paper-explicit behavior | Executable evidence |
|---|---|---|
| Architecture | BTP before retrieval and QA encoders, QTP before the QA encoder, CTP inside the decoder | `src/docprune/indexing.py:445-476`, `pipeline.py:75-108`, `qwen2vl/model.py:193-234` |
| BTP | Nonoverlapping patches, page-wide grayscale mode, strict pixel test, ratio, and keep rule `R_i <= tau_bg` | `src/docprune/btp.py:36-57` |
| QTP | Reuse retrieval document/question embeddings and sum cosine similarities over question tokens | `src/docprune/qtp.py:13-32`, `answerers.py:346-381` |
| QTP | Bilinear resize, then Gaussian smoothing, then inclusive `S'_i >= tau_qst` | `src/docprune/qtp.py:83-128,174-201` |
| Merger safety | BTP and QTP prune complete 2x2 spatial blocks at encoder input | `src/docprune/layout.py:61-78`, `btp.py:60-79`, `qtp.py:131-150`, `qwen2vl/vision.py:52-68` |
| CTP trigger | L2 norm of the last-token representation; first inclusive threshold crossing | `src/docprune/ctp.py:28-49`, `qwen2vl/decoder.py:123-148` |
| CTP mechanics | One selected layer, reduced last-token query recomputation, visual threshold inequality, pruning after that layer | `src/docprune/_legacy/qwen2vl/decoder.py:50-89,134-175`, `ctp.py:78-103` |
| Thresholds | Published 1/2/4-page BTP, QTP, comprehension, and attention values | `src/docprune/config.py:85-93`, `legacy/configs/docprune-m3docvqa.toml:19-41` |
| Models | Training-free Qwen2-VL-7B family for QA | `src/docprune/m3docvqa_factory.py:751-785` |
| Metrics | Dataset arithmetic means for EM/F1 and the paper's evidence/hop slices | `src/docprune/evaluation.py:217-240,330-395` |

“Guaranteed” here means that the high-level operation is faithful. For
example, Gaussian smoothing is guaranteed, while its unpublished sigma and
border rule are not.

## CTP: complete ambiguity inventory

### P0: the attention score is not mathematically specified enough to apply `tau_att`

Main Section 3.4 calls `a` “attention weights from the output token to all
visual tokens” and Equation 9 applies `a_i >= tau_att`. The supplement adds
only that attention is recomputed for the last token with a reduced query at
the selected layer. It does not define any of the following:

| Missing choice | Material alternatives consistent with some part of the text |
|---|---|
| Score stage | raw `QK^T/sqrt(d)` logits; post-softmax attention; a transformed or normalized importance score |
| Normalization domain | all causal keys; visual keys only; each page separately; each head or GQA group separately |
| Head reduction | mean, sum, maximum, selected heads, learned/fixed weights, union of per-head masks |
| Reduction order | aggregate logits then softmax; softmax each head then aggregate; threshold each head then combine masks |
| GQA treatment | repeat four KV heads across 28 query heads; aggregate the seven query heads per KV group; operate on four group maps |
| Scale | no multiplier; current visual count; original visual count; total sequence length; head count; per-page count; another normalization |
| Precision | model-dtype or float32 logits/softmax; exact boundary behavior under rounding |

The published thresholds make a missing transformation more than a cosmetic
detail. A mean post-softmax attention vector has total visual mass at most one,
so threshold `0.075` can retain at most 13 tokens and threshold `0.5` at most
two. The paper reports hundreds or thousands of retained tokens. Even a head
sum has bounded total mass and does not uniquely reconcile all reported
counts. Therefore literal mean normalized weights cannot be the whole author
procedure, but the paper does not identify what completes it.

The active canonical code chooses full-key softmax, visual selection, equal
mean across repeated query heads, and multiplication by the current visual
count (`src/docprune/ctp.py:52-75` and
`src/docprune/_legacy/qwen2vl/decoder.py:50-89`). That is a reconstruction choice, not
a paper guarantee.

### P0: “output token,” layer state, and timing

The paper alternates among “last token,” “output token,” and the last-token
representation at layer `l`. It never states:

- whether the query is the final prompt token during prefill or the first/later
  generated answer token;
- whether `x_N^l` is the input to block `l`, its residual output, a normalized
  state, or the state after the model's final norm;
- whether the norm and attention query come from the same boundary;
- whether attention is recomputed from the selected block's input or output;
- whether layer numbering is zero- or one-based; or
- what occurs if no layer crosses `tau_comp`.

Canonical code observes the norm after a complete block, then recomputes Q/K
from that block's input, using the last prompt token during prefill
(`src/docprune/_legacy/qwen2vl/decoder.py:61-89,123-175`). This is plausible but not
specified. The rejected first-generated/layer-output experiment proves that
one attractive timing interpretation is not sufficient; it does not prove the
canonical timing is the authors' choice.

### P1: decoder state after pruning

The paper says tokens are removed after the chosen layer but does not specify:

- whether the selected layer's K/V cache stays full or is retroactively
  compacted;
- whether deeper layers and later generated tokens use heterogeneous per-layer
  cache lengths;
- whether retained tokens preserve original M-ROPE positions or are reindexed;
- whether gaps in rotary positions are allowed;
- how `cache_position` relates to logical M-ROPE coordinates;
- the next generated token's position after compacting a multimodal prompt;
- whether nonvisual prompt/special tokens are always kept;
- whether zero visual survivors are allowed or a keep-one fallback is used; or
- batch, padding, and cache implementation behavior.

Local code keeps the selected/earlier layer caches full, compacts deeper caches,
preserves original visual positions, and sets the first new position to the
maximum retained position plus one (`src/docprune/_legacy/qwen2vl/decoder.py:134-201`,
`qwen2vl/model.py:188-259`). These choices can change every later token even
when the selected mask is identical.

### P1: the reported decoder drop does not define a 65% CTP target

Section 4.2 defines drop rate only as the proportion of tokens “removed across
the layers.” It does not give an equation, denominator, endpoint, layer
weighting, or sample aggregation. The top-4 table reports encoder drop `0.60`
and decoder drop `0.74`.

The commonly used conditional target

`(1 - 0.74) / (1 - 0.60) = 0.65`

is valid only if both columns are endpoint ratios against the same original
token denominator and if a ratio of reported dataset averages can stand in for
mean samplewise conditional retention. None of those assumptions is stated.
Consequently:

- 65% remains a useful cross-formula fingerprint;
- matching 65% is not proof of author parity; and
- failing to match it is evidence only under the endpoint interpretation.

The repository reports endpoint `1 - post_ctp/original` in
`src/docprune/_legacy/qwen2vl/model.py:264-275` and `src/docprune/metrics.py:218-255`.
It has not established equivalence to the paper's decoder column.

## BTP ambiguity inventory

The mathematical test in Equations 3-4 is implemented faithfully. The paper
does not specify:

- the numeric patch size `P` for each retrieval/QA processor;
- whether BTP sees original raster pixels or a model-resized raster;
- grayscale coefficients, color space, quantization, rounding, or dtype;
- the mode tie rule;
- how the four fine tokens in a required 2x2 block produce one block decision
  (`any`, `all`, mean ratio, maximum, or another rule);
- row/raster ordering and treatment of model special tokens;
- minimum-retention or empty-page behavior; or
- device/precision details at threshold boundaries.

Current code uses processor-resized pixels, BT.601 float32 conversion rounded
to uint8, `torch.mode`, and “retain the 2x2 block if any member passes”
(`src/docprune/btp.py:10-79`). These are paper-consistent but not author-proven.

Recent CPU reconstruction and bounded gates make the ordinary grayscale,
strict/inclusive pixel, and `mean`/`all` block variants unlikely explanations
for the controlled quality loss. That evidence deprioritizes these choices; it
does not turn them into paper-specified behavior.

## QTP ambiguity inventory

Equation 5, bilinear resize, Gaussian smoothing, and the inclusive keep rule
are explicit. The following are not:

- exactly which tokenizer/template/special question-token rows belong to
  `N_qst`;
- the exact ColPali checkpoint revision and processor contract beyond the
  named ColPali-v1 resource;
- embedding normalization epsilon, precision, and zero-vector behavior;
- the document-token-to-2D-raster mapping, source grid, row order, and handling
  of nonvisual embedding rows;
- bilinear coordinate convention (`align_corners`);
- Gaussian `sigma`, support/truncation, normalization, border padding, and
  dtype;
- the 2x2 block decision rule;
- whether QTP is scored over all retrieval visual tokens or only QA-BTP
  survivors;
- how retrieval-BTP holes are represented and resized;
- how BTP and QTP masks compose when their grids differ;
- degenerate empty-question/empty-map/zero-survivor behavior; and
- page ordering and tie handling.

Current code fixes a 32x32 row-major ColPali map, uses all active processor
query rows, `align_corners=False`, sigma 1.0, a radius of `ceil(3 sigma)`,
replicate padding, and any-member 2x2 retention
(`src/docprune/qtp.py:13-213`, `answerers.py:346-381`). Sparse holes become
zero before resize and are governed by a separately resized validity mask.
For canonical top-4, retrieval `tau_bg=1.0` makes that sparse-hole branch
irrelevant because all finite BTP ratios pass.

The 245-question QTP-mean block test was worse than current any-member QTP by
`-0.9673` F1 with an interval spanning zero; QTP-all was strongly harmful in
the calibration gate. Current any-member semantics are therefore the best
tested local choice, but the paper still does not select them.

## System and evaluation ambiguities

The paper names model families, dataset, one hardware configuration, and broad
metrics. It does not fully specify:

| Area | Missing author detail |
|---|---|
| Model/runtime | exact Qwen revision, dtype, Transformers/FlashAttention/CUDA versions, deterministic kernel settings |
| Images | PDF renderer/version, DPI, color conversion, document-page canonicalization, Qwen/ColPali resizing and resampling |
| Retrieval | exact index type and search parameters, token/page aggregation, tie order, corpus bytes and page order |
| Prompt | chat template, image ordering, exact instruction text and special tokens |
| Generation | greedy/sampling settings, token cap, beam count, EOS list, decode cleanup |
| Hyperparameter tuning | tuning split, objective, tie rule, whether quality/throughput was jointly optimized, leakage controls, random seeds |
| Quality metrics | exact normalization, list alignment, numeric handling, answer casting, per-question rounding |
| Efficiency | warmup, timing boundaries, synchronization, batch size, TFLOP profiler, memory definition, endpoint versus layer-integrated token drop |
| Statistics | number of runs, variance, confidence intervals, and whether reported values are single-run or averaged |

Some local choices reproduce the pinned released M3DocRAG evaluator, but that
does not make them explicit in the DocPrune paper.

## Paper-internal inconsistencies

These prevent publication figures from acting as exact implementation oracles:

1. The same Piranha example is `2508 -> 1340 -> 406 -> 168` in main Figure 10,
   but `2508 -> 1340 -> 406 -> 197` in supplement Figure C.
2. Main Figure 10 labels the post-QTP count `406`, while the accompanying
   paragraph says `460` for that same stage.
3. Main notation uses `tau_comp` and `tau_qst`; supplement Table B switches to
   `tau_info` and `tau_q` without explicitly declaring them aliases.
4. The prose says QA performance improves progressively as BTP, QTP, and CTP
   are added, but top-4 Table 5 drops from BTP F1 `37.1` to BTP+QTP `36.9`
   before rising to full `37.3`.

The first two discrepancies mean no exact token-count target should be taken
from the qualitative example without an author clarification.

## Direct conflicts and local control defects

These are not paper ambiguities and should not be hidden inside them.

### ColPali checkpoint conflict

Main Section 4.1 and the supplement's resource list identify
`vidore/colpali-v1`. The active configuration pins `vidore/colpali-v1.2`
(`legacy/configs/docprune-m3docvqa.toml:7-10`,
`src/docprune/m3docvqa_factory.py:751-764`). This changes retrieval and every
QTP embedding even when retrieved pages are held fixed.

### Canonical CTP operand conflict

The active code thresholds mean-head post-softmax attention multiplied by the
current visual-token count (`src/docprune/ctp.py:52-75`). Equation 9 states a
threshold on attention weights and never specifies that multiplier. Some
unstated transform appears necessary, but necessity does not make the local
transform author-specified.

### All-kept versus DocPrune EOS mismatch

All-kept calls stock `model.generate`, which uses the checkpoint generation
configuration. Manual DocPrune builds its EOS tuple from only an integer
`model.config.eos_token_id` (`src/docprune/answerers.py:250-265,419-447`). For
the pinned model, stock generation has EOS IDs `[151645, 151643]`, while the
manual path honors only `151645`. This can change answers even under an
all-token-kept mask and is a local paired-control defect.

### Preprocessing equivalence is unverified

All-kept sends source images to the full processor. DocPrune first performs a
smart resize, converts the prepared raster back to an image, and calls the
full processor again (`src/docprune/answerers.py:250-254,396-400`). The second
resize is expected to be idempotent but has not been proven byte/numerically
equivalent with the real checkpoint.

### Manifests can overstate executed semantics

The manifest records `reconstruction_defaults`, but the production factory
does not pass them to `DocPruneQwenAnswerer`
(`src/docprune/m3docvqa_factory.py:943-951,1251-1256`). The answerer silently
constructs defaults (`answerers.py:303-322`), and configured
`attention_head_aggregation`/`ctp_timing` are not wired into canonical
generation. Canonical values happen to equal the silent defaults, but an
alternate manifest can describe behavior that did not execute.

The shared comparison identity records prompt, token cap, sampling, and beam
count, but not the resolved EOS/generation-configuration semantics
(`src/docprune/m3docvqa_factory.py:652-698`,
`src/docprune/comparison.py:21-31,416-427`). It therefore cannot fail closed on
the real EOS mismatch.

## What recent experiments actually resolved

| Candidate or ambiguity | Result | Audit consequence |
|---|---|---|
| BTP grayscale/arithmetic variants | Tiny mask changes and essentially unchanged retention | Still unspecified; low priority |
| BTP inclusive pixel tolerance | Jaccard about `.988`, small retention movement | Literal strict equation remains preferred; not the controlled loss |
| BTP mean/all 2x2 reductions | Mean neutral in a small gate; all catastrophically sparse | Current any retained pragmatically, not author-proven |
| QTP mean/all 2x2 reductions | Mean worse on 245; all strongly harmful in calibration | Current any retained pragmatically, not author-proven |
| Current CTP | `-2.2980` F1 versus BTP+QTP on fixed 245 | Statistically supported local loss |
| Original-token CTP scale | Count/quality gate failed | Rejected local candidate |
| Mean raw logits -> visual softmax -> current-count scale | Improves current, but remains below BTP+QTP on 245 | Leading tested reconstruction, not resolved or paper-faithful |
| Sum raw logits before one softmax | Retained only `.1844%` at top-1 threshold `.5` | Rejected formula; does not reject post-softmax head sums or other reductions |
| Max-logit aggregation | Poor paper-count fingerprint | Rejected formula |
| KV-group/GQA scoring | Better count fingerprint, but tied BTP+QTP in 16-question QA gate | Rejected for expansion; GQA semantics remain unspecified |
| Total-sequence instead of visual-count scale | Changed only `111/56,235` selections | Deprioritized |
| First-generated, layer-output aggregate logits | Matched derived 65% count; failed QA gate | Rejected timing/formula pair; timing ambiguity remains |

The key distinction is that experiments can reject local candidates. They
cannot prove an unreported author choice unless a unique paper oracle exists.

## Prioritized conclusion

The code is close to the paper at the architectural and equation level for BTP
and QTP, and at the trigger/reduced-query level for CTP. The remaining quality
gap is not evidence that “everything in the paper was copied incorrectly.” It
is evidence that the paper does not contain enough information to recreate the
author's exact CTP masks and decoder state transitions, and that the repository
also contains independent paired-control/resource differences.

Before another large pruning job, the highest-value sequence is:

1. establish real-checkpoint manual all-kept equivalence, including both EOS
   IDs and preprocessing, so CTP quality is not measured through a confounded
   generation path;
2. treat 65% only as a heuristic fingerprint, not a promotion oracle;
3. use existing captures for any remaining CPU-only score analyses;
4. if another live CTP gate is justified, test one paper ambiguity at a time on
   the sealed cached pages; and
5. separately evaluate the named ColPali-v1 resource before claiming retrieval
   or QTP paper parity.

No new experiment is authorized or launched by this audit.
