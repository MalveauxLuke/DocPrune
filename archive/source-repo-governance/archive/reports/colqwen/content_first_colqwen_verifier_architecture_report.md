# Content-First ColQwen Evidence Verifier: Technical Architecture Report

## Purpose of This Report

This report reconciles the proposed architecture images, the earlier project design, and the subsequent critiques. The final recommendation is a content-first verifier over frozen ColQwen embeddings. The main model should not be framed as a spatial co-location model. Geometry can be tested, but it should be an ablation arm, not the foundation of the architecture.

The central project claim should be:

> ColQwen is a strong high-recall candidate generator, but its MaxSim score is not calibrated for page-level evidence applicability. We train a lightweight verifier over frozen ColQwen query/page embeddings and MaxSim alignment statistics to distinguish pages that are merely similar from pages that are actually useful evidence.

The motivating empirical result is the validated document-scoped ColQwen2 run:

| Setting | Gold@1 | Gold@3 | Gold@5 | Key implication |
| --- | ---: | ---: | ---: | --- |
| ColQwen2 document-scoped MMDocIR 4K | 70.28% | 91.88% | 97.22% | Gold is usually retrieved, but ranking/evidence calibration is imperfect. |

This means the project is not mainly about getting ColQwen to find the right document. In the document-scoped setting, it often finds the gold evidence page somewhere in the top candidates. The project is about improving the second-stage decision: among high-scoring candidate pages, which page or pages should be sent to the reader?

## Final Position After Reconciling the Critiques

The strongest critique was correct: the earlier geometry-heavy architecture overcorrected. If the motivation is "MaxSim fails because query tokens match scattered spatial regions," then the model needs coordinates or alignment geometry. But that is not the best thesis for this project.

The better thesis is content/query-conditioning:

> A page can contain query-like visual/textual evidence and receive a high MaxSim score without answering the query as a composed evidence need.

This failure is not necessarily geometric. A false positive may be visually similar, contain the same entities, contain the same chart labels, or strongly match one important query token while missing another. Conversely, a true gold page may have its evidence spread across a figure, caption, header, legend, table, or footnote. Therefore, a model that assumes useful evidence should be spatially clustered risks penalizing legitimate gold pages.

The architecture should therefore prioritize:

1. Query-conditioned content interaction.
2. Per-token MaxSim distribution shape.
3. Coverage of important query tokens.
4. A clean residual representation of the original query.
5. Controlled use of the MaxSim scalar as a calibration anchor.

The architecture should not prioritize:

1. Raw patch coordinates in the main model.
2. Best-match patch locations in the main model.
3. Geometry summaries in the main model.
4. Claims that the verifier performs spatial grounding or compositional layout reasoning.

Geometry remains scientifically interesting. It is just not the main model. It should be tested as a single ablation arm:

> Does explicit patch-location information add discriminative signal beyond what frozen ColQwen/Qwen2-VL embeddings already encode?

If it helps, that becomes a finding. If it does not help, the project still stands on the content-first verifier.

## Architecture Summary

The final architecture should match the content-first diagram:

```text
Pipeline:

Corpus page images
        |
ColQwen retriever
        |
top-k candidate pages
        |
Content-first evidence verifier
        |
selected evidence pages
        |
VLM reader / tool-augmented reader
        |
answer
```

Verifier internals:

```text
Frozen query tokens Q
Frozen page patch tokens P
        |
Pre-norm
        |
Cross-attention
Q attends over P
        |
Attended query states Z
        |
MaxSim-weighted pooling
        |
pooled attended evidence h_attn
        |
concat:
  h_attn
  scale-matched query residual h_query
  distributional alignment features
        |
content branch logit f_content
        |
late fusion with MaxSim scalar g(maxsim)
        |
final logit
        |
sigmoid relevance probability
```

Mathematically, for query tokens `Q = {q_i}` and page tokens `P = {p_j}`:

```text
S_ij = q_i dot p_j
s_i = max_j S_ij
MaxSim(Q, P) = sum_i s_i
```

The verifier does not use only `MaxSim(Q, P)`. It uses the richer structure around it:

- the query tokens `Q`
- the page tokens `P`
- the cross-attended query states `Z`
- the per-query-token max scores `{s_i}`
- distributional summaries of the similarity matrix
- the original MaxSim scalar, late-fused as a controlled calibration feature

## Component Classification

| Component | Classification | Main reason |
| --- | --- | --- |
| Frozen ColQwen encoder | Necessary | Gives reusable multimodal query/page token embeddings without training the large VLM. |
| Document/page candidate set from ColQwen | Necessary | Defines the hard candidate distribution the verifier must correct. |
| Query tokens Q | Necessary | The verifier must condition on the actual query, not just page content. |
| Page patch tokens P | Necessary | The verifier needs multimodal page evidence in frozen ColQwen space. |
| Q x P similarity matrix | Necessary | Source of MaxSim, pooling weights, and alignment features. |
| Pre-norm before attention | Necessary default | Stabilizes small trainable head over frozen embeddings. |
| Cross-attention Q over P | Main model | Learns query-conditioned content interaction beyond the MaxSim scalar. |
| MaxSim-weighted pooling | Main model, high-value | Focuses pooling on query tokens that drove retrieval instead of averaging filler tokens. |
| Query residual, scale-matched | Main model | Lets the classifier compare what was asked to what the page returned. |
| Distributional alignment features | Main model | Captures coverage, sharpness, entropy, and score shape discarded by the MaxSim sum. |
| Late-fused MaxSim scalar | Controlled-risk main model | Useful calibration anchor, but must be isolated and ablated. |
| 2-layer MLP | Necessary default | Gives enough capacity to combine content, query, and alignment features. |
| Sigmoid relevance probability | Necessary | Produces the binary evidence probability used for filtering/reranking. |
| Geometry/coordinates | Ablation only | Potentially useful, but target-misaligned and partly redundant with Qwen2-VL spatial encoding. |
| MaxSim-zero diagnostic | Mandatory diagnostic | Tests whether the model is a real content verifier or just a dressed-up reranker. |
| Downstream reader evaluation | Mandatory final validation | A classifier improvement matters only if it helps the full QA pipeline. |

## Detailed Component Explanations

### 1. Corpus Page Images

**Classification:** Necessary pipeline input.

The system begins with documents rendered as page images. This matters because ColQwen/ColPali-style retrieval is visual retrieval, not OCR-only retrieval. Page images preserve layout, visual structure, tables, charts, figures, captions, and other multimodal information that text extraction may damage or lose.

For this project, page-level representation is also important because the verifier's label is page-level evidence usefulness. The model is not being asked to identify a text span or crop. It is being asked whether a candidate page is useful evidence for a query.

**What it seeks to accomplish:**

- Preserve multimodal information.
- Let ColQwen encode each page in its native expected format.
- Support downstream reader/tool use after candidate selection.

**Why it is necessary:**

The entire project depends on frozen ColQwen page embeddings. Those embeddings are produced from page images. Without the page-image input, this is no longer a ColQwen-based visual retriever/verifier.

**Risk or limitation:**

The page is a coarse unit. Some pages may contain relevant and irrelevant regions. A page-level verifier can reduce page noise, but it cannot by itself solve region-level evidence extraction. That belongs either to the reader or to a later fine-grained tool module.

### 2. ColQwen Retriever

**Classification:** Necessary candidate generator.

ColQwen retrieves candidate pages using late interaction over query and page token embeddings. It is not being replaced. It is the left half of the system. The verifier is a second-stage module that operates on ColQwen's output distribution.

This is crucial. The verifier is trained on hard negatives drawn from pages ColQwen already considered plausible. That means the task is not "separate obviously irrelevant pages from relevant pages." It is "separate high-scoring candidates that actually support the query from high-scoring candidates that do not."

**What it seeks to accomplish:**

- Generate high-recall candidate pages.
- Provide frozen query/page token embeddings.
- Provide MaxSim scores and alignment information.
- Define the hard-negative distribution for training.

**Why it is necessary:**

The verifier is not useful without a candidate set. ColQwen gives the candidate set and the representation space. The project contribution is the verifier over that candidate set, not a new first-stage retriever.

**Current empirical support:**

The validated document-scoped run shows:

- Gold@1: 70.28%
- Gold@3: 91.88%
- Gold@5: 97.22%
- Non-gold candidates outranking gold when gold is present: 1797

This is exactly the pattern a verifier can exploit. ColQwen is good enough to find candidates, but imperfect enough that a reranker/verifier has room to improve the final selected evidence.

**Risk or limitation:**

If the gold page is not in the candidate set, the verifier cannot recover it. This is why candidate recall is a first-order metric. The verifier should be evaluated at a candidate depth that preserves high gold recall.

### 3. Candidate Depth: Top-k Pages

**Classification:** Necessary experimental choice.

The slide image says ColQwen retrieves top-20. The validated document-scoped run currently reports top-5. Architecturally, the verifier should be defined over a top-k candidate set where k is configurable.

There are two separate questions:

1. How deep should retrieval go to avoid missing the gold page?
2. How many selected pages should the verifier pass to the reader?

These should not be conflated. Retrieval depth can be high to preserve recall, while verifier output can be smaller to reduce reader noise.

**What it seeks to accomplish:**

- Preserve gold evidence in the candidate set.
- Provide enough hard negatives for learning.
- Let the verifier learn from the actual pages that would otherwise go to the reader.

**Why it is necessary:**

The classifier only learns the decision boundary induced by the candidate set. If training negatives are random pages, the classifier learns an easy task. If training negatives are top-k ColQwen candidates, the classifier learns the task that matters: distinguishing true evidence from plausible false positives.

**Design recommendation:**

Use a configurable retrieval depth. The report and slides can say "top-k" generally, then instantiate:

- top-5 for the current validated document-scoped pilot results
- top-20 for a fuller training/evaluation setting if recall and compute allow

**Risk or limitation:**

If top-k is too shallow, the verifier may appear precise while silently dropping queries where gold was never available. If top-k is too deep, the candidate set may include many weak negatives, making training easier but less representative of the hardest reranking cases. The right choice is empirical.

### 4. Frozen Query Tokens Q

**Classification:** Necessary.

The query is not represented as one vector. In ColQwen/ColPali-style late interaction, the query is a sequence of token embeddings:

```text
Q = {q_1, q_2, ..., q_n}
```

Each token can represent a word, subword, or special/query expansion token depending on the tokenizer and model. MaxSim operates over these query vectors individually. This token-level structure is one reason ColQwen works well: the model can match different parts of the query to different parts of a page.

**What it seeks to accomplish:**

- Preserve the multiple semantic pieces of the query.
- Let the verifier know which terms or concepts the page must satisfy.
- Support cross-attention from query tokens into page evidence.
- Support per-token alignment features such as coverage and sharpness.

**Why it is necessary:**

Evidence relevance is relational. A page is not relevant in isolation. It is relevant to a specific query. If the verifier only sees a page vector or only sees the MaxSim score, it loses the structured intent of the query.

**Why not pool Q immediately?**

Pooling the query too early hides which query tokens matter. A query such as "What was Germany revenue in 2021?" contains filler words and content-bearing words. Uniformly pooling the query can blur the entity, attribute, and time constraint. Keeping query tokens allows the verifier to preserve this structure until after cross-attention.

**Risk or limitation:**

Not all query tokens are equally important. Filler tokens can dilute the signal if the pooling step treats them equally. This is why MaxSim-weighted pooling and query-token coverage features matter.

### 5. Frozen Page Patch Tokens P

**Classification:** Necessary.

Each page is represented by a set of page token embeddings:

```text
P = {p_1, p_2, ..., p_m}
```

These are often described as patch embeddings, but the phrase can be misleading. They are not raw local crop descriptors. They are contextualized VLM outputs. In ColQwen2/Qwen2-VL-derived systems, page/image tokens are processed through a model that contains spatial and visual-language structure. Therefore, `p_j` should be treated as a contextualized page token, not as an isolated patch.

**What it seeks to accomplish:**

- Represent the page's multimodal content.
- Preserve layout, visual, table, chart, and figure information in the frozen model space.
- Provide the keys/values that query tokens attend over.
- Provide the candidate evidence that MaxSim and the verifier inspect.

**Why it is necessary:**

The verifier must decide whether a page supports the query. The page tokens are the frozen evidence representation. Without them, the verifier collapses into a score-only reranker.

**Important correction from the critique:**

Do not say:

> MaxSim matches independent local patches.

Say:

> MaxSim aggregates independent best matches over contextualized page tokens.

This keeps the real limitation without misdescribing the model.

**Risk or limitation:**

Because page tokens are already contextualized, adding explicit coordinates may duplicate information. This is one reason geometry should be an ablation, not a default design choice.

### 6. Q x P Similarity Matrix

**Classification:** Necessary internal object.

For each query-page pair, compute:

```text
S_ij = q_i dot p_j
```

This matrix is the source of MaxSim:

```text
s_i = max_j S_ij
MaxSim = sum_i s_i
```

But the matrix contains more information than the final score. It contains which query tokens matched strongly, whether those matches were sharp or diffuse, whether all key query tokens had support, and how much the final score depends on one token versus many.

**What it seeks to accomplish:**

- Provide the original retrieval score.
- Provide per-token max scores for pooling.
- Provide distributional alignment features.
- Provide diagnostics for whether the verifier is correcting or copying MaxSim.

**Why it is necessary:**

The main design idea is to recover useful information that MaxSim collapses. The similarity matrix is where that information lives.

**Risk or limitation:**

Because the similarity matrix is also the source of MaxSim, it can become a shortcut. If the model uses only similarity-derived features and ignores cross-attended content, it may become a more complicated version of the original retriever. This is why the MaxSim-zero diagnostic is mandatory.

### 7. MaxSim Scalar

**Classification:** Controlled-risk main-model input, late-fused only.

The MaxSim scalar is the original ColQwen retrieval score. It is useful because it tells the verifier where the query-page pair sits on the retrieval-confidence scale. In the current data, rank still carries signal: Gold@1 is around 70%, so the top-scoring page is often correct.

However, the MaxSim scalar is also the thing we are trying to correct. If the model receives MaxSim too early, it may learn a shortcut: high score means relevant. That would produce a classifier that looks good on easy distributions but fails to add a real content-verification capability.

**What it seeks to accomplish:**

- Provide a calibration anchor.
- Preserve useful rank/score information.
- Help the model distinguish borderline candidates from obviously strong ones.

**Why it is included:**

The goal is not to pretend MaxSim is useless. It is not useless. It is a strong retrieval signal. The goal is to prevent MaxSim from being the entire decision.

**How to include it safely:**

Use late fusion:

```text
final_logit = f_content(content_features) + g_score(MaxSim)
```

Here, `f_content` is the cross-attention/alignment branch and `g_score` is a small score branch. This separation matters. It lets us evaluate whether the content branch can stand without the score branch.

**Mandatory diagnostic:**

At test time, zero or remove the MaxSim scalar and evaluate again. If performance collapses, the model is relying too heavily on the retrieval score.

**Important caveat:**

MaxSim appears in multiple places:

1. pooling weights
2. distributional alignment features
3. late-fused scalar

This concentration is risky. It does not mean remove all MaxSim-derived signals. It means we must audit them carefully. The final report should never claim the verifier is independent of MaxSim. It is explicitly retrieval-aware.

### 8. Per-Token Max Scores

**Classification:** Necessary alignment feature and pooling input.

For each query token:

```text
s_i = max_j S_ij
```

The vector `{s_i}` says which query tokens found strong evidence somewhere on the page. Unlike the final MaxSim scalar, it preserves the distribution across query tokens.

**What it seeks to accomplish:**

- Identify which query tokens drove retrieval.
- Provide weights for pooling attended query states.
- Expose whether the score is balanced across query concepts or dominated by one token.

**Why it is necessary:**

The core failure mode is often not "nothing matched." High-scoring false positives matched something strongly. The question is whether the page matched the query as a complete evidence need. Per-token max scores let the verifier see whether all important parts of the query are represented.

**Example intuition:**

Two pages can have the same total MaxSim:

```text
Page A: revenue high, Germany high, 2021 high
Page B: revenue extremely high, Germany low, 2021 low
```

The final sum can hide this difference. The per-token vector exposes it.

**Risk or limitation:**

Per-token max scores are still similarity features. They do not by themselves prove answerability. They are strongest when combined with cross-attended content.

### 9. Distributional Alignment Features

**Classification:** Necessary main-model feature set and essential baseline feature set.

Distributional alignment features summarize the shape of the similarity matrix and per-token max distribution. They are cheap, parameter-free, and directly tied to what MaxSim discards.

Core features should include:

| Feature | What it measures | Why it matters |
| --- | --- | --- |
| per-token max vector | Which query tokens matched strongly | Preserves token-level match shape. |
| mean/max/min/std of `{s_i}` | Overall match distribution | Distinguishes balanced vs uneven match patterns. |
| query-token coverage | Fraction of important tokens above threshold | Tests whether the page covers the full query need. |
| entropy/sharpness per query token | Whether a token matched one clear page token or many diffuse tokens | Sharp evidence can differ from diffuse similarity. |
| top-k gap/margin | Difference between best and next-best page token matches | Measures confidence of alignment. |
| number of weakly grounded query tokens | How much of the query is unsupported | Useful for catching partial false positives. |

**What it seeks to accomplish:**

- Recover information lost in the MaxSim sum.
- Give the classifier explicit signals for partial vs complete matches.
- Provide a strong non-cross-attention baseline.
- Help identify whether cross-attention adds value beyond engineered alignment summaries.

**Why it is necessary:**

These features operationalize the revised thesis better than geometry does. The revised thesis is not "matches are far apart." It is "the page may match query-like content without satisfying the query-as-composed." Coverage, sharpness, and score distribution directly test that.

**Coverage is especially important:**

Coverage asks whether all key query tokens have evidence. This is a content-side measure of compositionality without relying on spatial assumptions. If a page strongly matches "revenue" but not the entity or year, coverage should expose that.

**Risk or limitation:**

These features may be strong enough that a small MLP over them matches the full cross-attention model. That would weaken the need for cross-attention. This is not a failure of the project; it is an important empirical result. But it would change the contribution from "cross-attention verifier" to "alignment-statistic verifier."

### 10. Pre-Norm

**Classification:** Necessary default.

Before cross-attention, apply LayerNorm or RMSNorm to the frozen query/page token embeddings:

```text
Q_norm = Norm(Q)
P_norm = Norm(P)
```

Frozen embeddings come from a large pretrained model and may have scale properties not ideal for a small trainable head. Pre-norm makes the trainable attention module easier to optimize.

**What it seeks to accomplish:**

- Stabilize attention training.
- Reduce sensitivity to frozen embedding scale.
- Make the small verifier less brittle.

**Why it is necessary:**

This is a low-cost stability measure. The trainable verifier is small relative to ColQwen and will be trained on a much smaller dataset. Avoiding avoidable scale instability is worth it.

**Risk or limitation:**

Normalization can remove some absolute scale information. That is acceptable because absolute retrieval confidence is separately available through the late-fused MaxSim scalar and distributional score features.

### 11. Cross-Attention: Q Attends Over P

**Classification:** Main model.

The cross-attention layer computes query-conditioned page evidence:

```text
Z = CrossAttention(Q_norm, K=P_norm, V=P_norm)
```

Each output state `z_i` corresponds to query token `q_i` after it has attended over the page tokens. This is not the same as MaxSim. MaxSim picks the single best page token per query token and sums scores. Cross-attention can form a soft mixture of page evidence for each query token, potentially capturing richer content relationships.

**What it seeks to accomplish:**

- Let each query token inspect the page evidence in a learned way.
- Produce attended query states that encode what the page offered in response to the query.
- Move beyond the final MaxSim scalar.
- Create a content branch that can, in principle, stand independently of retrieval score.

**Why it is included:**

The verifier's main question is relational: does this page support this query? Cross-attention is a natural mechanism for pairwise query-page interaction while keeping the heavy ColQwen encoder frozen.

**What cross-attention does not do by itself:**

It does not prove spatial co-location. It does not necessarily know whether two matched tokens came from the same chart, caption, or table. It does not perform full VLM reasoning. It is a lightweight attentive probe over frozen embeddings.

**Why the model uses Q over P instead of P over Q:**

The output should remain query-token indexed so pooling can be weighted by per-query-token MaxSim scores. Each query token asks the page: "what evidence do you have for me?" This matches the MaxSim structure and makes the later pooling step coherent.

**Number of heads:**

Four heads is a reasonable initial default for a small verifier. More heads increase capacity but also increase the chance of overfitting and make the model harder to interpret. The number of heads should be treated as a tunable implementation detail, not the main research claim.

**Risk or limitation:**

The cross-attention model may not beat an MLP over distributional alignment features. If it does not, then cross-attention is unnecessary complexity for the pilot. This is why B1 is the load-bearing baseline.

### 12. MaxSim-Weighted Pooling

**Classification:** Main model, high-value component.

Cross-attention produces one attended state per query token:

```text
Z = {z_1, z_2, ..., z_n}
```

The MLP needs a fixed-length vector, so the sequence must be pooled. Uniform mean pooling is the simplest option:

```text
h = (1/n) sum_i z_i
```

But uniform pooling gives filler tokens the same weight as important content tokens. MaxSim-weighted pooling uses each query token's best-match score:

```text
w_i = softmax(s_i / tau)
h_attn = sum_i w_i z_i
```

or another normalized weighting rule.

**What it seeks to accomplish:**

- Focus the pooled representation on query tokens that drove retrieval.
- Avoid diluting important evidence with filler tokens.
- Reuse ColQwen's useful token-level matching signal without making the final decision purely MaxSim.

**Why it is necessary:**

Pooling is not a minor implementation detail. It decides what information survives into the classifier. MaxSim-weighted pooling is the best default because it aligns the verifier's aggregation with the retrieval mechanism while still letting cross-attended content determine relevance.

**Key conceptual distinction:**

MaxSim-weighted pooling does not decide relevance. It decides where the verifier looks. On a false positive, important tokens may also have high MaxSim weights. That is why the attended content still matters.

The division of labor is:

```text
MaxSim weights: which query-token outputs matter most?
Cross-attended content: what did the page actually provide for those tokens?
MLP: does that evidence pattern support the query?
```

**Risk or limitation:**

Because the weights are MaxSim-derived, this is one of the three MaxSim pathways. It can contribute to shortcut learning. The diagnostic is to evaluate the model with MaxSim-derived weights replaced by uniform weights or with all MaxSim-derived inputs zeroed.

**Ablations:**

- uniform mean pooling
- learned attention pooling
- MaxSim-weighted pooling
- MaxSim-weighted pooling with temperature sweep

The pooling ablation is one of the most important architecture experiments.

### 13. Query Residual

**Classification:** Main model.

The query residual is a pooled representation of the original query tokens:

```text
h_query = pool(Q)
```

It is concatenated with the attended evidence vector. Its purpose is not simply to "preserve query information" in the abstract. Its purpose is to let the MLP compare:

```text
what was asked
vs.
what the page returned after attention
```

**What it seeks to accomplish:**

- Preserve the original query intent.
- Let the classifier evaluate whether the page response coherently satisfies that intent.
- Prevent the model from relying only on the page-conditioned attended vector.

**Why it is necessary:**

Evidence relevance is a comparison. A page can only be evidence relative to a query. The attended vector alone may encode what the page offered, but the MLP should also see a clean query baseline.

**Scale matching is mandatory:**

If `h_attn` comes from a normalized attention path but `h_query` is raw ColQwen scale, concatenation creates a silent scale mismatch. The larger-magnitude block can dominate the early gradients.

Therefore:

```text
h_query = Norm(pool(Q))
h_attn = Norm_or_attention_output_scale(h_attn)
```

or otherwise ensure both blocks are comparable.

**Risk or limitation:**

If the query residual is too strong and the page features are weak, the model may learn query priors instead of page relevance. This is why a "no query residual" ablation is useful.

### 14. Concatenation Layer

**Classification:** Necessary integration step.

The concatenation stage builds the final feature vector for the content branch:

```text
h_content = concat(
  h_attn,
  h_query,
  alignment_features
)
```

The MaxSim scalar should not be early-mixed here if late fusion is used. It should enter through a separate branch.

**What it seeks to accomplish:**

- Combine attended evidence, query intent, and alignment statistics.
- Give the MLP access to complementary views of the query-page pair.
- Keep feature groups explicit enough to ablate.

**Why it is necessary:**

No single component is sufficient. The attended vector gives learned content interaction. The query residual gives a clean query baseline. Alignment features give explicit retrieval-shape information. Concatenation is the simplest way to expose all three to the classifier.

**Risk or limitation:**

If too many features are mixed too early, interpretability and ablation clarity suffer. The report's recommended design keeps the MaxSim scalar separate specifically to reduce this problem.

### 15. 2-Layer MLP

**Classification:** Necessary default.

The MLP maps the concatenated content features to a logit:

```text
u = LayerNorm(h_content)
r = GELU(W1 u + b1)
r = Dropout(r)
logit_content = W2 r + b2
```

The final probability comes after late fusion and sigmoid.

**What it seeks to accomplish:**

- Learn nonlinear combinations of attended content, query residual, and alignment features.
- Model interactions such as "high coverage but diffuse evidence" or "strong entity match but weak attribute match."
- Produce a scalar evidence logit.

**Why 2 layers instead of 1:**

A linear classifier may be too weak because relevance is likely not linearly separable in the concatenated feature space. The model needs enough capacity to compare feature groups. A 2-layer MLP is a conservative default.

**Why not make it much deeper:**

The project should prove the value of the representation and alignment design, not bury everything in classifier capacity. A deeper MLP increases overfitting and makes it harder to interpret what component helped.

**Risk or limitation:**

The MLP may learn dataset artifacts if splits are not document-disjoint. Strong regularization, early stopping, and held-out documents are important.

### 16. Late Fusion of MaxSim Scalar

**Classification:** Controlled-risk main-model component.

Late fusion means the MaxSim scalar enters at the output stage, not as an early feature mixed into the content vector:

```text
logit_final = logit_content + g(MaxSim)
```

where `g` can be a tiny linear or shallow calibration branch.

**What it seeks to accomplish:**

- Use the retrieval score as a calibration anchor.
- Preserve score/rank signal without letting it contaminate the content branch.
- Make shortcut diagnostics cleaner.

**Why it is necessary or useful:**

MaxSim is not wrong. It is just insufficient. Removing it completely may throw away useful calibration. Late fusion is a controlled way to keep it.

**Why late fusion matters:**

If MaxSim is concatenated early with everything else, the MLP can use it as a shortcut. If it is late-fused, we can inspect:

```text
content branch alone
score branch alone
combined model
```

This makes the research cleaner.

**Risk or limitation:**

Even late-fused MaxSim can dominate if the content branch is weak. This must be audited by reporting content-only performance and MaxSim-ablated performance.

### 17. Sigmoid Relevance Probability

**Classification:** Necessary output.

The final logit becomes:

```text
p = sigmoid(logit_final)
```

This is interpreted as the model's estimated probability that the candidate page is useful evidence for the query.

**What it seeks to accomplish:**

- Produce a binary evidence score.
- Enable filtering by threshold.
- Enable reranking by probability.
- Enable variable-size selected page sets.

**Why it is necessary:**

The project goal is dynamic evidence selection. A calibrated probability is more flexible than a fixed top-k rank. It supports:

- keep pages above threshold
- keep at least one page
- cap maximum selected pages
- tune recall/precision tradeoff

**Risk or limitation:**

Sigmoid scores are not automatically calibrated. Calibration should be measured. Thresholds chosen on validation data may not transfer across domains unless evaluated carefully.

### 18. Thresholding and Candidate Selection

**Classification:** Necessary pipeline policy.

After the verifier scores each candidate page, the system must decide which pages to pass to the reader.

Possible policies:

1. threshold all pages with `p >= t`
2. top-r by verifier probability
3. threshold with minimum one page
4. threshold with maximum page cap
5. recall-preserving policy that keeps extra pages when confidence is uncertain

**What it seeks to accomplish:**

- Reduce reader context noise.
- Preserve gold evidence.
- Control reader cost.
- Support dynamic retrieval depth.

**Why it is necessary:**

The verifier's output only matters when it changes the reader input. The selection policy converts probabilities into a page set.

**Recommended initial policy:**

Use thresholding with a minimum-one constraint and a maximum cap:

```text
selected = pages with p >= t
if selected is empty: keep highest-p page
if selected is too large: keep top-r by p
```

This prevents catastrophic empty-context failures while still enabling dynamic filtering.

**Risk or limitation:**

False negatives are more damaging than false positives. Dropping the gold page is unrecoverable. Therefore the threshold should be tuned for high gold recall, not maximum raw accuracy.

### 19. VLM Reader

**Classification:** Necessary downstream evaluator, not part of verifier training at first.

The selected pages are passed to a reader model. The reader may be a VLM with tool calls, OCR zoom, crop tools, segmentation, or layout parsing. But the verifier should first be evaluated independently as a reranker/filter before adding full reader complexity.

**What it seeks to accomplish:**

- Produce the final answer.
- Test whether evidence selection actually helps document QA.
- Measure the real pipeline effect of filtering.

**Why it is necessary:**

A binary classifier can improve PR-AUC and still hurt the reader if it occasionally drops gold pages. The final claim must be based on downstream answer accuracy or at least downstream answerability proxies.

**Risk or limitation:**

Reader performance introduces another source of variance. The verifier should be debugged first with retrieval/reranking metrics, then evaluated in the full pipeline.

## Geometry: Why It Is Not in the Main Model

### Geometry Block Under Consideration

The geometry-heavy version included:

- patch coordinates `C`
- coordinate-embedded page tokens `P + coord_embedding(C)`
- best-match patch locations
- geometry summaries such as dispersion or clustering

These should be removed from the primary architecture and tested as an ablation.

### Why Geometry Is Tempting

MaxSim can produce interpretable heatmaps. It is natural to wonder whether false positives happen because query terms match different page regions that do not jointly answer the query. If that is the failure, location features could help.

### Why Geometry Is Risky as the Main Thesis

The critiques identify three issues:

1. **Target misspecification:** True evidence can be spatially distributed. A figure, legend, caption, and table can all contribute to the answer.
2. **Redundancy:** ColQwen2 is built on Qwen2-VL-style representations, which already encode spatial position through the underlying visual-language model. Raw coordinates may duplicate or distort information already present in `P`.
3. **Confounding:** If the full model improves with coordinates baked in, it becomes unclear whether the gain came from content query-conditioning or from explicit geometry.

### Correct Role of Geometry

Geometry should be one ablation:

```text
G1 = A2 content-first model + explicit coordinate/location features
```

If G1 beats A2 on document-disjoint held-out data, then we can claim:

> Explicit patch-location features add discriminative signal beyond frozen ColQwen embeddings and MaxSim alignment features.

If G1 does not beat A2, then geometry is unnecessary for the core contribution.

## MaxSim Shortcut Risk

The model is intentionally retrieval-aware, but that creates a shortcut risk.

MaxSim-derived information enters through:

1. pooling weights
2. distributional alignment features
3. late-fused MaxSim scalar

This is not automatically wrong. It is the design. But it means the model may learn to copy retrieval confidence rather than perform evidence verification.

### Required Diagnostics

| Diagnostic | What to do | What it tells us |
| --- | --- | --- |
| MaxSim scalar ablation | Remove/zero late-fused scalar | Tests dependence on original retrieval score. |
| All MaxSim-derived input ablation | Replace pooling weights with uniform and zero alignment features | Tests whether content branch learned anything independent of MaxSim. |
| Content branch only | Evaluate without score branch | Tests whether cross-attended content is useful. |
| Score branch only | Evaluate MaxSim scalar/rank alone | Establishes floor baseline. |
| Shuffle labels | Train on shuffled labels | Catches leakage/artifacts. |

The strongest warning sign would be:

```text
Full model performs well
MaxSim-derived ablated model collapses
B1 alignment-feature MLP matches full model
```

That would mean the model is mostly an expensive reranker over retrieval features. That may still be useful practically, but it weakens the cross-attention contribution.

## Baseline and Ablation Ladder

The architecture should be evaluated in a ladder that isolates each idea.

| ID | Model | Classification | Purpose |
| --- | --- | --- | --- |
| B0 | MaxSim scalar/rank logistic regression | Baseline | Tests whether original retrieval score/rank is enough. |
| B1 | MLP over MaxSim distribution features | Strong baseline | Tests whether cheap alignment statistics solve the task. |
| A0 | Cross-attention + uniform pooling | Ablation | Tests whether cross-attention alone helps. |
| A1 | Cross-attention + MaxSim-weighted pooling | Main-path ablation | Tests pooling choice. |
| A2 | Content-first full model without geometry | Main model | Query residual + alignment features + weighted pooling + late-fused scalar. |
| G1 | A2 + geometry/location features | Ablation only | Tests whether explicit spatial information helps. |
| R0 | A2 without query residual | Ablation | Tests query residual importance. |
| S0 | A2 without late-fused MaxSim scalar | Diagnostic/ablation | Tests score shortcut dependence. |
| D0 | A2 with all MaxSim-derived signals removed | Diagnostic | Tests whether content attention branch stands on its own. |

The load-bearing comparison is:

```text
A2 vs B1
```

If A2 beats B1, cross-attention adds value beyond summary alignment features. If it does not, the simpler model is the more honest design.

The geometry comparison is:

```text
G1 vs A2
```

If G1 beats A2, geometry adds value. If not, keep geometry out.

The downstream comparison is:

```text
reader with ColQwen top-k
vs.
reader with verifier-selected pages
```

This is the final pipeline test.

## Training Design

### Labels

For each query-candidate page pair:

```text
y = 1 if candidate page is annotated gold evidence
y = 0 otherwise
```

For multi-page evidence datasets, this can extend to multiple positive pages per query. For the pilot, single-page evidence is cleaner and easier to interpret.

### Negatives

Negatives should be hard negatives from ColQwen top-k candidates, not random pages. This is essential because the verifier must learn to correct ColQwen's mistakes among plausible candidates.

### Loss

Use binary cross-entropy:

```text
L = -[ y log(p) + (1-y) log(1-p) ]
```

Use positive class weighting or balanced sampling if negatives greatly outnumber positives.

### Split

Prefer document-disjoint splits:

```text
train documents != validation documents != test documents
```

This prevents the model from memorizing document-specific patterns.

### Initial Data Size

The 4K-query pilot is enough to test whether the approach has signal. It is not enough to establish final performance claims across broad domains. More data helps variance, but it does not fix a wrong inductive bias. That is why geometry stays as an ablation even if more data becomes available.

## Metrics

### Classifier Metrics

| Metric | Why it matters |
| --- | --- |
| PR-AUC | Better than accuracy under class imbalance. |
| F1 | Captures precision/recall tradeoff at a threshold. |
| Recall of gold pages | Most important safety metric. |
| Precision of selected pages | Measures context noise reduction. |
| Calibration / reliability | Tests whether sigmoid scores can be thresholded. |

### Reranking Metrics

| Metric | Why it matters |
| --- | --- |
| Gold@1 after verifier reranking | Main reranking improvement target. |
| Gold@3 / Gold@5 after verifier reranking | Tests whether gold remains highly ranked. |
| MRR | Measures rank movement. |
| nDCG@k | Useful if multiple positives or graded evidence labels exist. |

### Pipeline Metrics

| Metric | Why it matters |
| --- | --- |
| Downstream answer accuracy | Final verdict. |
| Average selected pages | Measures reader context reduction. |
| Gold retained after filtering | Safety metric. |
| Reader cost/latency | Practical benefit. |

The project should not claim success from classifier PR-AUC alone. A verifier that drops gold pages can improve classifier-looking metrics and still hurt QA.

## How to Explain the Architecture in Slides

Use this language:

> The verifier reuses frozen ColQwen query and page embeddings. Query tokens attend over candidate page tokens, then attended states are pooled using MaxSim-derived token weights. The classifier also receives a scale-matched query residual and alignment statistics such as coverage and match sharpness. The original MaxSim score is fused only at the output as a calibration anchor.

Avoid this language:

> The verifier detects whether evidence tokens are spatially co-located.

Avoid this unless G1 wins:

> Explicit geometry is necessary for relevance verification.

Use this for the research question:

> Can a lightweight, retrieval-aware verifier over frozen ColQwen embeddings improve page-level evidence selection from high-recall candidates?

Use this for the limitation:

> MaxSim is excellent for candidate retrieval, but the final score collapses token-level alignment into a scalar that is not calibrated for evidence applicability.

## Final Recommended Architecture

The final main model is:

```text
Inputs:
  Q: frozen ColQwen query token embeddings
  P: frozen ColQwen page token embeddings
  S: Q x P similarity matrix

Derived:
  MaxSim scalar
  per-token max scores
  distributional alignment features

Content branch:
  Q_norm, P_norm
  Z = CrossAttention(Q_norm over P_norm)
  h_attn = MaxSimWeightedPool(Z, per-token max scores)
  h_query = Norm(Pool(Q))
  h = concat(h_attn, h_query, alignment_features)
  logit_content = 2-layer MLP(LayerNorm(h))

Score branch:
  logit_score = small calibration function(MaxSim scalar)

Output:
  p = sigmoid(logit_content + logit_score)
```

Geometry is not part of the main model:

```text
G1 = main model + explicit coordinate/location features
```

The report's final classification is:

- **Necessary:** frozen ColQwen embeddings, candidate set, Q/P tokens, similarity matrix, cross-attention, weighted pooling, query residual, alignment features, MLP, sigmoid.
- **Controlled-risk necessary:** late-fused MaxSim scalar.
- **Mandatory diagnostics:** MaxSim-zero tests, content-only tests, score-only baselines, downstream answer evaluation.
- **Ablation only:** explicit coordinates, best-match locations, geometry summaries.
- **Load-bearing baseline:** MLP over MaxSim distribution features.

## Bottom Line

The architecture should be content-first and retrieval-aware:

> Use MaxSim to decide what to inspect, not to make the final relevance decision.

The verifier should learn from the content returned by query-page interaction, compare it to the original query, and use alignment statistics to detect partial or diffuse matches. Explicit geometry is a legitimate scientific hypothesis, but it is not the safest main model. The project is strongest when it claims calibrated evidence selection over high-recall candidates, not spatial reasoning over page patches.

