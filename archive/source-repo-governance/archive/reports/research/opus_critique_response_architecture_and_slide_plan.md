# Response to Architecture Critique and Revised Project Plan

## Executive Verdict

The critique is valid, but it is not fatal to the project. It is a serious hit to the current wording of the motivation, especially the claim that MaxSim is simply a "bag of independent local patches" and that the proposed cross-attention classifier can detect spatial co-location by itself.

The project remains strong if it is reframed as:

> ColQwen/ColPali is a high-recall candidate generator, but its MaxSim score is not calibrated for page-level evidence applicability. A learned verifier can use frozen query/page embeddings plus MaxSim alignment structure to rerank or filter retrieved pages before the expensive reader stage.

This framing is better supported by our current empirical result:

| Setting | Gold@1 | Gold@3 | Gold@5 |
| --- | ---: | ---: | ---: |
| ColQwen2, document-scoped MMDocIR 4K | 70.28% | 91.88% | 97.22% |

The key observation is not that ColQwen fails to retrieve evidence. It usually retrieves it. The issue is that the gold page is often not rank 1, and non-gold pages can outrank it. This makes the project a second-stage evidence verification/reranking problem.

## What the Critique Gets Right

### 1. ColQwen patch embeddings are contextualized

The slide claim that MaxSim is a "bag-of-best-matches" is too strong if it implies independent local image patches. ColQwen/ColPali patch embeddings are produced by a VLM backbone, so each page token already contains context from the page representation. A patch token is not equivalent to a raw crop.

Safe version:

> MaxSim performs independent best-match aggregation over contextualized page tokens. This is powerful for retrieval, but the final summed score still collapses a rich query-page alignment pattern into one scalar.

### 2. Cross-attention alone does not prove spatial co-location

A single layer where query tokens attend over page tokens can learn query-conditioned interactions, but without patch coordinates or alignment-map features, it does not explicitly know where matched page tokens are located. It can model content compatibility, but not reliably reason over match geometry.

So the current architecture should not be described as detecting:

- spatial co-location
- compositional layout
- coordinated regions
- spatial grounding

unless we add position/alignment features and evaluate whether they help.

### 3. Pooling is a core design choice

Uniform mean pooling can dilute the few query tokens that actually matter. Coordinate-wise max pooling can combine dimensions from different query tokens into a vector that no real token had. The pooling layer is not a small implementation detail. It determines what evidence signal survives after cross-attention.

This makes MaxSim-weighted pooling attractive because it preserves ColQwen's useful inductive bias: query tokens with strong retrieval matches should influence the verifier more than filler tokens.

## Revised Architecture

The revised model should be a retrieval-aware, geometry-aware attentive verifier over frozen ColQwen embeddings.

```text
Frozen ColQwen encoder
        |
        |-- Query token embeddings Q
        |-- Page patch embeddings P
        |-- Patch coordinates C
        |
Q x P similarity matrix
        |
        |-- global MaxSim score
        |-- per-query-token max similarity
        |-- best-match patch index per query token
        |-- best-match patch coordinates
        |-- match sharpness / entropy / margin features
        |
Geometry-aware page tokens
P + coord_embedding(C)
        |
Cross-attention
Q attends over geometry-aware P
        |
MaxSim-weighted pooling over query-token outputs
        |
Concatenate
  pooled attended evidence
  pooled query residual
  MaxSim / alignment summary features
  geometry summary features
        |
LayerNorm
        |
2-layer MLP
        |
sigmoid: P(page is useful evidence for query)
```

### Why these changes matter

**Patch coordinates** address the critique that the classifier has no explicit spatial signal. The model still should not be oversold as a spatial reasoner, but it now has access to the information required to learn whether match locations are clustered, scattered, or otherwise structured.

**MaxSim alignment features** exploit information ColQwen already computes but normally collapses into a scalar. The per-token max values, argmax patch indices, match margins, and entropy can reveal whether the page contains sharp evidence-like matches or diffuse partial matches.

**MaxSim-weighted pooling** avoids treating all query tokens equally. This matters because many query tokens are syntactic or low-value, while one or two entity/attribute tokens may determine whether a page is actually useful evidence.

**Query residual path** remains useful because binary evidence classification is relational. The classifier should see both what the page returns after attention and what the original query asked for.

**LayerNorm plus a 2-layer MLP** is the right default head. The MLP needs enough capacity to compare query residuals, attended evidence, and scalar alignment features. LayerNorm is useful because frozen embeddings and engineered features may have different scales.

## Baselines and Ablations

The project needs to prove that cross-attention is worth using. The critical baselines are:

| Row | Model | Purpose |
| --- | --- | --- |
| B0 | MaxSim scalar or rank logistic regression | Tests whether score/rank alone is enough. |
| B1 | MLP over MaxSim summary features | Tests whether alignment statistics solve the task without cross-attention. |
| A0 | Original cross-attention verifier | Tests the initial proposed architecture. |
| A1 | Cross-attention + MaxSim-weighted pooling | Tests whether pooling is the main improvement. |
| A2 | Cross-attention + MaxSim alignment features | Tests whether retrieval alignment features matter. |
| A3 | Cross-attention + patch coordinates | Tests whether explicit geometry helps. |
| A4 | Full model | Coordinates + alignment features + weighted pooling + query residual. |

The most important comparison is not A4 versus A0. It is A4 versus B1. If a small MLP over MaxSim summary features matches the full cross-attention model, the heavier architecture is not justified.

## Evaluation Protocol

The evaluation should be document-disjoint when possible. Query-level random splits can leak document-specific visual/textual patterns across train and test.

Primary metrics:

| Metric | Why it matters |
| --- | --- |
| Gold recall at selected page budget | Dropping the gold page is unrecoverable for the reader. |
| PR-AUC | Handles class imbalance better than raw accuracy. |
| F1 at operating threshold | Measures binary verifier quality at a chosen threshold. |
| Reranked Gold@1 / Gold@3 / Gold@5 | Measures whether the verifier improves page ordering. |
| Downstream answer accuracy | Final pipeline metric; verifier only matters if reader performance improves. |
| Average selected pages | Measures context reduction and reader cost. |

Diagnostics:

- Evaluate with MaxSim features zeroed at test time to detect shortcut dependence.
- Report at least three seeds for the trainable verifier.
- Include a label-shuffle sanity check.
- Track false negatives separately because they are more damaging than false positives.

## Revised Research Story

The project should be presented as a pipeline intervention:

1. Visual document QA needs retrieval because full-document context is expensive and noisy.
2. ColQwen/ColPali is strong at high-recall page candidate generation.
3. However, its late-interaction score is optimized for ranking retrieval candidates, not for calibrated binary evidence applicability.
4. Our document-scoped run shows this clearly: gold evidence is in the top 5 for 97.22% of queries, but only rank 1 for 70.28%.
5. Therefore, the useful question is: can a lightweight verifier over frozen ColQwen representations recover better evidence ordering or filtering from the retrieved candidate set?
6. The proposed verifier reuses frozen embeddings, MaxSim alignment structure, and patch geometry, avoiding additional VLM forward passes.
7. The experiment is publishable only if it beats strong MaxSim-feature baselines and improves reader-facing metrics, not just binary accuracy.

## Slide-by-Slide Changes

### Slide 2: Related Work and Limitations

Current RQ:

> Can a learned binary classifier operating on ColQwen's patch embeddings filter retrieved pages without additional VLM calls?

Replace with:

> Can a learned binary verifier over frozen ColQwen query/page embeddings improve page-level evidence selection from high-recall retrieved candidates, without additional VLM calls?

Reason: "verifier" and "evidence selection" better describe the task than generic classification.

### Slide 3: ColPali

Keep the slide mostly intact. It already says the VLM contextualizes each patch, which is important and technically protective.

Change:

> Page image -> visual patches

to:

> Page image -> contextualized visual patch embeddings

Change:

> each query token finds its highest-similarity page patch; these per-token maxima are summed to produce a scalar relevance score, preserving token-level evidence localization at scale.

to:

> each query token finds its highest-similarity contextualized page token; these per-token maxima are summed into a fast retrieval score.

Reason: "preserving token-level evidence localization" is a little too strong for the later argument. The heatmaps are useful, but the final scalar does not preserve the alignment pattern.

### Slide 4: Limitations of ColPali

This slide needs the largest rewrite.

Replace:

> MaxSim is a bag-of-best-matches: the score sums each query token's highest patch similarity independently. A page scores highly if it contains relevant tokens, but MaxSim does not verify those tokens are co-located or compositionally connected.

with:

> MaxSim collapses alignment: the score sums independent best matches for query tokens over contextualized page tokens. This is effective for retrieval, but a high scalar score does not necessarily mean the page satisfies the full evidence need.

Replace:

> a query for "Germany revenue in 2021" matches highly against a page where each term appears in separate, unrelated tables

with:

> a query may receive a high score from strong partial matches, diffuse matches, or matches to visually similar but non-answering content.

Reason: keep the intuition without making an unsupported claim about separate tables unless we have a real failure example.

Replace:

> score is uncalibrated for compositional relevance

with:

> score is not calibrated for page-level evidence applicability

Replace RQ:

> A binary classifier trained on ColQwen's patch embeddings may learn to distinguish compositionally relevant pages from high-scoring false positives.

with:

> A binary verifier may learn query-conditioned evidence patterns that separate useful evidence pages from high-scoring false positives.

Replace figure caption:

> token-local, not compositional

with:

> token-level matching does not guarantee page-level evidence support

### New Slide After Slide 4: Empirical Motivation

Add a compact table:

| Model / setting | Gold@1 | Gold@3 | Gold@5 | Interpretation |
| --- | ---: | ---: | ---: | --- |
| ColQwen2, document-scoped MMDocIR 4K | 70.28% | 91.88% | 97.22% | High recall, imperfect rank calibration |

Suggested takeaway text:

> ColQwen2 usually retrieves the gold page, but often does not rank it first. This creates room for a lightweight verifier/reranker over the candidate set.

### Slide 5: Proposed Architecture

Change:

> Binary Classifier -> Verified Pages

to:

> Evidence Verifier / Reranker -> Selected Evidence Pages

Reason: "verified" sounds stronger than the model can guarantee.

### Slide 6: Binary Relevance Classifier

The architecture should change to match the revised design.

Replace:

> Q attend over page patches via a single cross-attention layer with 4 heads. Output is concatenated with the query vector and passed through a 2-layer MLP.

with:

> Query tokens attend over contextualized page tokens augmented with patch coordinates. The verifier also receives MaxSim alignment features such as per-token max scores, best-match locations, and match sharpness. MaxSim-weighted pooling aggregates attended query states before a 2-layer MLP predicts evidence probability.

Replace:

> the model learns to separate true evidence from high-scoring false positives

with:

> the model learns to rerank or filter high-recall candidates by predicted evidence usefulness

Replace:

> verified-relevant subset

with:

> selected candidate subset predicted to be useful evidence

### Slide 7: Inference Reader

No major change required. If time is short, leave it as-is.

If tightening:

> verified-relevant pages

should become:

> selected evidence pages

## Final Architecture Recommendation

Use the full revised verifier as the main model:

- frozen ColQwen query tokens
- frozen ColQwen page tokens
- normalized patch coordinate embeddings
- Q x P similarity matrix
- MaxSim scalar and per-token alignment summaries
- query-to-page cross-attention
- MaxSim-weighted pooling
- pooled query residual
- 2-layer MLP with LayerNorm, dropout, and sigmoid

Keep the original cross-attention-only model as an ablation, not the final design.

The clean one-line project description becomes:

> We train a lightweight, retrieval-aware evidence verifier over frozen ColQwen embeddings to convert high-recall page retrieval into better calibrated page-level evidence selection.

