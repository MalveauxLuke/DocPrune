# V-JEPA Attentive Probing: Applicability to the ColQwen Binary Classifier

## Purpose

This note reviews the V-JEPA paper with special attention to Appendix D.1,
“Frozen classification,” where the authors describe attentive probing. It then
evaluates how much that design supports our planned ColQwen binary
page-relevance classifier.

The short answer: V-JEPA is a useful motivation for using a trainable
cross-attention readout over frozen token embeddings, but it does not directly
validate our full query-page cross-attention architecture. V-JEPA uses
cross-attention mainly as adaptive pooling over one frozen visual feature
sequence. Our task uses cross-attention as a query-conditioned interaction
between two frozen sequences: query tokens and page tokens.

## What V-JEPA Is Doing

V-JEPA is a self-supervised video representation learner. It trains a vision
transformer on videos using feature prediction rather than pixel reconstruction,
text supervision, contrastive negatives, or labels. The model sees a masked
video clip, predicts feature representations for missing spatio-temporal
regions, and compares those predictions to a target encoder representation.

The backbone is then evaluated in two main ways:

- frozen evaluation, where the V-JEPA encoder is frozen and only a lightweight
  task head is trained;
- full fine-tuning, where the model is adapted end-to-end.

The attentive probing mechanism belongs to the frozen evaluation setting. It is
not the pretraining objective itself. It is the downstream classifier head used
to test whether the frozen V-JEPA token sequence contains useful information.

## Appendix D.1: Frozen Classification

In Appendix D.1, the frozen V-JEPA encoder outputs a sequence of tokens:

```text
E(x) = [s1, s2, ..., sL]
```

Each `si` is a token embedding from the video/image feature map. Instead of
average-pooling these tokens into one vector, V-JEPA trains a small attentive
probe:

```text
learnable query q attends to frozen encoder tokens S
output is added back to q as a residual
two-layer MLP + GeLU + LayerNorm
linear classifier
```

The encoder stays frozen. The cross-attention block and classifier are learned
for the downstream task.

The paper says they use an attentive probe with `12` heads, each of dimension
`12`. This means their head-count choice is part of their frozen-evaluation
protocol, not a universal recommendation. It should not be copied blindly into
our setting.

The important point is functional: V-JEPA uses cross-attention to pool a token
sequence adaptively instead of applying a fixed average over all tokens.

## Why V-JEPA Uses Cross-Attention

V-JEPA uses cross-attention because frozen token features are not guaranteed to
be linearly separable after simple pooling. The paper explicitly motivates
attentive probing by saying that there is no reason the encoder's feature
prediction objective should yield a representation where average-pooled features
are already optimal for downstream classification.

Cross-attention helps because it gives the task head a learned way to select and
combine relevant tokens.

For video classification, not every spatio-temporal patch matters equally:

- some patches contain the action;
- some contain background;
- some contain objects useful for Kinetics-style appearance classification;
- some contain motion evidence useful for Something-Something-v2;
- some clips contain more relevant evidence than others.

Average pooling treats these all as equal. A learned query can attend more to
the task-relevant parts of the frozen feature sequence.

That is why the paper reports a large gap between average pooling and attentive
pooling in frozen evaluation. In Table 3, attentive pooling improves V-JEPA
frozen evaluation by about `+17.3` points on Kinetics-400 and `+16.1` points on
Something-Something-v2 compared with average pooling.

Appendix E reinforces that this is not only a V-JEPA-specific trick. VideoMAE,
DINOv2, and OpenCLIP also benefit from attentive probing in their frozen
evaluation protocol. That matters for us because it suggests the gain comes from
the readout mechanism being better than naive pooling, not only from the V-JEPA
pretraining objective.

## What Attentive Probing Is Not

V-JEPA attentive probing is not a full cross-encoder in the usual retrieval
sense.

It does not jointly encode a query and a document from raw inputs. It does not
perform deep bidirectional fusion between two separately meaningful sequences.
It does not update the frozen encoder.

It is closer to:

```text
frozen token sequence -> learned attention pooling -> classifier
```

The learnable query token is not a text query. It is a task-specific pooling
token. It asks, in effect:

```text
Which frozen visual tokens should matter for this downstream class decision?
```

This distinction is critical for our architecture.

## Our Setting

Our intended experiment is different:

```text
query -> known document -> rank pages within that document
```

For each query-page pair we have:

```text
Q = ColQwen query token embeddings       # Nq x 128
P = ColQwen page token embeddings        # Np x 128
```

ColQwen/ColPali was trained for multi-vector late interaction:

```text
for each query token:
  find max dot product over page tokens
sum those token-wise maxes
```

Our classifier asks a different question:

```text
Given the query and a candidate page from the known document,
is this the relevant/gold page?
```

The planned main model is:

```text
frozen ColQwen query tokens
frozen ColQwen page tokens
trainable query-to-page cross-attention
pool attended query states
binary classifier
```

This is inspired by attentive probing, but it is not identical to it.

## Where V-JEPA Applies Well

V-JEPA strongly supports the idea that frozen token embeddings often need a
learned readout. This maps well to our problem.

ColQwen's MaxSim score is also a readout:

```text
max over page tokens per query token, then sum
```

It is fixed, non-learned, and optimized for retrieval ranking, not calibrated
binary page relevance within a document. Our classifier can be understood as a
learned readout over frozen ColQwen evidence.

The analogy is:

| V-JEPA | Our ColQwen Classifier |
| --- | --- |
| frozen video/image encoder | frozen ColQwen query/page encoders |
| token sequence from one input | query token sequence plus page token sequence |
| average pooling is too rigid | MaxSim/rank may be too rigid |
| attentive probe learns adaptive pooling | cross-attention head learns adaptive relevance evidence |
| downstream classifier over frozen features | binary classifier over frozen features |

The most applicable lesson is:

```text
Do not assume fixed pooling/readout is enough.
Train a small readout over frozen token features before touching the backbone.
```

That supports our plan to freeze ColQwen and train a small cross-attention head
before considering any end-to-end fine-tuning.

## Where V-JEPA Does Not Directly Apply

The mismatch is also important.

V-JEPA attentive probing uses one learned query token to summarize a single
visual token sequence. Our problem has a real natural-language query. The query
is not merely a pooling token; it specifies what evidence should matter.

So copying V-JEPA literally would give us something like:

```text
learnable probe token attends to concat(query tokens, page tokens)
classifier
```

That would be possible, but it would ignore a useful structure already present
in ColQwen: query tokens and page tokens live in a shared retrieval space.

For us, the better adaptation is:

```text
ColQwen query tokens attend to ColQwen page tokens
```

This uses the query itself as the attention query. Each query token can ask:

```text
Which page tokens provide evidence for me?
```

That is closer to a trainable replacement for MaxSim than V-JEPA's original
attentive probe.

## Cross-Attention vs Cross-Encoder

We should be careful with the term “cross-encoder.”

A full cross-encoder usually means:

```text
raw query + raw document/page are fed together through a transformer
all layers can jointly attend across both inputs
```

That is powerful, but expensive. It also abandons much of the caching benefit
of ColQwen-style retrieval.

Our proposed model is not that. It is a lightweight interaction head:

```text
encode query once with frozen ColQwen
encode pages once with frozen ColQwen
train a small cross-attention classifier over cached embeddings
```

This preserves the practical benefits of late interaction:

- page embeddings can be cached;
- query embeddings can be cached for training;
- the heavy VLM stays frozen;
- the trainable head is small;
- the architecture tests whether ColQwen embeddings contain usable relevance
  evidence beyond the fixed MaxSim score.

So the right wording is probably:

```text
cross-attention reranker over frozen ColQwen embeddings
```

not:

```text
full cross-encoder
```

## Should We Use a V-JEPA-Style Learnable Query Token?

Maybe, but not as the first main model.

A V-JEPA-style learnable query token could be useful as a pooling mechanism
after query-page interaction. For example:

```text
Q attends to P -> attended query states
learnable pooling token attends to attended query states
MLP classifier
```

This would separate two jobs:

1. query-page evidence extraction;
2. adaptive pooling of extracted evidence.

But as a first model, this adds another learned mechanism. Mean-plus-max pooling
over attended query states is simpler and easier to debug:

```text
attended = CrossAttention(Q, P, P)
pooled = concat(mean_pool(attended), max_pool(attended))
classifier(pooled)
```

V-JEPA tells us learned pooling can help. It does not prove we need learned
pooling immediately.

## Should Query Tokens Attend to Each Other?

V-JEPA does not add self-attention among the frozen encoder tokens in the probe.
It replaces the self-attention part of a transformer block with cross-attention
from a query token to the frozen sequence. The assumption is that the frozen
encoder already contextualized its tokens.

That maps to our current intuition:

```text
ColQwen query tokens are already contextualized by ColQwen.
ColQwen page tokens are already contextualized by ColQwen.
```

Therefore, the first model should spend its capacity on query-page interaction,
not extra query self-attention.

A later ablation can add:

```text
Q = small self-attention adapter(Q)
P = small adapter(P)
CrossAttention(Q, P, P)
```

But the first experiment should stay closer to the frozen-readout question:

```text
Can a small learned interaction head read better relevance evidence from frozen
ColQwen tokens than MaxSim can?
```

## Attention Head Count

V-JEPA's Appendix D.1 says their attentive probe uses `12` heads of dimension
`12`. That is useful evidence that multi-head attention can be an effective
probe, but it is not directly portable.

Reasons not to copy `12` heads:

- V-JEPA's feature dimension and probe design are different.
- V-JEPA attends from a learned pooling token to a large visual/video token
  sequence.
- Our embeddings are `128`-dimensional ColQwen multi-vectors.
- Our dataset is much smaller than Kinetics/SSv2-style probe training.
- Our task is binary query-page relevance, not multi-class action or image
  classification.

For our first run:

```text
2 heads is the best default.
4 heads is the first capacity ablation.
8 heads is later if 4 heads clearly underfits.
```

The reasoning:

- `1` head is too restrictive except as a sanity ablation.
- `2` heads allows one head to specialize in sharp local evidence and another
  in broader contextual evidence.
- `4` heads may help with multi-clue queries involving text, figures, tables,
  layout, and numeric details.
- `8+` heads increases capacity and interpretability burden before we know
  whether the basic approach works.

V-JEPA supports “multi-head attention can be a good probe.” It does not support
“use 12 heads for this ColQwen binary classifier.”

## What V-JEPA Suggests for Our Architecture

The strongest V-JEPA-aligned architecture for our first main model is:

```text
Q = frozen ColQwen query tokens        # Nq x 128
P = frozen ColQwen page tokens         # Np x 128

Qp = LayerNorm/Linear(Q)
K  = Linear(P)
V  = Linear(P)

A = MultiHeadCrossAttention(Qp, K, V)  # 2 heads first
Z = LayerNorm(Qp + A)                  # residual, V-JEPA-style principle

h = concat(mean_pool(Z), max_pool(Z))
logit = MLP(h)
```

Then compare two variants:

```text
Main-A: cross-attention only
Main-B: cross-attention + ColQwen score/rank/gap features
```

Main-A tells us whether attention over frozen tokens is useful by itself.
Main-B tells us the best practical reranker when combined with the existing
ColQwen score signal.

## Practical Consequences for the Current Experiment

The corrected document-scoped retrieval setup matters here. V-JEPA's attentive
probe assumes the relevant evidence is in the frozen token sequence being
pooled. Our old global run often did not even place the gold page in the top-5
candidate set. That is a retrieval-corpus mismatch, not an attentive-probing
question.

For the cross-attention classifier, the right training table should be built
from the document-scoped run:

```text
for each query:
  candidate pages = pages from the annotated gold document
  label gold page = 1
  label other pages = 0
```

Then the model is testing:

```text
Can learned query-page attention identify the right page within the known
document?
```

This is much closer to V-JEPA's frozen-probe spirit: the relevant evidence is
inside the frozen tokens; the question is whether a learned readout can extract
it.

## Risks and Limits

V-JEPA does not remove the main risks in our setting.

First, ColQwen was trained for MaxSim late interaction, not binary relevance.
The embeddings may be shaped around token-wise maximum dot products. A
cross-attention head may need to learn around that geometry.

Second, labels may be incomplete. A non-gold page in the same document can
contain partially relevant evidence. That makes binary supervision noisy.

Third, our data scale is modest. V-JEPA's attentive probes are trained on large
classification datasets. Our pilot has about `4000` queries, so head size and
regularization matter.

Fourth, V-JEPA's attention is used for pooling, not direct text-conditioned
retrieval. It motivates the family of solution, but it is not direct empirical
proof for our architecture.

## Recommendation

Use V-JEPA as support for this claim:

```text
When using frozen token-level visual representations, learned attention-based
readouts can outperform fixed pooling/readout functions.
```

Do not use it to claim:

```text
V-JEPA proves our query-page cross-attention classifier will work.
```

The correct design implication is:

1. keep ColQwen frozen for the pilot;
2. do document-scoped retrieval first;
3. train a small 2-head query-to-page cross-attention classifier;
4. include residual connections and a small MLP head;
5. compare against Baseline 1 score/rank classifier;
6. only then try 4 heads, learned pooling tokens, query self-attention, or
   BERT-style query branches.

V-JEPA makes the cross-attention head a reasonable and well-motivated readout
experiment. It does not justify skipping the baseline or overbuilding the first
model.

## Final Position

Attentive probing is applicable to our setting at the level of principle:

```text
frozen encoder tokens + learned attention readout > fixed pooling/readout
```

It is only partially applicable at the level of architecture:

```text
V-JEPA: learned query token pools one frozen visual sequence
ours: natural-language query tokens attend to frozen page tokens
```

That difference is not a problem. In fact, using ColQwen query tokens as the
attention queries is probably the more faithful adaptation for our task because
the query is not just asking for a generic class decision; it defines the
specific page evidence we need.
