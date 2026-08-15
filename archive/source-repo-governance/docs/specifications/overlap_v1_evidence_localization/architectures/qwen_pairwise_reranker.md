# Qwen Pairwise Reranker Architecture

## Role and arms

The pairwise reranker supplies the simplest deployment baseline and the
matched independent-scoring control for MiniVGent:

- **R0:** released Qwen3-VL-Reranker-2B with no project fine-tuning;
- **R1:** the same immutable checkpoint after training-only audited
  hard-non-anchor adaptation.

Both score one original question and one frozen rendered candidate at a time:

```text
score(original_question, rendered_candidate) -> scalar relevance
```

“Stock” means released pretrained weights, not random initialization. R0/R1
are mandatory even if an optional Jina model performs better because they
share the backbone, prompt family, and relevance task used by MiniVGent.

## Model lock

The primary proposed lock is:

| Component | Binding proposal |
|---|---|
| Model | `Qwen/Qwen3-VL-Reranker-2B@4bd860ac4f15ad1897a214615cccc700f8f71818` |
| Official wrapper source | `QwenLM/Qwen3-VL-Embedding@393e2978d27852b0d0230d6994f37f9c15bed73c` |
| Python | `3.11` |
| PyTorch | `2.8.*` |
| Transformers | `4.57.3` |
| `qwen-vl-utils` | `0.0.14` |
| Attention backend | `sdpa` for first execution |
| Dtype | `bfloat16` |
| Cache | `use_cache=false` |
| Released-wrapper image budget | at most 1,800 merged visual tokens |

Stage 02 must resolve every Hub and package reference to actual immutable
artifacts and record licenses, preprocessing, device, dtype, prompt,
processor, attention backend, and score extraction. The listed pins are the
approved starting configuration, not a substitute for the measured model
lock.

The Qwen technical report establishes a pointwise multimodal yes/no relevance
model. It does not establish MiniVGent memory taps, ROI features, set decoding,
or this program's answer-anchor objectives.

## Prompt and input contract

Use the official wrapper formatting and freeze the task instruction:

```text
Given a search query, retrieve relevant candidates that answer the query.
```

The logical order is:

```text
system yes/no judgment instruction
task instruction
original query
Document prefix
rendered candidate image
assistant-generation prefix
```

R0/R1 receive only the original question and exact rendered candidate pixels.
They receive no answer, gold box, label, source, conflict flag, relation class,
candidate rank, or label-derived structure. Stock and tuned runs must use the
same prompt, processor, pixels, visual-token settings, score definition, and
candidate ordering.

## Official scalar score

The official wrapper retains `lm.model`, constructs a fixed bias-free head from
the output embeddings for `yes` and `no`, scores the final sequence position,
and applies sigmoid:

```text
h_final = lm.model(**inputs).last_hidden_state[:, -1, :]
w_score = lm_head[yes] - lm_head[no]
score = sigmoid(w_score dot h_final)
```

The implementation preserves the raw logit and probability. Stage 02 must
reproduce the official wrapper on identical prompt and pixels within
`atol=5e-3, rtol=5e-3` in BF16 on the same device. Parity failure stops all
Qwen-dependent arms.

## R0 stock benchmark

For every eligible question, R0 scores every frozen candidate on its supplied
page. Persist one create-once prediction row per
`(canonical_question_id, candidate_id)` with:

- program, stage, run, model, environment, prompt, view, and candidate hashes;
- question, document, page, candidate, and split IDs;
- raw logit, probability, deterministic rank, and cost fields; and
- no answer text or gold geometry in the model-input payload.

Resume uses the exact pair key, not line count. Process shards merge only when
all locks and input hashes match. Reject duplicate rows, mixed revisions,
missing eligible queries, non-finite scores, and primary-run holdout rows.

R0 is evaluated on validation before negative mining. Because R0/R1 now belong
to the larger unified program, internal-test predictions are not created or
opened until the Stage 05 confirmatory registration freezes every arm.

## Hard-negative mining

Mining occurs only on `train` and uses the frozen R0 scores and common rules in
`candidate_and_supervision.md`. Per question retain every positive, up to four
audited `hard_non_anchor` candidates, and up to four deterministic ordinary
non-anchor candidates.

Strong-negative guards remove:

- any positive or partial candidate;
- any candidate overlapping an accepted alternative;
- normalized accepted-answer-string matches;
- plausible unresolved context; and
- any row not certified by the audit view.

The artifact must say `hard_non_anchor`, never merely `irrelevant`. The
deterministic at-least-200-row stratified human/independent audit is completed
and its false-negative rate reported before R1 optimization.

## R1 training objective

Each training group contains at least one accepted positive and up to four hard
plus four ordinary verified non-anchors. For valid positive-negative pairs:

```text
L_pair = mean_query(
  mean_pairs_within_query(softplus(-(score_positive - score_negative)))
)
```

With multiple accepted positives, compare each retained positive against the
same verified-negative pool, average within the query, then average across
queries. Do not let queries with more candidates dominate the batch.

R1 must:

- initialize exactly from the R0 model lock;
- preserve the score head, processor, prompt, and candidate revision;
- train on `train` groups only;
- record seeds, optimizer, schedule, dtype, hardware, steps, checkpoints, and
  resume state;
- reject changed locks rather than silently resume;
- select checkpoints only on complete validation MRR, breaking ties by
  Recall@1 then earlier optimizer step; and
- never inspect internal-test or holdout metrics during optimization.

The implementation plan supplies exact file and test mechanics. Hyperparameters
not fixed by architecture are registered before Stage 01 starts and may change
only from validation under a new config hash.

## Matched R0/R1 comparison

Evaluate R0 and R1 on byte-identical validation manifests in Stage 01 and on
the frozen internal-test manifest only in Stage 05 alongside the other arms.
Require identical:

- V1 and eligible-question hashes;
- candidate revision and rendered-candidate hashes;
- question strings and candidate order;
- prompt, processor, score extraction, and evaluation code; and
- evaluation hardware class where feasible.

The primary pairwise effect is computed on validation first and registered
again on internal test in Stage 05:

```text
delta_pairwise = Recall@1(R1) - Recall@1(R0)
```

Also report Recall@3/5, MRR, nDCG@5, pairwise positive-over-negative accuracy,
latency, memory, wins/losses/unchanged, slices, and a document-clustered paired
bootstrap confidence interval. A positive result establishes task adaptation
for answer-anchor ranking only.

## Jina boundary

The previously allowed stock alternative was `jinaai/jina-reranker-m0`.
Jina Embeddings v4 is an embedding-retrieval architecture, not a silent
replacement for a multimodal reranker. Under the unified program:

- Qwen R0/R1 is mandatory;
- Jina Reranker M0 may be added as optional arm `J0` under the identical frozen
  candidate and validation manifest;
- a tuned Jina arm would require its own named architecture and matched
  before/after protocol; and
- Jina Embeddings v4 would be a separately registered retrieval arm with
  embedding-specific indexing and metrics.

Neither optional Jina result removes the Qwen control required to interpret
M0/M1. Optional arms cannot delay or change the registered primary contrasts.

## Why pairwise remains necessary

R0/R1 are cheaper, simpler, auditable, and naturally rank candidates. They are
the deployment default unless the shared-page MiniVGent treatment produces a
decisive matched gain. Interpret active POC comparisons as:

- `R1 > R0`: audited hard-negative adaptation helps independent scoring;
- `M0 > R1`: one question-conditioned full-page memory adds value beyond
  repeated independent crops;
- `M1 > M0`: off-diagonal candidate interaction adds value beyond shared
  memory; and
- all trained joint arms `<= R1`: retain the tuned pairwise system.

## Prohibited changes

R0/R1 do not receive full-page hidden-state taps, candidate interactions,
answer generation, count heads, box regression, answer-feedback rewards,
validation/test mining, or mutable candidate revisions. A change to their
input representation or scoring path creates a newly named arm rather than a
quiet baseline modification.
