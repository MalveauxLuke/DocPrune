# Overlap-First V1 Segment-Reranker Baseline

Status: owner-approved experiment design; implementation and model execution not started  
Decision date: 2026-08-11  
Immutable source: `/home/lmalveau/overlap_first_document_corpus/v1`  
Candidate source: a frozen DeepSeek-OCR-2 semantic-segmentation revision  
Task: rank one page segment against one original question

## 1. Goal

Measure whether task-specific hard-negative fine-tuning improves a stock
multimodal reranker at retrieving the segment that contains the annotated
answer location.

The experiment has one fixed sequence:

```text
immutable V1
  -> frozen semantic segments
  -> answer-anchor coverage eligibility
  -> stock pretrained reranker benchmark
  -> training-only hard-negative mining
  -> fine-tune the same reranker
  -> matched post-training benchmark
```

“Base untrained” means the released pretrained checkpoint with no project
fine-tuning. It does not mean a randomly initialized model.

## 2. Scope and boundaries

This experiment uses only the completed overlap-first V1. It does not alter,
rewrite, or delete rows from V1. All filtering produces a versioned derived
eligibility manifest with an exclusion reason for every rejected question.

The task is deliberately narrow:

```text
score(query, segment) -> scalar relevance score
```

The first stock benchmark receives only the original query and one rendered
segment. It receives no gold answer, gold box, source label, conflict flag,
answer string, candidate rank, or label-derived structural metadata.

This phase does not authorize MiniVGent training, DeepSeek-OCR2 fine-tuning,
synthetic multi-hop generation, answer-feedback RL, split redesign, conflict
adjudication, or use of the InfographicsVQA holdout for model selection.

## 3. Source contract

The source package is immutable and has these verified counts:

| Split | Usable questions | Page positions |
|---|---:|---:|
| train | 40,963 | 15,584 |
| validation | 5,023 | 1,911 |
| internal test | 4,748 | 1,743 |
| InfographicsVQA holdout | 15,053 | 3,805 |

Consumers must require `usable_in_v1=true`. Document-grouped V1 splits are
preserved exactly. Training uses only `train`; threshold selection, model
choice, and early stopping use only `validation`; `test` is evaluated only by
the frozen before/after comparison. `infographicsvqa_holdout` remains sealed
until the primary experiment and its decisions are frozen.

## 4. Candidate and eligibility contract

### 4.1 Frozen candidate revision

Materialize one deterministic segment universe from the packaged
DeepSeek-OCR-2 output. Begin from the repository's
`deepseek_semantic_segmentation.md` rules and assign a `candidate_revision` to
the exact parser, normalization rules, rendering parameters, and source
hashes. Do not regenerate candidates after seeing reranker scores.

Each candidate must retain:

- stable `candidate_id`, `canonical_page_id`, and `candidate_revision`;
- the union of its member boxes rather than only an enclosing rectangle;
- its rendered crop or masked-page view and SHA-256;
- raw OCR text, type, reading order, and member provenance for auditing; and
- no query-relative label in the reusable candidate table.

### 4.2 Correct-location gate

The owner's “segment in the correct location” rule is interpreted as candidate
coverage of a valid answer-location annotation, not downstream answer-model
success.

For a gold answer-location group `G` and candidate member-box union `C`:

```text
gold_coverage(G, C) = area(G intersect C) / area(G)
```

A candidate is a positive answer-anchor segment when
`gold_coverage >= 0.70` for one accepted alternative answer-location group.
Alternative occurrences are OR alternatives. Multiple boxes that jointly
form one answer are evaluated as a group rather than as independent evidence
requirements.

A question is eligible only when:

- it is usable in the relevant V1 split;
- it has at least one accepted answer-location group on a packaged page;
- it has no unresolved `answer_disagreement`, `page_disagreement`, or
  `box_disagreement` affecting the target; and
- at least one frozen candidate passes the coverage threshold.

If no candidate passes, retain the question in the derived audit with
`exclusion_reason=no_anchor_covering_segment`. Never remove it from V1.
Questions with partial coverage only are retained as
`exclusion_reason=partial_anchor_coverage`. Candidate-generation failures and
invalid OCR pages receive separate reasons.

Report candidate-oracle recall before any reranker metric, overall and by
dataset family, split, OCR quality flag, and answer-box topology.

### 4.3 Label meaning

The current V1 boxes are answer-bearing anchors, not audited complete evidence.
Therefore labels mean:

```text
positive: contains an accepted answer location
negative: does not contain an accepted answer location
```

They must not be described as complete-evidence versus irrelevant labels.
Plausible contextual segments may be excluded from loss or retained as an
explicit `unverified_context` class for analysis; they are not automatically
strong negatives.

## 5. Model-choice gate

The two allowed initial candidates are:

- `Qwen/Qwen3-VL-Reranker-2B`;
- `jinaai/jina-reranker-m0`.

Jina Embeddings v4 is not substituted silently for `jina-reranker-m0`; an
embedding-retrieval arm would be a different experiment. Exact model and code
revisions, license, preprocessing, prompt/template, dtype, image resolution,
and score extraction must be pinned before execution.

If resources permit, run both stock checkpoints on the identical validation
manifest and choose one before opening the internal test results. Otherwise,
the owner selects one checkpoint before the baseline run. Fine-tuning must use
the same selected model, input format, candidate revision, and score
definition as its stock baseline.

## 6. Stock baseline

For each eligible query, score every frozen candidate on its mapped page using
only `(segment, original_query)`. Preserve every raw score and deterministic
rank.

Primary metrics are query-macro:

- answer-anchor Recall@1, Recall@3, and Recall@5;
- mean reciprocal rank;
- nDCG@5 with all accepted alternative positives;
- pairwise positive-over-negative accuracy;
- candidate-oracle recall;
- candidate count, rendered visual tokens, latency, and peak memory; and
- results by source, OCR quality, candidate count, answer-box size, and
  single-box versus grouped-box topology.

Fit no threshold on the internal test. A top-k benchmark requires no score
calibration; any binary threshold is fitted on validation and then frozen.

## 7. Hard-negative construction

Mine negatives only from the training split, using the frozen stock checkpoint
and frozen candidate universe. For each query:

1. retain every accepted positive candidate;
2. remove every candidate overlapping any accepted answer alternative;
3. remove candidates containing a normalized accepted answer string from the
   strong-negative pool;
4. rank the remaining same-page candidates with the stock reranker;
5. retain up to four highest-scoring candidates as `hard_non_anchor`;
6. retain up to four deterministic ordinary same-page candidates as
   `ordinary_non_anchor`; and
7. record score, rank, geometry, OCR text, and every rejection reason.

Before fine-tuning, audit a deterministic stratified sample of at least 200
mined hard negatives across source families and mark likely false negatives.
The fine-tuning manifest excludes audited false negatives and records the
measured false-negative rate. The phrase `hard_non_anchor` is mandatory in
artifacts and reports because V1 does not prove complete evidence
irrelevance.

## 8. Fine-tuning contract

Fine-tune the selected stock model on training-only query groups. Each group
contains at least one positive, up to four `hard_non_anchor` candidates, and
up to four `ordinary_non_anchor` candidates.

Use a scalar-score pairwise logistic objective:

```text
L_pair = mean(softplus(-(score_positive - score_negative)))
```

If multiple accepted positives exist, compare each retained positive with the
same negative pool and average within the query before averaging across
queries. Use validation MRR as the early-stopping criterion and report
Recall@1 as the primary outcome. All optimization parameters, random seeds,
checkpoint selection, and stopped step must be recorded in the run manifest.

No validation or test candidate may enter hard-negative mining or optimizer
updates.

## 9. Matched post-training benchmark

Evaluate the stock and fine-tuned checkpoints on byte-identical validation and
internal-test manifests with the same preprocessing, prompt, candidate order,
score extraction, and hardware class.

The primary effect is:

```text
delta Recall@1 = tuned Recall@1 - stock Recall@1
```

Also report paired deltas for Recall@3, Recall@5, MRR, nDCG@5, latency, and
memory. Compute a document-clustered bootstrap 95% confidence interval for
the primary delta so repeated questions from one document do not masquerade
as independent evidence.

Publish wins, losses, unchanged cases, and regressions by source and candidate
topology. A positive result means hard-negative task adaptation improved
answer-anchor segment ranking. It does not establish multi-hop reasoning,
complete-evidence selection, or MiniVGent set reasoning.

## 10. Required artifacts

Each run must preserve:

- V1 manifest hash and source file hashes;
- candidate revision and candidate-oracle report;
- eligibility and exclusion manifests;
- exact model revision and license record;
- stock predictions and metrics;
- hard-negative mining manifest and audit;
- fine-tuning config, environment, logs, and checkpoint identity;
- tuned predictions and metrics; and
- a paired before/after report.

Large crops, caches, weights, and predictions belong in the approved
experiment workspace or scratch root. Git receives only small configs,
manifests, tests, summaries, reports, and provenance.

## 11. Stop conditions

Do not start model training when:

- the experiment workspace, branch, scratch root, or SOL handoff is unresolved;
- candidate-oracle recall has not been measured;
- the candidate revision changes after baseline scoring;
- the model revision or input formatting differs between stock and tuned runs;
- hard-negative mining uses validation, test, or holdout records; or
- the audited hard-negative false-negative rate is not reported.

If candidate coverage is poor, repair and refreeze candidate generation, then
rerun the stock benchmark under a new candidate revision. Do not attribute a
candidate miss to reranker quality.
