# Negative Contrastive Query Rewriting

## Proposal metadata

- Status: `unvalidated_proposal`
- Last updated: 2026-08-04
- Possible consumers: direct semantic-unit rerankers; multimodal rerankers;
  future set-aware evidence selectors
- Capability tags: hard-negative generation; contrastive supervision; semantic
  sensitivity; robustness and preservation
- Active-task effect: none. This proposal does not alter or authorize work
  beyond the current Evidence-DINO-Units Stage 0–4 contract.

## Primary methodological anchor

This proposal is the hard-negative branch of the broader
[query-rewriting proposal](query_rewriting.md). Its closest published analogue
is
[DocReRank: Single-Page Hard Negative Query Generation for Training Multi-Modal RAG Rerankers](https://arxiv.org/abs/2505.22584v1).

**Evidence level: `paper_reported`.**

DocReRank fixes a positive page-query pair, generates 12 similar-looking query
variants that seek different information, and retains candidates only when two
VLM verification prompts both judge them unanswerable from the page. Its
training datasets keep three verified query negatives per positive page-query
pair. A targeted finance variant changes exactly one fine-grained property,
such as the year, company, value, metric, or business segment.

The paper trains a binary `True`/`False` reranker with weighted cross-entropy.
It does not use InfoNCE or demonstrate a segment-level contrastive objective.
Here, “contrastive” describes the training contrast between an authoritative
positive query and nearby meaning-altered negative queries. The exact loss
remains an open project choice.

## Motivation

An evidence reranker should respond to the meaning of a question rather than
only its vocabulary. Random negative questions are often too easy: they can be
rejected through topic or lexical mismatch without learning the relationships
that make evidence correct. A harder training signal may come from rewrites
that remain fluent and lexically close to the original question while changing
one meaning-bearing constraint.

Examples of possible changes include entity, relation, polarity, quantity,
time, comparison direction, scope, modality, or answerability. The resulting
rewrite should be plausible but semantically incorrect for the original
positive evidence. This is a candidate source of hard negatives, not a claim
that every generated rewrite is truly negative.

## Loose research direction

A future pipeline might:

1. start from an original question with authoritative positive evidence;
2. generate one or more minimally changed negative rewrites;
3. record which semantic constraint was changed;
4. verify that the original evidence does not answer the rewritten question;
5. pair the rewrite with the original evidence as a hard negative; and
6. train or evaluate a reranker on the contrast between the original and
   altered question.

The contrastive signal could eventually complement ordinary evidence labels,
but it should not replace them. The unmodified original question should remain
the anchor, and generated negatives should be filtered or ignored whenever
their validity is uncertain.

DocReRank suggests two useful starting constraints without settling our full
design: generate query-side negatives against fixed authoritative evidence,
and separate linguistic generation from evidence-grounded answerability
verification. We should initially retain ordinary mined document/unit
negatives as a control rather than assume generated query negatives replace
them.

## Possible project connections

- Direct semantic-unit reranking could learn sensitivity to small semantic
  changes rather than keyword overlap.
- A 2B VLM reranker could be tested for whether its yes/no relevance score
  separates original queries from meaning-altered rewrites.
- A VGent-inspired set decoder could use the negatives to test whether global
  candidate interaction improves semantic discrimination.
- Query-rewriting research could share generation and intent-preservation
  checks while keeping positive and negative objectives separate.

## Main risks

- A generated “negative” may still be answered by the same evidence, creating
  false-negative supervision.
- Surface artifacts may make generated negatives easier than natural user
  questions.
- Changing too much can collapse the task into topic discrimination; changing
  too little can preserve the original answer.
- Document annotations may establish answer-bearing evidence without proving
  that a nearby alternative question is unanswerable.
- Training only on synthetic perturbations may harm performance on ordinary
  queries or teach the generator's style rather than semantic precision.

## Open questions

- Which minimal semantic changes yield difficult but defensible negatives?
- What combination of automatic checks and human review is scientifically
  clean enough for a first dataset?
- Should negatives be generated without seeing the evidence, conditioned on
  the evidence, or both?
- How should uncertain, partially answerable, or contradictory rewrites be
  represented?
- Should the first experiment train with these negatives or use them only as a
  diagnostic challenge set?
- What preservation objective is needed to prevent regressions on questions
  the original reranker already handles correctly?

## Non-decisions

This skeleton does not commit to InfoNCE or another contrastive loss, a rewrite
taxonomy, a generator, verification model, review budget, negative ratio,
dataset size, or experiment gate. Further research should resolve those items
before this project becomes a specification.

## Adoption status and decision history

- 2026-08-04 — `unvalidated_proposal`: loose project skeleton recorded; no
  experiment or implementation approved.
- 2026-08-04 — `unvalidated_proposal`: grounded the direction in DocReRank
  `2505.22584v1` while leaving model, verification, loss, granularity, and data
  mixture decisions open.
