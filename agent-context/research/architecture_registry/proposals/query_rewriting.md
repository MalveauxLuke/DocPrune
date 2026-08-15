# Query Rewriting

## Proposal metadata

- Status: `unvalidated_proposal`
- Last updated: 2026-08-04
- Possible consumers: coarse page retrieval; semantic-unit reranking; future
  evidence-selection architectures
- Capability tags: training-query augmentation; query reformulation; hard
  negative generation; semantic robustness; implicit and multi-evidence
  questions
- Active-task effect: none. This proposal does not alter or authorize work
  beyond the current Evidence-DINO-Units Stage 0–4 contract.

## Primary methodological anchor

- Paper:
  [DocReRank: Single-Page Hard Negative Query Generation for Training Multi-Modal RAG Rerankers](https://arxiv.org/abs/2505.22584v1)
- Reviewed version: arXiv `2505.22584v1`
- Related artifacts:
  [Hugging Face paper page](https://huggingface.co/papers/2505.22584),
  [ColHNQue dataset](https://huggingface.co/datasets/DocReRank/ColHNQue-ColPaliHardNegativeQueries),
  [FinHNQue dataset](https://huggingface.co/datasets/DocReRank/FinHNQue-FinanceHardNegativeQueries),
  and
  [rephrased positive queries](https://huggingface.co/datasets/DocReRank/RephColHNQue-RephrasedColPaliHardNegativeQueries)

**Evidence level: `paper_reported`.**

DocReRank reverses conventional hard-negative mining. Instead of holding a
query fixed and retrieving similar but irrelevant pages, it holds one page
fixed and generates queries that resemble the positive query but are not
answerable from that page. Its full pipeline can also generate and verify a
positive query from a page; when an authoritative positive query already
exists, the hard-negative branch can begin directly from that query.

For general hard negatives, the paper uses Qwen2.5-7B-Instruct to generate 12
query variants that are similar in topic and form but seek different
information. Qwen2.5-VL-7B-Instruct then checks every candidate against the
page using two verification prompts. A candidate is retained only when both
checks classify it as unanswerable. The released ColHNQue construction keeps
the original positive page-query pair and three verified negative queries for
that page. A finance-specific branch generates more targeted negatives by
changing one property such as year, company, numerical value, financial
metric, or business segment.

The paper also creates positive rewrite variants: it rephrases 50% of the
positive queries while preserving their meaning, with the goal of discouraging
shallow matching to page wording. The reranker itself is Qwen2-VL-2B-Instruct
with LoRA on the language transformer and a frozen visual encoder. It is
trained to score page-query answerability through weighted cross-entropy over
`True` and `False` token logits. This is hard-negative reranker training, not
an InfoNCE objective.

At matched model and training settings, DocReRank-Base combines traditional
hard-negative pages with the generated hard-negative queries. Compared with
training only on document-mined negatives, the paper reports average NDCG@5
gains of `+2.8` on ViDoReV2 and `+4.7` on Real-MM-RAG after ColQwen retrieval.
This supports adding generated query negatives alongside conventional
negatives; it does not show that conventional negatives should be removed.

## Our working definition of query rewriting

**Evidence level: `unvalidated_proposal`.**

For this project, **query rewriting is primarily a training-data construction
process**. Starting from an authoritative query and its positive page or
evidence unit, we generate a controlled family of related queries so the
training distribution casts a wider net than the single original wording.

“Wider net” refers to coverage across the training set: more vocabulary,
syntax, framing, specificity, and evidence-seeking formulations of the same
underlying information need. It does not mean weakening the semantic meaning
of one query, treating a broad query as equivalent to a precise one, or
silently replacing the user's runtime question.

The umbrella contains two different outputs:

| Rewrite type | Meaning relative to the original | Relation to the fixed positive page/evidence | Training role |
|---|---|---|---|
| Intent-preserving positive rewrite | Same information need, expressed differently | Must remain answerable from the same authoritative evidence | Additional positive view that broadens phrasing coverage |
| Hard-negative rewrite | Similar topic and form, but one or more answer-defining constraints change | Must be unanswerable from the fixed evidence | Difficult negative that tests semantic precision rather than keyword overlap |

The dedicated
[negative contrastive query-rewriting proposal](negative_contrastive_query_rewriting.md)
develops the second branch. Keeping the labels separate is essential: a
positive rewrite widens coverage without changing truth conditions, whereas a
negative rewrite deliberately changes those truth conditions.

## Motivation

The user's original question is the authoritative expression of intent, but it
is only one sample from a much larger space of valid ways to request the same
information. Training on that wording alone can reward lexical mirroring and
leave a reranker brittle to paraphrase, indirect phrasing, and modality-aware
questions. Intent-preserving rewrites widen the positive training distribution;
hard-negative rewrites fill the nearby semantic boundary with examples that
look relevant but ask for something the evidence does not contain.

## Loose research direction

Potential positive rewrites could vary:

- terminology and synonymous expressions;
- syntax, voice, and question form;
- explicit versus implicit evidence-seeking formulations;
- decomposition into smaller information needs;
- alternate formulations for textual, tabular, or visual evidence;
- relation-focused wording for implicit questions; and
- specificity, provided that the same evidence remains sufficient.

Potential negative rewrites could modify entity, relation, polarity, quantity,
time, comparison direction, scope, modality, or answerability while preserving
as much surface similarity as possible.

The original query remains the anchor and authoritative source. Rewrites are
derived training examples whose labels depend on evidence-grounded
verification. Any rewrite that is ambiguous, changes too many dimensions, or
cannot be verified should be ignored rather than forced into a positive or
negative class.

## Possible project connections

- Coarse page retrievers could be trained or evaluated on multiple valid
  formulations of the same information need.
- A semantic-unit reranker could receive positive paraphrase augmentation and
  difficult query-side negatives for the same fixed unit.
- A 2B multimodal reranker could use the DocReRank page-query construction as
  a direct baseline before custom mini-VGent training.
- A VGent-inspired set decoder could use the same rewrite groups to test
  whether candidate interaction improves semantic discrimination.
- A later inference-time rewrite experiment remains possible, but it is
  separate from this primary training-data proposal.

## Main risks

- A rewrite can subtly change the requested entity, relation, scope, or
  answerability.
- VLM verification can still misclassify answerability and introduce false
  positive or false-negative labels.
- Synthetic rewrite style may become a shortcut that does not transfer to
  natural user variation.
- A rewriting model may merely expand vocabulary without improving evidence
  reasoning.
- Broad positive rewrites may no longer be supported by exactly the same
  minimal evidence set.
- Improvements can be difficult to attribute if rewriting and fusion change
  in the same experiment.

## Open questions

- Should the first study use only offline training augmentation, reserving
  inference-time rewriting for a later experiment?
- How do we verify that a positive rewrite preserves both intent and evidence
  sufficiency?
- Which semantic properties yield the hardest defensible negatives for our
  document domains?
- Should verification operate on the full page, the authoritative semantic
  units, or both?
- How many rewrite variants add meaningful distributional coverage before
  synthetic redundancy dominates?
- What controls distinguish a real semantic benefit from simple query
  expansion?

## Non-decisions

This proposal does not adopt DocReRank's exact models, 12-candidate generation
count, three-negative retention count, prompts, binary loss, dataset mixture,
or page-level granularity. It also does not choose our rewriting model,
verification stack, review budget, dataset, or success threshold. Those choices
remain open pending research.

## Adoption status and decision history

- 2026-08-04 — `unvalidated_proposal`: loose project skeleton recorded; no
  experiment or implementation approved.
- 2026-08-04 — `unvalidated_proposal`: DocReRank `2505.22584v1` added as the
  primary methodological anchor; query rewriting clarified as a training-data
  umbrella containing separately labeled positive and hard-negative rewrites.
