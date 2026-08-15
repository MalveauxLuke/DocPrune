# Document-Specific Multimodal Reranker Baseline

## Proposal metadata

- Status: `adopted_for_overlap_v1_baseline`
- Last updated: 2026-08-04
- Possible consumer: future Evidence-DINO-Units model experiments
- Capability tags: question-conditioned unit ranking; document-image
  relevance; fixed-candidate evidence selection; calibration
- Active-task effect: the owner approved the narrow stock-versus-hard-negative
  V1 baseline in
  `../../../../docs/specifications/overlap_v1_segment_reranker_baseline.md`.
  That canonical specification, not this proposal, defines the authority.

## Candidate models and verified boundary

Primary model records:

- [Qwen3-VL-Reranker-2B](https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B)
  is a 2B multimodal reranker that scores query-document pairs across text,
  images, screenshots, video, and mixed inputs. Its model family and reported
  evaluation include visually rich document retrieval, but it is not trained
  specifically for complete supporting-evidence selection over page units.
- [jina-reranker-m0](https://huggingface.co/jinaai/jina-reranker-m0) is a
  2.4B multilingual multimodal document reranker based on Qwen2-VL-2B with a
  ranking head. Its model card describes training with pairwise and listwise
  ranking losses over visually rich documents. Its license is CC BY-NC 4.0.

The official Qwen3-VL reranker line currently provides 2B and 8B models, not a
4B checkpoint. “Move to approximately 4B” should therefore mean testing a
different model family or a custom checkpoint. The direct official Qwen size
escalation is 8B.

## Project baseline

**Evidence level: `unvalidated_proposal`.**

Use a pretrained multimodal reranker to assign one relevance score to each
question-unit pair without generating an answer or coordinates.

```text
question + one DeepSeek semantic unit
  -> pretrained multimodal reranker
  -> relevance score

scores for all page units
  -> calibrated threshold or top-k
  -> deterministic structural closure
  -> downstream answerer
```

The input can include the unit crop plus inference-available OCR, unit type,
geometry, reading-order position, parent heading, and linked structural
context. The first frozen-model screen should keep this formatting fixed and
avoid label-derived metadata.

This is the strongest simple semantic baseline because it uses document-aware
multimodal pretraining while keeping geometry and candidate construction
deterministic.

## Zero-shot and post-training sequence

Start with the 2B frozen or stock checkpoint to measure how much relevant
document semantics already transfer. If calibration or evidence recall is
inadequate, post-train the same 2B model before increasing model size.

A defensible post-training mixture should include:

- direct evidence positives;
- linked contextual positives where the annotation supports them;
- same-page hard negatives;
- semantically similar wrong units and wrong pages;
- no-evidence page-question pairs; and
- ignored ambiguous ancestors rather than forced negatives.

Only escalate size after the 2B learning curve and error analysis indicate a
semantic-capacity ceiling rather than poor supervision, candidate coverage,
input formatting, or calibration. A larger model is not a substitute for a
fair 2B baseline.

## Strengths

- No coordinate-generation failure mode.
- Uses stable, inspectable document units.
- Minimal architecture work compared with mini-VGent.
- Direct scalar scores are easy to calibrate and audit.
- The same scorer can be applied to parser units, fallback regions, and other
  proposal sources.
- Post-training can target the exact question-to-unit relevance formulation.

## Limitations

Independent unit scoring estimates something like:

```text
relevance(unit | question)
```

It does not directly estimate the conditional value of a unit after other
units have been selected. Consequently it cannot naturally represent:

- two weak units that are useful only together;
- a caption becoming necessary when a figure is selected;
- a second row completing a comparison;
- redundancy between a parent table and its child cells; or
- competition among several plausible evidence sets.

Deterministic structural closure can recover known relationships, and a
listwise loss can improve ranking, but neither is the same as joint
set-conditioned inference across all candidates. Repeatedly scoring many unit
crops may also duplicate image encoding and lose full-page context.

## Matched comparison

The reranker, Evidence-DINO-Units scorer, and mini-VGent treatment must use:

- identical candidate units and candidate caps;
- identical questions and document-disjoint splits;
- identical positive, contextual, ignored, and negative mappings;
- identical structural closure and fallback policies;
- identical threshold-fitting data; and
- the same downstream answerer and prompt.

Report candidate-oracle recall before interpreting model differences. Primary
metrics should include strict complete-evidence recall at matched retained
tokens, no-evidence false positives and false negatives, evidence-only answer
accuracy, latency, memory, and performance by direct versus implicit and
single-unit versus multi-unit questions.

The existing [VGent registry record](../papers/vgent-2512.11099.md) defines the
corresponding mini-VGent treatment. Comparing the independent reranker with
that treatment isolates whether joint set reasoning and shared semantic memory
add value beyond a strong pretrained relevance score.

## Decision boundary

This is a required strong baseline, not an assumed final architecture. Retain
the 2B reranker if it matches more complex treatments on complete evidence and
downstream answering. Post-train it if errors look like task mismatch. Move to
a larger model only if controlled evidence shows that model capacity, rather
than candidate or supervision quality, is the limiting factor.
