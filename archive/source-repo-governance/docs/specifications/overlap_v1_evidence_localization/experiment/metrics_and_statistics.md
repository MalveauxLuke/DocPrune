# Active POC Metrics and Statistics

## Evaluation unit

Metrics are query-macro. Compute per question, then average questions equally.
For uncertainty, resample `canonical_document_id` and keep all of its questions
together. Every report begins with candidate-oracle coverage and exclusions.

## Alternative-anchor semantics

Accepted anchor occurrences are OR-equivalent. For ranking, a hit occurs when
any accepted positive is retrieved. For optional selected-set analysis, score
the best accepted alternative rather than the union. Eligible gold sets are
nonempty; predicted empty sets receive zero recall/F1 and an explicit flag.

## Candidate-oracle metrics

- eligibility and exclusion rates;
- answer-anchor oracle hit/recall;
- strict grouped-box and alternative coverage;
- coverage before/after any cap;
- candidate/member count percentiles;
- partial-anchor rate; and
- source, OCR, type, box-size, and topology slices.

## Primary ranking metrics

For R0/R1/M0/M1:

- Recall@1, Recall@3, Recall@5;
- MRR;
- nDCG@5 with accepted alternatives;
- pairwise positive-over-verified-negative accuracy; and
- raw score/rank distributions.

Ranking needs no threshold. Any binary selected-set view is fit on validation
and frozen before internal test.

## Optional anchor-set analysis

- best-alternative anchor precision, recall, and F1;
- anchor hit rate;
- selected-candidate count mean/median/p90/p95/max;
- metrics at matched average selected count;
- selected OCR tokens and visual area; and
- empty-set rate.

These are anchor-set metrics, not answer-sufficiency metrics. Complete evidence,
proper-subset necessity, answer correctness from selected-only evidence,
multi-hop, and abstention are reserved for the later experiment.

## MiniVGent diagnostics

- candidate-order permutation equivariance error;
- M0 off-diagonal isolation;
- official score and selected-tap parity deltas;
- image-token/grid validity;
- ROI overlay outcomes;
- backbone gradient/optimizer violations;
- finite-gradient and exact-parameter checks;
- checkpoint round trip;
- per-component loss; and
- answer-string-present versus absent performance.

## Required slices

- source family and split;
- candidate type/count strata;
- member-count/disconnected-member strata;
- OCR quality;
- answer-box size/topology;
- single versus grouped boxes;
- alternative occurrences;
- answer string present versus absent in candidate OCR;
- MiniVGent depth/config/seed; and
- validation-fitted selection threshold where used.

Source/slice labels never enter the model.

## Cost and systems metrics

- total/trainable parameters;
- model/optimizer and peak allocated/reserved memory;
- wall time, latency, and throughput;
- frozen-Qwen versus decoder time;
- VLM passes per question;
- visual/language input tokens;
- selected tokens/area;
- checkpoint/cache/storage size; and
- OOM, skipped, and failed rows.

Report medians and tails, not only means. R0/R1 use repeated candidate crops;
M0/M1 use one full-page forward, so cost and representation differences are
reported explicitly.

## Paired uncertainty

Use seed `1729` and 10,000 document-cluster bootstrap replicates. Resample
documents with replacement, keep their questions, and report 2.5th/97.5th
percentiles for treatment-minus-control effects. Primary:

```text
M1 Recall@1 - M0 Recall@1
```

Also report R1-R0 and M0-R1 intervals.

## Three-seed aggregation

Stage 04 reports every M0/M1 seed and aggregate mean/dispersion. Primary
promotion requires positive M1-M0 direction in all three seeds, not only a
positive pooled mean.

## Matched-input validity

Reject comparisons with mismatched V1, eligibility, candidate, relation,
held-out, question, target, or evaluator hashes. Candidate order must match
except an explicit permutation test. R0/R1 candidate crops versus M0/M1 full
pages are declared architecture differences, not accidental input drift.

## Promotion and noninferiority

Primary MiniVGent promotion is at least +2.0 absolute Recall@1 points for M1
over M0, positive in all three seeds, with a document-clustered 95% interval
excluding zero and a gain on answer-string-absent rows.

The binding noninferiority margin is 2.0 absolute Recall@1 points: M1 may not
trail R1 by more than this margin. Both values are project decisions, not paper
results.

## Reporting language

Use “answer-anchor localization,” “hard non-anchor,” “shared-page benefit” for
M0-R1, and “off-diagonal candidate-interaction benefit” for M1-M0. A V1 M1-M0
gain supports single-hop competition/redundancy, not complete evidence or
multi-evidence complementarity.
