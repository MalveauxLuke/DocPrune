# Corrective region selection

Authority: [the owner's ExperimentPlan.md](../../ExperimentPlan.md). This workspace
organizes implementation preparation; it does not replace or silently amend the plan.

## Working contracts

- Freeze the complete instance: question, page identities/order/rendering, regions,
  reader, intervention and decoding. Cached replay uses the recorded page sequence.
- Encode all admitted pages, then physically delete visual positions before the
  answer decoder. Selected positions carry original reader features, not selector states.
- Partition eligible visual positions completely and uniquely into atomic regions;
  semantic regions need bounded local fallback coverage and disjoint token costs.
- Preserve G (accepted-reference support), S (fixed all-kept-answer support), and
  C = G − S. Gold-aware preferences guard gold support before optimizing contrast;
  likelihood is not proof of answer correctness.
- Start with diversified mask banks of roughly 16–32 proposals as a design target,
  not a frozen quota. ContextCite fitting and pair discovery are optional diagnostics.
- Stage 2 compares Compact, 2B-Pooled, and 2B-Rich on a common bank. Both 2B arms use
  question-first, unpooled shared-input states. Rich performs regional read, compare,
  re-read and compare; pooled pools once. Rich representation does not consume mask z.
- Start with a shared additive policy head and budgeted allocation. The optional
  set-conditioned head requires its own later test and does not define pure interactions.
- Measure generated-answer correctness, rescue and harm, plus total system cost;
  oracle mask headroom and label fit do not establish deployed selector performance.

## Stage map

| Stage | Purpose | Current status |
| --- | --- | --- |
| 0 | Existing-evidence alignment, retrieval/action audit, feature compatibility | 17-case input audit passed; SOL feature package active; remaining Stage 0 pending |
| 1 | Fresh data, reader and teacher feasibility; proposed 150–300 examples | Not implemented |
| 2 | Central rich/pooled/compact comparison; proposed 500–1,000 training examples | Not implemented |
| 3A–3C | Acquisition/features, representation/optional head, ranking/regression | Conditional later comparisons |
| 4 | Diversity versus mask count; bounded adaptive acquisition | Conditional scale-up |
| 5 | Locked confirmation and transfer | Future held-out evaluation |

See [Stage 0 inputs](STAGE0_INPUTS.md), [readiness](READINESS.md),
[resolved retention inventory](../../../legacy/RETENTION.md), and [configuration area](../../../configs/corrective-selection/README.md).

Active bounded task: [SOL Colfeatures17](COLFEATURES17.md).
