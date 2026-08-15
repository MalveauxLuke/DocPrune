# Matched Qwen Autoregressive-ID Control

## Status

A1 is preserved for the later HierDoc/answer-sufficiency experiment. It is not
an active POC arm or SOL task. The initial POC compares R0/R1 with M0/M1.

## Role

**A1** is a small generative Qwen-family candidate-ID selector. It separates
the useful action design in HierDoc from the effects of an 8B backbone,
paper-specific prompt, generative decoding, and GRPO.

MiniVGent versus HierDoc alone is a system comparison, not an
architecture-isolated comparison. A1 supplies the closer diagnostic:

```text
same V1 page + same aliases + same candidate list + similar backbone scale
  -> autoregressive candidate-ID set
```

## Model-choice gate

Before later A1 implementation or data execution, later stage L1 must freeze:

- one generative Qwen-family checkpoint at approximately MiniVGent's backbone
  scale;
- immutable Hub/code revision and license;
- exact prompt, processor, image budget, decoding, and environment;
- trainable parameter policy;
- whether training is SFT, structured-set policy optimization, or an explicitly
  staged SFT-plus-policy procedure; and
- a compute/update matching rule.

The current design does not assume that a compatible checkpoint exists. If no
acceptable model and objective are frozen, A1 is omitted with an explicit gate
failure; H0/H1 versus M0/M1 remains a system-level comparison.

## Input and action

A1 uses the exact H0/H1 derived view:

- original question;
- identical alias-overlaid supplied page pixels;
- identical `alias | type | OCR/table hint` rows;
- identical alias mapping and action grammar;
- identical candidate omissions/masks; and
- identical strict parser, invalid-ID accounting, and reflection rule if
  reflection is enabled for A1.

It emits exactly one `<evidence_region>...</evidence_region>` candidate-ID set.
No coordinate generation, answer generation, gold metadata, or free-form
evidence prose is allowed.

## Training and matching

Train A1 on the same eligible/reward-complete training rows used by the
structured selector comparison. Match H1 or M1 optimizer updates where the
different objective allows, and report rather than hide differences in:

- rollouts versus supervised examples;
- input/generated tokens;
- trainable/loaded parameters;
- image resolution;
- optimizer state and update count;
- peak memory and latency; and
- reflection or decoding passes.

If A1 uses H1's reward, it inherits the reward-completeness gate and exact
alternative-aware structured-set reward. If it uses SFT, serialize one
deterministic accepted target action without treating alternative occurrences
as a mandatory union. Any SFT warm-up creates a separately named training
phase in the model lock and report.

## Evaluation

A1 uses common selected-set metrics, action-format diagnostics,
alias-renaming sensitivity, cost reporting, and matched-average-selection
comparisons. It does not naturally emit comparable candidate logits unless the
chosen checkpoint/interface explicitly exposes a registered score; ranking
metrics are therefore not silently fabricated.

## Interpretation

- `A1 ~= H1`: most of HierDoc's benefit transfers to the smaller
  autoregressive-ID design.
- `H1 > A1`: the gap may reflect model scale, paper initialization, or
  optimization—not merely action format.
- `M1 > A1`: parallel tensorized candidate selection helps beyond a matched
  autoregressive action.
- `A1 > M1`: retain A1 unless MiniVGent has decisive cost, reliability, or
  robustness advantages.

A1 is not called “HierDoc-small.” It is a project control sharing HierDoc's
action representation.
