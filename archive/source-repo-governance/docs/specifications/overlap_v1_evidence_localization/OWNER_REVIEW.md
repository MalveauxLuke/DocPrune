# Owner Review: Overlap-First V1 MiniVGent POC

**Reviewed:** 2026-08-14  
**Specification:** `docs/specifications/overlap_v1_evidence_localization/`  
**Reviewed revision:** `74e9f2e`  
**Decision:** Approved as the basis for the binding POC implementation plan. This review does not activate SOL or authorize model execution.

## My assessment

The specification is ready to move into implementation planning. It asks a
focused and useful first question: on one supplied document page and one frozen
semantic-candidate universe, does MiniVGent improve answer-anchor localization
over a matched Qwen pairwise reranker?

The four-arm design is appropriate. R0 measures stock pairwise Qwen, R1
measures audited hard-negative adaptation, M0 isolates shared full-page Qwen
memory without candidate interaction, and M1 adds bidirectional candidate
interaction. The cleanest architectural result is M1 versus M0; M0 versus R1
is an operational system comparison because their visual representations
differ. Qwen R0/R1 must remain the mandatory baseline. Jina may be registered
as an optional side experiment, but it must not replace the matched Qwen
comparison.

Starting with single-hop, answer-anchor supervision is the right POC. The V1
labels can test candidate coverage and answer-bearing localization, but they do
not prove that a selected set is complete, minimal, necessary, or sufficient
for answering. A successful M1 result would therefore establish improved
single-hop candidate competition or redundancy handling—not multi-hop
reasoning or complete document understanding.

## Decisions I approve

- Preserve V1 as immutable and derive every candidate, exclusion, label, and
  prediction view with recorded hashes.
- Measure candidate-oracle coverage before judging any model.
- Compare R0/R1/M0/M1 on the same frozen candidates, questions, splits, and
  accepted answer-anchor alternatives.
- Treat uncertain, partial, or plausible context as unknown rather than
  silently converting it into a negative; audit at least 200 mined hard
  candidates before training depends on them.
- Keep Qwen frozen for M0/M1 and require the complete hidden-state, visual-grid,
  ROI, gradient, checkpoint, parameter-count, and real-overfit preflight.
- Use the bounded one-seed screen only for validation decisions, then freeze
  the three-seed confirmatory registration before opening internal test once.
- Keep InfographicsVQA sealed until the optional Stage 05 robustness run and
  never use it to select the model.
- Accept a negative result: if MiniVGent does not beat its controls, retain the
  simpler tuned pairwise reranker.

## Promotion standard

MiniVGent is promoted only if M1 exceeds M0 Recall@1 by at least 2.0 absolute
points, improves in all three registered seeds, has a document-clustered 95%
confidence interval excluding zero, retains a gain when the accepted answer
string is absent from candidate OCR, remains within 2.0 Recall@1 points of R1,
and passes every integrity and systems check. These thresholds are project
decisions, not claims from the VGent paper.

## What remains to be resolved in the implementation plan

1. Record the execution checkout, branch, scratch root, recovery authority,
   exact Qwen/model/package locks, resource request, and commands for each
   stage.
2. Translate the candidate contract into deterministic builders and report the
   actual candidate-count distribution, oracle coverage, exclusions, and any
   post-cap oracle loss before scoring models.
3. Demonstrate official Qwen score parity and selective hidden-state tap parity
   on the real pinned checkpoint before implementing data training.
4. Freeze batching, accumulation, runtime budgets, checkpoint rules, and
   create-once artifact schemas from measured Stage 02 profiling rather than
   estimates.

HierDoc H0/H1 and the matched A1 autoregressive control are deliberately
deferred. They belong in the later answer-sufficiency experiment after the
project has independently audited complete or sufficient evidence sets, or a
validated frozen-answerer sufficiency procedure. Completing this POC does not
automatically activate that work.

## Review conclusion

I approve the written POC specification for implementation-plan preparation.
The next deliverable is one binding, staged implementation plan that faithfully
implements the approved architecture and experiment files. After that plan is
reviewed, only Stage 00 may receive a separate SOL handoff; later stages remain
inactive until their predecessor reports pass review.
