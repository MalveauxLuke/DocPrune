# Source-Coverage and Preservation Audit

## Purpose

This file prevents the unified program from losing information during
consolidation. The older files remain unchanged as provenance. The new design
rewrites and separates their scientific content rather than treating one old
document as the new monolith.

Two kinds of information are intentionally separated:

- architecture and experiment facts are present in this specification family;
- exact implementation file lists, test code, CLI invocations, and commit
  boundaries remain in the two existing plans until the owner reviews this
  written specification and a POC implementation plan supersedes them.

No old implementation plan is declared obsolete before that replacement plan
passes its own source-coverage audit.

## Frozen source inventory

Hashes were computed on 2026-08-14 before consolidation.

| Source | SHA-256 | Role |
|---|---|---|
| `docs/specifications/overlap_v1_segment_reranker_baseline.md` | `f434b85ede3cbce187fef3d42762fb9154a40f0f527b478b9dc8c4688b3576f9` | Approved pairwise baseline design |
| `docs/superpowers/plans/2026-08-11-overlap-v1-segment-reranker-baseline.md` | `4355fd37c259c0441d87404775075d497812dcf41dd26cfc76c8296797428202` | Pairwise implementation tasks and tests |
| `agent-context/research/architecture_registry/proposals/minivgent_answer_anchor_poc.md` | `01f8308bbee923a1b887366eefc8e673f909eeb296e3b2d9feba90a202ef5a3e` | MiniVGent, HierDoc, A1, arms, phases, metrics, and interpretation |
| `agent-context/research/architecture_registry/proposals/minivgent_qwen_implementation_readiness.md` | `c4192cc21553e5b7c9abc91645dc265544592740f8422197265f0d938d4e6a57` | Exact Qwen and MiniVGent tensor/runtime contract |
| `docs/superpowers/plans/2026-08-14-minivgent-qwen-implementation.md` | `3f79c1a134e2cd7485dcef40842db7ca893a908fdddd20c8ebc98493941d6680` | MiniVGent implementation tasks and tests |
| `agent-context/research/architecture_registry/papers/hierdoc-2607.29638.md` | `b8e164df2e451f9ce0c72c067f0800c43a48b4179887de19354e4652d80b3ddf` | Primary-source HierDoc review and adaptation analysis |
| `docs/specifications/deepseek_semantic_segmentation.md` | `64b67231844d5f216d55b93fa0d444963611e9d4512507bb7381fd3d37381bf0` | Candidate-generation contract |
| `docs/specifications/deepseek_ocr2_minivgent_evidence_localization/single_hop_poc_research_report.md` | `da0d02a22d7c4b2fabd6f2e5e0eaa8175f729b6cacdf8a279927ddfab2cfe638` | Corpus interpretation and staged research boundary |

## Baseline design coverage

| Existing section | New destination |
|---|---|
| Goal; scope and boundaries | `README.md`; `experiment/program.md` |
| Source contract and verified counts | `architectures/candidate_and_supervision.md` |
| Frozen candidate revision | `architectures/candidate_and_supervision.md`; `experiment/artifact_contract.md` |
| Correct-location gate and alternative anchors | `architectures/candidate_and_supervision.md` |
| Answer-anchor label meaning | `README.md`; `architectures/candidate_and_supervision.md` |
| Qwen/Jina model-choice gate | `architectures/qwen_pairwise_reranker.md`; Jina retained as optional, not a Qwen replacement |
| Stock baseline | `architectures/qwen_pairwise_reranker.md`; Stage 01 |
| Training-only hard-negative construction and 200-row audit | `architectures/qwen_pairwise_reranker.md`; Stage 01 |
| Pairwise fine-tuning objective | `architectures/qwen_pairwise_reranker.md`; Stage 01 |
| Byte-identical matched benchmark | Validation comparison in Stage 01; first internal-test comparison in Stage 04 after every POC decision is frozen |
| Required artifacts | `experiment/artifact_contract.md` |
| Stop conditions | `experiment/gates.md` |

## Pairwise implementation-plan coverage

| Existing task | Current design destination | Unified-plan requirement |
|---|---|---|
| Candidate/eligibility manifests | Candidate architecture; Stage 00 | Preserve exact records, union geometry, tests, smoke CLI, and commit boundary |
| Shared metrics and paired comparison | Metrics file | Preserve alternative-positive tests, document bootstrap, validation, and output schemas |
| Stock Qwen and Jina adapters | Pairwise architecture | Preserve immutable model locks, adapter protocol, resume keys, gold-field exclusion, and Qwen smoke |
| Hard-negative mining and audit | Pairwise architecture; Stage 01 | Preserve removal rules, deterministic ordinary negatives, and stratified audit schema |
| Fine-tuning | Pairwise architecture; Stage 01 | Preserve query-balanced loss, validation selection, resume, and leakage tests |
| Matched benchmark and report | Stages 01 and 04 | Preserve strict input parity, paired deltas, reporting, control-plane updates, and policy tests |

## MiniVGent/HierDoc proposal coverage

| Existing section | New destination |
|---|---|
| Proposal metadata and decision | `README.md`; `experiment/program.md` |
| VGent fact and transfer boundary | `architectures/minivgent.md` |
| HierDoc fact and V1 adaptation | Preserved in `architectures/hierdoc_region_policy.md` and deferred by `experiment/later_answer_sufficiency/README.md` |
| Dataset and label boundary | `architectures/candidate_and_supervision.md` |
| Candidate/action contract | Active candidate architecture; later HierDoc architecture |
| R0/R1 | Pairwise architecture; program arms |
| H0/H1 | Later HierDoc architecture and answer-sufficiency plan; not active POC arms |
| A1 | Later autoregressive-control architecture and answer-sufficiency plan |
| M0/M1 | MiniVGent architecture; active POC arms |
| Tensorized architecture boundary | MiniVGent architecture |
| MiniVGent OR/ranking/negative objective | MiniVGent architecture |
| HierDoc structured-set reward | Preserved for the later answer-sufficiency experiment |
| Phases 0-6 | R0/R1/M0/M1 work becomes active Stages 00-05; H0/H1/A1 work moves to later L0-L4 |
| Former Phase 7 future work | `experiment/program.md` non-goals and future boundary |
| Metrics | Active POC metrics plus later answer-sufficiency plan |
| Failure interpretation | Active POC gates/program plus later architecture interpretation |
| Future repository shape | `docs/superpowers/plans/2026-08-14-overlap-v1-minivgent-poc.md` |
| Adoption boundary | Replaced by `README.md` authority and staged activation model |

## Exact Qwen/MiniVGent readiness coverage

| Existing section | New destination |
|---|---|
| Hidden-state availability and selective hooks | `architectures/minivgent.md` |
| Paper/checkpoint/project distinction | MiniVGent architecture |
| Proposed source/environment pins | MiniVGent architecture; artifact contract |
| Official yes/no score parity | Pairwise and MiniVGent architectures; Stage 02 |
| Prompt and causal order | Pairwise and MiniVGent architectures |
| Exact PageMemory fields and tap semantics | MiniVGent architecture |
| Visual grid and member ROI rule | MiniVGent architecture |
| Exact candidate tensors, features, vocabulary, and encoder | MiniVGent architecture |
| Exact decoder, M0/M1 masks, and parameter counts | MiniVGent architecture |
| Target masks and numerically stable losses | MiniVGent architecture |
| Online versus cached execution | MiniVGent architecture; artifact contract |
| Thirteen mandatory pre-training tests | Stage 02 and `experiment/gates.md` |
| Implementation-agent gates | README; artifact contract; gates |

## MiniVGent implementation-plan coverage

The later unified plan must preserve each task as a reviewable implementation
unit, even if file names are consolidated:

1. frozen configuration and typed tensor contracts;
2. official Qwen prompt and score path;
3. selective Qwen layer-memory capture;
4. visual-grid reconstruction and member-box ROI pooling;
5. exact candidate encoder;
6. M0/M1 set decoders;
7. stable answer-anchor losses;
8. composition, backbone freezing, parameter counts, and added-weight-only
   checkpoints;
9. real-Qwen GPU preflight and overlay audit;
10. frozen V1 candidate-view integration;
11. online frozen-Qwen training;
12. matched M0/M1 evaluation and screen; and
13. full verification and authority handoff.

The replacement plan must retain the concrete interfaces, shapes, tests,
expected failures, commands, package pins, smoke schedules, resume rules,
checkpoint hashes, and commit boundaries from the source plan. Until then,
the hashed source plan remains required reading for implementation.

## HierDoc review coverage

| Existing section | New destination |
|---|---|
| Paper metadata, source, and factual capsule | `architectures/hierdoc_region_policy.md` |
| Page/region/answer mechanism | Later HierDoc architecture |
| 16-page windows and reflection threshold 8 | HierDoc architecture |
| Candidate aliases, matching, reward, penalties, and reported settings | HierDoc architecture |
| Reported data and ablation results | HierDoc architecture as paper context, not local evidence |
| Project interpretation and comparison table | MiniVGent and later HierDoc/A1 architecture files |
| Reusable mechanisms and integration points | Later HierDoc architecture and answer-sufficiency plan |
| Required adaptations, advantages, risks, and open questions | Later HierDoc architecture and answer-sufficiency plan |
| Smallest decisive experiment | Deferred L0-L3; active POC first establishes R0/R1/M0/M1 |
| Adoption history | Preserved in the original registry record; current authority lives in README |

## DeepSeek segmentation coverage

| Existing section | New destination |
|---|---|
| Purpose and academic-layout boundary | `architectures/candidate_and_supervision.md` |
| Python sources of truth | Candidate architecture |
| Pinned OCR-2 inference and provenance | Candidate architecture; artifact contract |
| Heading/preamble and paragraph fallback rules | Candidate architecture |
| Visual runs, caption search, geometry fallback, and bundles | Candidate architecture |
| Member-union geometry and query-relative labels | Candidate architecture |
| Quarantine behavior | Candidate architecture and gates, with V1-specific derived-view reconciliation |
| Training/evaluation context and MinerU boundary | Candidate architecture |
| Change control | Candidate architecture; artifact contract |

## Single-hop research-report coverage

| Existing section | New destination |
|---|---|
| Verified corpus counts and supervision limitation | Candidate architecture; README |
| VGent mechanism and frozen-backbone evidence | MiniVGent architecture |
| Candidate-oracle-first recommendation | Stage 00 and gates |
| 15k-20k single-hop pilot proposal | Stage 03 records it as historical guidance; binding screen cap is 10k |
| Content-rich candidate queries and positive-unlabeled supervision | MiniVGent architecture |
| Same-page audited multi-evidence pilot | Future boundary in program; not authorized here |
| Answerer search/distillation before RL | Future boundary in program; not authorized here |
| Genuine dependency-based multi-hop | Future boundary in program; not authorized here |
| Decisive-experiment interpretation | Program hypotheses and gates |

## Reconciled differences

The merge makes these choices explicit rather than silently choosing one old
statement:

- **Qwen versus Jina:** Qwen R0/R1 is mandatory for a matched MiniVGent
  comparison. Jina remains optional and separately named.
- **Candidate hierarchy:** the frozen DeepSeek contract provides reading order
  and types, not trusted parent-child hierarchy. No model receives invented
  hierarchy edges.
- **DeepSeek legacy uniqueness:** the older 4K contract quarantined any query
  with multiple positive candidates. V1 instead preserves accepted alternative
  occurrences with OR semantics and excludes unresolved conflicts; it does not
  force exactly one positive.
- **Decoder depth:** two blocks are the engineering smoke and one-seed screen;
  four blocks require validation promotion. Both exact parameter counts are
  retained.
- **Relative bias:** the first M0/M1 implementation has none. A typed relation
  bias is a later ablation only if those relation types exist in the frozen
  manifest.
- **Candidate-oracle thresholds:** `0.70` gold coverage is binding. Historical
  suggestions of at least 95% overall and 90% strict multi-box candidate
  coverage are planning guidance, not fabricated measured gates. Stage 00 must
  report actual coverage; the owner evaluates whether it supports the claim.
- **Pilot size:** Stage 03 uses a deterministic balanced sample of up to 10,000
  eligible training questions. Earlier 15,000-20,000 estimates remain
  historical proposals, not validated thresholds.
- **Broad candidate proposals:** the research report mentioned words/lines,
  table cells, headers, figures, and fallbacks as a possible universe. The
  binding first revision uses the actual DeepSeek semantic-section builder
  (`document_preamble`, `headed_text`, `deepseek_paragraph`, and
  `visual_bundle`) and does not invent hierarchy or candidate types.
- **Internal-test timing:** the former standalone R0/R1 baseline could open its
  internal test after pairwise training. Once merged into the POC, that would
  leak test information into MiniVGent decisions. Stage 01 is validation-only;
  Stage 04 opens internal test once for R0/R1/M0/M1.
- **HierDoc timing:** the source proposal placed H0/H1 and A1 in the first
  comparison. The owner narrowed the active POC to matched Qwen R0/R1 versus
  MiniVGent M0/M1. HierDoc/A1 content remains preserved but is deferred until
  complete or independently verified answer-sufficiency supervision can
  support precision-bearing set rewards.
- **InfographicsVQA:** it remains sealed through model choice and confirmatory
  testing. Stage 05 is a single locked robustness evaluation and cannot alter
  the chosen system.

## Audit completion rule

The consolidation is preservation-complete only when:

1. every row above points to an existing destination;
2. all relative links resolve;
3. no source heading lacks a destination or explicit provenance-only reason;
4. architecture and experiment files contain no placeholder markers or
   implicit scope expansion;
5. `docs/superpowers/plans/2026-08-14-overlap-v1-minivgent-poc.md` maps both
   source plans task by task;
6. repository documentation tests pass; and
7. the owner reviews the written specification before SOL activation.
