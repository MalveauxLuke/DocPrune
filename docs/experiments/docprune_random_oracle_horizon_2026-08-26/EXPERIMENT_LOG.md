# DocPrune Random/Coverage/Attribution Experiment Log

Scientific contract: [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md)
Execution checklist: [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)
Created: 2026-08-26

## Logging rules

This is the canonical append-oriented ledger for this experiment.

- Add a dated entry for every source acquisition, implementation decision,
  validation, submitted job, failed/rejected run, result, and course change.
- Never replace a failed result with a retry. Link both and state which one is
  canonical.
- Record exact commit, environment, input hashes, page/feature provenance, job
  ID, GPU, artifact path, file hashes, sample size, and decision where relevant.
- Mark fresh-retrieval or deep-global-index access as non-canonical. Never merge
  it with fixed-page controls.
- Update the implementation plan when a task changes status. Update the
  scientific plan only through an approved, dated amendment.
- Historical results below predate the corrected generation contract unless
  explicitly labeled otherwise. They select hypotheses and controls; they are
  not final comparisons for a new method.

## Current state

Status: Tasks 1–5 and Task 8 are complete. Task 9's four-mask smoke, original
64+32 development run, B13/input 256+64 diagnostics, and unified generated-
response and accepted-answer analyses are complete. The accepted-answer B13
surrogate has sufficient global predictive fidelity for a controlled oracle
pilot. On 2026-08-31 the user superseded the old `0.8` exact selected-set
Jaccard admission rule and approved preparation of a 48-question
answer-conditioned causal-selection pilot. A preliminary 24-correct/24-wrong
stratified-random cohort ran before the enriched panel and is complete.

The separate Task 6 fixed-page holdout root contains 1,213 sealed QIDs and
1,213 result files, each intended to hold the aggregate-score-versus-20-random
matrix. It is existing random-versus-DocPrune evidence, not a Task 9 comparator
arm; Task 9 will not rerun its uniform or coverage-matched random families.
The preliminary pilot includes only one same-action-space region-size-aware
random comparator. Task 6 branch/code and final analysis authority are
consolidated separately.

Unique accepted Task 6–9 work is consolidated on the active Task 9 branch.
Next action: prepare and seal the enriched cohort and remaining arms. No
method holdout, retrieval, feature rebuild, or Task 10 job is currently
authorized. Task 10 remains separately approval gated.

## Frozen local provenance

| Item | Value |
|---|---|
| Working repository | `/home/lmalveau/DocPrune-fix-evaluate-measurement` |
| Branch at creation | `fix/evaluate-measurement` |
| HEAD at creation | `dd5f000a909a718826541df816a4b65396764e1c` |
| Production environment | `/home/lmalveau/mamba-envs/docprune-sol` |
| Corrected baseline runtime | `/home/lmalveau/DocPrune-ctp-baseline-dd5f000` |
| M3DocVQA corpus | `/scratch/lmalveau/docprune/datasets/m3docvqa` |
| Historical benchmark root | `/scratch/lmalveau/docprune/benchmark-4e2473b/attempt-2` |
| Qwen revision | `eed13092ef92e448dd6875b2a00151bd3f7db0ac` |
| ColPali revision used locally | `vidore/colpali-v1.2@961b51745de3e9adb3468ac5c9ccca0ac626c217` |
| M3DocRAG revision | `29e6ac2294d6b87075a1d45b8a8df175b214248a` |
| Fixed retrieval rule | cached ordered top-4 pages and persisted page features only |

The local ColPali v1.2 revision differs from the original ColPali generation
reported for the paper-era M3DocRAG setup. This can affect absolute paper/local
scores, but it cannot explain paired losses once pages and features are held
fixed.

## Paper-code availability audit — 2026-08-26

This table records links and audited target pins. Task 1 acquired the exact
snapshots; the immutable inventory and hashes are recorded at the end of this
log.

| Source | Code availability and target pin | Status |
|---|---|---|
| DocPrune | No paper-linked author repository found; local implementation is a reconstruction. | Limitation recorded |
| Wen et al. | No author implementation linked in arXiv v2. FastV is a studied baseline, not Wen's code: [repository](https://github.com/pkunlp-icler/FastV), `d1659729b5bf1be225e99ee15783deeea80f63b1`. | Limitation recorded |
| Zhang et al. | [VisPruner](https://github.com/Theia-4869/VisPruner), `aefa01adc7c7ce6334e880c88225e90cede760d1`. | Exact read-only snapshot acquired and audited |
| Wang et al. | [Information-Horizon](https://github.com/YahongWang1/Information-Horizon), `75909b2936a13d7214e83514f0bc0cb9cc91139e`. | Exact read-only random/standalone-deletion source acquired and audited |
| Endo et al. | [FEATHER](https://github.com/markendo/FEATHER), `c2a09b2765967601054c7b1fd513ac3e94ee4fc9`. | Exact read-only snapshot acquired and audited |
| ContextCite | [context-cite](https://github.com/MadryLab/context-cite), `c11f8ace6e68ba0121b2e2f1f5c896da9e4156f4`. | Exact source acquired, audited, and installed in isolation |
| MinerU independent tool | [MinerU](https://github.com/opendatalab/MinerU), tag `mineru-3.1.0-released`, commit `d9cd58add047c2364c1198eefcb1ee9cd63a971a`; [MinerU2.5-Pro-2604-1.2B](https://huggingface.co/opendatalab/MinerU2.5-Pro-2604-1.2B) revision `d3f5e08d073c21466bbabe21c71bb1e9c2e595da`. | Exact source/model acquired; isolated CPU-safe setup passed; not attributed to a pruning paper |

Verified Wang source behavior from the initial read-only audit:

- the published LLaVA random branch samples uniformly without replacement with
  Python `random.sample`, restores sorted sequence indices, and wrapper scripts
  repeat the evaluation three times;
- the branch assumes a hard-coded LLaVA image span/offset and fixed token count,
  and its layer/cache/position handling is not directly compatible with the
  local Qwen2-VL post-BTP+QTP decoder;
- the local port therefore preserves the distribution, exact count, sorted
  order, and three repetitions while adapting the token population, boundary,
  M-RoPE, cache, and deterministic seed recording; and
- correction after exact local-source acquisition: `cal_info` physically
  deletes all visual states or retains one 2x2 window, and `info_prune` ranks
  `P(first gold token | only this window remains)`. It broadcasts each window's
  score to four tokens and does not subtract the generated no-visual baseline.
  The earlier zero-mask characterization was incorrect.

Verified ContextCite source defaults from the initial read-only audit:

- 64 ablations;
- ablation keep probability `0.5`;
- Lasso regularization `0.01`;
- intercept enabled; and
- random state `0`.

## Corrected generation-path admission — canonical control

Date: 2026-08-26
Decision: **pass; fair controlled decoder baseline admitted**

The stock Qwen path and all-kept manual DocPrune decoder passed exact
preprocessing and exact full generated-suffix equality under BF16
FlashAttention-2 and EOS IDs `[151645, 151643]`. First-step logits passed at the
declared L40S kernel tolerance `rtol=0.02`, `atol=0.07`.

| Field | Canonical value |
|---|---|
| Job | `62219618` |
| GPU | NVIDIA L40S, 46,068 MiB, driver 595.71.05 |
| Runtime | `dd5f000a909a718826541df816a4b65396764e1c` |
| Artifact root | `/scratch/lmalveau/docprune/qwen-all-kept-parity-dd5f000-v4-l40s/result` |
| `artifacts.sha256` digest | `84bda742007f3702dd752d536c8257022af89645540846243f5a79df52275d43` |
| `junit.xml` digest | `a5d265cbd8ee13e8ac6221e60320b687985ed3016fe6ed532d47d382c133c9e6` |
| `pytest.log` digest | `801e1eb0dc7e74ece3624dd6a2c269021c5efd86a355306e3242891bd5318848` |
| Repository verification | 442 passed, one opt-in real-model test skipped, two known stale-authority tests deselected; Ruff passed |

Numerical-drift probes also passed exact generated suffix and their declared
logit checks:

| GPU | Job | `artifacts.sha256` digest |
|---|---:|---|
| A30 | `62212255` | `52f25ba9c28d50976d502fc419c830abc6a9329a4c7a2ef60adc6dbd37a22a7b` |
| A100-40GB | `62212258` | `69f4581290689467f364a3e137b1c15e6ffe0c15a2ca204614142fd1d12c1ab9` |
| H100 | `62212256` | `166d2ebef05aa76673c9b2a62ab26ba47bef9cae4ff7416202d175d66271fc1a` |

Consequence: all future quality arms must be rerun under this corrected
generation contract. Historical scores below may not be directly compared with
a new corrected-runtime method.

## Paper target and historical full top-4 recreation

Paper Table 2 top-4 target:

| Mode | EM | F1 | Reported delta from all-kept |
|---|---:|---:|---:|
| Paper all-kept Qwen | 31.5 | 36.3 | — |
| Paper DocPrune | 33.0 | 37.3 | +1.5 EM, +1.0 F1 |

Local completed top-4 recreation at historical runtime
`4e2473bdbbc2e4eca0e92c30d4a0633044501ccf`:

| Slice | N | All-kept F1 | Literal DocPrune F1 | Paired F1 delta | 95% CI |
|---|---:|---:|---:|---:|---:|
| Overall | 2,441 | 37.7911 | 36.7603 | -1.0307 | [-2.1188, +0.0606] |
| Single-hop | 1,461 | 46.4497 | 44.8303 | -1.6194 | [-3.0910, -0.1609] |
| Multi-hop | 980 | 24.8827 | 24.7296 | -0.1531 | [-1.7571, +1.4592] |
| Retrieval-identical | 2,126 | — | — | -1.1980 | [-2.3471, -0.0640] |

Overall EM was `32.4867` all-kept and `31.8722` literal DocPrune, a paired
delta of `-0.6145`.

Canonical historical quality analysis:
`/scratch/lmalveau/docprune/benchmark-4e2473b/attempt-2/analysis/top4-paired-quality.json`

- whole-file SHA-256:
  `e07cc815a2847e6a9af21761fc29f0b1241f4e411b54132bb65b04cf47c3b51a`
- internal analysis digest:
  `2770fc16f66cc5f33d856179032335a03f8da8a0abc5e967780eff5dedd6ea14`
- all-kept result/summary/manifest SHA-256:
  `ace39b9ece689a58d9c3302c726a5494f7d91dd20abf759846dce0d99b3907b1`,
  `9e634aa1047b1f6357c81fa7f7d08cb6157b6fea9f9e10ce8acdc66ad9d3649c`,
  `6ef2c6bba1465e974d4614e279c0a377c0d866657c332f8d56a2bdbe9a601189`
- literal DocPrune result/summary/manifest SHA-256:
  `fb148d6ce33f92ef788aa0c65ada4a95f9ed81f575e9fe3e3097b54f37d1b217`,
  `5dc265588ca87d98664629d2243c992b233fd0d588754fc0c5ac6a22f441bdce`,
  `bb54cb34ca23b3ba65b59f5b70fee3e36ccd34d1a59ae67bfedac0c39ea7d95b`

Historical literal DocPrune visual-token trajectory:

| Stage | Tokens | Retained from original |
|---|---:|---:|
| Original | 24,488,112 | 100.0000% |
| Post-BTP | 13,031,372 | 53.2151% |
| Post-QTP | 11,289,528 | 46.1021% |
| Post-CTP | 4,477,474 | 18.2843% |

Historical selected zero-based block counts were `14:1491`, `15:147`,
`16:797`, and `17:6`.

Interpretation: local absolute all-kept F1 is close to the paper, but the
controlled CTP direction is reversed. The full result motivates the revised
ranking/dependence experiment but is not a corrected-runtime final baseline.

## Sealed 245-question stage diagnostic

Population: all 245 paired single-hop questions available to the controlled
diagnostic; ordered pages are identical across stages.

| Stage | F1 |
|---|---:|
| All-kept | 44.8612 |
| BTP only | 43.1143 |
| BTP+QTP | 43.6327 |
| Literal/current full CTP | 41.3347 |

The literal CTP transition was `-2.2980` F1 with paired 95% CI
`[-4.2980, -0.6286]`. This localizes a statistically supported controlled loss
to the current CTP stage on this cohort.

Historical analysis:
`/scratch/lmalveau/docprune/benchmark-4e2473b/attempt-2/diagnostics/stage245-v1-incremental/stage245-analysis.json`
(SHA-256 `d1a2479528aa775acbe8af7bb6e94718af034f2687867ae310ff2238dfb1b3af`).

## Aggregate-logit CTP diagnostic

Decision: **retain as primary best-fit local DocPrune reference; do not call it
author-confirmed**.

On the same 245 controlled questions:

| Mode | F1 | Conditional CTP retention |
|---|---:|---:|
| BTP+QTP | 43.6327 | 100% |
| Literal/current CTP | 41.3347 | 41.3774% |
| Aggregate-logit CTP | 42.4163 | 67.2929% |

Aggregate-logit minus literal was `+1.0816` F1, 95% CI
`[-0.1102, 2.5796]`. Aggregate-logit minus BTP+QTP was `-1.2163`, 95% CI
`[-2.8571, 0.1673]`. Its retention is close to the paper-derived top-4 target
of 65%, but its point estimate remains below BTP+QTP.

Canonical historical analysis:
`/scratch/lmalveau/docprune/ctp-aggregate-logit-stage245-fixed-v1/analysis-final.json`

- file SHA-256:
  `0932ff61144774a66f589afb52c78366bb55424f42d176c1a50e12c4cc802e82`
- internal digest:
  `6d2cf097defd1cdc120ddd34f7d262c12e53072dddc677ae1f65f65867d51467`

## Rejected or non-promoted diagnostics

These results prevent repeated work; none becomes an arm in the new plan.

| Diagnostic | Cohort/result | Decision and evidence |
|---|---|---|
| Original-token attention scaling | 16-question calibration retention `58.0067%`; F1 `43.0833` versus current `46.8333`, delta `-3.75`, CI `[-11.25, 0]` | Rejected. Restoring count selected the wrong identities. Analysis `/scratch/lmalveau/docprune/pruning-semantics-v1/ctp-original-relative-v1/analysis.json`, SHA-256 `200a811b248ea5215e7d543c8811ec6ad7d459c065f181d2ed6fd1e392792139`. |
| QTP mean aggregation | 245 questions, F1 `42.6653` versus current BTP+QTP `43.6327`, delta `-0.9673`, CI `[-3.1755, +1.0408]` | Rejected; keep current QTP. Analysis `/scratch/lmalveau/docprune/pruning-semantics-v1/qtp-mean-stage245-v1/analysis.json`, SHA-256 `ae0d551a7bf1957163cb43de310438a8a13f9d7ecf01c839f0a3d5928de92828`. |
| Grouped-query-head aggregate logits | 12 calibration questions, retention `68.5143%`, F1 `43.50`, tied BTP+QTP and below current CTP `46.8333` | Rejected; retention fingerprint alone is insufficient. Analysis `/scratch/lmalveau/docprune/ctp-kv-group-probe-v1/l40/analysis.json`, SHA-256 `acb4fb27f39fff126384d2443b2a6af6fa61f7ad05357af9fbbeaf69785c681d`. |
| First-generated-token/layer-output aggregate logits | 12 calibration questions, retention `65.0360%`; answers identical to BTP+QTP on all 16; candidate-current F1 delta `-3.3333`, CI `[-10.0, 0.0]` | Rejected; near-perfect retention matching did not create the paper's gain. Analysis `/scratch/lmalveau/docprune/ctp-generated-query-qa-v1/l40s/analysis.json`, SHA-256 `80573b02f2a3b450ad7020eb8c86e1ace30a02179e3db68ddb8a40e5b57dd073`. |
| Total-sequence rather than post-QTP scale | CPU replay changed only `111/56,235` available tokens; Jaccard `0.997081` | Rejected without GPU expansion; too small to explain residual loss. |

## Superseded initial design decisions — 2026-08-26

The decisions in this subsection document the initial plan. They were
superseded before implementation or new experiment runs by the approved review
amendment immediately below.

1. Primary DocPrune reference is aggregate-logit CTP; literal/current CTP remains
   a required secondary reference.
2. Primary random follows Wang's uniform-without-replacement core with three
   repetitions, adapted to the local dynamic post-BTP+QTP population,
   per-question `M`, `B_l*`, M-RoPE, and caches.
3. Fixed-boundary interventions use explicit `B_K = after block K, before block
   K+1`; the existing threshold-selected-only decoder needs a forced-boundary
   diagnostic hook.
4. MinerU is not paper-derived pruning code. It provides page-space regions
   that become binary masks over the actual deep visual tokens inside each
   region. A segment is never treated as one token.
5. The primary oracle is an adapted ContextCite sparse surrogate over MinerU
   region masks, targeted to the first canonical-gold-token log-probability.
6. If ContextCite fails its pre-QA feasibility/fidelity gate, it is replaced by
   Wang's published standalone token-information implementation. The fallback
   compares the first-gold-token probability with only token `v` versus no
   visual token at the same boundary.
7. No information horizon may be claimed until a numeric paired
   non-inferiority margin is approved before sweep outcomes are inspected.
8. Random/oracle/horizon implementation and jobs had not started. The next
   authorized activity is tool/source setup only.

## Approved review-driven design amendment — 2026-08-26

Authority: user approved all ten review packages before implementation. No new
experiment output had been generated, so the revision is pre-outcome for this
experiment.

### Findings accepted and project-specific corrections

1. The existing 64-question prefix is contained in the 245-question diagnostic,
   and aggregate-logit CTP was selected/evaluated using all 245. Therefore all
   16-, 64-, and 245-question cohorts are development data; the remaining 181
   are not untouched confirmation.
2. A confirmatory method holdout will be selected deterministically from the
   remaining single-hop full top-4 QIDs after excluding every diagnostic QID.
   Eligibility and selection may use only QID/support metadata and authenticated
   cached DocPrune-mode pages/features, never historical scores. Required `N`
   comes from an 80%-power developmental calculation. The result is a method
   holdout, not a pristine benchmark test, because historical all-kept/literal
   outcomes exist for all 2,441 questions.
3. Native literal/aggregate threshold policies and forced score top-M rankings
   are separate experiment families. Native policy results are not budget
   matched; ranking-only results share boundary and budget.
4. The primary estimand is aggregate-score top-M minus expected global-random
   top-M token F1 at native `B_l*` and aggregate-native `M`. Equivalence margin
   is `±1.0 F1`; TOST uses a 90% interval, superiority uses a two-sided 95%
   interval, and otherwise the result is unresolved.
5. Statistical random evaluation starts with 10 masks per developmental
   question and freezes 20 before holdout sealing if mask Monte Carlo standard
   error exceeds `0.25 F1`. Three repetitions remain an author-code-fidelity
   report, not the inferential justification. Terminal analysis uses nested
   mask resampling, support-document connected-component clustering, 100,000
   draws, and Holm adjustment for secondary families.
6. Coverage is experimentally controlled with global random, page-stratified
   random, normalized 4×4 grid-stratified random, and aggregate-occupancy-matched
   identity shuffling. Descriptive coverage correlations alone cannot support
   a causal mechanism claim.
7. ContextCite/MinerU is renamed exploratory gold-answer-conditioned regional
   attribution. Whole MinerU regions are intervention and deployment units;
   coefficients are never broadcast to tokens and raster/original index never
   splits a region. Whole-region knapsack yields achieved budget `M′`, and all
   comparators use that `M′`.
8. Regional attribution targets normalized teacher-forced full-answer
   log-likelihood over accepted references, with the unpruned model response as
   a separate target. Admission requires both LDS and Spearman points at least
   `0.5`, both 95% lower bounds above `0.2`, error better than constant,
   selected-region Jaccard at least `0.8` across five refits, and direct
   deployed top-versus-reverse mask validation.
9. Wang's zero-mask score is a conditional standalone-contribution study, not
   an automatic fallback or oracle. It requires separate approval after a cost
   microbenchmark, is initially limited to one boundary and the smaller of
   eight questions or 40,000 continuations, and must demonstrate transfer to
   physical deletion.
10. The layer experiment is an explicit-visual-state removal/dependence curve at
    `B_input`, `B_0`, `B_6`, `B_13`, `B_20`, `B_23`, and `B_26`. Its paired
    reference is BTP+QTP/no CTP; simultaneous one-sided 95% lower bounds must
    exceed `-1.0 F1` at a candidate and every later tested boundary. It is not
    called an information horizon without near-zero standalone information.
11. Secondary assay-sensitivity strata use support-document recall@4, no-CTP
    EM/F1 success, and `B_input` visual sensitivity. These do not prove exact
    answer evidence survived BTP/QTP and do not authorize grounding claims.
12. Intervention validation now includes all-kept replay at every forced
    boundary, physical-deletion/zero-mask comparison, cache and M-RoPE checks,
    edge budgets, failure-to-cross, region mapping, deterministic seeds, and
    fixed-page/global-index fail-closed enforcement.

### Revised phase order

1. Native-policy corrected-runtime controls.
2. Ranking-only score/random/coverage comparison.
3. Explicit-visual-state removal curve.
4. Exploratory regional attribution after its development gate.
5. Separately approved Wang standalone contribution only after cost/transfer
   review.

### Claim changes

- Use `equivalent` only when the powered equivalence procedure passes; otherwise
  use `unresolved` when appropriate.
- Use `privileged answer-conditioned regional attribution advantage`, not
  `oracle gap`.
- Use `explicit-visual-state dependence boundary`, not `information horizon`.
- Use causal coverage language only for controlled coverage contrasts.
- Keep every claim bound to local literal/best-fit reconstructions, the tested
  Qwen model, cached pages, token population, layer, budget, and cohort.
- No grounding, official-DocPrune, end-to-end retrieval, multi-hop, cross-model,
  or cross-dataset claim is authorized.

## Approved Wang semantics amendment — 2026-08-27

Authority: on 2026-08-27 the user approved this exact course change before any
Task 2/3 outcome existed:

> Wang's faithful released-code diagnostic uses physical deletion of 2x2
> visual-token windows. Zero masking remains a separate local diagnostic and
> must not be described as Wang's implementation.

Rationale: Task 1's pinned-source audit found that the Wang paper describes a
zero-masked standalone-token quantity with a no-visual baseline, while the
released `cal_info`/`info_prune` paths physically delete all visual states or
retain one 2x2 window. Released ordering ranks the first-gold-token probability
for the retained window, broadcasts that score to its four members, and does
not use the saved no-visual baseline. The faithful released-code arm is
therefore the 2x2-window physical-deletion diagnostic. Any local zero-mask
intervention is a separately labeled transfer/estimand diagnostic and is not
Wang implementation fidelity.

Timing and scope: no affected Task 2 or Task 3 outcome existed or was inspected
before this approval. The amendment resolves Task 10's scientific-semantics
blocker but does not activate Task 10: it remains separately approval gated
after Tasks 1 and 3. Tasks 2–3 are now authorized, with Task 2 active. No
threshold, cohort, unrelated method, source pin, code, test, artifact, or job
changed as part of this amendment.

## Experiment-specific run ledger

Task 3 admission is complete; no current run is active.

When the first run is authorized, append rows with this schema:

| Date | Run/status | Runtime | Inputs | Policy/boundary/budget | Job/GPU | Artifact and hashes | Result/decision |
|---|---|---|---|---|---|---|---|
| 2026-08-27 | Task 3 forced all-kept parity; completed/admitted | `/home/lmalveau/DocPrune-forced-boundary-runtime-36cb771` @ `36cb771db3af91ee7a0b76803ec099b12dba31e4` | Env `/home/lmalveau/mamba-envs/docprune-sol`; runtime launcher SHA-256 `bda211dc9c2ef958842d1323ef481fd94b2a9a1af0be344c88d3eae00b099054`; cached Qwen revision `eed13092ef92e448dd6875b2a00151bd3f7db0ac`; fixed probe SHA-256 `3ae33f3bc9064df02ef3535a3e7ed6a2b2dafbbc25519cea38db78b393ee19d4` | Forced `physical_delete`; all-kept `M=|V|`; `B_input`, `B_0`, `B_6`, `B_13`, `B_20`, `B_23`, `B_26`; all identity/parity verdicts true | `62265662`; completed `0:0` on `scg017`; NVIDIA L40S 46,068 MiB, driver `595.71.05`; 4 CPUs, 64 GiB, 20 min requested | Result root; verified `artifacts.sha256` `649bde404af0cc63a30436ba884a990ef8a0abfa99dc0377e5b0c69745528b1a`; boundary JSON `32748c311eaa993aa88128e87393fa36b5a2a10d5776cd871ae9e32e8a9d8302`; JUnit `6e037e92570f2069b6eeacbf35828a94aa64a4605b69f3f7a72404816502edc9`; log `58f1314ce14880520ff2a43748dc863e5a0847db02d10dddf084a90c130478b7` | Admitted: `1 passed` in `19.88s`; JUnit 1/0/0/0; Task 3 complete, no further submission |

## Source acquisition ledger

### Task 1 complete — 2026-08-26

Scope: lightwork CPU only. No retrieval, global index, document parsing, model
inference, experiment policy implementation, GPU job, or production-environment
change occurred.

Source root:
`/scratch/lmalveau/docprune/paper-code/random-coverage-attribution-v2`.
All six repositories are detached, clean, and read-only (`0550`). The generated
inventory is
`/scratch/lmalveau/docprune/tool-envs/random-coverage-attribution-v2/manifests/source-repositories.tsv`,
SHA-256 `9bc189377212a8854f65ed94d7a38c8063782f88329c17cd9a558a1e81c30734`.

| Snapshot | URL | Commit | Git tree | `git ls-tree -r` SHA-256 | License |
|---|---|---|---|---|---|
| Information-Horizon | `https://github.com/YahongWang1/Information-Horizon.git` | `75909b2936a13d7214e83514f0bc0cb9cc91139e` | `3dfca14664f976e8f6606215c07cea1ad278c9a4` | `8aac106bf0477c32ba5840afb66124d3c2f73acd8dcbe05dbb453a75b013d31c` | Apache-2.0 |
| context-cite | `https://github.com/MadryLab/context-cite.git` | `c11f8ace6e68ba0121b2e2f1f5c896da9e4156f4` | `61ac1e995f0d1b9f17a20576791fc83cf485291c` | `9cbaf1dd11324f2a401e3134fca061a25f6259a56a1ffd9ef07c6e4d26f72643` | MIT |
| MinerU | `https://github.com/opendatalab/MinerU.git` | `d9cd58add047c2364c1198eefcb1ee9cd63a971a` | `a411f821f3f26a893661a729d9e6301873a4e44f` | `2963131e09b0bc04845a7b2eceb516ae677afbd3b513b26f2d8a4402c4e6c114` | MinerU Open Source License (Apache-2.0 base plus additional terms) |
| VisPruner | `https://github.com/Theia-4869/VisPruner.git` | `aefa01adc7c7ce6334e880c88225e90cede760d1` | `8d5a67267e6332d2146368ffb46ac73a214ce850` | `20fea342bc8581ea0a64076bfa487abba4c27683cde636d362e4a29319f101b0` | Apache-2.0 |
| FEATHER | `https://github.com/markendo/FEATHER.git` | `c2a09b2765967601054c7b1fd513ac3e94ee4fc9` | `5d2f3334a89234c551ee4f19ded9e834af55f2da` | `08f1874fd3bce6716ab9ac654f4b39f7fedc9eed8f416c506a4917a9de3c583f` | MIT |
| FastV | `https://github.com/pkunlp-icler/FastV.git` | `d1659729b5bf1be225e99ee15783deeea80f63b1` | `d0fdb44ec67ac95c74136adc867562068c616102` | `eb55dc7b033b21a3d3aaba109df8d41ce9124dba3959843b8a22252cab861474` | No root license; vendored LLaVA and Transformers trees are Apache-2.0 |

No paper source has been ported into the local implementation yet. The audited
files most likely to inform later ports or adaptations have these SHA-256s:

| Source | Directly relevant audited files |
|---|---|
| Information-Horizon | `modeling_llama_self.py` `528f828e62530b619dbb05692d4e3caf7d463d92bc85c5fa3380d5dc674c6458`; `llava_arch.py` `39af15991284562536b56e4ec052cef9e7d1a8a1efb62bac55633578c68fc39b`; `llava_llama.py` `41ca12bd7ca09b8307592e6f6245273de73c9a2164a5d19a175b04ee44132c31`; `model_vqa_loader.py` `5d98cf34ca9965b6f1cf61948ea98940c738b513f11a77e5de2d712229640e05`; `cal_info/textvqa.sh` `55723af43d35e0ce1cbd2779213b45a743866ea4531c83fa9d92bc8bdb45746c`; `info_prune/textvqa.sh` `6b152180893cc557b1d95db58e46f331fce7fd7b0b926801434358efe8be03a7`; `info_prune/textvqa_gt.sh` `bfb0eb8d2881b6165ceede5e6d6803da76995c1261a3c8c963410e68ef12ab01`; `allbench_dart_random.sh` `fd217cf255d8334ed2ff8cfd35fe76e327bef8ea4d775cca59e45122f2a3d987`; `allbench_divprune_random.sh` `68235059467d28a8fe355d24da9c9d1170e1f03292a91ce8362d480de9a36999` |
| context-cite | `context_citer.py` `958b2444f8245ec54ecfdf17aca6b3e7334dab0444c21cee8e06ec8a3dfe855c`; `solver.py` `9c3de5c4b06b08a82245431105a58aecada0944a7bb38f506b6eb23f434fd37c`; `utils.py` `d3825dc3292886e0fc9e4c4f0d397f45a84ce1a1ee387ce6985d3006e1c28a4b`; `context_partitioner.py` `a13c418eed5a5710503e109549f4b56221157f1c510239e34d8e4439e33c39aa` |
| MinerU | `pyproject.toml` `354ee12e71c9a59ff1afcbd0c76d0e069f53a0d6560796dd52f1c5a53ef734a9`; `output_files.md` `07c2154f7059c2502eab7c383d3c1f31b59e0d802f91a97e3dd313503d72dcf3`; pipeline `model_json_to_middle_json.py` `51b8180cfd984286ad155f83052957b181bdddb4950e70a16faa3f22f78757b8`; VLM `model_output_to_middle_json.py` `06651d31202a48132725e23cb3baf7510bdcd8e7a114615917a63a3a1c812576`; `models_download_utils.py` `7280e42a6d7c9f50f317754d55d10445a541b66ea9128dd5950c5a737eda1228`; `hash_utils.py` `50f4bff0d441995ba67723afd7a2e6bfa2f2db12bc3950c911ee601ba80a2059` |
| FastV | `modeling_llama.py` `b06cde51267b635100763d58f5f1b9b5933c769bd203936146d34ecd3c242b08`; `configuration_llama.py` `c4f33506a0cf5cc6c75e89afc08d1243d9d441ee0af0c22b03504ce03534c801`; `inference_aokvqa.py` `127045787f521e755cbd5d1eb3ce75290a252127407c1784a3c943a2c10be937` |
| VisPruner | `llava_arch.py` `5559d52a05fd77e39aa82cfe4e05835adea787604fdab8ad4588291bb6e843b2`; `clip_encoder.py` `b11b6f6b37e839be81e5de200fceb2dad2e8212e9ac9e4e9087d4f835a314268`; `llava_llama.py` `a4ad3bdec743d66329b0b7ffc419b7c6696b71fe71c422746d8884535f3fb066` |
| FEATHER | `llama2_models.py` `1ef95649c182b12f682b16cd9144cc767e45ae2f786ee726c7a7f4850defb7b7`; `base_vision.py` `736491aea310bca01c891dfcf2c04ca0f6003527fc2487ca78b35dcefa96ab17`; `patch_cluster.py` `ec4a9e1a4f86f5aa38cc8774a9ec7405107d638f0694dfd9d7d0389eab46448a`; `prismatic.py` `8b5e87521b599dc593ae315b73180bebdf50d5c0f043eeeb08f5b7e6d82491bd` |

### Source behavior and adaptation decisions

- Information-Horizon random uses unsown Python `random.sample` uniformly
  without replacement, exact retained counts, sorted survivor order, and three
  whole-evaluation wrapper repetitions. Its fixed LLaVA span (`35`, `576`,
  `24x24`), scalar positions, batch-one cache path, and before-layer indexing
  cannot be copied into Qwen. A source layer `20` is closest to local `B_19`.
- The Wang `SIZE` paths physically delete all visual states or retain one 2x2
  window. The shipped ranking assigns the same first-gold-token probability to
  each of the window's four tokens and does not use its no-visual logits in the
  ordering. Therefore the approved plan's zero-mask/token-score premise is an
  unresolved contradiction requiring a pre-outcome amendment; Task 10 is
  blocked meanwhile.
- ContextCite faithfully supplies 64 independent masks at keep probability
  `0.5`, deterministic seeds `0..63`, Lasso `alpha=0.01`, intercept, and a
  fixed-response likelihood/log-odds target. It supplies no LDS, Spearman,
  held-out fidelity, or deep-visual-region implementation; those remain local
  evaluation adaptations.
- MinerU provides page identifiers, region labels/types, reading order, scores,
  and axis-aligned bboxes. It does not emit deep-token masks or a full runtime
  provenance sidecar. Region-to-Qwen-token rasterization is a local adaptation.
- The pinned MinerU Git tag/README says `3.1.0`, but `mineru/version.py` and the
  source-built distribution report `3.0.9`; both facts are retained rather than
  normalizing the discrepancy away.
- FastV contains scalar last-query mean-head attention top-k over a fixed LLaVA
  visual span, not Wen's Window FastV. Wen links no author code. The normalized
  per-page 4x4 policy is therefore a named local coverage control, not a
  Window-FastV reproduction. VisPruner is feature-diversity interpretation;
  FEATHER uses source-specific square stride/27x27 geometry; neither is the
  local 4x4 arm.

### Isolated tool and model setup

Tool root:
`/scratch/lmalveau/docprune/tool-envs/random-coverage-attribution-v2`.

| Item | Exact setup | CPU-safe verification |
|---|---|---|
| ContextCite | `contextcite/`; Python `3.10.20`; source distribution `0.0.4`; Torch `2.4.1`; Transformers `4.46.3`; Hub `0.36.2`; Tokenizers `0.20.3` | Torch is visible to Transformers; `context_cite` and `utils` import; NLTK `punkt_tab` present; synthetic Lasso score `0.99936`; `pip check` clean |
| MinerU | `mineru/`; Python `3.11.15`; source distribution `3.0.9`; Torch `2.13.0+cu130`; Transformers `4.57.6`; Hub `0.36.2` | no-model `str_sha256` import/result matches stdlib; `pip check` clean; no parser/model inference run |
| MinerU model | `hf-cache/models--opendatalab--MinerU2.5-Pro-2604-1.2B/snapshots/d3f5e08d073c21466bbabe21c71bb1e9c2e595da`; 13 files, 2,328,026,289 bytes | revision matches Hub commit; no `.incomplete`; `model.safetensors` SHA-256 `f2650d91aaa619534980445034f62cde27fc3fa0430aaf5c3302b91179cad0c5` |

ContextCite's unconstrained upstream dependencies initially resolved
Transformers `5.16.1`, which disables Torch `2.4.1`; a failing
`transformers.is_torch_available()` assertion exposed the incompatibility.
The isolated environment was corrected to the production-known compatible
Transformers/Hub/Tokenizers versions shown above. Production itself was not
modified. ContextCite module metadata also says `0.0.1` while its built
distribution says `0.0.4`; the distribution version is used in the manifest.

| Manifest | SHA-256 |
|---|---|
| `contextcite-conda-explicit.txt` | `48ef672901e4e955e561879009a5edffbbd74a524d75c65f3777ed15697a8f09` |
| `contextcite-pip-freeze.txt` | `2332525e1d7b16e8fbf7af4f00018efe71c2d05639a7b11e3fe19f65dfeab3dd` |
| `mineru-conda-explicit.txt` | `982aa7fdc7cb9f6d2eb7a1a0989d8d35816f40935b6dcd6e2e362f4bef32f32b` |
| `mineru-pip-freeze.txt` | `aa0aae3bff5ba08e2c6d5d8da236374d6e01d10ff8c75fd2ae2e1d7840fd1ccb` |
| `mineru-model-sha256.txt` | `5ec9100ca6d70984dfae755a8a89fd60ffac663fcc5ae9b31f4a958ebe2846e1` |

Task 1 acceptance decision: **pass for source reproducibility and isolated
setup**. This does not admit any experiment method. Task 2 had not started at
the time of this Task 1 decision.

## Task 2 statistical, cohort, and artifact contracts — 2026-08-27

Task 2 acceptance decision: **pass**. Added the pure-Python
`src/docprune/experiment_design.py`, focused synthetic tests, and the durable
`development-qid-registry.json`. The registry uses explicit QID-only
projections from three immutable sources; it contains 12 diagnostic labels and
254 unique developmental QIDs. The 64-QID cohort is contained by the 245-QID
cohort; the historical 16-QID diagnostic contributes nine QIDs outside the
245-QID cohort, confirming that exclusion must use the complete union.

| Registry source | Source SHA-256 | QID projection SHA-256 | Count |
|---|---|---|---:|
| `/scratch/lmalveau/docprune/ctp-normalization-diagnostic-v1/cohort.json` | `2ada86e06b320240862b38f99de823d49f692514208db442d714dd0a71165b37` | `ee8d9014a92cd998b2e731e90bbf935995134742d70f9d662fcd216533bd853f` | 16 |
| `/scratch/lmalveau/docprune/benchmark-4e2473b/attempt-2/diagnostics/stage64-v1/reference.json` | `381e19b033fa3e75e16d13bf6f13002bee7db2fc5102a9c6f374268131070642` | `ca03da09cfd9febb72f9a371e53c28457be79f0159ba04a3a604cefc377acaf8` | 64 |
| `/scratch/lmalveau/docprune/benchmark-4e2473b/attempt-2/diagnostics/stage245-v1-incremental/reference.json` | `96c4bb855cd16daca1ea10719617cea311eda6a2a2b65e0d2d210833cd7dfd1a` | `883d82f9dd1d675e2292d8d2fe01a98ae80015d51c604545bf4f83fc6af204e3` | 245 |

Registry union SHA-256:
`dd8d39df9e92335f1c130703ab2fd9341ea8257a18bad7453201a58eb72dc93e`.
Canonical registry digest:
`8a61cca82fa7e0d241e94f5802bc664b17302082dc06d98a161af690a78e6022`.
File SHA-256:
`46431c02445d94bd80a4d92d6c4e64155195d6be21cb0e3476943e553eaf79f9`.

Implemented outcome-key-rejecting eligibility; exact versioned holdout order;
support-document components; the paired aggregate-minus-within-QID-random F1
estimand; deterministic nested mask/component bootstrap with 100,000 terminal
draws by default; 90% TOST and 95% superiority intervals; orthogonal result
flags and precedence labels; Holm adjustment; the prescribed ten-mask MCSE
calibration; clustered normal-TOST power planning; atomic no-replace holdout
sealing and fail-closed launch validation; and the Task 3 intervention artifact
schema with distinct native-threshold and score-top-M naming.

No final method holdout was selected or sealed because developmental random
variance and repetition calibration do not yet exist. No retrieval/index was
loaded, no page was retrieved, no feature was built, no model was loaded, no
candidate outcome was supplied to eligibility, and no job was submitted.

### Task 2 independent-review disposition — 2026-08-27

The initial Task 2 acceptance decision above is superseded by an independent
**BLOCK** pending correction and re-review. Blocking findings cover true atomic
no-replace publication and parent-symlink handling; mandatory durable-registry
identity/labels and semantic launch revalidation; byte authentication of every
bound external path; and semantic parsing of both QID member files. Task 2 is
reopened, Task 3 must not start, and the final holdout remains unsealed.

### Task 2 independent-review corrections ready — 2026-08-27

The B1–B4 corrections and N1/N2 test/documentation corrections are implemented
and locally verified. Publication now uses Linux
`renameat2(RENAME_NOREPLACE)` through a validated non-symlink parent directory
descriptor. Seal and launch require the explicit durable registry path and
exact label identity, authenticate the registry and all bound external bytes,
rerun the eligibility/development-exclusion/selection/component semantics, and
require exact canonical bytes for both QID members. The nested-bootstrap test
now uses unequal multi-mask repetitions and repeated outer components with
hand-derived deterministic draws.

Disposition remains **BLOCKED PENDING INDEPENDENT RE-REVIEW**. These local
correction checks do not restore Task 2 completion, authorize Task 3, create a
final method holdout, inspect candidate outcomes, or authorize any job.

### Task 2 final independent acceptance — 2026-08-27

Final independent review decision: **PASS**. All prior B1–B5 blockers are
cleared: atomic publication is descriptor-anchored true no-replace; the exact
development registry and eligibility semantics are mandatory and revalidated;
all bound external bytes are authenticated; both QID members are exact-byte
and semantic-bound; and potentially large external files are hashed
incrementally without whole-file buffering. The N1 terminology correction and
N2 nested-bootstrap characterization also remain accepted.

Fresh independent verification reported 36 focused Task 2 tests passing, Ruff
checks passing, and both files formatted. Task 2 is complete and Task 3 is now
active only for its CPU test-first implementation. The final method holdout
remains unselected and unsealed until Task 6 receives the required development
random variance and repetition-count inputs. No holdout, model, retrieval or
index load, feature build, experiment, or job action was run by the review or
this disposition update.

### Task 3 CPU decoder-intervention implementation — 2026-08-27

Historical initial Task 3 CPU implementation decision: **complete; live parity
admission pending**. This status is superseded by the independent-review
correction entry below. Added a forced decoder intervention defined by `B_input` or
`B_K = after zero-based block K`, mode, and selected compact pre-CTP visual
population ordinals. The forced path skips native comprehension/attention
selection. Physical deletion preserves nonvisual state/order and all three
M-RoPE axes, leaves caches through `K` full, and compacts only later layers.
Local zero mask clones and zeros only unretained boundary hidden-state rows;
it never compacts sequence positions or cache topology. Both true no-op edges
(`M=|V|` and empty visual population) and `M=0` are CPU covered.

Native `CTPDecision` and the legacy five-field `PruningTrace` retain their
existing meanings. Forced runs instead return a distinct record carrying
canonical boundary, mode, selection kind, visual population, requested/achieved
budget, sorted stable pre-CTP visual IDs, logical retained sequence IDs, and
prefill cache lengths. The Task 2 intervention artifact now requires and
validates explicit `mode` and `selection_kind`; it never infers a forced/native
or deletion/zero-mask identity.

Focused tests cover input, block-zero, interior, and final tiny boundaries;
physical versus zero-mask topology; M-RoPE/order/dtype/device; edge budgets;
native non-crossing regression; adapter plumbing; and the fixed synthetic
deletion-versus-zero distinction. An opt-in real-model test covers all-kept
`B_input`, `B_0`, `B_6`, `B_13`, `B_20`, `B_23`, and `B_26` against stock suffix
and the admitted first-logit tolerance. It was deliberately **not run**.

Added draft-only launcher
`examples/sbatch/32_docprune_qwen_forced_boundary_parity_draft.sbatch` and
draft handoff `sol/handoffs/DOCPRUNE_QWEN_FORCED_BOUNDARY_PARITY_DRAFT.md`.
Both are explicitly NOT AUTHORIZED/NOT SUBMITTED; the launcher exits before
any model load and neither changes `sol/CURRENT_SOL_TASK.md`. No model,
retrieval/index, feature rebuild, real-model test, GPU job, or experiment run
occurred during this implementation.

### Task 3 independent-review correction work — 2026-08-27

Disposition: **BLOCKED PENDING INDEPENDENT RE-REVIEW.** The review identified
four durable-contract gaps and an ambiguous direct-decoder `visual_indices`
input. Local correction work adds fail-closed native artifact identity: only
the `native-threshold` family with physical deletion and native selection may
claim native identity, and it must bind a non-null `native_layer` to exactly
`B_<native_layer>`, never `B_input`. A forced record remains
`selection_kind="forced"` and may carry an observational native layer without
using it to identify its forced boundary.

Forced records now pass separately from `GenerationResult` through
`DocPruneQwenAnswerer`, `AnswerOutput`, and `SampleResult`, serializing via the
record's small `to_dict()` payload while the legacy five-field `PruningTrace`
is unchanged. The record now also includes the actual retained M-RoPE tensor
shape plus deterministic canonical SHA-256. The decoder rejects rank-other-
than-one, duplicate, reordered, or out-of-range visual-index vectors before a
layer can execute.

The CPU matrix now executes `M=0`, `M=|V|`, and empty populations for both
modes at `B_input` and `B_1`, including cache/order/position contracts and
pointer-identity no-op helpers. Tiny adapter parity covers all-kept physical
deletion and zero mask at `B_input`, `B_0`, `B_1`, `B_2`, and `B_3` against
native no-crossing generated IDs and first logits. The opt-in seven-boundary
live node is extended to require a fresh explicit no-replace JSON path and, if
separately authorized, emit/validate boundary, record, cache/order/M-RoPE,
suffix, tolerance, and useful logit-diff evidence. The draft launcher still
exits 64 but its unreachable future section supplies and hashes that JSON.

No real model, retrieval/index, feature work, GPU/job, or experiment was run.
`sol/CURRENT_SOL_TASK.md` remains untouched. These local corrections do not
restore Task 3 completion or authorize live parity; independent re-review is
the only next action.

### Task 3 final independent correction acceptance — 2026-08-27

Decision: **PASS — Task 3 CPU implementation accepted.** Independent re-review
accepted B1 native artifact identity, B2 forced-record transport/serialization,
B3 literal edge/no-op and tiny-boundary parity coverage, B4 no-replace live
evidence emission, and direct `visual_indices` preflight validation. The
legacy five-field `PruningTrace` remains unchanged.

Independent focused verification: `119 passed, 2 skipped`; the skips are the
two opt-in real-model probes. Accepted repository verification: `527 passed,
2 skipped, 2 deselected` with the two known stale authority-document nodes
deselected. Ruff check/format and draft-launcher shell syntax were accepted.

No model, retrieval/index, feature build, experiment, GPU, or job was run.
The draft handoff/launcher remains **NOT AUTHORIZED/NOT SUBMITTED** and exits
64 before any model action. Live seven-boundary parity remains a separately
authorized future gate; Task 4 is not started.

### Task 3 sealed seven-boundary parity submission authority — 2026-08-27

Status: **completed and admitted as job `62265662`.** The future-gate status
immediately above is superseded only for this completed Task 3 live all-kept
parity admission.
The active handoff is
[`DOCPRUNE_QWEN_FORCED_BOUNDARY_PARITY_2026-08-27.md`](../../../sol/handoffs/DOCPRUNE_QWEN_FORCED_BOUNDARY_PARITY_2026-08-27.md),
which supersedes the draft as submission authority without changing the draft
files.

The sealed clean runtime is
`/home/lmalveau/DocPrune-forced-boundary-runtime-36cb771` at
`36cb771db3af91ee7a0b76803ec099b12dba31e4`. The sole authorized launcher is
`examples/sbatch/33_docprune_qwen_forced_boundary_parity.sbatch` with SHA-256
`bda211dc9c2ef958842d1323ef481fd94b2a9a1af0be344c88d3eae00b099054`. It is
bound to `/home/lmalveau/mamba-envs/docprune-sol`, the cached Qwen snapshot
and revision `eed13092ef92e448dd6875b2a00151bd3f7db0ac`, and fixed probe
`/scratch/lmalveau/docprune/datasets/m3docvqa/probe/000ebcad8623837eb1384098a8c31d40-page-1.png`
(SHA-256 `3ae33f3bc9064df02ef3535a3e7ed6a2b2dafbbc25519cea38db78b393ee19d4`).

The one non-array L40S submission requests 4 CPUs, 64 GiB, and 20 minutes and
writes only to fresh root
`/scratch/lmalveau/docprune/qwen-forced-boundary-parity-36cb771-v1`, with
`result/forced-boundary-parity.json` as its fresh no-replace evidence path.
Before sealing, the runtime was clean at the exact commit, the launcher/probe
hashes matched, the cached model/environment paths existed, and the fresh root
was absent. The command runs only
`tests/qwen2vl/test_model.py::test_real_model_forced_all_kept_boundaries_match_stock_without_download`
under local-only model settings; retrieval, index access, feature rebuilding,
benchmark work, and all other test nodes are forbidden.

The sealed command was submitted once as job `62265662`; no scheduler or
outcome inspection was performed while recording this submission. On either
outcome, retain the root and record the job ID, evidence hashes, and verdict;
no retry or other Task 3 action is authorized by this entry.

### Task 3 seven-boundary all-kept parity submission — 2026-08-27

Job `62265662` was submitted under the sealed handoff. Its outcome is recorded
in the completion entry below. It
binds runtime `/home/lmalveau/DocPrune-forced-boundary-runtime-36cb771` at
`36cb771db3af91ee7a0b76803ec099b12dba31e4`, runtime launcher SHA-256
`bda211dc9c2ef958842d1323ef481fd94b2a9a1af0be344c88d3eae00b099054`,
environment `/home/lmalveau/mamba-envs/docprune-sol`, cached Qwen revision
`eed13092ef92e448dd6875b2a00151bd3f7db0ac`, and fixed probe SHA-256
`3ae33f3bc9064df02ef3535a3e7ed6a2b2dafbbc25519cea38db78b393ee19d4`.

The only test node is the seven-boundary forced all-kept physical-delete gate:
`B_input`, `B_0`, `B_6`, `B_13`, `B_20`, `B_23`, and `B_26`, each with
all-kept budget `M=|V|`. It requests one L40S GPU, 4 CPUs, 64 GiB, and 20
minutes and writes only to fresh root
`/scratch/lmalveau/docprune/qwen-forced-boundary-parity-36cb771-v1/result`,
including `forced-boundary-parity.json` and the declared evidence hashes.

Decision at submission: **pending; no outcome inspected.** No scheduler
inspection, model action, retrieval/index access, feature build, or other
change was performed while recording this submission.

### Task 3 seven-boundary all-kept parity completion and admission — 2026-08-27

Decision: **PASS — Task 3 live parity admitted; Task 3 complete.** Job
`62265662` completed `0:0` on `scg017` in `00:00:26` on NVIDIA L40S (46,068
MiB, driver `595.71.05`). The sole pytest node passed: `1 passed` in `19.88s`,
with three non-failing generation-configuration warnings. JUnit reports one
test with zero failures, errors, and skips.

Every `artifacts.sha256` entry verified at
`/scratch/lmalveau/docprune/qwen-forced-boundary-parity-36cb771-v1/result`.
Whole-file SHA-256 values are `artifacts.sha256`
`649bde404af0cc63a30436ba884a990ef8a0abfa99dc0377e5b0c69745528b1a`,
`forced-boundary-parity.json`
`32748c311eaa993aa88128e87393fa36b5a2a10d5776cd871ae9e32e8a9d8302`,
`junit.xml` `6e037e92570f2069b6eeacbf35828a94aa64a4605b69f3f7a72404816502edc9`,
and `pytest.log` plus Slurm output
`58f1314ce14880520ff2a43748dc863e5a0847db02d10dddf084a90c130478b7`.

Schema-1 boundary evidence has the exact ordered `B_input`, `B_0`, `B_6`,
`B_13`, `B_20`, `B_23`, and `B_26` records. Each is
`physical_delete`/`forced`, has visual population/requested/achieved `2508`,
retains every visual ID, and passes all six identity/parity verdicts. Every
record has M-RoPE digest
`1af5ac1559e09c2c1bc83fc52ce1c0fe28cb381f7b737998468bbec23b4ab077`.
The maximum and mean absolute first-step logit differences were `0.125` and
`0.012369606643915176`; the declared relative-tolerance verdict passed.

The probe remained SHA-256
`3ae33f3bc9064df02ef3535a3e7ed6a2b2dafbbc25519cea38db78b393ee19d4`, and
runtime remained `36cb771db3af91ee7a0b76803ec099b12dba31e4`. No further Task
3 submission is authorized by this result.

## Task 4 CPU implementation activation — 2026-08-27

The user explicitly activated Task 4 after the Task 3 live admission passed.
Scope is the pure CPU `src/docprune/ctp_controls.py` policy layer and focused
tests: deterministic aggregate/literal score top-M, Wang-core global random,
page-stratified random, normalized per-page 4x4 grid-stratified random,
coverage-matched identity shuffle, frozen seed identity, and coverage metrics.
Task 5 decoder/factory integration and every retrieval, feature, holdout, model,
GPU, launcher, or job action remain out of scope. No experiment outcome exists
for this implementation activation.

## Task 4 CPU implementation acceptance — 2026-08-27

Task 4 is complete. The first implementation attempt was superseded because its
initial RED command imported a different editable checkout; its later
delete/restore check was not accepted as retrospective repair. Production was
removed while the corrected tests and frozen rulings were retained. A fresh
implementer then observed a valid local `ModuleNotFoundError` with
`PYTHONPATH=src` before independently creating `src/docprune/ctp_controls.py`.

The accepted surface provides aggregate/literal stable score top-M,
Wang-core global uniform sampling without replacement, page-stratified random,
normalized per-page 4x4 grid stratification, exact page-by-cell
coverage-matched identity shuffling, canonical five-field SHA-256 seed
identity, sorted compact visual ordinals, and coverage measurements. This is a
local Qwen/DocPrune adaptation of Wang's random core and named local coverage
controls; it is not decoder integration and does not produce a QA result.

Independent review returned **ACCEPT** with no Critical or Important findings.
Final verification produced `26 passed` focused; the accepted repository suite
produced `553 passed, 2 skipped, 2 deselected`. The two skips are opt-in cached
real-model probes. The two deselections are pre-existing stale-authority tests:
the unmodified full run produced exactly those two failures and otherwise
`553 passed, 2 skipped`. Ruff lint and format checks passed. Explicit no-index
whitespace checks emitted no diagnostics for both untracked files.

Accepted artifact hashes:

- `src/docprune/ctp_controls.py` —
  `916a52be8eacf751c592f9d533ad21be2a5d1314bc4c6208a547f7b0b797967d`
- `tests/test_ctp_controls.py` —
  `cfe80a441eafc9158173f3d658f9731bff147995cd3dda622c418b9c2d4fbd30`
- frozen Task 4 rulings —
  `be230482858c6b98c52ac0e6df2c6fb7b85e06dfb05b03a14ad7a7aace800655`

No retrieval/index, feature, holdout, model, GPU, launcher, or job action was
performed. Task 5 is unstarted and requires a new continuation decision.

## Task 5 CPU integration activation — 2026-08-27

The user activated Task 5 after Task 4 acceptance. Scope is corrected-runtime
CPU code and tests that keep BTP+QTP/no CTP, literal native threshold,
aggregate-logit native threshold, score-top-M, and developmental fixed-retention
policy identities separate and durable. The accepted prior aggregate-logit
primitive will be ported from clean local diagnostic checkout
`/home/lmalveau/DocPrune-ctp-stage64-analysis` at
`441d433c879b8a9b9e523b633530350467616c8f`; that is a best-fit local
reconstruction source, not paper-author code. Task 3's admitted forced record
and legacy trace remain unchanged.

Preflight relevant CPU verification is `198 passed, 2 skipped`; both skips are
the opt-in cached real-model probes. Exact integration rulings are recorded in
the SDD Task 5 ledger. No retrieval/index, feature, holdout, model, GPU,
launcher, job, or experiment action occurred.

## Task 5 CPU integration final acceptance — 2026-08-27

Decision: **PASS — Task 5 complete and independently accepted.** The accepted
runtime has five explicit corrected-policy factories: BTP+QTP/no CTP, literal
native threshold, aggregate-logit native threshold, literal score top-M, and
aggregate score top-M. Native threshold and ranking-only evidence use distinct
policy identities and a separate durable selection record; Task 3's forced
record and five-field `PruningTrace` remain separate.

Both literal and aggregate vectors are derived from the same raw final-prompt
query logits at the first comprehension crossing. Literal uses full-key
float32 softmax, query-head mean, and post-QTP count scaling. Aggregate uses
mean raw query-head logits over visual keys, visual-only float32 softmax, and
the same count scaling. Primary top-M uses the aggregate-native threshold
count; untied aggregate divergence fails closed, while tied IDs and the exact
symmetric difference are serialized. Fixed 55/65/80% arms use exact
`11/20`, `13/20`, and `4/5` half-up budgets.

Policy name, family, selection kind, score semantics, budget source, and fixed
fraction form a closed identity table. Manifests bind that identity before
model construction. Random/coverage context binds canonical experiment
version, repetition, source QID, and optional geometry digest/count to the
unchanged Task 4 five-field seed. Global random is operational; page/grid and
coverage policies fail closed unless truthful post-QTP geometry is supplied.
Fresh and resumed JSONL records are cross-checked against manifest policy,
trace population/budget/layer semantics, no-crossing behavior, native/fixed
budget rules, and forced physical-delete versus zero-mask consequences.

Independent review required three correction rounds and then returned
**Specification ACCEPT** and **Code-quality ACCEPT**, with no Critical,
Important, or Minor findings. Fresh verification evidence:

- Task 5/relevant preflight: `232 passed, 2 skipped`; both skips are the
  opt-in cached real-model probes.
- Full repository run: `594 passed, 2 skipped, 2 failed`; the failures are the
  two pre-existing stale-authority-document checks in
  `tests/test_m3docvqa_launchers.py`.
- Accepted repository run with only those two known nodes deselected:
  `594 passed, 2 skipped, 2 deselected`.
- Ruff lint passed across `src` and `tests`; the 15 Task 5 Python files pass
  Ruff format. Repository-wide format check still reports 22 unrelated
  pre-existing files that would be reformatted. Tracked and explicit
  untracked Task 5 whitespace checks pass.

Core accepted SHA-256 values:

- `src/docprune/ctp_policy.py` —
  `86a63de8f56c43bcc25389bf6de1c457d8bf032a16bd281154b2dd5f26e8da61`
- `src/docprune/ctp.py` —
  `83eaf71c0f4b0bb4587359bad344a54e26b74b12694967ab58e078b41bbb2731`
- `src/docprune/qwen2vl/decoder.py` —
  `e8f1e018a6de24dbc4a37d07d281424d731339de422f8acf5d185035eca82f4a`
- `src/docprune/m3docvqa_factory.py` —
  `7fb2a3b9c9a657d71a69233c51e18b171fe52e50d93f1e952cb4d2613284f57a`
- `tests/test_ctp_policy.py` —
  `da8432720710cc5eed35ed841fe5df02e724222b516f07d204f05025d845797b`
- frozen Task 5 rulings —
  `e9b270f1f9bffd51785c0efa85344536ed8b99e011ac8a567a3c0a63bb478dd5`
- implementation report —
  `18d095032f811f9d71df52de86e063cbaadd80a8d7335b150ea9be20ee69d91e`
- independent review —
  `0c679535fc25a8009516e190c2fbffad1cf074706b734a4ece81655b22f860d1`

No retrieval/index, feature, holdout, model, GPU, launcher, job, or experiment
action occurred. Task 6 is unstarted and requires a new continuation decision.

## Task 6 CPU runtime, fixed-page gate, and smoke acceptance — 2026-08-27

Task 6 CPU implementation is independently accepted for a four-GPU smoke.
The fixed branch no longer constructs `M3DocVQADevDataset`, enumerates the PDF
corpus, or computes the corpus-wide source-order hash. It reads questions and
answers only from the authenticated 64-QID eligible JSONL and authenticates
page-source and feature-shard bytes only for the selected QID. The smoke QID
has exactly four distinct PDFs and four selected document shards. Global
FAISS, embeddings, and token-map loads remain unreachable; the only live query
operation is the pinned ColPali query encoder over the already sealed pages.

The runtime now separates historical `feature_build_runtime_commit` from the
clean `runtime_commit` executing Task 6 and binds the historical feature source
order independently. It records dynamic post-BTP/QTP geometry, cache lengths,
and retained M-RoPE shape/digest per result. The A100-40 role now enforces
`39000 <= memory.total < 50000 MiB`. Conditional native repetitions 10–19 are
an executable 40-cell `native-extension` matrix and remain dormant unless the
development MCSE exceeds `0.25 F1`.

New no-replace scratch artifacts (older versions preserved):

- fixture v2:
  `/scratch/lmalveau/docprune/task6-fixed-page-gate-v1/fixture-stage64-top4-v2.json`,
  SHA-256 `32b3ddd6a1f608db509f002f9541dbc92317b59ac769f0b41fcb20bcc536652b`;
- gate v4:
  `/scratch/lmalveau/docprune/task6-fixed-page-gate-v1/gate-manifest-v4.json`,
  SHA-256 `4d44e297081b152c2e492ec68be58445c7ac17c8d106ceb2ddb1f6331fd57761`;
- bound eligible JSONL SHA-256
  `99d5e45a6ab55c298d8262cd23e996d78babf59cf45a453e3ba837fa40920795`;
- fixed-page reference SHA-256
  `381e19b033fa3e75e16d13bf6f13002bee7db2fc5102a9c6f374268131070642`;
- schema-5 feature manifest SHA-256
  `ffa5979b3bf157adefcc132b0438af295ddafb2243377db4cdb8fcb8eaafe5da`.

Fresh verification after the isolation/provenance corrections produced `619
passed, 2 skipped, 2 deselected`; the skips are opt-in real-model probes and
the deselections are the two known stale-authority checks pending the exact
Task 6 handoff. Ruff, shell syntax, compile checks, and the focused runtime
suite passed. Independent review returned technical **GO** and identified only
the intentional procedural gate: create the clean commit and exact SOL
handoff before submission. No model, GPU, scheduler, retrieval, index, or
feature-build action occurred in this CPU phase.

### Task 6 portability smoke submission — 2026-08-27

The exact sealed handoff
`sol/handoffs/DOCPRUNE_TASK6_PORTABILITY_SMOKE_2026-08-27.md` was submitted
once from clean runtime `b0c8742d358319b7b617b9b1d36ba1f5d1ea6c86` into
fresh root `/scratch/lmalveau/docprune/task6-smoke-b0c8742-v1`. Returned job
IDs, in the exact submission order, are:

- `62277597` — A30;
- `62277598` — A100-40GB;
- `62277599` — H100;
- `62277600` — L40S.

All four use the same sealed one-QID, nine-cell fixed-page smoke. No result was
inspected before recording these IDs. No development matrix, sensitivity,
retrieval, index, feature-build, or holdout job was submitted.

### Task 6 portability smoke failure — 2026-08-27

All four smoke jobs failed before their first answer with the same runtime
interface error: `62277597` A30 in `00:00:24`, `62277598` A100-40GB in
`00:00:24`, `62277599` H100 in `00:00:33`, and `62277600` L40S in
`00:00:24`, each at exit `1:0`. ColPali loaded and encoded the one question;
Qwen did not load and no `results.jsonl` was created.

Root cause is localized to `AuthenticatedFixedPageRetriever.retrieve`.
`_ColPaliQueryAdapter.encode_queries` follows the normal retriever contract and
returns a list containing one `[tokens,128]` tensor per query. The fixed
retriever incorrectly called `torch.as_tensor` on that list instead of
selecting its sole batch element. CPU tests used a synthetic encoder returning
a rank-3 tensor and therefore missed the production interface. Preserve all
four failed roots under
`/scratch/lmalveau/docprune/task6-smoke-b0c8742-v1`; the handoff grants no
retry. The 64-QID and sensitivity matrices remain blocked pending a test-first
fix, clean successor commit, and successor smoke handoff.

## 2026-08-28 — Task 9 smoke admission and 96-mask development preparation

Canonical exact-L40S four-mask smoke job `62315446` completed `0:0` in 31
seconds on `scg022` and passed the outcome-blind validator. It used QID
`e1e6ed53f9ad11813845088f4cf2f6b1`, `B_13`, fit seeds `0..3`, and runtime
`4e2f44c7dde6f6f9b47e2cb3351207adbc9ed306`. Shared-prefix parity maximum
absolute error was `0.0`; projected 96-branch decoder time was
`19.65922513604164` seconds; peak allocated GPU memory was `17765844992`
bytes; Slurm MaxRSS was `2386936 KiB`. Artifact root:
`/scratch/lmalveau/docprune/task9-regional-smoke-4e2f44c-v1/output`.
File hashes were `a034ad922dc83befa77bc79de6c16d340503c6ba7bca9d54ed749a60541a9ea9`,
`53b7610e62a23c705cb790036b3105c32a19fe2a2dda495f3def027be06eda3a`,
and `d4279e2a770218f9f48060db613671b713953b7e77c66742617b14adc80aa1c4`.
This smoke proves parity and cost only; its four outcomes are excluded from
attribution fitting and judgment.

Diagnostic A100-80 job `62315546` was canceled while pending after the L40S
gate passed. Slurm recorded `CANCELLED by 2644339`, elapsed `00:00:00`, with
no node assigned. It consumed no GPU time and is not authorized for
resubmission absent a new portability question.

Successor `e9310cd32067781952afa46f6b25aee4fd7e889c` added exact unpruned
generated-response token capture. It removes only a terminal EOS and appends
the remaining exact IDs to the same teacher-forced mask-scoring call; it does
not decode and retokenize the response.

Clean preparation commit
`7616b29b4dc5ba33584a6e26371281be6886188f` adds the exact ordered 96-mask
runner, dual-target no-replace publication, terminal raw-replay validator, and
post-validation CPU analysis. The exact clean detached runtime is
`/home/lmalveau/DocPrune-task9-development-runtime-7616b29`. The validator
authenticates both target identities, generated IDs/EOS/no-CTP trace, all 96
physical cache records, full-to-compact topology, Qwen M-RoPE shapes, exact
inputs and L40S, and no retrieval/global index. One-question analysis reports
per-question LDS/Spearman, fit-target-mean constant RMSE, and five-refit
stability for both targets. It produces no LDS confidence interval; that
remains deferred to a later sealed multi-question support-component analysis.

Verification evidence: production `110 passed, 12 skipped`; pinned solver
environment `24 passed`; Ruff, changed-file formatting, compile, shell syntax,
and diff checks passed. Exact real-input `--validate-only` authenticated all
96 seeds and created no output. Launcher
`examples/sbatch/39_docprune_task9_regional_development.sbatch` has SHA-256
`963522965d13c530ab9f4a2ecc8c424882919cdbe00025e25f76ad57409753df`.
It requests one exact L40S on HTC, 8 CPUs, 24 GiB, and 10 minutes, with no
array, shard, or requeue. Fresh root
`/scratch/lmalveau/docprune/task9-regional-development-7616b29-v1` was absent.
Exact prepared handoff:
`sol/handoffs/DOCPRUNE_TASK9_REGIONAL_DEVELOPMENT_L40S_2026-08-28.md`.

After explicit user approval, the exact command in the sealed handoff returned
`Submitted batch job 62323129` at `2026-08-28T19:57:28`. The first Slurm check
showed `PENDING (Priority)` with partition `htc`, QOS `public`, one node, 8
CPUs, 24 GiB, one GPU, feature `l40s&public`, a 10-minute limit, and no requeue.
No analysis, retrieval, global-index load, feature build, A100 replacement, or
method-holdout experiment was launched. Do not resubmit or inspect partial
target output; require terminal admission first.

### Task 9 one-question result and reliability decision — 2026-08-31

Job `62323129` completed `0:0` on L40S node `scg027` in `00:01:15` with MaxRSS
`2338552K`. The terminal validator returned
`admitted-task9-regional-development` for all 96 masks. Completion-manifest
file SHA-256 was
`5b2f3bcba250640f6172a9cf513c50cc4fb112019cd497bc56a47445e1c3b71c`;
the primary and secondary target dataset identities were respectively
`f7253f720d3810a5caf16cbf791c7fdf8bb1dc08763afec2aab912bfbda7f59b`
and `d5f14dd3120a041269e21c04e5fc3af1b30ec17b407c1b331dffe0403a642c2b`.

The frozen CPU analysis ran once at requested and achieved budget `M=2689`.
Primary accepted-reference LDS/Spearman was `0.8878299120234603`; held-out RMSE
was `0.24043597646895476` versus fit-target-mean constant RMSE
`1.0162899764963313`. Its five-refit coefficient-Spearman mean/minimum were
`0.39520522916779327`/`0.24705580145431552`, and selection-Jaccard
mean/minimum were `0.7007899670180094`/`0.6363636363636364`.

Secondary generated-response LDS/Spearman was `0.5879765395894427`; held-out
RMSE was `0.9825739081690632` versus constant RMSE `0.9332444565813031`, so
the surrogate did not beat the constant. Its coefficient-Spearman mean/minimum
were `0.4873276608584233`/`0.36699061531116595`, and selection-Jaccard
mean/minimum were `0.7628912547637502`/`0.7111111111111111`.

One secondary bootstrap refit, seed `3`, reached the pinned upstream default
`max_iter=1000` and emitted the sole convergence warning. A fit-by-fit replay
confirmed that the canonical secondary fit and all primary fits converged.
The warning therefore does not explain the secondary constant-baseline failure
or the independently failing primary selection stability.

Analysis path:
`/scratch/lmalveau/docprune/task9-regional-development-7616b29-v1/analysis.json`.
File SHA-256 is
`1dd375f5e52374b17b3efc3b8cad97e50e159e138f90a8270d406641e752de40`;
internal analysis SHA-256 is
`95ead2d334542606bdaa0fac517a851fb163bc84325b7c751048f348e5e5224b`.
All three internal analysis hashes replayed, and the file contains zero
confidence-interval fields.

Decision: the first one-question dual-target reliability prerequisite failed.
Do not seal additional Task 9 development questions, launch a large holdout,
retry the job, or tune a replacement surrogate against this question. A new
pre-outcome scientific decision is required before any further Task 9 work.

### Task 9 paired 256-mask diagnostics — 2026-08-31

Before new outcomes, the user approved two same-question diagnostics with 256
fit masks and 64 unseen masks. Runtime `39aa5f5142cd1ef87291e72f30bcfb23d813195f`
implements variable frozen mask counts and the B13 launcher; exact-L40S job
`62423463` was submitted to fresh root
`/scratch/lmalveau/docprune/task9-b13-256-39aa5f5-v1`. Runtime
`f1d163c8ef3922311f35db4deabd8420621c4e58` adds the paired `B_input`
launcher; job `62423876` was submitted to fresh root
`/scratch/lmalveau/docprune/task9-input-256-f1d163c-v1`. Each requests one
L40S, 8 CPUs, 24 GiB, 10 minutes, and no requeue or shards. At the first joint
check, B13 was running on `scg027` and B_input was pending for priority. Do not
inspect partial outputs; the decision target is the exact generated-response
ContextCite scale reconstructed from terminally admitted raw results.

Input attempt `62423876` failed preflight in six seconds with exit `1:0` and
created no output because its launcher misspelled the pinned M3DocRAG commit.
Successor `8e956be8259e4d72ba1aa7b534649c2f411ef16c` corrects only that character,
adds a regression assertion, and uses fresh root
`/scratch/lmalveau/docprune/task9-input-256-8e956be-v2`.
Corrected successor job `62424211` was submitted and initially queued for
priority with the frozen L40S/8-CPU/24-GiB/10-minute contract. B13 job
`62423463` completed `0:0` in `00:07:23` and terminally admitted all 320 masks;
no attribution metrics were computed during this admission check.

### Task 9 paired diagnostic completion and analyses — 2026-08-31

Corrected input job `62424211` completed `0:0` in `00:04:23` on `scg027` and
terminally admitted all 320 masks. Its completion-manifest file SHA-256 is
`8c2051e559064fcbc47ae317e123aaccb2cb685a3930c39c61b6ea9a6efe752e`.
The B13 completion-manifest file SHA-256 is
`db46773f64c45fdbcbc9109af6f92458a96378ec177e6a3c61c978b61d0dec86`.
The failed input attempt `62423876` remains a preserved six-second preflight
failure with no model load or output; it is not silently replaced.

The canonical generated-response comparison is:

```text
path: /scratch/lmalveau/docprune/task9-paired-256-diagnostics-c49abb5-v2/analysis.json
analysis runtime: c49abb5eac5bd800d75efcca2e4fc571024ef2b7
file SHA-256: cfe9b9456b0040aa9bcee1ce7f332d67401f203a8a53646402a905f009798a18
internal analysis SHA-256: 4d5f27cfd8a069ed8fff976a24b1529780b6f0a9430c5b53949bd9ddf3e40605
B13 LDS / RMSE / constant: 0.7159341 / 0.979053 / 1.340796
B13 selection Jaccard mean / minimum: 0.622318 / 0.546667
B_input LDS / RMSE / constant: 0.6488553 / 0.889320 / 1.196005
B_input selection Jaccard mean / minimum: 0.610084 / 0.530864
```

Both boundaries pass the predictive-fidelity checks and fail the old exact-set
stability check. Moving deletion to `B_input` does not improve the diagnostic,
so B13 remains the controlled-pilot boundary.

The accepted-answer reconstruction uses the same terminally admitted raw rows:

```text
path: /scratch/lmalveau/docprune/task9-paired-256-accepted-answer-0d40fad-v1/analysis.json
analysis runtime: 0d40fadb3de6001b4ab3974c7053443552e1f749
file SHA-256: 5c67c85a0223bbd8ee84d8787a40789ac2f3cf19a4bde8dd706fcde1ad8e8d98
internal analysis SHA-256: 2e695fbb4413392bf8dae633294174881458c4d2fcfe3ced67a3996cdd3e65fc
B13 LDS / RMSE / constant: 0.9148352 / 0.193681 / 0.994279
B13 coefficient Spearman mean / minimum: 0.669753 / 0.574119
B13 selection Jaccard mean / minimum: 0.707792 / 0.650000
B13 nonzero / selected / sources: 44 / 71 / 95
B_input LDS / RMSE / constant: 0.8995421 / 0.219347 / 1.009503
B_input coefficient Spearman mean / minimum: 0.652819 / 0.589037
B_input selection Jaccard mean / minimum: 0.683227 / 0.626506
B_input nonzero / selected / sources: 50 / 68 / 95
solver warnings: none
```

This result validates the accepted-answer intervention function globally on
the one development question. It does not validate a population effect, a
deployable selector, or the exact identity of the large weak-tail keep set.

### Approved Task 9 oracle-pilot amendment — 2026-08-31

The user supplied two external recommendations and explicitly approved an
experiment-level revision before any new question-level ContextCite outcomes.
The source file SHA-256 values are
`ffeaa63b8dbb0f989c31862beec838376176254c438d4d8f77631c16d990141b`
for the oracle-progression review and
`64fac4a15f971390c64b337144a2453be9cfd5efe330eca582056d686c855699`
for the distractor-enriched pilot review. They are advisory inputs; project-
specific user decisions and authenticated prior results control conflicts.

Accepted changes:

1. Treat the accepted-answer B13 procedure as a privileged causal-selection
   reference oracle and bypass the old minimum `0.8` support-identity Jaccard.
   Preserve Jaccard descriptively and evaluate actual budgeted-set outcomes.
2. Keep physical deletion, B13, original positions, whole-region actions, exact
   region-cost knapsack, and 256 fit masks. Add 32 global and 32 primary-budget-
   local holdouts, five deterministic 80% refits, direct selected-set regret,
   and nested 64/128/192/256 calibration on a sealed subset.
3. Seal a 48-question panel before ContextCite scoring: 16 uniformly sampled
   eligible questions, 16 traceable distractor errors, 8 high-ambiguity correct
   questions, and 8 clean controls. Sampling within frozen pools is random;
   attention and ContextCite outcomes cannot define eligibility.
4. Compare query-only attention-region, privilege-matched gold-answer attention-
   region, robust and canonical accepted-answer ContextCite-region, reverse
   ContextCite, unpruned, and a traceable-only audited gold-in/distractor-out
   diagnostic at matched 55%, 65%, and 80% budgets. Generated-answer F1 is the
   primary outcome; rescue, harm, gold-vs-wrong likelihood margin, and region
   retain/remove states diagnose distractor removal.
5. Keep mechanism and population estimands separate. The enriched pilot cannot
   estimate population prevalence. The completed 1,213-question Task 6
   random-versus-DocPrune run stays separate and will not be rerun in Task 9.

Rejected as redundant or out of scope for this pilot: FastV, a new uniform-
random pruning arm, and a new coverage-matched-random pruning arm. The uniform
anchor in item 3 is a sampling stratum, not a pruning comparator. No pilot GPU
job is authorized until the consolidated implementation passes a bounded smoke
and a fresh exact handoff is reviewed.

This amendment supersedes only the 2026-08-26 prohibition on the phrase
`oracle gap`: `answer-conditioned causal-selection reference oracle` and
`oracle headroom` are now allowed when the answer privilege and nondeployable
claim boundary appear with the result. An unqualified `oracle`, token oracle,
or deployable-method claim remains prohibited.

### Task 6–9 branch consolidation — 2026-08-31

All local Task 6–9 branches were compared by patch identity before integration.
The active branch `codex/task9-analysis-cpu-20260828` now contains every unique
accepted change through Task 6 merge `689432e`, Task 7 merge `c5b212f`, and
Task 8 merge `c38cd9c`; the remaining Task 9 branches were already patch-
equivalent. Historical branches and worktrees remain preserved.

Focused post-merge verification passed: 83 Task 7 tests, 99 Task 8 tests, and
the consolidated Task 6–9 suite with 322 passes and 19 expected dependency/
opt-in skips. Injecting the pinned solver made the Task 9 solver-dependent
subset pass 30/30. Targeted Ruff, shell-syntax, and compile checks also passed.
A repository-wide sweep exposed two stale benchmark-documentation assertions
that incorrectly treated changing current-task files as benchmark authority;
their scope was narrowed to the immutable benchmark documents, and both focused
regressions pass. No experiment was launched during consolidation.

### Approved preliminary stratified-random Task 9 pilot and cohort seal — 2026-08-31

Before the enriched mechanism panel, the user approved a simpler 48-question
discovery/calibration pilot: 24 baseline-correct and 24 baseline-wrong questions
sampled randomly within stratum. The exact eligible authority is the completed
245-question BTP+QTP/no-CTP stage-localization pool because it matches Task 9's
unpruned post-QTP state. Canonical list EM partitions that pool into 90 correct
and 155 wrong questions.

The pilot compares unpruned, query-only DocPrune attention-region, accepted-
answer gold-support ContextCite, conditional gold-margin ContextCite when the
unpruned response is a distinct non-gold alternative, and one deterministic
region-size-aware random comparator at matched 55/65/80% achieved whole-region
costs. FastV remains excluded. The matched regional random arm is a local same-
action-space comparator; the 1,213-question Task 6 random study is not rerun.
Correct and wrong strata are primary reports. Any pooled descriptive estimate
uses the frozen 90/245 and 155/245 weights; uncertainty across budgets clusters
by question.

CPU selector tests were written before implementation and pass 3/3. The cohort
was sealed without model execution or attribution outcomes using seed
`docprune-task9-preliminary-random48-v1`:

```text
path: /scratch/lmalveau/docprune/task9-preliminary-random48-v1/cohort.json
selected: 48 unique QIDs; 24 baseline correct; 24 baseline wrong
eligible: 245 QIDs; 90 baseline correct; 155 baseline wrong
source result files: 14 authenticated BTP+QTP JSONL members
file SHA-256: 123607a6a1226b4e3436f43cb82d45e64e8a3008e9ab6deefd7526efeabd0273
internal cohort SHA-256: 465fcf6e8e0adee6e79845db8cb1d6f1fbc97e01d02b7cf5c1e28c7c33c3c5f9
```

No pilot job is submitted. Next action is implementation of the sealed
preliminary cohort's regional mappings, masks, matched arms, and unified
analysis, followed by a bounded smoke and a new executable handoff.

### Task 9 preliminary preprocessing GPU-constraint correction — 2026-08-31

The initial one-question MinerU and geometry smoke jobs `62441167` and
`62441168` requested exact L40S GPUs. HTC left both pending at zero elapsed
time and estimated L40S availability around 2026-09-07. The user rejected that
hardware-name restriction because neither deterministic MinerU extraction nor
post-BTP+QTP geometry identity scientifically requires an L40S. Both untouched
jobs were cancelled before using GPU time.

Replacement smoke jobs `62444442` (MinerU) and `62444443` (geometry) request
the first available single CUDA GPU and retain the actual requirements: pinned
offline model/tool revisions, one visible CUDA device, at least 12 GiB device
memory, authenticated fixed pages, no retrieval, and output GPU identity. The
Task 9 mapping path accepts any authenticated CUDA GPU; the historical Task 8
mapping default continues to require L40S, preserving its completed contract.
Absolute runtime is not compared across GPU families. These jobs are bounded
preprocessing smokes, not the 48-question pilot.

The remaining 47 preprocessing questions are not launched as 47 independent
array tasks. Their frozen execution shape is 12 batch jobs: four questions per
job and three in the final job, with per-question authenticated output roots
inside each batch. MinerU batches request 90 minutes; geometry batches request
30 minutes. Both use at most six concurrent batch jobs and no GPU-model-name
constraint.

The first batched geometry attempt (`62445270`) showed that the Task 6
post-QTP count is not an exact replay identity for this new capture: for probe
QID `9b17c72e6c59db83dcaeeba6be8417e2`, Task 6 recorded 5,669 tokens while a
current A100 MIG capture produced 5,674. This is not a GPU-name effect, as both
A100 and L40S tasks included matches and mismatches. The preliminary contract
therefore keeps the authenticated Task 6 row as the sampling-stratum and
fixed-page reference, but freezes the newly captured BTP/QTP geometry as the
common action space for every pilot arm. Original and post-BTP counts must
still match the reference; exact Task 8 replay behavior remains unchanged.

### Task 9 native-comparator correction and pilot launch — 2026-09-01

The user approved replacing the fixed-B13 regional-attention comparison with a
fair native DocPrune comparison. For each question, native aggregate-threshold
DocPrune now runs first and independently selects its crossing layer `l*_q`,
exact retained count `M_q`, and retained-token IDs. That layer is frozen for
ContextCite and regional random. Their whole-region knapsack uses the closest
attainable common cost `M′_q <= M_q`; all pruned arms use the same physical-
deletion implementation and original M-RoPE positions. The 256 fit masks, 32
global holdouts, and 32 holdouts local to the native budget remain unchanged.
FastV, new uniform-random, and coverage-matched-random arms remain excluded.

Fixed-B13 array `62463802` was cancelled after this correction was approved.
Its completed or partial outputs, together with completed batch-0 job
`62463676`, remain preserved as secondary fixed-layer diagnostics and are not
mixed into the corrected pilot.

Implementation commit:
`2a66d79c105d28ba4ddcd53b9a3015b5db624b67`. Focused verification passed 37
attribution/intervention tests with the pinned ContextCite solver, Ruff, Python
compilation, and the batch-0 no-model input/mapping validation. The runtime was
clean at submission.

Corrected HTC array `62463982` was submitted as 12 four-question batches with
up to 12 concurrent tasks, first-available supported CUDA GPU including A30,
24 GiB host RAM, 75-minute limits, and no requeue. Initial scheduler state had
batches 0–3 running and 4–11 pending for resources. Fresh artifact root:
`/scratch/lmalveau/docprune/task9-preliminary-dynamic48-2a66d79-v1`. Do not
interpret partial scientific outcomes; wait for all batches, validate all 48
question artifacts, and then emit the unified analysis JSON.

### Task 9 preliminary dynamic-48 completion and unified analysis — 2026-09-01

Array `62463982` did not produce admissible final outcomes: its first four tasks
reached a validator incompatibility and later tasks encountered the temporary
dirty-runtime guard. The raw root is preserved at
`/scratch/lmalveau/docprune/task9-preliminary-dynamic48-2a66d79-v1` and is not
mixed into the result. Validator fix commit
`90f27d7ed8b99ad10f1a5fe405c131127456ae5d` admitted the native dynamic
selection schema. Replacement array `62464099` ran the 12 four-question
batches at
`/scratch/lmalveau/docprune/task9-preliminary-dynamic48-90f27d7-v1`.
Batches 5–7 timed out after two questions each without code errors or OOMs.
Targeted retry array `62466113` ran only missing ordinals 22, 23, 26, 27, 30,
and 31 as one-question jobs with the original 75-minute limit; all completed at
`/scratch/lmalveau/docprune/task9-preliminary-dynamic48-90f27d7-retry-v1`.

The final verifier admitted exactly the sealed 48 unique QIDs across those two
roots and checked cohort, selected-arm, analysis, per-question-analysis, and
individual arm-result signatures. Canonical unified output:

```text
path: /scratch/lmalveau/docprune/task9-preliminary-dynamic48-90f27d7-unified-v1/analysis.json
internal SHA-256: 985a837b2094b5a925730eeb4b9bf07c594344f9021c4a718c49c11aa655a8c5
bootstrap: 10,000 within-stratum question resamples, seed 20260901
```

Primary gold-support ContextCite results versus native dynamic DocPrune:

- baseline-correct (`n=24`): both mean token-F1 and EM were `1.0`; paired
  F1/EM delta `0`, win/tie/loss `0/24/0`, rescue/harm `0/0`;
- baseline-wrong (`n=24`): ContextCite mean F1 `0.38125` versus DocPrune
  `0.13750` (paired delta `+0.24375`), EM `4/24` versus `0/24`,
  win/tie/loss `9/15/0`, rescues/harms `4/0`;
- balanced descriptive delta: F1 `+0.121875` (question-bootstrap 95% interval
  `[0.051042, 0.203125]`) and EM `+0.083333` (`[0.020833, 0.166667]`);
- natural-pool reweighted descriptive delta using 90/245 correct and 155/245
  wrong: F1 `+0.154209` (`[0.064583, 0.257015]`) and EM `+0.105442`
  (`[0.026361, 0.210884]`).

Against the matched region-size-aware random control, balanced paired F1 delta
was `+0.155208` and natural-reweighted delta `+0.174277`. On baseline-wrong
questions, gold-margin ContextCite achieved mean F1 `0.36875` and EM `5/24`,
versus gold-support F1 `0.38125` and EM `4/24`; it remains a conditional
diagnostic, not a full-cohort arm.

Surrogate diagnostics are supportive but heterogeneous: global LDS mean/min
`0.775738`/`0.308651`, with 44/48 at least `0.5` and RMSE beating constant on
48/48; native-budget-local LDS mean/min `0.622393`/`-0.103006`, with 35/48 at
least `0.5` and RMSE beating constant on 45/48. Therefore the direct generated
outcomes are primary and question-level local-fidelity warnings remain in the
unified JSON. This is development-pilot evidence for an answer-conditioned
regional surrogate, not vanilla ContextCite, a token oracle, or a held-out
population estimate. The enriched developmental panel is next; no method
holdout is authorized.

### Approved preliminary-48 mask-count ablation — 2026-09-01

The user approved making a reuse-only mask-count ablation the immediate next
Task 9 phase before enriched-panel sealing. Across all 48 completed preliminary
questions, reuse the already-scored 256 fitting masks and the same 32 global
plus 32 native-budget-local holdouts. For each `N` in `64, 96, 128, 192`, fit
five deterministic independently sampled without-replacement subsets; compare
with the canonical 256 fit. The CPU analysis reports held-out fidelity,
coefficient agreement, selected-region/token agreement, and seals unique
reduced-mask selections. It performs no new model inference.

A later bounded GPU step compares those sealed selections with the existing
canonical-256 and native DocPrune responses. The required functional checks are
the four canonical rescues, baseline-correct preservation, wrong-stratum mean
F1, selected-context accepted-answer likelihood, and win/tie/loss. A reduced
count may be adopted only under a decision rule frozen after CPU analysis and
before reduced-mask generation outcomes are read; 64 is not adopted merely for
cost. No method holdout is authorized.
A later bounded GPU step compares those sealed selections with the existing
canonical-256 and native DocPrune responses. The required functional checks are
the four canonical rescues, baseline-correct preservation, wrong-stratum mean
F1, selected-context accepted-answer likelihood, and win/tie/loss. A reduced
count may be adopted only under a decision rule frozen after CPU analysis and
before reduced-mask generation outcomes are read; 64 is not adopted merely for
cost. No method holdout is authorized.

### Preliminary-48 mask-count CPU result and frozen GPU gate — 2026-09-01

CPU-only array `62471599` ran 12 four-question batches at clean implementation
commit `d6de9d8f2e2357ec13d6e674eddc7dc0e9a22cb5`, 4 CPUs and 8 GiB per task.
All 12 tasks completed `0:0` in 21–37 seconds. The verifier admitted all 48
questions and 1,008 fits. Unified result:

```text
path: /scratch/lmalveau/docprune/task9-mask-count-ablation-d6de9d8-v1/analysis.json
internal SHA-256: 9d6a9ac6ece6712ca9cdad33bba0bc61caf122121435c3d99a6fb40e460edc51
```

Across 240 repeated fits per reduced count, mean/median token-cost-weighted
selection Jaccard versus canonical 256 was `0.78595/0.79150` at 64,
`0.81254/0.84820` at 96, `0.84224/0.86602` at 128, and
`0.89808/0.95053` at 192. Exact region-set reproduction was respectively 14,
18, 22, and 32 of 240 fits. Mean global LDS rose from `0.71564` at 64 to
`0.77777` at 192 versus canonical `0.77574`; mean budget-local LDS rose from
`0.57021` to `0.61665` versus canonical `0.62239`. The CPU result does not
establish response equivalence: even 192 changes most exact regional sets,
including appreciable variation on two of the four canonical rescue questions.

The sealed inventory contains 205 unique 192-mask selections: 13 exactly reuse
canonical-256 sets and 192 require new generation. Stage 192 first. Repeat 0
must preserve all 4 rescues and all 24 baseline-correct exact answers, with
wrong-stratum mean F1 no more than `0.02` below canonical and mean gold
likelihood no more than `0.05` nats/token below canonical. Across five repeats,
preserve at least 18/20 rescue opportunities and 118/120 correct-stratum
answers and satisfy the same average F1/likelihood tolerances. If 192 fails,
retain 256; if it passes, evaluate 128 under the same frozen rule. These gates
were recorded before reduced-mask GPU outcomes were read.

### Task 9 H200 baseline-wrong confirmation preparation — 2026-09-02

The user approved a second experiment on 100 new ordinary baseline-wrong
questions, separate from the preliminary 48. The frozen 192-mask functional
comparison did not justify reducing the estimator, so this experiment retains
256 fitting masks. It removes the 32 global and 32 native-budget-local holdouts
because downstream generated-answer outcomes—not surrogate LDS—are the
confirmation endpoint.

The cohort sealer joined the authenticated Task 6 holdout result rows with its
sealed support-document records, explicitly excluded the preliminary-48 QIDs,
and preferred one question per support-document component. Sealed local output:

```text
path: /home/lmalveau/task9-h200-artifacts/task9-baseline-wrong100-confirmation-v1/cohort.json
internal cohort SHA-256: 0bdfdde29b568f54ea541453abeca196cfb49f4b6630f084f80e6c21f158514e
eligible baseline-wrong: 731
eligible support components: 724
selected: 100
document fallback used: false
retrieval run: false
```

Because 724 independent eligible components were available, all 100 selected
questions are support-component independent and no fallback fill was needed.
The source cached-results file contains 2,441 rows; the sealer restricts it to
the exact 1,213 sealed Task 6 holdout QIDs before baseline classification. No
H200 survey, environment mutation, model load, smoke, or GPU job occurred
during this preparation.

### Task 9 source/H200 execution-boundary correction — 2026-09-02

Before any confirmation outcomes were produced, the user corrected the
operational division of work. The source/SOL computer must seal the 100 fixed
inputs, run pinned MinerU, capture frozen post-BTP/QTP geometry, build and
model-free validate exactly 100 mappings, and package the authenticated bundle.
The H200 agent must not recreate those artifacts or install MinerU; it only
surveys and configures the host, receives and authenticates the completed
bundle, performs CPU-only transferred-input validation, then runs smoke,
production, retries if required, and aggregation. This changes packaging and
execution ownership only; the cohort, 256/0 mask design, arms, model, decoding,
and analysis contract are unchanged.

### Task 9 MinerU reuse inventory — 2026-09-02

Before fixed-input sealing or preprocessing, the user approved retaining the
already sealed random 100-question cohort rather than selecting on artifact
availability. Across the 731 eligible new baseline-wrong questions, the 48
authenticated preliminary-pilot MinerU completions cover 192 unique pages, but
zero eligible questions have all four pages covered: 713 have zero covered
pages, 17 have one, one has two, and none has three or four. The sealed random
100 contains one reusable page slot and 99 questions with no page overlap.

Source preprocessing is therefore reuse-first by exact page and rendered-byte
identity, but cohort membership remains unchanged and random. MinerU may run
only for missing unique pages; no retrieval or global index access is allowed.

The first CPU fixed-input sealing attempt at runtime `7be2137` failed closed
before publication because the reused Task 6 fixture builder required the
question cohort itself to be outcome-blind. Baseline-wrong selection is
intentionally outcome-stratified, while its already cached top-4 page selection
remains outcome-blind. The builder now retains its strict default but accepts an
explicit confirmation-only override only when
`fixed_page_selection_is_outcome_blind` is true. The failed output root was
removed automatically; no fixture, retrieval, model, or GPU outcome was
published. Focused old/new fixture tests pass.

The corrected clean runtime `c0c9beef76f327ff80f73bb971a8bc772c2a95ec`
sealed the fixed inputs in CPU-only SOL job `62497332` (completed in `00:05:36`,
peak RSS `309460K`, exit `0:0`). Canonical artifact root:

```text
/scratch/lmalveau/docprune/task9-baseline-wrong100-inputs-c0c9bee-v1
preprocessing-manifest SHA-256: 037b5479f5c05c3aede984f8cdcfd0e63935031a7e30ac99960b56b11917d61a
fixture SHA-256: 9b54c41787c0ccc34205c8e3fc23ca50701b93dc1c74289e64b57740c3ed9a71
selected-source-results SHA-256: e32613223575949c8822b3fbc1ab057e2c546ba31e65fd9e6be6854f085eac0c
questions: 100
page slots: 400
size: 170M
retrieval_run: false
global_index_loaded: false
```

The two earlier interactive attempts were externally terminated before terminal
publication; their partial roots were automatically removed and are
non-canonical. No GPU or model execution occurred during fixed-input sealing.
