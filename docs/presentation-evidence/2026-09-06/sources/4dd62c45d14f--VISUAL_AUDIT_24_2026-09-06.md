# Visual and quantitative audit of the development pilot

6 September 2026 · frozen Qwen2-VL-7B-Instruct regional ContextCite/DocPrune track

The saved numerical advantage is reproducible, but it overstates the number of document-supported corrections. Gold-support ContextCite has four exact rescues among 24 baseline-wrong questions: **one is visibly supported by the retrieved table (Q29), two cannot be verified from the supplied evidence (Q36/Q37), and one reproduces stale gold contradicted by the supplied pages (Q43)**. The audit supports investigating content competition and coherent evidence preservation, while leaving their causal contribution and predecoder transfer unresolved.

Deliverables: [browsable gallery](/home/lmalveau/DocPrune/outputs/regional-audit-2026-09-06/index.html), [30-row case table](/home/lmalveau/DocPrune/outputs/regional-audit-2026-09-06/cases.csv), [full case JSON](/home/lmalveau/DocPrune/outputs/regional-audit-2026-09-06/cases.json), [144 arm measurements with token IDs](/home/lmalveau/DocPrune/outputs/regional-audit-2026-09-06/arm-measurements.csv), and [cohort identities and overlaps](/home/lmalveau/DocPrune/outputs/regional-audit-2026-09-06/cohort-audit.json).

## Scope and inspection integrity

All 24 baseline-wrong cases, ordinals 24–47, and six baseline-correct controls, 00/02/03/04/08/18, were audited. Controls were chosen from questions and layouts before pruning outcomes: horse history/infobox, cartoon graphic, movie poster/filmography, numerical judge tables, geographic prose/map, and sports biography/portrait. These are layout varieties within Wikipedia-derived documents, not six independently sampled document genres. The complete 24-correct aggregate was read for reconciliation; only the six controls received visual case audits.

For every audited case, the question and all four saved page images were inspected before case-specific gold, generations, likelihoods or masks. The known baseline stratum and published aggregate results were not blinded. [Initial notes](/home/lmalveau/DocPrune/outputs/regional-audit-2026-09-06/initial-observations.json) were written before the outcome extraction and are retained separately, with their SHA in each case. First-pass full-page contact sheets were downscaled by the display; later **native-resolution detail and overlay inspection was unblinded**. This distinction matters: the native-detail interpretation is not an independent blinded adjudication.

All 120 page instances have original rasters, reconstructed processor-size views, MinerU PDF renders, parser/region overlays, and actual retained-token footprints for every saved arm. Original pages were 1224×1584; processor geometry is 1232×1596 on all 120. Bicubic resizing follows the pinned preprocessing code, but these are raster reconstructions, not a recovered normalized tensor. Native details use source pixels; no conclusion depends on extra high-resolution-only evidence. No extra crops entered the answerer.

Footprints are rendered CPU-side from saved selected IDs, mapping geometry and MinerU page PDFs. A footprint cell is one merged visual-token location (28×28 input pixels), **not a whole parser box**. Green is retained, red is deleted at the decoder boundary, and gray is already absent after BTP/QTP. Gray cannot distinguish the two upstream stages spatially. Pink boxes identify MinerU sources, ochre residual bins, blue empty parser regions. These overlays illustrate token deletion; they are not literal edited images fed to cached continuation. Every case exposes exact source IDs, token IDs, page order, PDF paths and hashes.

No training, production inference, GPU work, retrieval search or new mask sweep was launched. The 600 preparation was not modified. Only identity/split/retrieved-page metadata from the 100 and 600 collections was used; their answers, pruning results and confirmation outcomes were not used for tuning.

## Reconciled saved measurements

| Arm | Wrong cohort mean F1 (n=24) | Exact answers /24 | Correct cohort exact /24 |
|---|---:|---:|---:|
| Unpruned, post-BTP/QTP, no CTP | 0.165833 | 0 | 24 |
| Native DocPrune | 0.137500 | 0 | 24 |
| Gold-support ContextCite | 0.381250 | 4 | 24 |
| Gold-versus-fixed-self margin | 0.368750 | 5 | Not a preservation arm |
| Region-size-aware random | 0.154167 | 0 | 22 |

Gold support versus native has **9/24 F1 wins, 15/24 ties, 0/24 losses**. Versus unpruned it has **7/24 wins, 17/24 ties, 0/24 losses**. Q31 and Q38 recover damage introduced by native pruning; neither improves on unpruned. All six visually audited controls remain exact under support, native and random; this does not contradict the two random failures elsewhere in the 24-correct cohort. These are development measurements, without a new generalization claim or confidence interval based on treating masks as independent questions.

All pruning arms in the 30 audited cases exactly match native DocPrune’s **achieved token count**. Among the wrong cases these budgets range from 1,990 to 4,011 visual tokens, representing 49.19–85.85% of the post-QTP population. Equal region counts would not be equivalent. Native DocPrune remains the main comparator; random is additional. Gold support selects whole nonempty mapped sources in these cases, whereas native footprints often remove scattered tokens within a source.

The full machine-readable table records each original and pruned generation, F1/EM, raw G/S, G−S, ΔG, ΔS, Δ(G−S), both F1 comparisons, retained/deleted IDs and budgets. No answer or historical score was silently changed.

### Likelihood interpretation and evaluation defects

G is implemented as `max-accepted-reference-mean-loglikelihood`, not always the likelihood of a complete answer. In **5/24 cases** (Q30/Q34/Q41/Q46/Q47), a question requests a set but its gold list members are scored as alternative sequences and maximized. Q30 has 12 reference sequences. This rewards one rural gmina rather than enumeration of all rural gminas. Q41 returns both correct 2015 roles before pruning, yet omitting Tommy raises F1 from .4 to .5; the scorer treats the generated comma-separated string as one prediction span against two gold spans. These are target/evaluation-contract findings, not a decision to rewrite the benchmark.

Under the support-selected masks, G increases while fixed-self S decreases in **16/24** cases. This does not guarantee a correct generation: Q42 has ΔG=+2.37571 and ΔS=−2.65757, but answers Ben Moore instead of Audra McDonald. Under margin-selected masks, **5/24** cases (Q26/Q31/Q32/Q33/Q34) have a positive margin change while both sequences become less likely. For Q26, G goes −0.92045→−1.05457 and S goes −0.90859→−1.59490: Δmargin=+0.55220 while gold support declines. Q29 instead shows the desired support pattern: G −0.40038→−0.07282, S −0.12803→−0.41965 under gold support.

S is always the saved original unpruned generation, including its actual wording, not the generation produced by each mask. Deltas use the same case’s no-CTP full reference **after BTP/QTP**, not the initial 10,032-token raster population. Where gold and self are exactly the same token sequence, G−S is identically zero and cannot measure preservation. Semantically equivalent strings with different tokenizations need not have equal scores. Correct cases have no margin-selection arm here; their preservation is assessed using G and generations.

Additional evaluation concerns are visible without rerunning a model: Q25’s Calgary and Lillehammer satisfy an open host-city question despite gold St. Moritz; Q26’s Mustache is a valid spelling of moustache, whereas the support answer beard is visually worse; Q27’s margin-only rescue changes Smiling to Smile; Q39 inherits Mamma from the question rather than document Mama; Q40’s Kush is a valid short form of Kingdom of Kush. Q43’s 2024 is supported by the supplied document, while gold 1981 is stale. The report flags these while preserving all saved scores.

## Visual findings, mechanisms and counterevidence

These are overlapping audit categories, not estimated population frequencies or proven causes. [Case membership](/home/lmalveau/DocPrune/outputs/regional-audit-2026-09-06/audit-counts.json) makes each denominator and label inspectable.

| Visual/evaluation category | Wrong cases /24 | Interpretation |
|---|---:|---|
| Required evidence visibly present | 10 | Presence does not guarantee complete generation |
| Required evidence missing | 7 | Requested image, table or fact absent from supplied pages |
| Partial/indirect evidence | 6 | At least one relation or qualifier insufficiently grounded |
| Visual source/pose uncertain | 1 | Q45, stylized hands/image identity |
| Definite retrieval or missing-edge concern | 12 | Includes six partial cases except Q31, which has indirect date references |
| Recognizable competing content | 15 | Candidate hypothesis, not individually harmful-region identification |
| Relation binding or set reasoning | 14 | Entity/year/column/role/condition needs linking |
| Mixed indispensable/competing source | 7 | Large tables or lists include several roles/candidates |
| Explicit header/continuation dependency candidate | 2 | Q29 and Q38; broader relationships also occur elsewhere |
| Redundancy candidate | 2 | Q31 and Q40; plus controls Q00/Q08/Q18 |
| Evaluation artifact or granularity concern | 11 | Includes ambiguity, spelling, stale gold and set scoring |
| Multi-item G target mismatch | 5 | Distinct from a visual segmentation defect |

**Competition is recognizable in informative cases, but its removal is not sufficient.** Q37 deletes an Abbey Road paragraph about McCartney holding a cigarette in his right hand (R021, 383 tokens); the answer changes Cigarette→Gun, but the Black Hawk Down poster is absent. Q36 deletes MO Béjaïa’s 2015 history (R018, 230 tokens) and changes 2015→2010, but the Titans’ 17,877 attendance evidence is absent. Q44 deletes Isla Fisher’s Hollywood relocation (R020, 342 tokens), yet the retained New York reference concerns a film release, not Matthew Newton’s relocation. Q42 removes an Alan Menken passage but fails to choose the retained collaborator Audra McDonald. These distinguish wrong-entity suppression from completing the required relation.

**Q29 is the clearest real correction and the best depth/dependency case.** Gold support retains the rank-1 Fast/Del Mar race table R011 (601 tokens) and returns Seven Furlongs. It deletes rank-0 table R010 (370), containing a competing Del Mar/Firm row **and the headers**. Other Fast rows with different distances remain in R011. Thus coherent retention helps relative to scattered native deletion, but the saved mask neither isolates the competing row nor proves headers were dispensable before decoding. The prefix already saw them.

**Redundancy and complementarity are visible but not causally identified.** Q38 keeps both table headers R024 (357 tokens) and continuation R025 (770), restoring the unpruned response while still confusing entrant/driver initials. Q40 keeps Kawa support R015 (298) and overlapping architecture support R021 (280); either could suffice, or their roles could differ. Control Q18 deletes all nonempty image sources yet answers rugby from a biography/identity cue; the supplied portrait is an interview, not a playing action. Only joint and individual deletions can discriminate interchangeability from complementary evidence. No pairwise synergy or redundant sufficient set is certified by this audit.

**Segmentation and upstream loss matter before the 600 segmentation is finalized.** All 30 mappings form an exclusive, exhaustive partition of surviving compact tokens; no duplicate or missing compact-token owner was found. Box overlap alone is not an ownership error. There are 28 parser regions with no surviving owned tokens across 10/30 cases (10 empty regions in 6/24 wrong cases); empty means unavailable as a selector source, not necessarily a parser omission. Original population is 10,032 tokens per case; wrong-case BTP losses are 3,607–6,701, QTP losses 20–2,481, leaving 2,865–6,178. Tables and text have upstream holes even when their source is fully retained. The saved geometry does not expose separate BTP/QTP spatial footprints, so the audit cannot assign each missing cell to a stage or prove a particular lost word caused failure.

Large table regions mix target rows, competing rows, headers and units (Q29/Q30/Q34/Q38/Q41/Q47); overlapping residual bins also make rectangles misleading. Q30 and Q41 retain their entire needed table source but fail at enumeration/completeness. Q26 discards the correct portrait rather than failing to segment it. These observations motivate auditing row/header links and source-image identity, **not an unrequested boundary rewrite**. A later selector cannot restore removed visual tokens. Any revised segmentation changes the mask intervention; old labels cannot automatically be reused.

## Surrogate evidence and its limits

Each wrong case has 256 unique fit masks over 29–103 nonempty sources, plus 32 global and 32 budget-local held-out masks. Per-source inclusion counts span 102–148 across the banks; the smallest joint cell across region pairs is 34–44 per case. This verifies coverage of all four pair states, but those states occur with many other regions changing: they are not matched pair interventions.

Gold-support local RMSE beats the fit-mean constant in 24/24 wrong cases, yet local Spearman/LDS is −.103 for Q30, .083 for Q38 and .111 for Q47. Five bootstrap refits give a per-case mean pairwise selection Jaccard of .479–.854, median .640; it is source-set agreement, not a confidence interval for causal effects or answer accuracy. A low coefficient does not establish semantic irrelevance. A joint successful mask can combine helpful, harmless and harmful deletions. An additive surrogate does not certify interactions or a sufficient set, and weak local ranking cannot establish absence of useful selection headroom. The existing development mask-count ablation supports retaining **256 masks per reliable oracle refit**; fewer masks over more training questions is a separate shared-learning hypothesis.

ContextCite motivates checking held-out surrogate faithfulness and cautions about redundancy and nonlinear response behavior; a pruning intervention is an empirical behavior change, not a proof that every removed source was misleading. This project adapts its method to answer-conditioned visual regions and cached deletion. [ContextCite](https://arxiv.org/html/2409.00729)

SPEX models interactions using a sparse Fourier representation and a purpose-built recovery procedure. That precedent supports asking interaction questions; it does not establish that our Bernoulli bank, fitted additively, identifies them. [SPEX](https://arxiv.org/html/2502.13870)

## Actual decoder boundary and input contract

The wrong cases use **B_14 in 16/24 and B_16 in 8/24**. Across all 48: B_14=34, B_15=1, B_16=13. They are not uniformly B13. The implementation defines B_K as **after zero-based block K**, so B_14 follows 15 completed decoder blocks; B_input is before the first block, and B_0 is after it. Every case’s boundary is recorded in the case table.

Pinned prompt construction places all four images in retrieval order, then `question: …\noutput only answer.`, followed by the assistant generation prompt. Text positions remain; only selected visual positions are physically removed. The processor uses merge-safe 2×2 patch groups. Compact sequences preserve original multimodal rotary positions and original logical token identities rather than renumbering the retained page into a new dense image. Run manifests record prompt IDs/hash/shape, and selected-arm records contain actual logical IDs, positional hashes and cache-length vectors.

At an intermediate boundary, surviving visual and question/text states have already processed the full post-QTP visual context. Earlier-layer KV caches are cloned and retain full prefix lengths; later layers run on the reduced sequence. During later generation those earlier caches can still carry deleted-token information. Consequently even a perfect late retained set does not prove sufficiency at B_input. The shared vision encoder also mixes spatial information before either decoder-boundary choice; B_input removes decoder processing of deleted tokens, not all prior cross-patch information flow.

Existing [forced-boundary parity evidence](/scratch/lmalveau/docprune/qwen-forced-boundary-parity-36cb771-v1/result/forced-boundary-parity.json) passes exact generated suffix, logical order, positional identity, cache-vector and configured logit-tolerance checks at B_input/B_0/B_6/B_13/B_20/B_23/B_26 on its fixture. Maximum absolute logit differences are .125, so this is not bitwise logit equality. This is a separate earlier fixture, not case-by-case all-keep replay at the pilot’s B_14/B_16. No saved per-case parity panel at those actual boundaries was located. That exact evidence is missing; no rerun was launched.

The runtime versions of `qwen2vl/decoder.py`, `sequence.py`, `preprocessing.py` and `vision.py` are byte-identical to the checked-out files. [Decoder capture/resume](/home/lmalveau/DocPrune/src/docprune/qwen2vl/decoder.py), [prompt construction](/home/lmalveau/DocPrune/src/docprune/answerers.py:110), [target construction](/home/lmalveau/DocPrune/src/docprune/task9_attribution.py), and [sparse vision/merger](/home/lmalveau/DocPrune/src/docprune/qwen2vl/vision.py) provide implementation evidence. Native DocPrune’s BTP, QTP and decoder CTP have distinct roles; this implementation’s adapted regional oracle is not a replacement definition of the paper’s baseline. [DocPrune](https://arxiv.org/html/2604.22281)

## Cohort roles and independence

The owner’s current operational update is **600 questions transferred to H200, segmentation pending**. This audit accepts that status and does not revive an older staging count or recast this as an uncertainty diagnostic. The available sealed SOL transfer bundle has 600 questions/2,400 page instances; its cohort and split-file hashes match the local files. Its intended use is shared-probe training/development, while the 100 baseline-wrong cohort is confirmation.

There is a concrete manifest/design discrepancy. The available `task9-shared-probe-random600-v1/splits.json` contains 360 balanced train, 120 balanced validation and 120 balanced test (300 correct/300 wrong overall), 571 supporting-document components, and a `teacher_boundary: B13` field. The current canonical plan instead requires 360 train/120 validation/60 balanced primary test/60 natural-prevalence secondary test and exclusion of the 48 and locked 100. The saved transfer manifest does not establish that revised contract, and its B13 field is not evidence that the pilot ran at B13. A newer authoritative H200 split/manifest, if one exists, is the exact missing artifact. No remote running preparation was inspected or changed.

| Collections | Shared question IDs | Shared gold-support document IDs | Shared retrieved document IDs |
|---|---:|---:|---:|
| 48 development vs 100 confirmation | 0 | 0 | 9 |
| 24 wrong development vs 100 confirmation | 0 | 0 | 5 |
| 48 development vs 600 | 16 | 19 | 77 |
| 24 wrong development vs 600 | 5 | 6 | 35 |
| 100 confirmation vs 600 | 17 | 25 | 125 |

The 600 train/validation/test subsets contain respectively 8/2/6 pilot QIDs and 10/4/3 confirmation QIDs. Within the saved 600, supporting-document overlap between splits is zero, but retrieved-document overlap is **96 train–test, 68 train–validation and 27 validation–test**. A selector seeing all retrieved tokens could therefore see the same document on both sides. Supporting-document disjointness is insufficient for the requested full-input document-level separation. These overlaps are identities, not outcome analyses; the full lists are in the cohort JSON. The three collections cannot be called independent as currently manifested. This report does not reassign questions or redefine the running preparation.

Before learned-selector evidence is interpreted, the owner should reconcile the manifest and freeze separation over the document inputs actually available to the selector, including confirmation exclusions. Multiple masks from a document remain dependent examples; group related documents/questions in split and uncertainty calculations.

## Implications for selector architectures

The early deployment contract remains: initial BTP/QTP, one shared vision pass, all remaining visual tokens with region identities available to the selector, and selected **original** visual tokens alone entering the frozen answer decoder. Gold, fixed self and oracle masks construct offline targets only. There is no full-page fallback, additional answerer crop or second vision pass.

| Architecture | Inputs and interaction capacity | Learned semantics and unresolved evidence |
|---|---|---|
| Attached selector | Frozen intermediate visual/text states with region identities; query-conditioned cross-region attention can model relationships | Incorrect generations do not imply useless states, but decodability is unmeasured. Late attachment pays decoder-prefix cost and cannot establish the early contract. It deserves supervision at its own boundary. |
| Compact independent | All surviving shared-vision tokens plus question tokens, geometry and region IDs; region pooling plus a small cross-region transformer can represent joins | Seeing every token and having attention establishes access/capacity, not learned ability to bind year/entity/column or enumerate sets. Small-cohort training may not teach those semantics. |
| Pretrained approximately 2B | A compatible text/multimodal backbone could supply semantic priors and contextual interaction over region tokens | No compatible shared-encoder/connector pair was established from these artifacts. Qwen’s shared features are available before/after its merger, but a 2B model’s expected feature space, dimensions, normalization and positional contract need explicit verification. A separate encoder would violate the contract. |
| Hybrid | Compact spatial/region pooling plus pretrained question/region interaction, with a utility head over original source IDs | Can combine inexpensive visual access with semantic knowledge; alignment and relational skill must be learned or demonstrated, not assumed from a pretrained name. Additional latency/adapter training could erase the benefit. |

Learning What Matters §5 shows that router-owned embeddings and a pooling bottleneck can erase relationship information, while an interaction-capable router can solve the synthetic task. Its pretrained multi-hop results also show that available information and architectural capacity do not guarantee learned selection semantics. Appendix M instead pools model input embeddings with early hidden states and uses a small transformer. Its matched-supervision comparison holds that representation/model setup fixed; it is **not an embeddings-versus-hidden-states ablation**. These findings support separating input access, interaction capacity and learned ability, rather than declaring a compact model sufficient or an attached model hopeless. [Learning What Matters v4](https://arxiv.org/html/2607.21692v4)

LAST provides a shared-encoder, compact-guidance precedent using compatible encoder/token layouts and model-specific connectors. That supports the one-vision-pass idea, not plug compatibility with our frozen Qwen2-VL interface or evidence of multi-region document reasoning in a new selector. [LAST](https://arxiv.org/html/2607.27952)

Human recognition of a wrong person, date or table column is a hypothesis about useful selector features. Some distractions can be rejected cheaply by entity identity; others require part of the same join as answering. Neither “distractor detection is always easier” nor “selection always requires solving the entire question” follows from this sample.

## Artifact provenance and reproducibility

Repository HEAD at audit start: `3c25f805863ece09f7d5bea82311baa9b1bd9742`. Instructions followed: repository `AGENTS.md`, `agent-context/CURRENT_TASK.md`, workspace routing, regional `INDEX.md` and canonical experiment plan. This is the frozen-Qwen regional attribution track, not MiniVGent answer-anchor work.

All source roots below are under `/scratch/lmalveau/docprune/`:

| Evidence | Artifact root / revision |
|---|---|
| Unified 48 | `task9-preliminary-dynamic48-90f27d7-unified-v1/analysis.json` |
| Original 42 + six retries | `task9-preliminary-dynamic48-90f27d7-v1/` and `task9-preliminary-dynamic48-90f27d7-retry-v1/`; each case follows unified `source.selected_arms_path` |
| All region mappings | `task9-preliminary-mappings-ecd1a87-v1/` |
| Questions, original answers, pages, PDFs | `task9-preliminary-mapping-inputs-5f7ff93-v1/` |
| MinerU Q00 / Q01–47 | `task9-preliminary-mineru-2a85d9e-v1/` / `task9-preliminary-mineru-b4-ef3202e-v1/` |
| Geometry Q00 / Q01–47 | `task9-preliminary-geometry-2a85d9e-v1/` / `task9-preliminary-geometry-b4-ad2b5c6-v1/` |
| Pilot cohort | `task9-preliminary-random48-v1/cohort.json` |
| 600 cohort/splits and transfer identity | `task9-shared-probe-random600-v1/`; `task9-shared-probe-random600-transfer-v1/bundle.json` |
| Confirmation identities only | `task9-baseline-wrong100-inputs-c0c9bee-v1/fixture.json` |

Runtime in all audited selected/raw/run manifests: `90f27d7ed8b99ad10f1a5fe405c131127456ae5d`. Qwen model revision: `eed13092ef92e448dd6875b2a00151bd3f7db0ac`. ContextCite pin: `c11f8ace6e68ba0121b2e2f1f5c896da9e4156f4`. MinerU code/model pins: `d9cd58add047c2364c1198eefcb1ee9cd63a971a` / `d3f5e08d073c21466bbabe21c71bb1e9c2e595da`. The 600 cohort-file SHA is `6c56c3a9b84818888ccf9a55601abfef0529be206a496874f05b588d979365a8`; split-file SHA is `2c1b99b51a087618f1b04818161098004b04c76e84f7e33c6085e88588385087`.

All 30 selected-arm files were checked against the unified SHA, mappings against run-manifest SHA, and token IDs against compact geometry/population. Scripts `prepare_blind.py`, `analyze.py`, `render.py`, `interpret.py`, and `publish.py` are preserved beside the outputs. Existing artifacts were reused; interpretation is stored separately from measurements. Missing source evidence is listed per case, not replaced with OCR inference or new retrieval.

## Case register

| Case | Boundary; tokens | F1 unpruned → native → support | Evidence-linked interpretation |
|---|---|---|---|
| [Q00](/home/lmalveau/DocPrune/outputs/regional-audit-2026-09-06/00-5fb253761e72a6048b8b759f04c3b784/index.html) | B_14; 4175 | 1.00 → 1.00 → 1.00 | The lead and infobox support Breeders’ Cup Mile; all arms remain exact. |
| [Q02](/home/lmalveau/DocPrune/outputs/regional-audit-2026-09-06/02-ac88e905caf79065a86ea107d4f0e5d0/index.html) | B_16; 4175 | 1.00 → 1.00 → 1.00 | The black-and-white ground below Mario is visibly checkered. |
| [Q03](/home/lmalveau/DocPrune/outputs/regional-audit-2026-09-06/03-2d72990e471477b01f909752e70127ab/index.html) | B_14; 3591 | 1.00 → 1.00 → 1.00 | The poster visibly shows a woman using a stethoscope and the title Tammy and the Doctor; the filmography supplies the Joan Marshall link. |
| [Q04](/home/lmalveau/DocPrune/outputs/regional-audit-2026-09-06/04-7f3e9cd662a95a07a0512e2b0f074193/index.html) | B_14; 3536 | 1.00 → 1.00 → 1.00 | The 1973 vintage identifies Stag’s Leap Wine Cellars within judge tables; the pictured grade is 14 rather than 17. |
| [Q08](/home/lmalveau/DocPrune/outputs/regional-audit-2026-09-06/08-14f49a6306b7b12ca89db44d501f340d/index.html) | B_16; 3557 | 1.00 → 1.00 → 1.00 | The event paragraph explicitly names Fontana Dam and the other geography pages corroborate location. |
| [Q18](/home/lmalveau/DocPrune/outputs/regional-audit-2026-09-06/18-2e7fa01a2e1cc5b43f3adf8e44d82d76/index.html) | B_14; 2897 | 1.00 → 1.00 → 1.00 | The supplied portrait is an interview image, while the biography explicitly establishes rugby. |
| [Q24](/home/lmalveau/DocPrune/outputs/regional-audit-2026-09-06/24-c6952949da6dae986391015dee092555/index.html) | B_14; 2606 | 0.40 → 0.40 → 0.40 | All main arms answer Kallang Stadium (F1 .4). |
| [Q25](/home/lmalveau/DocPrune/outputs/regional-audit-2026-09-06/25-bf78b88d5a21f31e0851b07066c76442/index.html) | B_14; 2785 | 0.00 → 0.00 → 0.00 | Calgary and the support/margin output Lillehammer both satisfy the open question ‘name a city that has hosted winter olympics’. |
| [Q26](/home/lmalveau/DocPrune/outputs/regional-audit-2026-09-06/26-3369d59ffc3d22c62e0bfa048a9cea21/index.html) | B_14; 2504 | 0.00 → 0.00 → 0.00 | The Graham Hill portrait visibly shows a moustache. |
| [Q27](/home/lmalveau/DocPrune/outputs/regional-audit-2026-09-06/27-b6a62e8b1c1cb1587d75b0af4e1ae981/index.html) | B_14; 3331 | 0.00 → 0.00 → 0.00 | Reba is visibly smiling. |
| [Q28](/home/lmalveau/DocPrune/outputs/regional-audit-2026-09-06/28-0f72c820c39f6fe6f6724b3338a4e134/index.html) | B_14; 2974 | 0.00 → 0.00 → 0.00 | All arms say 2 rather than gold 6. |
| [Q29](/home/lmalveau/DocPrune/outputs/regional-audit-2026-09-06/29-d82d381af11d76d8a70601251ecb8ca5/index.html) | B_14; 2861 | 0.00 → 0.00 → 1.00 | A real table correction: One and One-Sixteenth Miles becomes Seven Furlongs (EM/F1 1). |
| [Q30](/home/lmalveau/DocPrune/outputs/regional-audit-2026-09-06/30-774a3681d03347efd977caaf730ca97a/index.html) | B_14; 2144 | 0.08 → 0.08 → 0.08 | The readable rank-3 table shows Gmina Łabunie at 87.5 km², type rural, and the other rural rows. |
| [Q31](/home/lmalveau/DocPrune/outputs/regional-audit-2026-09-06/31-303da73f74cb98c1e853073417b9bdc9/index.html) | B_14; 3039 | 0.50 → 0.00 → 0.50 | Support restores October 28, 2010 (F1 .5), the unpruned answer, after native changes to March 3, 2026 (F1 0). |
| [Q32](/home/lmalveau/DocPrune/outputs/regional-audit-2026-09-06/32-ee3d5e6aaa766f4a7f1611950f12bac8/index.html) | B_16; 2811 | 0.00 → 0.00 → 0.00 | No arm is correct. |
| [Q33](/home/lmalveau/DocPrune/outputs/regional-audit-2026-09-06/33-5217a8617917f1ca031e915f17f9f326/index.html) | B_14; 3477 | 0.50 → 0.50 → 0.50 | All arms give 1851 (F1 .5) against September 18, 1851. |
| [Q34](/home/lmalveau/DocPrune/outputs/regional-audit-2026-09-06/34-2f34ec15d30206535baa99d709903307/index.html) | B_14; 2959 | 0.00 → 0.22 → 0.27 | Unpruned emits a camera specification; native returns 512 MB/4 GB (F1 .22), support 4 GB/8 GB (.27). |
| [Q35](/home/lmalveau/DocPrune/outputs/regional-audit-2026-09-06/35-23dec6930edc341ae25864a97cd28a89/index.html) | B_16; 3925 | 0.00 → 0.00 → 0.00 | Plemons’ filmography visibly includes Like Mike → Ox on rank 2; the Battleship connection is not present. |
| [Q36](/home/lmalveau/DocPrune/outputs/regional-audit-2026-09-06/36-e5d9ec29613dce539c3c381ff17bb77a/index.html) | B_14; 1990 | 0.00 → 0.00 → 1.00 | Support/margin change 2015 to gold 2010 exactly. |
| [Q37](/home/lmalveau/DocPrune/outputs/regional-audit-2026-09-06/37-1db882f16ee4721cc4c3fe9cf55e07e0/index.html) | B_16; 3859 | 0.00 → 0.00 → 1.00 | Support/margin change Cigarette to Gun exactly. |
| [Q38](/home/lmalveau/DocPrune/outputs/regional-audit-2026-09-06/38-8e205410446916c5f3a7522923db8660/index.html) | B_14; 2398 | 0.80 → 0.40 → 0.80 | Support restores K. |
| [Q39](/home/lmalveau/DocPrune/outputs/regional-audit-2026-09-06/39-146153932281542396eea1b476040e1c/index.html) | B_14; 2989 | 0.80 → 0.80 → 0.80 | All arms correctly choose the newer album, but output Mamma (as in the question) rather than document/gold Mama, scoring .8. |
| [Q40](/home/lmalveau/DocPrune/outputs/regional-audit-2026-09-06/40-7bd23de6ac7fe9fbdc7949da55628f69/index.html) | B_16; 2448 | 0.50 → 0.50 → 0.50 | All arms answer Kush (.5) against Kingdom of Kush. |
| [Q41](/home/lmalveau/DocPrune/outputs/regional-audit-2026-09-06/41-26ca9466876a6e17844791762c62ec0a/index.html) | B_14; 3963 | 0.40 → 0.40 → 0.50 | The full answer Zhang Ling, Tommy names both visible 2015 roles but scores .4. |
| [Q42](/home/lmalveau/DocPrune/outputs/regional-audit-2026-09-06/42-cff1106bdd553b3ef2b8b74419606ce0/index.html) | B_16; 2274 | 0.00 → 0.00 → 0.00 | Ben Moore’s collaborator paragraph R005 (251) explicitly includes Audra McDonald and is retained. |
| [Q43](/home/lmalveau/DocPrune/outputs/regional-audit-2026-09-06/43-77240911edd7c3cc1d48a0623e54d32d/index.html) | B_14; 3654 | 0.00 → 0.00 → 1.00 | Support/margin change the document-correct 2024 to stale gold 1981, a nominal exact rescue. |
| [Q44](/home/lmalveau/DocPrune/outputs/regional-audit-2026-09-06/44-eea285074866dd36b5f5a7333b74d09a/index.html) | B_16; 4003 | 0.00 → 0.00 → 0.80 | Support changes Hollywood to New York (.8); random does too, while margin says Sydney. |
| [Q45](/home/lmalveau/DocPrune/outputs/regional-audit-2026-09-06/45-d0b8f1a8c94566aabb1596c2f5ce9066/index.html) | B_14; 4011 | 0.00 → 0.00 → 0.00 | Unpruned/native say Ray Charles; support identifies James Brown but says hands in pockets, still zero against fists. |
| [Q46](/home/lmalveau/DocPrune/outputs/regional-audit-2026-09-06/46-3284364d73d82667ce3e2d6a38b05f08/index.html) | B_16; 3947 | 0.00 → 0.00 → 0.00 | All arms remain wrong; outputs switch among song titles. |
| [Q47](/home/lmalveau/DocPrune/outputs/regional-audit-2026-09-06/47-179569f6e6dba0729ddcf5e0acb23753/index.html) | B_16; 2995 | 0.00 → 0.00 → 0.00 | Support changes Dimitar Berbatov to Aykut Kocaman, still wrong; Aykut is a coach. |

## One recommended next research action

**Run a small, separately supervised boundary comparison before choosing a learned-selector architecture**—proposed here, not launched. Use six informative development cases: Q26 (correct portrait discarded), Q29 (real table correction/header deletion), Q38 (header/continuation relation), Q40 (redundant support), and controls Q02 (visual pattern) and Q04 (table constraint). The subset is selected after this audit and is exploratory, not a new confirmation cohort.

Compare actual **B_input, before the first decoder block**, against each case’s saved boundary: B_14 for Q26/Q29/Q38/Q04 and B_16 for Q40/Q02. Hold fixed retrieved pages/order, preprocessing and BTP/QTP IDs, shared vision output, original region definitions, original token positions, question/prompt, fixed gold/self strings, decoding settings, and achieved native token budget. Use corresponding 256 fit-mask vectors and corresponding 32 global/32 local holdouts at both depths, but **generate separate outcomes and fit separate oracles at each depth**. Do not transfer late labels to the early selector. Confirm all-keep continuation parity at these actual case boundaries first. Keep native DocPrune as the main achieved-budget comparator and random as an additional control.

Include only the minimal discriminating interventions: Q29 add R010 back to its successful set versus a 370-token control addition, plus R010-only removal from full context; Q26 add its 119-token correct portrait versus comparable added tokens, and swap/remove the retained 187-token wrong-person portrait; Q40 remove R015/R021 separately and together. Within each contrast hold all other sources fixed. Add-backs change length: use equal-token control additions or matched-budget swaps of original visual tokens, report any residual cost mismatch, and do not conflate content with budget. These controls are distinct from whole-region oracle masks if a control uses a token subset. Do not relabel old masks after splitting headers or rows.

Preserve historical F1/EM and add explicit document-validity annotations for the known spelling/alias issues in Q26/Q38/Q40; no benchmark gold is silently edited. Resolve the manifest discrepancy and document exclusions before this develops into a shared learned training experiment. The 100 remains confirmation-only.

**What would change the recommendation:** if the independently fit early oracle preserves the real table gain and controls at matched budgets with credible local fidelity/stability, proceed to an early learned-selector feasibility test rather than paying a decoder prefix. Its concrete candidate would consume shared-vision tokens, question tokens, original geometry and region IDs; use question-conditioned within-region pooling followed by cross-region transformer interaction; predict per-region coefficients from the separately fitted B_input gold-support oracle as offline targets; and maximize summed predicted utility with deterministic 0/1 token-cost knapsack selection (stable source-ID tie breaks), constrained by the native achieved token budget. Log any budget shortfall, avoid splitting sources to fill it, and pass only selected original tokens into B_input. Compact, compatible pretrained ~2B and hybrid variants would need the same own-boundary supervision and document-disjoint training/development inputs. If early own-depth oracles lose the verified gain despite reliable fits, while intermediate oracles retain it, prioritize attached selection and explicitly account for prefix compute/cache dependence. If both fits are unreliable, improve the oracle/intervention evidence before treating either architecture as disproven. No architecture choice is justified by the four nominal rescues alone.
