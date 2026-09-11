# Regional oracle: 48-question development evidence and what the audit changes

Prepared 2026-09-06 from existing artifacts. This chapter supplies presentation evidence, not slides or a new experiment. Numerical outcomes below were reconciled against the actual unified JSON; the 192-mask generated outcomes were independently joined by saved selection/repeat bindings. Visual interpretations reuse the completed audit rather than claim a new blinded inspection.

## The role of this experiment in the story

The earlier native-versus-random comparisons ask whether the tested pruning heuristic identifies useful content. The one-question regional diagnostic asks whether interventions can be modeled faithfully. This 48-question pilot asks the next question: **with privileged answer-conditioned supervision, can another retained set improve the frozen answerer's behavior at the native token budget?** The answer is yes under the saved development scoring, with a clear table correction as a concrete example. That establishes a reason to study learned selection, not a demonstrated deployable architecture.

The subsequent visual audit sharpens the objective: train and evaluate correction of genuinely wrong answers using evidence actually available to the model. Better normalization alone cannot repair absent evidence, stale gold, or a likelihood target that mistakes members of a required set for alternative complete answers. None of these findings erases the observed capacity to influence generation; they narrow which changes count as verified correction.

## What was held fixed and what was optimized

Frozen Qwen2-VL-7B-Instruct received four retrieved pages in order, then the question and answer-only instruction. Initial BTP/QTP already removed visual tokens. At a cached intermediate decoder boundary, binary masks physically deleted visual-token positions belonging to semantic regions. Remaining original positions and logical IDs were preserved; text positions were not deleted. Each case used 256 unique fitting masks plus 32 global and 32 budget-local holdouts. The additive oracle fitted accepted-answer support G(m); the secondary oracle fitted G(m) minus the likelihood S(m) of the **fixed original no-CTP generation**.

Native DocPrune is the primary comparator. Region-size-aware random is an additional control. All pruning arms in the 30 visually audited cases exactly matched the native achieved token count, not just a number of regions. Wrong-case budgets were 1,990–4,011 visual tokens, 49.19–85.85% of the post-QTP population. Gold-support selections retained whole nonempty mapped sources; native deletion could scatter holes within them.

**Boundary convention matters.** Wrong cases used B_14 in 16/24 and B_16 in 8/24; all 48 used B_14=34, B_15=1, B_16=13. B_K is after zero-based block K, so B_14 follows 15 completed blocks. B_input is before the first block; B_0 is after it. The pilot is not uniformly B13.

## Main measurement panel

F1 is shown on a 0–1 scale; multiply by 100 for percentage-point plots.

| Arm | Wrong mean F1, n=24 | Wrong exact /24 | Correct exact /24 |
|---|---:|---:|---:|
| Post-BTP/QTP, no CTP reference | 0.165833 | 0 | 24 |
| Native DocPrune | 0.137500 | 0 | 24 |
| Gold-support regional ContextCite | 0.381250 | 4 | 24 |
| Gold-versus-fixed-self margin | 0.368750 | 5 | Not evaluated as preservation arm |
| Region-size-aware random | 0.154167 | 0 | 22 |

Gold support versus native: 9/24 F1 wins, 15/24 ties, 0/24 losses, delta +0.24375. Against the no-CTP reference: 7/24 wins, 17/24 ties, 0/24 losses, delta +0.215417. Q31 and Q38 restore damage introduced by native pruning; they do not improve the original answer. Across the deliberately balanced 48, support minus native is +0.121875 F1 and support minus regional random is +0.155208.

The saved 10,000-draw within-stratum question bootstrap gives a descriptive balanced support-minus-native interval [0.051042, 0.203125]. A reweighting using the source pool's 90/245 correct and 155/245 wrong prevalence gives +0.154209, interval [0.064583, 0.257015]. This is the same pilot reweighted, not a second observed cohort or independent confirmation. Document overlap and development selection further limit generalization; do not headline the interval as proof of deployment performance.

## Best visual examples and counterexamples

| Case | Saved answer change and score | Document-grounded interpretation | Presentation use |
|---|---|---|---|
| Q29 | One and One-Sixteenth Miles → Seven Furlongs; F1 0→1 | Actual rank-1 table contains Fast + Del Mar → Seven Furlongs. Support retains R011 (601 tokens), deletes R010 (370) containing a competing Del Mar/Firm row and headers. | Lead positive example: selection can produce a real correction. Qualify causal and depth interpretation. |
| Q26 | Mustache → beard; F1 0→0 | Original portrait visibly has a moustache. Support deletes correct Graham Hill portrait R013 (119) and retains bearded Wes Johnson R038 (187). | Why a scored tie and “no harms” can hide semantic deterioration. |
| Q37 | Cigarette → Gun; F1 0→1 | Requested Black Hawk Down poster is absent from all four inputs. Deleted Abbey Road paragraph R021 (383) says McCartney holds a cigarette in his right hand. | Strong competing-relation hypothesis; not proof of reading the requested object. |
| Q43 | 2024 → 1981; F1 0→1 | Supplied page says the last two meetings were 1981 and 2024. Gold is stale; original answer is document-correct. Other 2024 passages remain after deletion. | Why an oracle can optimize the specified target while worsening factual correctness. |
| Q41 | Zhang Ling, Tommy → Zhang Ling; F1 .4→.5 | Both 2015 roles are visible in retained R018 (741). Gold likelihood maximizes separate role strings rather than scoring a complete set. | Target-contract failure, not missing evidence or genuine correction. |

Thus the four nominal support exact rescues decompose into **1/4 visibly supported correction (Q29), 2/4 missing required input evidence (Q36/Q37), and 1/4 stale-gold match (Q43)**. These are audit judgments alongside unchanged historical scores, not a replacement benchmark score. The later source-PDF inspection found the Q37 gold poster on source page 0 and confirmed the gun; that page was not among the original four. Showing it without this label would falsely imply input availability.

Four recommended asset paths, captions, page ranks, hashes and matching retained-token overlays are in `03-pilot48.json` under `visual_assets`. Q29 rank 1, Q26 rank 0 and Q43 rank 0 have native detail crops; Q37 should show the four-page overview to establish the missing-input issue. The [bundled visual gallery](index.html) contains selected evidence views and footprints; the case JSONs retain source/token references. The full gallery remains at the SOL path recorded in provenance. Overlays illustrate token deletion; they were not literal edited images fed into continuation. Green/red/gray denote retained/deleted/already absent after BTP/QTP. One cell represents a merged visual token, 28×28 input pixels.

## What likelihoods and surrogates establish

Under gold-support selections, G increases and fixed-self S decreases in 16/24 wrong cases. Q29 shows G −0.40038→−0.07282 and S −0.12803→−0.41965, consistent with the desired direction. But Q42 has ΔG=+2.37571 and ΔS=−2.65757 and still generates Ben Moore rather than Audra McDonald: favorable fixed-sequence likelihood is not a correctness certificate.

Under the margin selections, 5/24 cases improve G−S while **both G and S fall**. Q26 has G −0.92045→−1.05457, S −0.90859→−1.59490, yielding Δmargin +0.55220. Margin gain therefore does not imply more gold support. Where gold and self are exactly the same scored sequence, the margin is identically zero and cannot measure preservation. The five margin exact matches also include the spelling/granularity change Smiling→Smile; margin does not dominate support in mean F1.

Global LDS mean/min is .775738/.308651; budget-local LDS mean/min is .622393/−.103006. Global RMSE beats the constant baseline in 48/48, local RMSE in 45/48. Local LDS reaches .5 in 35/48. Within wrong cases local RMSE wins 24/24, while Q30/Q38/Q47 local LDS is −.103/.083/.111. Five bootstrap refits have case-level mean pairwise source-set Jaccard .479–.854, median .640. Ranking nearby candidate sets can be unreliable despite good global predictive fit.

Each wrong-case bank covers 29–103 sources; per-source inclusion counts span 102–148, and the minimum joint pair cell is 34–44. This is coverage, not identification of pairwise interactions: other sources vary simultaneously. Additive coefficients cannot establish that every deleted region is harmful, that a retained set is semantically sufficient, or that redundancy is absent. Plausible competition appears in 15/24, relation binding/set reasoning in 14/24, mixed indispensable/competing regions in 7/24, and redundancy candidates in 2/24. These overlapping visual categories are hypotheses from a small development sample.

## Why 256 masks remained the configuration

The existing mask-count ablation refitted 64/96/128/192 subsets with five repeats each; their subsets were independently sampled, not a sealed nested schedule. Canonical 256 had one fit per question. Mean global LDS at 64/96/128/192/256 is .71564/.74864/.75749/.77777/.77574; mean token-weighted Jaccard versus 256 is .78595/.81254/.84224/.89808/1. High global fidelity at 192 did not guarantee identical generated outcomes.

| 192-mask repeat | Correct preserved /24 | Canonical rescues preserved /4 | Wrong mean F1 |
|---|---:|---:|---:|
| 0 | 24 | 3 | .29500 |
| 1 | 24 | 3 | .32833 |
| 2 | 24 | 4 | .33667 |
| 3 | 24 | 3 | .32833 |
| 4 | 24 | 3 | .32833 |
| Canonical 256 | 24 | 4 | .38125 |

Pooling repeated evaluations gives 120/120 preservation, 16/20 rescue retention and wrong F1 .32333, with mean G change −.01391 nats/token. These are **24 questions × five repeats**, not 120 independent questions, and four canonical rescued questions × five repeats, not 20 independent rescues. The exact-rescue losses are all Q36; any-F1 losses comprise 11/240 repeated rows from only Q34/Q36/Q44. The saved functional gate failed, retaining 256 fitting masks. The 64 heldouts make 320 intervention evaluations per canonical question.

**Cross-experiment inference:** the only exact rescue lost at 192, Q36, is later identified as missing the required retrieved evidence. The clearest verified correction Q29 survives all five 192 repeats. Therefore the old gate documents scorer-level sensitivity, not demonstrated loss of a document-verified rescue. This nuance is useful for the story: evaluation quality affects interpretation even of compute ablations. It does not validate switching to 192 or justify retuning a gate after seeing outcomes.

The failing Q36 repeats had global LDS .887–.908, local LDS .951–.967 and nested 128→192 token Jaccard .950–.966. Ordinary low-fidelity or unstable-selection alarms would not reliably identify it. The exploratory grouped screens lacked enough distinct positive questions; critical-loss recall cannot be cross-validated with one positive QID. Outcome-aware escalation rules use gold-scored answers and must not be presented as deployable uncertainty diagnostics. Retain 256 for reliable per-question refits for now; fewer masks across more training questions is a different shared-learning hypothesis.

## Bridge to the next research phase

The most defensible bridge is: **we have demonstrated controllable generation and some correction headroom, and now need clean correction targets plus an own-boundary selector comparison.** The pilot does not choose between attached, compact independent, pretrained approximately 2B or hybrid selectors. Input access, interaction capacity and learned semantic ability are separate questions.

Intermediate surviving states have already seen deleted regions, and earlier-layer KV caches retain full prefix context. Successful late deletion cannot establish that the same retained evidence works before the decoder. Existing forced-boundary all-keep parity passes its separate fixture at B_input/B_0/B_6/B_13/B_20/B_23/B_26 within configured logit tolerance (max absolute difference .125), but there is no located per-pilot-case panel at actual B_14/B_16. This is a bounded missing verification, not evidence that continuation failed.

A future comparison should fit separate gold-support oracles at actual B_input and the relevant intermediate boundary, using corresponding masks, native achieved budgets and 256 fitting masks at each depth. Do not give an early selector late labels by assumption. Preserve the deployment contract: BTP/QTP, one shared vision pass, all surviving visual tokens plus question/region identities accessible to the selector, and only selected original visual tokens entering the frozen decoder. Connector compatibility and semantic ability of a pretrained 2B selector remain unverified. Gold, fixed self and oracle masks are offline supervision only.

Useful proposed counterfactuals, not launched: Q29 add back R010 versus a 370-token control addition and remove R010 alone from full context; Q26 add back the correct portrait versus comparable tokens and swap out the wrong-person portrait; Q40 remove redundant-looking R015/R021 individually and jointly. Fix all other tokens, prompt, preprocessing and generation settings; use matched-budget swaps or equal-size additions to separate content from sequence-length effects. Region-boundary changes require new labels.

The available 48/100/600 manifests are not independent collections: 48–100 share 9 retrieved document IDs despite zero QID overlap; 48–600 share 16 QIDs/77 retrieved documents; 100–600 share 17 QIDs/125 retrieved documents. These are identity audits only, not confirmation outcome reads. Future splitting must group all document inputs, not masks or supporting pages alone. The owner reports 600 transferred to H200 with segmentation pending; no preparation was modified for this package.

## Provenance and writer guardrails

Primary unified result: `/scratch/lmalveau/docprune/task9-preliminary-dynamic48-90f27d7-unified-v1/analysis.json`, file SHA `541327e8fed25be50d00e6ab0716697c3a39611ae286389b21da8a8c681cc526`, internal digest `985a837b2094b5a925730eeb4b9bf07c594344f9021c4a718c49c11aa655a8c5`. Runtime `90f27d7ed8b99ad10f1a5fe405c131127456ae5d`; original 42 and six retries are joined by the unified source records.

Mask ablation: `/scratch/lmalveau/docprune/task9-mask-count-ablation-d6de9d8-v1/analysis.json`, file SHA `a03a7d4c8db697486d4c16bc6c8540d185c861dd2a97d096f5909d6003d5746a`, internal digest `9d6a9ac6ece6712ca9cdad33bba0bc61caf122121435c3d99a6fb40e460edc51`. Generated selections under `/scratch/lmalveau/docprune/task9-mask192-generation-356f118-v1/` report runtime `356f11842916243308f67c0134254f988041f5e8`. This extraction independently reconstructs all 240 repeat rows, including canonical reuse and duplicate selection bindings, rather than treating them as 240 new generations.

The companion JSON preserves aggregate outcomes, fidelity metrics, all 240 joined rows with source-file hashes, selected case measurements and four recommended visual assets. Detailed method and earlier validation: [visual audit](sources/4dd62c45d14f--VISUAL_AUDIT_24_2026-09-06.md), [mask-count analysis](sources/4c49f0160738--mask-count-and-nesting-2026-09-02.md), [development ledger](sources/8629c2196732--DEVELOPMENT_48.md).

Use “four scored exact rescues,” not “four verified corrections”; “no scored F1 losses,” not “no semantic harm”; “late answer-conditioned regional oracle,” not “deployable selector”; and “development observations,” not “confirmed generalization.” No new GPU inference, training, retrieval or mask sweep was performed.
