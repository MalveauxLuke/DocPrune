# One-question pilot: can regional interventions be modeled reliably?

Suggested story: before scaling the oracle, we tested whether an additive regional surrogate could predict unseen interventions. The gold-answer target was much more predictable than the fixed generated-answer target on this question. Both tested depths supported useful prediction at 256 masks, but the identity of the selected set remained unstable. This justified evaluating actual selected-set answers in the 48-question development pilot; it did not select a deployable architecture.

## What was held fixed

Question `e1e6ed53f9ad11813845088f4cf2f6b1`: **Which Club(s), in Career statistics | Club of Dženis Beganović, have a flower on the logo?** Saved gold: **FK Tuzla City**. One question, 95 region sources, four fixed retrieved pages. The image supporting-document ID in the original question annotation is `52830c5f8b6e8b19add8756c8f56576d`; it is not one of the four retrieved document IDs. This is a provenance warning, not a new visual claim about whether the flower is visible in the retrieved table. No independent visual adjudication was performed for this chapter.

The answerer is frozen Qwen2-VL-7B-Instruct. Original 10,032 visual tokens became 4,336 after BTP and 3,586 after QTP. Both depths used this same 3,586-token population and the same region mapping. Prompt input IDs had shape `[1, 10090]` before this pruning; the post-QTP prefix length was 3,644, including 58 nonvisual positions. Saved generation traces show `ctp_layer: null` and post-CTP count equal to post-QTP count. Thus “unpruned generated answer” here means no CTP/selector deletion after BTP/QTP, not all original visual tokens kept.

Actual boundary definitions: **B_input** deletes before decoder block 0, after vision encoding and BTP/QTP; **B_13** deletes after zero-based block 13, so 14 decoder blocks have already run. Raw B13 checkpoints contain 14 cache lengths of 3,644; input checkpoints contain none. B_input is not deletion after the first decoder block. Physical deletion preserves original logical positions and M-RoPE identities; it is not a blacked-out page image. Surviving B13 states and earlier-layer caches can retain information from deleted content.

## Saved quantitative results

LDS here is Spearman rank correlation, with average ranks for ties, between the surrogate prediction and measured likelihood on unseen masks. It is not answer accuracy. RMSE is compared with always predicting the fitting-target mean. Lower RMSE/constant ratios are better.

| Fit / held-out masks | Target | Boundary | LDS | RMSE | Constant RMSE | Ratio | Coefficient refit mean / min | Selected-set Jaccard mean / min |
|---|---|---|---:|---:|---:|---:|---|---|
| 64 / 32 | Accepted gold | B_13 | 0.887830 | 0.240436 | 1.016290 | 0.2366 | 0.3952 / 0.2471 | 0.7008 / 0.6364 |
| 64 / 32 | Fixed generated answer | B_13 | 0.587977 | 0.982574 | 0.933244 | 1.0529 | 0.4873 / 0.3670 | 0.7629 / 0.7111 |
| 256 / 64 | Fixed generated answer | B_13 | 0.715934 | 0.979053 | 1.340796 | 0.7302 | 0.5447 / 0.4093 | 0.6223 / 0.5467 |
| 256 / 64 | Fixed generated answer | B_input | 0.648855 | 0.889320 | 1.196005 | 0.7436 | 0.4721 / 0.3178 | 0.6101 / 0.5309 |
| 256 / 64 | Accepted gold | B_13 | 0.914835 | 0.193681 | 0.994279 | 0.1948 | 0.6698 / 0.5741 | 0.7078 / 0.6500 |
| 256 / 64 | Accepted gold | B_input | 0.899542 | 0.219347 | 1.009503 | 0.2173 | 0.6528 / 0.5890 | 0.6832 / 0.6265 |

All stability summaries use five bootstrap refits with replacement, drawing as many rows as the fit bank (64 or 256); mean/min summarize the ten refit pairs. These are not the later pilot’s deterministic 80% refits. Selected sets used exact region-cost knapsack at requested and achieved M=2,689 visual tokens, about 75% of the post-QTP population. This selection budget is distinct from the independently sampled Bernoulli masks, whose region keep probability was 0.5. At 256 masks, fixed-generated selected 64 regions at B13 and 65 at input; accepted-gold selected 71 and 68. Matching token budgets does not imply matching region counts.

## Valid comparisons and the target-scale trap

The paired 256-mask conditions use the same 256 fitting masks (seeds 0–255) and 64 held-out masks (256–319), verified by identical fit and held-out mask hashes across both depths and targets. Question, ordered pages, prompt, region mapping, token population, output sequence IDs, mask design and budget are fixed. The decoder boundary changes, outcomes are generated at each boundary, and each surrogate is separately fitted. Accepted-gold and generated-answer analyses reuse the same terminally admitted raw rows. They are not four different layers.

Accepted gold is maximum accepted-reference mean log-likelihood; this case has exactly one reference sequence. The generated target holds the original seven non-EOS token IDs fixed across masks; no re-decoding or retokenization is used to construct the target. The initial 64-mask secondary analysis uses mean token log-probability. The canonical 256-mask generated analysis instead uses ContextCite sequence logit divided by seven: `(log p(sequence) − log(1−p(sequence))) / 7`. Its maximum correction relative to the mean-logprob input was 0.03109 at B13 and 0.03668 at input. Therefore the 64→256 generated comparison changes both bank size/holdouts and target transform. Do not call it a controlled pure mask-count effect. Even the accepted-gold 64→256 comparison uses different held-out masks (64–95 versus 256–319); the former holdouts become fitting rows in the later diagnostic.

Within the paired 256-mask experiment, B13 minus input LDS is +0.06708 for the generated target and +0.01529 for accepted gold. Both boundaries beat their own constant RMSE baselines. B13 is numerically better by LDS in this one case; this does not establish population superiority, better correction accuracy, or an architecture preference. Raw RMSE across targets or boundaries has different outcome distributions; use the within-condition constant comparison.

## What the sequence of experiments taught us

1. **Initial warning:** accepted-gold LDS was high (0.888), but its coefficient ranking and set identity were unstable. The generated target had positive LDS (0.588) while its RMSE was worse than constant (ratio 1.053). Rank correlation alone was an insufficient reliability check. One secondary bootstrap fit had a convergence warning; the canonical fit and primary fits converged, so that warning does not explain the whole failure.
2. **Paired diagnostic:** at 256 masks both depths cleared the recorded predictive gates (LDS ≥0.5 and RMSE better than constant), while every condition failed the old minimum selected-set Jaccard ≥0.8 gate. Accepted-gold RMSE ratios of 0.195 and 0.217 indicate substantially more predictable intervention functions than the fixed-generated ratios 0.730 and 0.744 on this question. This is an empirical difference between target functions, not proof gold supervision will always be easier.
3. **Scientific decision:** the old exact-set-identity gate was explicitly relaxed before new question-level outcomes. The next pilot treated the accepted-answer procedure as a privileged oracle, retained stability descriptively, and tested actual budgeted-set generations. A large weakly separated coefficient tail is a plausible explanation for changing set identity despite good global prediction, but low coefficients do not certify irrelevant content or interchangeable support.
4. **Architecture bridge:** the input result keeps early selection plausible: its own gold-support intervention function is predictable here. The late result cannot establish sufficiency before decoding. A useful next comparison would generate separate supervision at each deployment depth and use real correction outcomes on an evidence-verified corpus. No attached/compact/2B/hybrid model was trained or compared in these diagnostics.

## Parity, limitations and presentation use

The preceding regional smoke saved first-mask cached-versus-independent continuation parity with maximum absolute likelihood error 0.0. That result is specifically a first-mask shared-prefix check, not an all-keep measurement. The three full-run completion manifests and all their listed member file hashes were verified while assembling this chapter. Their raw files do not contain an all-keep parity metric; do not relabel the smoke as one. They record frozen cached pages reused, no retrieval search and no global index load.

There is no cross-question confidence interval, population validation, interaction identification, or sufficient-set certificate here. Strong global LDS can coexist with poor budget-local selection. These diagnostics do not independently establish a corrected generated answer. The best presentation visual is a two-depth grouped table or plot with separate gold and self series at 256 masks; show the initial 64-mask diagnostic separately, clearly labeled.

## Provenance

Analysis files (file hashes freshly computed; internal hashes and raw runtime revisions are in the JSON companion):
- `/scratch/lmalveau/docprune/task9-regional-development-7616b29-v1/analysis.json` — SHA-256 `1dd375f5e52374b17b3efc3b8cad97e50e159e138f90a8270d406641e752de40`
- `/scratch/lmalveau/docprune/task9-paired-256-diagnostics-c49abb5-v2/analysis.json` — SHA-256 `cfe9b9456b0040aa9bcee1ce7f332d67401f203a8a53646402a905f009798a18`
- `/scratch/lmalveau/docprune/task9-paired-256-accepted-answer-0d40fad-v1/analysis.json` — SHA-256 `5c67c85a0223bbd8ee84d8787a40789ac2f3cf19a4bde8dd706fcde1ad8e8d98`

Raw runtimes: initial `7616b29b4dc5ba33584a6e26371281be6886188f`; B13 `39aa5f5142cd1ef87291e72f30bcfb23d813195f`; corrected input `8e956be8259e4d72ba1aa7b534649c2f411ef16c`. CPU analysis revisions: generated `c49abb5eac5bd800d75efcca2e4fc571024ef2b7`; accepted `0d40fadb3de6001b4ab3974c7053443552e1f749`.

History and decision ledger: `docs/experiments/regional-attribution/archive/legacy-governance-2026-09-05/EXPERIMENT_LOG.md`, one-question and paired-diagnostic entries around lines 1032–1170. Boundary validation: `src/docprune/task9_live.py` expected-prefix check. Mask sampling: `src/docprune/task9_attribution.py` pinned Bernoulli source ablation. Source mapping: `/scratch/lmalveau/docprune/task8-region-mapping-f3f5de7-v1/region-token-mapping.json`. Ordered page fixture and source PDFs are linked in the JSON companion.
