# Evidence chapter 1: Why move beyond attention ranking?

Prepared 2026-09-06. This is presentation source material, not a new experiment or a claim about the unpublished official implementation. All F1 and EM in this chapter use **0–100**, unlike the 48-question pilot’s commonly quoted 0–1 scale. Exact saved extracts, file SHA-256 hashes, and absolute source paths are in [01-docprune-random.json](01-docprune-random.json).

## Main story

The local reproduction made the selection problem concrete: aggressive token deletion was feasible, but the final attention-based selection step did not reliably improve answer quality. Once boundary and token budget were controlled, attention ranking was close to random on the small development study. That justified asking a different question: **does an answer-conditioned intervention oracle identify useful selections that these attention scores miss?** It did not establish that random and DocPrune are equivalent, that attention carries no information, or that a deployable learned selector would necessarily succeed.

## 1. Initial reproduction and stage localization

Saved full top-4 comparison (`full_top4` source): 2,441 questions, all-kept F1 **37.791069** versus full local DocPrune **36.760344**. Overall paired delta **−1.030725**, 95% interval **[−2.118804, 0.060631]**. Only **2,126/2,441** had exactly identical ordered retrieved pages; within that subset the delta was **−1.198024**, interval **[−2.347131, −0.063970]**. The subset supports the concern independently of observed page-identity drift, but is not a replacement randomized experiment. Timing was explicitly excluded from this analysis because shard hardware differed.

The controlled 245-question single-hop diagnostic (`stage245`) fixed retrieval across stages:

| Stage | F1 | Original visual tokens retained, pooled-token ratio | Incremental F1 delta and paired 95% interval |
|---|---:|---:|---|
| All-kept | 44.861224 | 100% | Reference |
| BTP only | 43.114286 | 52.830819% | −1.746939 [−4.681633, 1.073469] |
| BTP + QTP | 43.632653 | 45.431354% | +0.518367 [−2.138776, 3.232653] |
| BTP + QTP + literal CTP | 41.334694 | 18.798335% | −2.297959 [−4.297959, −0.628571] |

The strongest stage-local signal was the incremental CTP loss; the BTP and QTP intervals included zero. This does **not** prove that earlier filtering never removes indispensable evidence. Descriptively, the table modality fell from 39.8795 to 34.7831 at CTP, while image modality was approximately flat (32.4746 to 32.6441); this is an exploratory subgroup observation, not a corrected modality-specific test.

Runtime for the historical reconstruction was `4e2473bdbbc2e4eca0e92c30d4a0633044501ccf`. The fair-baseline protocol explicitly says these historical numbers predate the corrected shared EOS contract. Use them to explain the diagnostic journey; do not compare their absolute quality with later corrected-runtime results as if only the selector changed.

## 2. Reconstruction semantics mattered, but did not settle ranking quality

Literal CTP averaged post-softmax attention probabilities across query heads and used a locally added post-QTP token-count scale. Aggregate-logit CTP instead averaged raw query-key logits, applied visual-only softmax, then scaled by post-QTP visual count. The latter was selected because it better matched the paper-derived retention fingerprint, not because the author implementation was available.

On the same 245-question development set (`aggregate245`), aggregate-logit F1 was **42.416327**: **+1.081633** over literal CTP but **−1.216327** below BTP+QTP. Its conditional CTP retention was **67.292917%**, compared with literal **41.377448%** and a paper-derived target of 65%. Thus the native-policy comparison changes both selected identities and amount retained. It does not isolate better ranking. Candidate runtime: `c7a749801ad487c885e225354ccd2052fca29814`. The saved analysis itself calls neither candidate-versus-current improvement nor residual candidate harm conclusive at 95%.

All 16-, 64-, and 245-question cohorts are development data: the 64 prefix is inside the 245, and semantics were selected using all 245. Do not call the remaining 181 an untouched holdout.

## 3. The controlled 64-question random comparison

The admitted R20 analysis (`random64_r20`) used the same cached ordered top-four pages, shared BTP/QTP inputs, and per-question native decoder boundary. Ranking arms retained the **same exact M tokens per question**, where M was aggregate-native retention; the mean per-question conditional retention was **66.055800% of post-QTP tokens**. This is intermediate decoder deletion, not predecoder pruning. The random control samples uniformly without replacement and restores sequence order; it is a local Qwen adaptation of the random sampling core, not a reproduction of an entire different model pipeline.

| Policy | F1 | EM | Repetitions per question | Budget interpretation |
|---|---:|---:|---:|---|
| BTP+QTP, no CTP | 39.562500 | 32.812500 | 1 | Keep all post-QTP tokens |
| Aggregate native threshold / aggregate top-M | 39.921875 | 32.812500 | 1 | Same selected set at native M |
| Literal top-M | 39.812500 | 32.812500 | 1 | Matched M |
| Global uniform random | 39.427344 | 32.187500 | 20 | Matched M |
| Coverage-matched identity shuffle | 39.582031 | 32.890625 | 20 | Matched M and aggregate occupancy |
| Page-stratified random | 38.702344 | 30.859375 | 20 | Matched M |
| Grid-stratified random | 38.637500 | 31.484375 | 20 | Matched M |
| Literal native threshold | 39.531250 | 32.812500 | 1 | **Unmatched:** retains 40.928049% |

The primary estimand was the question-average aggregate score minus the within-question random mean. Its value was **+0.494531 F1 points**, with nested-bootstrap 95% interval **[−1.900801, 3.246895]**. The 90% equivalence interval was **[−1.526563, 2.727344]**, outside the prespecified ±1-point margin. Classification: **unresolved, neither demonstrated superiority nor demonstrated equivalence**. Support-document components were 64 singleton components in this development cohort; masks are repeated measurements, not 1,280 independent questions. The analysis used 100,000 draws, seed 20260827.

The exact matched-budget aggregate-minus-literal gain was only **0.109375 points**. This is useful counterevidence to claiming that the larger native-policy gain identifies a superior ranking mechanism. Aggregate also exceeds no-CTP by only **0.359375 points** on this cohort; that is a descriptive contrast, not the primary random comparison.

Random repetition increased from 10 to 20 because R10 global-random Monte Carlo standard error was **0.560332**, above the 0.25-point trigger. At R20 it remained **0.393565**; 20 was the frozen maximum, not evidence that Monte Carlo uncertainty vanished. Native runtime `2d2bea2722774e29009269637aa225131f170fd4`, native job `62279046`; corrected extension root uses `8d2231f`, job `62313075`. The saved admitted union contains 2,880 native rows plus 2,560 extension rows across 64 shards.

A separate fixed-budget sensitivity (`fixed64`) used 55/65/80% post-QTP retention, with only **3 random repetitions**, and was labeled descriptive-development-only:

| Conditional retention | Aggregate top-M F1 | Literal top-M | Global random | Coverage shuffle |
|---|---:|---:|---:|---:|
| 55% | 39.890625 | 39.906250 | 37.911458 | 40.625000 |
| 65% | 39.937500 | 39.796875 | 37.463542 | 39.572917 |
| 80% | 39.890625 | 39.984375 | 40.541667 | 39.291667 |

Coverage-matched random beating aggregate at 55%, and global random beating it at 80%, caution against a simple monotonic story. These noisy three-repeat results cannot identify a content-versus-coverage mechanism.

## 4. Larger random comparison: useful, still preliminary

The project results index reports **1,213 questions**, aggregate F1 **45.9744**, global-random mean **44.6437**, difference **+1.3307 points**, with 20 random repetitions. Treat these as **reported preliminary descriptive numbers**. They were not re-scored for this packet.

Direct inspection found **1,213 result files**, totaling **1,037,200,521 bytes**, under `/scratch/lmalveau/docprune/task6-holdout-primary-1213-v1/`. The reference enumerates 1,213 QIDs. The seal also contains 1,213 QIDs. These equal counts do not authenticate the exact union or all 21 cells per question, cached geometry, runtime provenance, or member hashes. A first-row identity traversal terminated before publishing; it was not repeated because full authentication is outside this bounded evidence-gathering pass.

Only the **252-question interim** aggregate analysis and its **10,000-draw descriptive nested-cluster interval** were found beneath that root; no terminal prespecified 100,000-draw clustered analysis was found. Interim F1 was 47.134921 versus 45.083929, delta 2.050992, interval [0.331510, 3.767482], over 248 supporting-document components. This interval belongs to the interim 252, **not the final 1,213**. A quick IID interval previously reported for the full pool must not be substituted for the prespecified analysis.

Selection was documented as outcome-blind with respect to method development and excluded diagnostic QIDs, but historical all-kept/literal outcomes already existed for the full benchmark. It is a **method holdout**, not a pristine benchmark test. Keep its historical aggregate results separate from subsequent confirmation-cohort outcomes.

The larger point estimate is counterevidence to “attention is no better than random.” A defensible transition is: **the local ranking showed a small and uncertain advantage in development and a positive preliminary larger-pool signal; we then asked how much useful selection headroom a stronger offline oracle could reveal.**

## Most useful presentation assets

1. A 245-question stage chart with F1 and retained-original-token percentage shown separately. Caption it historical local reconstruction; include CTP paired interval.
2. A 64-question matched-budget comparison, emphasizing aggregate, literal top-M, global random, and coverage shuffle. Plot the primary paired difference interval, not invented per-arm error bars.
3. A two-row table distinguishing admitted 64-question unresolved inference from preliminary 1,213-question descriptive means. Do not present it as a learning curve: cohorts differ.
4. A small experiment-controls diagram: fixed retrieved pages → shared BTP/QTP → common native boundary → exact matched M → frozen continuation. It explains why native policy and ranking-only comparisons differ.

## Writer claim guardrails

- Say **local DocPrune reconstruction**; unpublished author code was not verified.
- Say **unresolved** for the 64-question comparison, never “proven equivalent to random.”
- Keep retention denominators explicit: original visual tokens versus post-QTP tokens; pooled-token ratio versus mean question ratio.
- Gold-conditioned ContextCite later supplies an offline oracle. It is not a deployable comparator with the same inputs.
- These observations motivate testing an oracle; they do not by themselves choose attached, compact independent, approximately 2B, or hybrid selector architecture.
- Missing final 1,213 authentication and prespecified inference remain missing evidence, not permission to launch another GPU run.
