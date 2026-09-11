# Experimental-Design Specification  
## Learning Corrective Region Selection for Frozen Document VLMs

**Program objective:** determine whether a learned region selector can improve the complete answers of a frozen document vision-language model while reducing—or otherwise justifying—total inference cost, and whether supervision about the reader’s original answer adds value beyond matched gold-only utility supervision.

**Scope:** a coordinated research program, not a fixed implementation or a commitment to include every proposed component in the final system.

**Experimental principle:** begin with existing evidence and small reusable measurement banks; test the central learning hypothesis before extensive architectural refinement; expand data, context length, and adaptive acquisition only when earlier results justify the expense.

---

## 1. Scientific motivation and contribution

### 1.1 The problem

A document retriever can supply material that plausibly matches a question without ensuring that a downstream reader interprets it correctly. Several regions may concern the right topic while differing in entity, category, scope, time period, table field, or relationship. Other regions may be complementary: neither is sufficient alone, but their combination supports the complete answer.

The proposed selector should learn to distinguish these situations and choose a useful set of original visual regions under a token budget.

The intended deployment is:

\[
\text{question and candidate pages}
\rightarrow
\text{selector}
\rightarrow
\text{retained original visual tokens}
\rightarrow
\text{frozen answerer}.
\]

The selector does **not** receive the gold answer or the reader’s original answer at deployment. Those answers are used to construct offline supervision. Learning the selection policy does not require an explicit prediction of what the reader would otherwise answer.

### 1.2 What the existing experiments establish

The historical comparison reports a preliminary 1.33-point F1 advantage for local DocPrune ranking over matched random ranking on 1,213 questions. This motivates investigating better selection; it does not establish equivalence to random or that attention is useless. Its execution pipeline differs from the later correction study. 

The later study used frozen Qwen2.5-VL-7B-Instruct with four fixed pages and 10,032 visual tokens. On 17 curated, initially incorrect cases, gold-conditioned selection rescued 11 cases with input-level deletion, compared with five for one regional-random selection per question. These are error-cohort correction counts, not learned-selector performance or population accuracy estimates.  

The examples identify plausible transferable distinctions, including species versus species-and-subspecies, Championship versus Series results, and a photograph versus a competing textual cue. They also expose limitations: jointly changed contexts do not isolate one region’s causal role, and strong likelihood prediction can coexist with incomplete answers.  

The pilot therefore supports **the existence of useful selection actions in this development cohort**. It does not yet establish:

- generalization to unseen documents;
- preservation of initially correct answers;
- superiority of contrastive supervision over gold-only supervision;
- adequate region granularity;
- or a favorable learned-system quality/cost tradeoff.

The 17 cases remain a diagnostic resource throughout the program, not a validation set on which to select the final method.

### 1.3 Positioning relative to prior work

The contribution must be narrower than “selection trained from downstream utility.” RECOMP already trains sentence selection from downstream utility, including decoded-answer exact match for QA. Influence Guided Context Selection also learns generator-aware context selection from intervention-derived feedback. 

Target-versus-foil subtraction is established in contrastive explanations. ContextCite provides fixed-response mask-based attribution, and AT2 demonstrates shared learning directly from ablation outcomes rather than requiring precise independent attribution recovery for every example. None establishes that the proposed gold/self signal improves this visual selection task. 

The prospective contribution is:

> **A gold-free, model-relative visual-region selector learns corrective context decisions from matched intervention feedback, improves complete answers on unseen documents, and offers a useful total quality/cost tradeoff.**

The added contribution of the original-answer signal must be demonstrated separately. If gold-only supervision performs equally well, the broader selection result can remain valuable, but the stronger contrastive-supervision claim should be dropped.

---

## 2. Hypotheses and status of the proposed components

### 2.1 Research questions

| ID | Hypothesis or question | Decisive evidence |
|---|---|---|
| **H1: Corrective selection** | A learned selector can improve a frozen reader’s complete answers, not merely shorten its context. | Held-out answer gains with acceptable damage to initially correct answers and competitive total cost. |
| **H2: Added value of original-answer supervision** | Gold/self feedback improves learning beyond matched gold-only utility. | Better answer quality, preservation, or data efficiency with architecture, masks, inputs, and training effort controlled. |
| **H3: Retrieval-informed measurement and representation** | Col-style matches identify consequential candidate regions and provide useful features for selection. | Better measurement yield and learned decisions—not only correlation with retrieval labels. Sampling and feature benefits are isolated. |
| **H4: Transfer across questions** | Fewer measurements per question can suffice when training spans more independent documents and recurring evidence relationships. | Favorable learning curves at matched teacher cost, rather than accurate reconstruction of every per-question surrogate. |
| **H5: Representation and capacity** | Query-conditioned compact readouts may capture useful distinctions; a pretrained approximately 2B backbone may add value when compact processing is insufficient. | Controlled representation comparisons and whole-system quality/cost results. |
| **H6: Optional structured learning and acquisition** | Set-dependent training or oracle-verified adaptive masks improve the deployed policy. | Gains beyond a direct-only policy or static data expansion under matched additional cost. |

The mechanism “plausible-looking but query-mismatched material” is a hypothesis to investigate, not a universal interpretation of every negative effect or residual.

### 2.2 Design status

| Status | Commitments |
|---|---|
| **Core proposed approach** | Frozen answerer; direct mask-outcome supervision across questions; region-level actions; explicit query conditioning; one-pass direct selection as the default deployment; actual-answer evaluation. |
| **Preferred initial formulation** | Gold-aware contrastive preferences; compact and approximately 2B encoder variants; a small shared policy MLP; frozen Col-style retrieval and matching features. |
| **Necessary validation** | Teacher execution, evidence availability, segmentation coverage, target-to-answer alignment, visual-interface compatibility, and document-disjoint generalization. |
| **Scientific comparisons** | Gold-only versus contrastive supervision; retrieval-guided versus agnostic masks; feature use; compact versus 2B; direct versus set-dependent training; preference versus numerical supervision. |
| **Conditional extensions** | Adaptive acquisition, ContextCite-guided proposals, external-model proposals, richer readouts, small-head reranking, hierarchical admission, and policy distillation. |
| **Implementation defaults, not hypotheses** | Exact widths, layer counts, optimization settings, batching, and initial mask quotas. These should be chosen conservatively rather than swept exhaustively. |

Two substantive tensions are resolved by assigning different roles, not deleting earlier ideas.

**Targeted factorial masks** remain valuable for interaction diagnosis, but are not the default discovery mechanism because reliable pair nomination is unproven. **Read–compare–re-read architectures** remain meaningful alternatives, but question-first processing removes the need to use them merely to repair missing query access in the 2B backbone.  

---

## 3. Experimental unit and fixed execution contract

### 3.1 Define the instance, not just the question

A teacher instance is

\[
H=
(q,\ P,\ o,\ r,\ \mathcal R,\ F,\ \mathcal I,\ \mathcal D),
\]

where \(q\) is the question, \(P\) the candidate pages, \(o\) their order, \(r\) the rendering/preprocessing contract, \(\mathcal R\) the region-to-token map, \(F\) the frozen reader, \(\mathcal I\) the intervention operator, and \(\mathcal D\) the decoding/scoring contract.

Changing the answerer, candidate pages, page order, rendering, or relevant token mapping can change the response surface. Such changes must not silently inherit teacher scores from the old instance.

The selector’s input contains only deployment-available information: the question, page representations, region structure, permitted metadata, and optional retrieval features. Gold answers, the fixed self answer, correctness labels, and inspected evidence annotations are supervision or evaluation resources—not selector inputs.

### 3.2 Primary intervention

The main program uses **deletion of visual decoder-input positions after full vision encoding and before answer-decoder processing**.

The answerer receives retained original visual features, not the selector’s contextual hidden states or learned region slots. Nonvisual prompt/control tokens are handled under a fixed contract.

This is not equivalent to removing all information originating from deleted pixels. Vision features may already be contextualized. Claims concern the specified token intervention and its effect on the reader.

Intermediate deletion remains an optional comparison. It defines a different execution regime because earlier full-context computation and caches may remain. It requires its own measured targets and cost accounting; the one-case late advantage in the pilot is not a reason to make it primary.

### 3.3 Reader and shared-vision compatibility gate

Choose one frozen answerer before generating the first substantial teacher bank. The historical Qwen2.5-VL-7B bank remains useful for legacy diagnostics. Qwen3-VL-8B-Instruct is a candidate for the new program, not an automatic continuation of the old labels.

Qwen3-VL-Reranker-2B remains the preferred approximately 2B initialization to test. However, its released vision configuration differs from Qwen3-VL-8B-Instruct: among other differences, the published vision depths are 24 and 27, with different hidden widths and DeepStack taps. Exact shared visual computation is therefore not available merely by pairing these released checkpoints. 

The compatibility study should compare:

**Native selector vision path:** a diagnostic reference that preserves the reranker’s expected interface.

**Shared answerer vision plus selector-side adaptation:** the intended efficiency route, covering both initial visual embeddings and any additional visual streams.

The frozen answerer’s native pathway must remain unchanged. A feature-alignment warm-up is permissible, but must be validated behaviorally rather than by reconstruction loss alone.

All-keep agreement, sparse-mask positional correctness, cache handling, and consistency across visual streams are necessary checks. Bound this study: shared-vision engineering should not indefinitely block the compact selector’s scientific test. A two-encoder reference can establish whether a failure is caused by interface transfer, while being reported honestly as a more expensive system.

---

## 4. Dataset design: properties before benchmark names

### 4.1 Distinguish three meanings of “long”

Record separately:

\[
\text{source length},\qquad
\text{admitted visual context length},\qquad
\text{required evidence extent}.
\]

A long report with one localized answer is a difficult search setting but not necessarily a multi-evidence question. Two pages containing a linking passage and a table may require more relational selection.

Localized questions still matter. A single answer region with a plausible competitor directly tests the corrective hypothesis. The corpus should not consist exclusively of either trivial localization or unusually elaborate multi-hop questions.

### 4.2 Evidence strata

The audit should assign overlapping tags for:

| Stratum | Scientific role |
|---|---|
| **Localized, low competition** | Ordinary localization, compression, and preservation control. |
| **Localized with plausible competitors** | Entity, scope, category, time, or field disambiguation. |
| **Complementary or distributed evidence** | Preservation of bridges, continuations, headers, and multi-page relationships. |
| **Redundant alternative evidence** | Whether selection can choose an adequate path without treating all relevant regions as necessary. |
| **Broad aggregation** | Boundary of sparse selection when many pages or regions genuinely contribute. |
| **Unanswerable or evidence-missing** | Abstention and hallucination behavior, handled under a separate contract. |

Visual diversity should include prose, tables, charts, photographs, diagrams, scans, and mixed layouts. Answer diversity should include entities, numerical values with units, comparisons, yes/no questions, and complete lists.

The main training mixture should emphasize the first four strata. Broad aggregation and unanswerability can initially be smaller stress-test strata rather than dominating an objective developed for answer-preserving compression.

### 4.3 Audit requirements

For each audited question, establish the answer contract, candidate evidence sets, visual dependence, plausible competing material, retrieval completeness, and action/budget feasibility.

Distinguish:

- several **relevant** pages from several **necessary** pages;
- one annotated evidence path from all possible sufficient paths;
- document-layout boxes from question-specific evidence annotations;
- a source-answerable question from one answerable within the admitted pages.

A model answering correctly from one page or no context is a useful diagnostic, not proof that evidence is unnecessary. Use content review and source annotations together. Mark unresolved evidence requirements as unknown rather than forcing binary labels.

Use a random audit sample to estimate prevalence, plus a separately reported targeted sample of multi-evidence, visual, and complete-list questions. Do not let targeted sampling inflate claims about the natural dataset distribution.

### 4.4 Recommended source roles

| Resource | Initial role | Reason and limitation |
|---|---|---|
| **Existing 17-question pilot** | Mechanism, mapping, retrieval/intervention alignment, and regression-test diagnostics. | Already measured, but curated and heavily inspected. Not a learned-model validation set. |
| **SlideVQA training split** | Primary candidate for cross-page and visually varied training. | Contains 14,484 questions overall, including 2,018 created for multi-hop reasoning; multi-hop is therefore a minority. Decks are split disjointly. Edited bridge questions require artifact checks. |
| **DUDE training split** | Complementary real-document diversity and preservation cases. | Includes varied answer types and reasoning/evidence annotations; audit actual evidence extent rather than relying on the multipage label. |
| **TAT-DQA** | Smaller table/text and complete-answer stratum. | Its released examples contain at most three pages and originate from 182 reports. Useful relational data, not thousands of independent long reports. |
| **MP-DocVQA** | Localization and retrieval-control stratum. | Reuses Single-Page DocVQA questions with neighboring pages added, up to twenty pages. It should not be the sole test of distributed evidence. |
| **M3DocVQA** | Cross-document evaluation or carefully quarantined supplementary material. | M3DocRAG is the framework; M3DocVQA derives from MultiModalQA development questions. Historical project exposure and source overlap must be audited. |
| **ViDoRe** | Retriever validation; V3 additionally supports answer and grounding evaluation. | Do not assume every version provides equivalent answer/evidence supervision. |
| **MMLongBench-Doc** | Reserved long-document transfer and stress evaluation. | Includes single-page, cross-page, and unanswerable questions. It is not a large selector-training corpus. |

These source roles follow the primary releases: SlideVQA supplies deck-level splits and explicit multi-hop construction; DUDE supplies varied question/evidence categories; TAT-DQA has a small parent-report population; MP-DocVQA’s questions originate in its single-page predecessor. 

M3DocVQA’s connection to MultiModalQA means those resources cannot be treated as automatically independent. The release also describes additional training PDFs, but availability and alignment should be verified before committing to a new rendered training corpus. 

ViDoRe V3 adds answers, grounding, and broader query types, but its authors report that extractive queries still predominate and multi-hop examples were harder to scale. MMLongBench-Doc contains 1,082 questions over 135 documents, with distinct localized, cross-page, and unanswerable categories. Both therefore still require stratum-aware interpretation. 

**Initial source recommendation:** begin fresh learning with SlideVQA plus a complementary document source such as DUDE. Add a modest TAT-DQA or MP-DocVQA stratum only where it resolves a particular uncertainty. Do not construct a large heterogeneous mixture before testing whether the learning formulation works.

Using SlideVQA training data makes its held-out decks an **in-domain generalization evaluation**, not an unseen-source transfer benchmark. Reserve other sources for that claim.

### 4.5 Data integrity and split policy

Split by the highest meaningful source unit: report, deck, source article, or document family. Keep related versions, translations, derived questions, and near-duplicate pages together. For TAT-DQA, report-level grouping may require a separate custom evaluation alongside the official split.

Training retrieval indices must not introduce held-out pages as background distractors. A page seen during training is exposed even when no test answer accompanies it.

Audit dataset-parent overlap, permitted usage, asset availability, and evidence-label provenance before ingestion. Freeze benchmark versions and annotation corrections explicitly. Do not mix scores from different annotation versions.

Claims should be “held out from selector training,” not “unseen by every pretrained component,” unless upstream exposure is known.

---

## 5. Retrieval-to-selection interface

### 5.1 Separate retrieval breadth, admitted pages, and answer budget

Use:

\[
R=\text{initial retrieval-pool size},
\qquad
K=\text{pages admitted to the selector},
\qquad
B=\text{answerer visual-token budget}.
\]

A top-20 retrieval pool does not require twenty pages to be processed by the 2B selector. Likewise, admitting eight pages does not imply passing all eight pages to the answerer.

The initial study should cache a reasonably broad ranked pool—top twenty is a sensible starting point—and compare practical joint-admission sizes such as four and eight. Inspect complete-evidence recall at larger values before paying for full teacher banks there.

The original M3DocRAG four-page limit was explicitly tied to GPU capacity, not established as the optimal scientific setting. 

### 5.2 Select the interface using complete-evidence recall and cost

Where sufficient evidence sets are known, let \(\mathcal E_q\) contain alternative sufficient page sets. Measure

\[
\operatorname{CompleteRecall@K}
=
\Pr\!\left[
\exists E\in\mathcal E_q:\ E\subseteq P_K(q)
\right].
\]

If annotations provide only one evidence path, label this as recall of the annotated path. Do not require every relevant page when several are redundant alternatives.

For each candidate \(K\), measure evidence completeness by stratum, unpruned answer quality, selector cost, and teacher branch cost. Hold rendering fidelity fixed. In comparisons asking whether wider access helps, hold the final answer budget \(B\) fixed as well.

Questions scoped to one supplied report or deck should initially retrieve within that scope. Moving an underspecified question into a global collection can change its meaning. Cross-document retrieval should use questions whose scope supports that setting.

### 5.3 Primary and diagnostic populations

The main comparison holds retrieved pages fixed across methods and reports:

**Natural population:** all eligible questions under the frozen retrieval/admission pipeline.

**Evidence-complete stratum:** independently verified cases where admitted pages contain a sufficient evidence set.

**Controlled diagnostic contexts:** tagged contexts in which verified evidence is inserted alongside plausible retrieved alternatives.

The third view measures headroom and mechanisms, not natural end-to-end performance. Neither evidence completeness nor evaluation eligibility should be defined by whether an oracle or learned selector succeeds.

### 5.4 Retriever selection

Treat the interface as **Col-style query–patch matching**, not a permanent commitment to the historical ColPali checkpoint.

Use a small development comparison containing a documented reference such as ColQwen2.5 and an appropriate newer multivector model. Nemotron ColEmbed V2 supplies newer variants suitable for consideration; other ColQwen3 candidates remain eligible after checkpoint and interface verification. Choose using complete-evidence recall, spatially mappable features, query latency, indexing cost, and usage constraints—not a generic leaderboard rank alone. 

Freeze the chosen retriever and page lists for the central learning comparison. Changing retrieval while changing supervision would confound admission and selection.

### 5.5 Wider and hierarchical selection

Prefer joint admission when wider evidence access is necessary and affordable.

If not, evaluate a frozen page-admission stage from a wider pool to the feasible \(K\). Report evidence lost there as admission failure.

Hierarchical or grouped selection is a later extension, justified only if wider evidence recall is materially valuable and joint encoding is prohibitive. Independent group calls are not automatically interchangeable with joint processing: they miss cross-group comparisons, incur additional cost, and need score calibration or a global aggregation stage.

Do not concatenate scores from independent groups and assume they are globally comparable.

---

## 6. Region/action space and coverage

### 6.1 Atomic region contract

Every selectable visual token must belong to exactly one atomic action region:

\[
\bigcup_{i=1}^{n} I_i=\{1,\ldots,N\},
\qquad
I_i\cap I_j=\varnothing\quad(i\ne j).
\]

Use semantic regions where available. Partition uncovered tokens into bounded, spatially local fallback regions, such as grid intersections or connected uncovered pieces.

A page-wide catch-all region is not the default because it can combine distant missed content into one expensive indivisible action. Automatically retaining all uncovered content is also not the default because it silently changes the budget and protects potentially consequential material.

Resolve overlaps deterministically. Hierarchical relationships may remain metadata, but should not cause duplicate token accounting. Required nonvisual control tokens are distinct from selectable visual content.

### 6.2 Coverage and granularity study

The initial audit measures uncovered-token fraction, fallback sizes, evidence inside fallbacks, oversized semantic regions, and evidence sets that cannot fit under the chosen budget.

Compare the semantic-plus-fallback partition with a simple spatial partition on a small diagnostic subset. This asks whether semantic boundaries help beyond a generic spatial action space.

If large tables or combined sections frequently make useful actions impossible, test a deterministic finer partition—such as row groups where reliable structure exists—before scaling labels. Do not refine only hand-inspected failures.

An always-retained residual region and a single catch-all background region may be retained as bounded controls, but all their tokens count toward the budget.

Changing the partition changes feasible actions and usually requires new measurements. Old scores can be reused only for masks that select exactly the same original token positions under the same execution contract.

---

## 7. Teacher signals and supervision

### 7.1 Measured targets

For accepted complete answer string \(a\), define

\[
L_H(a,z)
=
\frac{1}{|a|}
\sum_{t=1}^{|a|}
\log p_F(a_t\mid q,\mathcal I_H(z),a_{<t}).
\]

The initial convention follows the existing mean continuation log-likelihood definition:

\[
G_H(z)=\max_{a\in\mathcal A_H}L_H(a,z),
\]

\[
S_H(z)=L_H(a_0,z),
\qquad
C_H(z)=G_H(z)-S_H(z),
\]

where \(a_0\) is the fixed original all-keep answer for this exact instance.

A branch supplies likelihood measurements, not necessarily a free decoded answer. G/S prefixes can share context processing, but their continuations still have computational cost.

Mean likelihood is not correctness probability. With unequal answer lengths, \(G-S\) is not generally a log probability ratio. The maximum over complete accepted strings is not the total probability mass of all accepted answers.

### 7.2 Why pure contrast is an ablation, not the sole success criterion

For two masks,

\[
D_C(z_a,z_b)
=
[G(z_a)-G(z_b)]-[S(z_a)-S(z_b)].
\]

The same contrast gain can result from improving G or from damaging both targets while suppressing S more. Pairwise differences do not restore the information lost by subtraction.

Therefore, retain both measured G and S even when a model predicts only their contrast. Retrieval features do not repair this target ambiguity.

### 7.3 Preferred initial gold-aware preference rule

Let \(z_0=\mathbf1\) be the common all-keep reference and

\[
g(z)=G(z)-G(z_0),
\qquad
c(z)=C(z)-C(z_0),
\qquad
a(z)=\mathbf1\{g(z)\ge-\epsilon\}.
\]

For initially incorrect examples, the preferred initial ordering is lexicographic:

\[
\kappa(z)
=
\left(
a(z),\
a(z)c(z)+(1-a(z))g(z)
\right).
\]

Thus, an admissible mask is preferred to an inadmissible mask; admissible masks are ordered by contrastive improvement; inadmissible masks are ordered by gold support.

This is a **proposed teacher decision rule**, not a correctness guarantee. Audit the tolerance, comparison margins, and admissible-mask yield before freezing them. Comparisons too close to numerical or annotation uncertainty should remain unordered.

If no measured feasible mask is admissible, retain that fact. The best observed inadmissible mask is not evidence that an acceptable one exists—or that none exists globally.

For initially correct examples, use gold-preservation ordering rather than suppression of a valid self answer. This applies even when the accepted answer is a paraphrase of the canonical gold. Baseline correctness affects label construction, not deployment inputs. The supplied design explicitly preserves this distinction. 

### 7.4 Required supervision comparisons

The early learned study compares, under the same bank and ranking-loss family:

| Supervision | Wrong baseline | Correct baseline |
|---|---|---|
| **Gold-only** | Order by G. | Gold preservation. |
| **Pure contrast** | Order by C without the gold-admissibility protection. | Same gold-preservation treatment for a clean comparison. |
| **Gold-aware contrastive** | The audited ordering above. | Same gold preservation. |

The literal “contrast on every example” formulation can be a diagnostic, but it should not be the principal contrastive control because coincident gold/self functions provide zero preservation signal.

If the admissibility condition rejects almost every mask, or the proposed ordering differs negligibly from G-only, the program must investigate budget feasibility, label definition, and statistical power before claiming a meaningful test of S.

### 7.5 Scoring alignment study

On a small subset, decode a common candidate pool and compare teacher preferences against complete-answer outcomes.

Inspect list serialization, accepted alternatives, EOS/stopping behavior, units, repetition transformations, and output truncation. Preserve the initial scoring definition for continuity while recording sufficient token-level diagnostics.

Only promote an EOS-aware, token-margin, complete-item, or decoded-utility target if this audit identifies a meaningful benefit. A replacement should be compared against matched gold-only supervision as well; otherwise S may receive credit for compensating for a weak G target.

Do not independently normalize G and S before subtraction. For regression, any common scaling should be fitted on training data and reported. Document-level offsets are removed through differences rather than by conflating the two target channels.

---

## 8. Teacher-mask construction as an experimental family

### 8.1 Default: coverage before pair discovery

The default bank consists of diversified, budget-relevant masks. It does not require identifying a small set of consequential regions first.

Generate a mixture of retrieval-agnostic randomized selections, softly retrieval-biased selections, and local variants of several different base contexts. Vary edit sizes: exclusively tiny changes can miss consequential material; exclusively large changes can create mostly broken contexts.

High retrieval priority should increase opportunities to observe a region both retained and removed—not make it mandatory.

An initial bank of roughly 16–32 masks per question is a reasonable test point. For example, twelve diverse feasible contexts plus four local variants illustrates the intended balance, but is not a fixed quota. The later design explicitly demotes repeated guessed-pair experiments from the main bank. 

Before scoring, check duplicate token masks, token-weighted diversity, actual costs, and coverage of plausible regions. Also inspect groups that always toggle together. These are inexpensive safeguards, not proof that the bank contains useful supervision.

### 8.2 Policy comparisons versus structural measurements

**Policy comparisons** involve feasible, similarly budgeted contexts representative of deployment decisions. They drive the direct policy loss.

**Structural measurements** explore effects under varied backgrounds, costs, and region combinations. They diagnose the response surface and may support the mask-conditioned branch.

The distinction prevents a large number of counterfactual interaction labels from overwhelming a policy whose job is to choose one set, not fit every possible context ranking.

For a fixed score vector,

\[
A(H,w+i)-A(H,w+j)=u_i(H,B)-u_j(H,B).
\]

If measured preference for the same exchange reverses across retained backgrounds, no increase in encoder capacity can make this particular additive mask-value expression fit both simultaneously. This does not imply that the policy cannot choose a useful joint set. Record reversal rates rather than discarding inconvenient observations. 

### 8.3 Controlled exchanges and interaction audits

For a local comparison,

\[
z_a=w\cup A,\qquad z_b=w\cup B,
\]

hold background \(w\) fixed and exchange budget-compatible bundles. Randomize backgrounds across comparisons.

For selected diagnostic pairs, evaluate all four configurations and calculate

\[
I^T_{ij}(w)
=
T(w,1,1)-T(w,1,0)-T(w,0,1)+T(w,0,0),
\quad T\in\{G,S,C\}.
\]

These blocks can distinguish possible redundancy, complementarity, and competition. Their semantic interpretation still requires content and answer review.

A fixed background and only two changing regions generally imply different token costs across the four corners. Cost-compensated variants instead measure the entire compensated exchange. Do not label them isolated single-region effects.

The existing privileged pair results justify this as a diagnostic; they do not validate automatic pair discovery. Most per-question measurements should not be concentrated on two guessed pairs unless prospective proposal-yield evidence supports that choice. 

### 8.4 ContextCite’s role

ContextCite remains a diagnostic estimator and optional proposer, not the primary source of region labels.

Fit G/S surrogates to already measured masks where useful. Compare coefficient concentration, stability, ranking, and selected-set prediction against raw measurements. ContextCite approximations should not be promoted to causal ground truth, particularly under interactions or optimization-induced distribution shift. 

A surrogate can propose a new mask, but that mask must be measured before its outcome is used as reader-grounded supervision.

### 8.5 Retrieval/intervention alignment audit

Begin on the existing 17 cases, using the exact revised questions and page inputs. No new large answerer bank is needed to compare retrieval features with existing measurements.

Evaluate whether retrieval priorities cover high-magnitude positive and negative effects in G and S, not only the highest positive G region. Examine contrast concentration separately: concentrated G and S coefficients do not establish a concentrated contrast or identical important-region sets.

Compare against size-aware randomized priorities and inspect known supporting/competing regions. Useful metrics include coverage of consequential candidates at a fixed region or token fraction, magnitude-weighted coverage, profile differences among plausible alternatives, and blind spots by modality and size.

The audit should distinguish three results:

**Retrieval finds consequential regions.**

**Retrieval separates helpful from harmful regions.**

**Retrieval proposes useful combinations.**

The first can justify measurement guidance without establishing the other two. Prospective targeted measurements are required to test new proposals not present in the saved bank.

---

## 9. Selector representations, heads, and inference

### 9.1 Shared input and feature contract

Both encoder designs receive the same permitted semantic inputs and region actions. Col-style features may be enabled or ablated independently of mask guidance.

For region \(i\),

\[
M_{t,i}
=
\max_{j\in\operatorname{patches}(i)}
(q_t^{C})^\top d_j^{C}.
\]

Retain query-vector identity/representation, region geometry, page/document identity, and token cost. Relative matches and strongest-match locations are optional additions.

Associate retriever patches with answerer tokens through spatial coordinates; their grids and embedding spaces need not coincide.

A small contextual matching encoder may learn query- and region-dependent aggregation. Do not use globally fixed weights for query positions or interpret learned weights as causal word necessity. ColPali’s vectors are contextualized, and its formulation includes learned query-augmentation tokens. 

Retain raw regional scores initially, with patch count and size features. Do not divide MaxSim by region size without a decision-relevant audit. Fixed nonnegative weighted MaxSim remains monotone when fixed-bank vectors are added, so weighting alone is not a general model of harmful added context.

Retrieval-guided acquisition and retrieval-feature use are separate experimental factors. Neither changes the measured G/S labels.

### 9.2 Compact selector

The compact design reads the frozen answerer’s visual memory through a small query-conditioned mechanism:

\[
\text{token-level question}
+
\text{visual memory}
\rightarrow
\text{region-anchored readouts}
\rightarrow
\text{bidirectional cross-region comparison}
\rightarrow
e_i.
\]

Region slots are shared learned readers anchored by membership and geometry, not region-ID-specific parameters. They retain access to original visual detail.

A second read after global comparison is a controlled refinement: it may let a region revisit a table after another region identifies the relevant entity or relation. Compare this with extra region mixing at similar cost.

Query-independent pooling followed by the same downstream mixer is the cheap control. Match the number of regional summaries where practical so the experiment does not confuse query conditioning with representational capacity.

The compact selector may reuse compatible pretrained language components, but must not rely on an uncounted full answerer prefill.

### 9.3 Approximately 2B selectors: rich regional reading and pooled readout

#### Experimental role

The approximately 2B study contains two explicit architecture arms:

| Variant | Representation pathway | Role |
|---|---|---|
| **2B-Rich** | Full unpooled proxy prefill → persistent visual and question memories → region-anchored query-conditioned reading → bidirectional cross-page comparison → re-reading of visual detail → region scores. | **Primary early architecture experiment.** |
| **2B-Pooled** | The same proxy prefill → straightforward regional summaries → question/retrieval-feature fusion → small bidirectional region mixer → region scores. | **Matched simplification comparison and potential lower-cost alternative.** |

The central question is:

> Does useful corrective selection require continued access to detailed visual-token representations after cross-region comparison, or do pooled query-conditioned proxy states already expose enough information?

**2B-Rich is included in the first learned-selector stage. It is not conditional on the compact or pooled model succeeding first.** A failure of simpler readouts would not, by itself, adequately test whether the supervision is learnable with a richer representation mechanism.

Both variants retain the same teacher signals, supervision alternatives, region action space, policy-head design, and budgeted selection procedure. Their primary comparison changes the representation/readout architecture, not the definition of a desirable selection.

#### Shared backbone and input contract

The preferred initial backbone candidate remains Qwen3-VL-Reranker-2B. The general-purpose 2B initialization remains a separate control. Shared-vision compatibility is governed by the validation in §3.3; neither readout assumes that visual features are interchangeable between released checkpoints.

Both 2B variants jointly process all admitted pages and the question in **one full-token proxy prefill**. They do not encode each region independently or run separate proxy calls for individual candidate masks.

At a chosen proxy layer \(\ell\), retain

\[
V^{(\ell)}
=
\{v_1^{(\ell)},\ldots,v_N^{(\ell)}\},
\]

and

\[
Q^{(\ell)}
=
\{q_1^{(\ell)},\ldots,q_T^{(\ell)}\}.
\]

Here \(V^{(\ell)}\) is the unpooled visual-position memory, and \(Q^{(\ell)}\) preserves the question as a sequence. Region mapping \(I_i\) identifies the visual positions belonging to region \(i\).

The readout receives page/document identity, token coordinates, region geometry and type where available, original token costs, and query-associated Col-style matching features. All mapped regions, including fallback and low-retrieval regions, remain accessible.

Use one chosen representation layer initially. Multi-layer concatenation or trajectories remain separate refinements rather than part of the basic rich-versus-pooled comparison.

#### Question ordering

Use **question-first ordering in both variants for the primary architecture comparison**. This allows visual-position decoder states to depend on the query while keeping serialization matched.

Question-first ordering does not make the rich reader redundant. It removes the missing causal query-to-visual path; the rich reader additionally tests learned detail extraction, multiple regional working representations, and renewed visual access after cross-page comparison.

The original visual-first/question-last rich configuration remains an ordering alternative. Under that ordering, question-position states can incorporate preceding visual information, while the external reader explicitly conditions extraction on the question. Under question-first ordering, leading question states must not be described as already document-conditioned through the causal backbone.

Do not compare a visual-first rich model with a question-first pooled model and attribute the entire difference to the readout.

#### 2B-Rich: persistent visual memory and region-anchored working slots

For each region, instantiate several working slots:

\[
R_i^{(0)}
=
\{r_{i1}^{(0)},\ldots,r_{is}^{(0)}\}.
\]

Slot parameters are shared across regions. Their instance-specific identity comes from the region's geometry, membership mapping, metadata, and an optional content-dependent initialization—not from learned parameters associated with arbitrary region IDs.

A pooled visual preview may initialize the slots, but **the original visual memory remains accessible afterward**. It is not replaced by that preview.

Several slots allow different details within a table, paragraph, or figure to remain available before final aggregation. Their meanings are learned; no slot is assumed in advance to represent an entity, distractor, answer, or evidence role.

The precise slot count and width are implementation choices. The scientifically important distinction is between an early irreversible summary and a small working representation that can repeatedly consult detailed memory.

#### Query-conditioned local reading

At read stage \(t\), condition the region slots on the complete question sequence and retrieval-matching features:

\[
\widetilde R_i^{(t)}
=
\operatorname{QuestionCondition}
\left(
R_i^{(t)},Q^{(\ell)},M_i,\mu_i,B
\right),
\]

where \(M_i\) denotes query-associated retrieval features and \(\mu_i\) denotes region metadata.

Then read the corresponding visual memory:

\[
L_i^{(t)}
=
\operatorname{CrossAttention}
\left(
\widetilde R_i^{(t)},
V_{I_i}^{(\ell)},
V_{I_i}^{(\ell)}
\right).
\]

The implementation includes learned projections, residual connections, nonlinear processing, and access to local spatial information.

Retrieval features guide extraction softly. They must not hard-restrict the readout to the patches with the highest similarity scores.

This stage allows the same region to be read differently depending on whether the question asks for a year, an entity, a category, a comparison, or all entries satisfying a condition.

#### Bidirectional cross-page comparison

Combine the region slots from all admitted pages:

\[
\{R_i^{(t+1)}\}_{i=1}^{n}
=
\operatorname{GlobalCompare}
\left(
\{L_i^{(t)}\}_{i=1}^{n},
Q^{(\ell)},
\{\mu_i\}_{i=1}^{n},
B
\right).
\]

This module is bidirectional. Earlier-page regions can receive information from later-page regions before their selection scores are produced.

The representation should support comparison of competing evidence and preservation of complementary evidence. A page summary may supplement the computation, but it must not be the mandatory bottleneck through which all cross-page information passes.

All regions participate in this comparison; region boundaries organize reading and actions without asserting semantic independence.

#### Re-reading after comparison

After the first global comparison, the updated region slots read the **same detailed visual memory again**. A further cross-region fusion then precedes the final regional readout.

The intended computation is:

\[
\text{read local detail}
\rightarrow
\text{compare across regions}
\rightarrow
\text{re-read with that context}
\rightarrow
\text{compare and summarize}.
\]

The hypothesis is that information discovered elsewhere changes which local detail should be extracted. For example, a linking passage may identify the song relevant to a question; a table reader can then revisit its original visual tokens to inspect the corresponding theme.

This is a proposed computational advantage, not a guarantee of successful reasoning or a claim that it will rescue a specific pilot case. It addresses the possibility that an initial summary discards a detail whose importance becomes apparent only after comparison.

A shallow sequence containing one re-read is a reasonable starting point. Exact depth, slot count, and width remain flexible.

#### Final regional representations and existing selection heads

After reading and contextualization, aggregate each region's slots:

\[
e_i
=
\operatorname{RegionReadout}
\left(
R_i^{\mathrm{final}},
Q^{(\ell)},
\mu_i,
B
\right).
\]

These representations feed the existing shared policy head:

\[
u_i^\pi(H,B)
=
\operatorname{PolicyHead}_\theta(e_i).
\]

The direct policy score and allocation rule remain:

\[
A_\theta(H,z;B)
=
\sum_i z_i u_i^\pi(H,B),
\]

\[
\hat z
=
\arg\max_{z\in\mathcal Z_B}
A_\theta(H,z;B).
\]

The teacher preferences, losses, and budget contract in §§7 and 9.4–9.7 remain unchanged.

Pooling is permitted at the final readout. The issue under investigation is premature loss of detail, not pooling in every form.

The selector outputs a mask over the original region/token mapping. Its learned slots and proxy hidden states are not passed to the frozen answerer as replacement visual tokens.

#### 2B-Pooled: matched simpler readout

The pooled variant starts from the same proxy layer:

\[
\bar e_i
=
\operatorname{Pool}
\left(
V_{I_i}^{(\ell)}
\right).
\]

It fuses these summaries with the question, retrieval features, geometry, and budget information, applies a small bidirectional region mixer, and uses the same policy-head design.

Its defining restriction is that the downstream region-processing module **does not re-access detailed visual memory after pooling**.

Because the backbone is question-first, the summaries may already be query-conditioned. This is a legitimate simplification comparison, not a deliberately question-blind control.

The initial rich-versus-pooled comparison evaluates complete readout designs. If a difference appears, focused ablations distinguish continued visual access from additional capacity or processing depth.

#### Relationship to the compact selector

The compact selector remains the lower-cost representation alternative in §9.2. Where practical, compare it with 2B-Rich using related region-reader interfaces and the same output heads.

The distinction is the detailed memory supplied to the reader:

\[
\text{Compact: shared vision features + compact question pathway},
\]

\[
\text{2B-Rich: full-token proxy visual states + proxy question states}.
\]

This asks whether pretrained proxy computation makes corrective distinctions more accessible than a compact reader operating directly on shared vision features.

These are system comparisons, not parameter-matched probes. Do not attribute a difference to model size alone; initialization, visual-interface compatibility, and representation processing also differ.

#### Rich regional reading is not optional Head 2

The rich reader belongs to the representation pathway feeding Head 1:

\[
E=\operatorname{EncoderReadout}_\theta(H,B).
\]

It does not consume the candidate keep/drop mask \(z\). The same \(E\) is reused across training-mask comparisons.

The optional mask-conditioned correction remains a separate component:

\[
P_{\mathrm{total}}(H,z)
=
A_\theta(H,z;B)+K_\phi(E,z).
\]

The initial rich-versus-pooled comparison should disable this correction in both arms. This isolates representation quality from additional mask-value modeling.

**2B-Rich with Head 1 alone remains a one-pass direct selector.** Re-reading stored visual states does not rerun the proxy backbone and does not require candidate-context reranking.

Record its cost as

\[
C_{\mathrm{2B\text{-}Rich}}
=
C_{\mathrm{proxy\ prefill}}
+
C_{\mathrm{regional\ reader}}
+
C_{\mathrm{policy\ head}}
+
C_{\mathrm{allocation}}.
\]

The regional reader's additional latency and memory must be measured rather than assumed negligible.

A richer representation also does not remove the additive policy's counterfactual limitation: a fixed score vector cannot reproduce every background-dependent ordering of masks. The experiment tests whether it nevertheless produces more useful deployed selections.

### 9.4 Head 1: direct policy

Let

\[
E=\operatorname{Enc}_\theta(H,B),
\qquad
u^\pi_\theta(H,B)\in\mathbb R^n.
\]

Use a small shared MLP over contextual region representations as the initial readout; a linear readout is the first cheap ablation.

Define

\[
A_\theta(H,z;B)=\sum_i z_i u^\pi_{\theta,i}(H,B).
\]

For teacher-preferred mask \(z_+\) and disfavored mask \(z_-\),

\[
\ell_{\mathrm{rank}}(A)
=
\log\left[
1+\exp\left(
-\frac{A(H,z_+)-A(H,z_-)}{\tau}
\right)
\right].
\]

Balance training by question or source group rather than allowing dense banks to dominate solely through pair count. Multiple comparisons from one bank are correlated supervision, not independent teacher observations.

### 9.5 Optional Head 2: set-dependent correction

Define

\[
P_{\mathrm{total}}(H,z)
=
A_\theta(H,z)+K_\phi(E,z).
\]

The correction is a shallow attention-based set model or pooled nonlinear set model receiving keep/drop indicators. It operates over region representations, not a new full visual-token encoder pass.

The initial joint loss is

\[
\mathcal L
=
\mathbb E\,\ell_{\mathrm{rank}}(A)
+
\lambda\mathbb E\,\ell_{\mathrm{rank}}(P_{\mathrm{total}}).
\]

Both use the same gold-aware ordering where applicable. The direct term is primary because Head 1 is the default deployed policy.

Call \(K_\phi\) a **set-dependent correction**, not a uniquely identified interaction function. For any \(v(H)\), replacing \(u\) by \(u+v\) and \(K_\phi\) by \(K_\phi-z^\top v\) leaves the total unchanged. Auxiliary training does not by itself produce a pure main-effect/interaction decomposition. 

Compare an independent set head only if the linked correction helps and the linkage itself needs explanation. VGent is an analogy for useful auxiliary supervision, not evidence that this decomposition or dropping Head 2 will succeed. 

### 9.6 Numerical effect regression alternative

Preserve direct numerical learning as a substantive alternative:

\[
F_T(H,z)
=
b_T(H)+\sum_i z_i u_{i,T}(H,B)+K_T(E,z),
\quad T\in\{G,C\},
\]

with

\[
\mathcal L_{\mathrm{reg}}
=
\sum_{T\in\{G,C\}}
\operatorname{Huber}
\left(
[F_T(H,z_a)-F_T(H,z_b)]-
[T(z_a)-T(z_b)]
\right).
\]

The G/S parameterization is equivalent in information content. Pure C regression remains a diagnostic comparison.

If deployment uses a predicted gold-preservation constraint relative to all-keep, include measurements anchoring that difference. Equal-budget comparisons alone do not identify the absolute relationship to an unobserved reference.

Predictions must influence decisions through both channels. An unused G auxiliary output does not fix unconditional C maximization. Compare regression and preferences separately from the presence or absence of S so the effects are interpretable.

### 9.7 Default inference and budget

Choose a common achievable token cost \(B^\star\le B\) from region costs, independent of gold information:

\[
\mathcal Z_B
=
\left\{
z\in\{0,1\}^n:
\sum_i c_i z_i=B^\star
\right\}.
\]

Then

\[
\hat z
=
\arg\max_{z\in\mathcal Z_B}A_\theta(H,z;B).
\]

This requires one encoder invocation and one budget-allocation step, not a candidate-context scoring loop.

The optional reranking mode evaluates a small candidate pool using \(P_{\mathrm{total}}\), reusing \(E\). Its cost is approximately one encoder plus multiple small-head evaluations. It is not multiple 2B prefills unless masks enter the encoder itself.

Scores trained only through equal-cost comparisons are not automatically calibrated to choose how much content to retain or compare independent page groups. Adaptive “at most B” selection and grouped-score fusion are separate extensions.

---

## 10. Coordinated experimental program

### 10.1 Stage overview

All quantities below are **planning ranges**, not minimum sample sizes or fixed hyperparameters.

| Stage | Main question | Data and approximate scale | Reuse and additional cost |
|---|---|---|---|
| **0. Legacy audit** | Do retrieval features and existing intervention evidence align enough to motivate guidance? | Existing 17 cases; the 18 originally correct cases where assets permit preservation checks. | Reuse saved masks, outcomes, and mappings. New retrieval features and a few targeted measurements only as needed. |
| **1. Setting and teacher validation** | Are data, retrieval, actions, and teacher labels valid for the proposed study? | Roughly 150–300 fresh audited questions across many documents; score a smaller representative subset. | Cache pages/features. Small 16–32-mask banks; denser measurements only on a diagnostic subset. |
| **2. Early central learning test** | Does learned corrective selection work, does S add value, and does a rich 2B readout improve over pooled proxy states? | Approximately 500–1,000 training questions plus separate document-disjoint development/evaluation groups of a few hundred questions. Include a bounded **2B-Rich** arm, its **2B-Pooled** comparison, and the compact selector. | Reuse one common teacher bank across architectures and supervision arms. The rich model is not contingent on simpler-model success; detailed component ablations follow only when informative. |
| **3. Measurement and representation studies** | Which sampling/features/readouts or heads are justified? | Reuse Stage 2 bank; expand toward 1,000–2,000 questions only when needed. | Most architecture and loss comparisons need no new teacher scoring. Sampler comparisons need matched prospective banks. |
| **4. Scale and acquisition** | Is the next unit of budget better spent on questions, masks, or adaptive proposals? | Nested training cohorts; optional growth toward 5,000–10,000 questions. Adaptive pilot on a smaller training subset. | Reuse existing rows; count all additional proposals, branches, and retraining. |
| **5. Confirmatory and transfer evaluation** | Does the selected system generalize and justify its total cost? | Locked in-domain test plus reserved sources/benchmarks; use full eligible sets where feasible. | Answer generation for selected systems; teacher masks only on a separately identified diagnostic subset. |

Document independence matters more than nominal question counts. A few hundred evaluation questions can establish direction but may be unable to resolve modest gains. Final sample size should follow observed paired variability and the minimum practically important effect.

### 10.2 Stage 0 — existing-evidence alignment and action audit

**Question.** Do Col-style features expose regions and relationships that existing measurements identify as consequential?

**Design.** Recompute retrieval features for the exact pilot questions. Compare raw region profiles and simple priority rules against G, S, contrast, surrogate stability, and conditional measurements. Inspect positive, negative, and counterexample regions—not only successful distractor stories.

Use the 17 cases for spatial mapping, coverage, size effects, score interpretation, and sample-mask feasibility. A small amount of targeted add-back or exchange measurement can validate the most consequential interpretations.

**Interpretation.** Broad coverage of large-magnitude effects supports retrieval as a measurement prior. Poor signed correlation with G alone does not disqualify it: a useful prior may identify both helpful and harmful regions. Failure to cover visual evidence or necessary bridges weakens hard targeting and motivates stronger exploration.

**Dependency.** This study informs the initial sampler and feature defaults. It does not decide the final architecture or validate generalization. Retrieval guidance is not a prerequisite for continuing H1/H2.

### 10.3 Stage 1 — data, reader, and teacher feasibility

**Question.** Does the new experimental population contain enough answerable, budget-feasible corrective decisions to support learning?

**Design.** Audit fresh questions by evidence topology. Freeze candidate retrieval, \(K\), rendering, action partition, answerer, and scoring contract on a bounded subset. Measure unpruned correctness and selected candidates from simple random/retrieval proposals.

Compare page-admission sizes using cached retrieval results before constructing full banks. Validate all-keep execution and the shared-vision path. Decode a small common mask pool to examine gold-aware preference alignment.

**Evidence favoring progression.** Nontrivial useful selections exist across independent documents, preservation data are available, masks produce informative preferences, and the target is reasonably aligned with complete answers.

**Evidence requiring repair.** Missing evidence dominates; coarse actions prevent useful selections; the gate has negligible yield; references are unreliable; or all apparent gains depend on revised/repaired fixtures.

**Dependency.** Fix the teacher instance contract and initial source mixture before Stage 2. Do not require every secondary question—such as ideal query-feature mixing—to be settled.

### 10.4 Stage 2 — test the central idea with an explicit rich 2B arm

**Question.** Can a learned one-pass selector improve complete answers on unseen documents, does original-answer supervision add value beyond G-only, and does a rich region reader expose useful distinctions that pooled proxy states miss?

**Data.** Use the small fresh training and document-disjoint evaluation pools defined in Stage 1 and the stage overview. The existing 17 cases remain implementation and mechanism diagnostics; they do not select the winning architecture.

**Teacher bank.** Build one diversified bank whose proposal procedure does not depend on S. Static retrieval guidance is permissible. Score both G and S so the measurements can be reused across supervision and architecture comparisons.

Hold the answerer, candidate pages, rendering, region actions, budgets, and decoding contract fixed.

#### Early architecture arms

Include:

| Arm | Role |
|---|---|
| **2B-Rich** | Primary high-capacity test using persistent visual memory and read–compare–re-read regional extraction. |
| **2B-Pooled** | Matched simplification comparison using the same proxy initialization and question-first prefill. |
| **Compact reader** | Lower-cost alternative using shared vision features and explicit query-conditioned regional reading. |

**Do not require the pooled or compact model to demonstrate success before running 2B-Rich.** Otherwise an information bottleneck in the simpler representation could become the only test of the learning hypothesis.

The visual-interface compatibility gate remains applicable. A native-vision diagnostic may help isolate interface failures, but its cost and role must be reported separately.

#### Separate supervision and architecture comparisons

The early study should answer two questions without requiring an exhaustive Cartesian product.

**Supervision:** compare G-only with gold-aware contrastive preferences within 2B-Rich under matched data, inputs, adaptation regime, policy heads, and training effort. Retain pure contrast as the specified target-ambiguity diagnostic. Use matched supervision controls for any compact result used to support the added-value-of-S claim.

**Readout:** compare 2B-Rich with 2B-Pooled under the same supervision, question-first ordering, backbone initialization, feature access, budget, and output-head design. Initially disable the optional mask-conditioned correction branch in both.

If the supervision effect appears architecture-dependent, extend the G-only versus gold-aware comparison to both 2B readouts. Do not claim that S helps by comparing a rich contrastive model against a pooled G-only model.

The compact-versus-2B comparison is a whole-system comparison. It need not be parameter-matched, but teacher data and evaluation conditions must be shared.

#### Evaluation

Include unpruned answering, repeated regional-random selection, and simple retrieval-based region selection. Evaluate complete-answer quality, rescue, damage to initially correct answers, and total inference cost under the existing budget contract.

Held-out mask preference accuracy is a supporting diagnostic. Better training fit or mask ranking alone does not establish successful corrective selection.

Use paired evaluation on the same questions and report evidence-topology strata, particularly plausible competition and complementary/distributed evidence.

#### Reuse and training controls

All three architecture arms reuse the same teacher measurements. Changing the selector readout does not require recollecting G/S labels.

Frozen proxy representations may be cached for preliminary readout screening. Confirm the selected configurations with the intended trainable components enabled; stale cached states cannot represent an adapting backbone.

Record both training exposure and actual computation. The richer model's additional readout cost should be measured, not hidden by equal epoch counts.

#### Progression and interpretation

A held-out answer gain, or a credible improving learning curve, justifies the next study. A small inconclusive result does not establish equivalence or impossibility.

If 2B-Rich succeeds where pooled or compact variants fail, retain the rich arm and investigate whether continued detail access, cross-page comparison, or pretrained processing explains the difference.

If 2B-Pooled matches the rich model at lower cost, prefer the simpler system for later scaling.

If G-only matches or exceeds gold-aware contrastive supervision, retain any useful selection result while weakening the S-specific claim.

If neither 2B readout nor the compact model learns, inspect teacher alignment, action feasibility, interface fidelity, comparison reversals, training fit, and data coverage before committing to a large corpus.

**Dependency.** Stage 2 determines which architecture and supervision combinations warrant scale or refinement. It does not require selecting every readout hyperparameter or enabling adaptive acquisition.

### 10.5 Stage 3A — mask proposal and feature contributions

**Question.** Do retrieval-informed measurements or features improve learning efficiency?

**Design.** First compare agnostic versus softly guided diversified banks on the same questions, with matched families, costs, and measurement counts. Evaluate masks prospectively; retrospective analysis cannot score proposals never evaluated by the reader.

Then compare feature absence versus presence on a fixed bank. Complete the small two-by-two only if the individual effects warrant it:

| | No retrieval profiles | Retrieval profiles |
|---|---|---|
| Agnostic sampling | Baseline | Feature effect |
| Guided sampling | Acquisition effect | Combination |

Within feature use, compare scalar regional scores, query-associated profiles, and learned contextual aggregation sequentially. Investigate size calibration only if raw-score admission errors are demonstrated.

**Favoring evidence.** Better held-out selections per teacher cost, useful coverage across questions, and benefits on relevant evidence strata.

**Weakening evidence.** Only larger score variance or more easy catastrophic-deletion labels; no downstream benefit; or a guidance mechanism that repeatedly misses important low-score regions.

**Dependency.** Select the sampler/features for scaling. No requirement that retrieval guidance win.

### 10.6 Stage 3B — explain representation gains and simplify where justified

**Question.** Which aspects of the regional representation and head design provide useful capability, and which can be removed without losing held-out answer quality?

The primary 2B-Rich versus 2B-Pooled comparison begins in Stage 2. Stage 3B explains any difference, tests targeted simplifications, and determines whether optional training components improve the policy that actually runs.

Do not treat this stage as a gate that must be passed before 2B-Rich is evaluated.

#### Focused comparisons

| Comparison | What must remain controlled | Scientific interpretation |
|---|---|---|
| **Re-reading versus extra mixing without renewed visual access** | Same backbone, initial readout, teacher bank, supervision, and similar processing capacity where practical. | Whether returning to detailed memory after comparison helps beyond adding depth. |
| **One versus several regional working summaries** | Same visual memory, question access, downstream head, and comparable summary capacity where practical. | Whether multiple slots preserve useful distinctions before final pooling. |
| **Global versus page-local regional comparison** | Same local reading and comparable processing depth. | Whether explicit cross-page communication adds value. |
| **Question-first versus question-last** | Same backbone initialization, readout family, teacher bank, and training procedure. | Where query conditioning is most useful; do not confound ordering with rich versus pooled architecture. |
| **Linear versus small MLP policy readout** | Identical contextual regional representations. | Whether final score accessibility, rather than upstream representation quality, limits selection. |
| **Reranker versus general-purpose 2B initialization** | Same ordering, regional reader, feature access, supervision, and adaptation budget. | Whether relevance pretraining improves corrective learning. |
| **Direct-only versus additive-linked correction training** | Same encoder/readout and direct-policy objective. | Whether mask-conditioned auxiliary learning improves the deployed direct head. |

The comparisons are not a mandatory Cartesian product. Choose the next one based on Stage 2 outcomes and observed failure modes.

For example, if rich and pooled models are indistinguishable, extensive re-reading and slot-count studies have low priority. If re-reading helps, compare it against additional region mixing before attributing the gain specifically to persistent visual access.

In the 2B model, removing the external global mixer does not eliminate all cross-page information: proxy states may already contain information from preceding pages. Interpret this as the value of the explicit readout-level pathway, not a clean test of all cross-page processing.

The same regional-reading family can be used with the compact selector where practical. This helps separate the contribution of detailed proxy processing from the contribution of the reader itself.

#### Other readout controls

LAST-style attention-weighted pooling remains an inexpensive comparison when the required attention signal is available. Preserve a serialization in which the readout/query position can attend to the visual sequence. A leading question token under causal attention does not supply the same signal as a trailing query readout.

Appended learned region tokens remain an alternative if the external reader is an implementation or representation bottleneck. Their causal ordering and access to all pages must be explicit.

Multi-layer feature trajectories, broad depth/width sweeps, and new page-level bottlenecks are not default experiments. Introduce them only in response to evidence that the existing representation path is inadequate.

#### Reuse and interpretation

Reuse the Stage 2 teacher bank for architecture and loss comparisons. New teacher measurements are required only when the underlying teacher instance, action space, or evaluated mask changes—not because the selector architecture changes.

Cached frozen-feature screens may eliminate clearly unhelpful readouts. Confirm important results with the intended trainable components enabled.

An improvement in the combined mask-conditioned scorer does not establish an improvement in the direct policy. If Head 2 helps only when retained at inference, evaluate that as a separate reranking system with its full cost.

Primary evidence remains held-out complete-answer quality and preservation, accompanied by latency and memory. Preference accuracy, representation probes, and training fit explain outcomes but do not replace answer-level evaluation.

**Dependency.** This stage identifies the smallest justified representation and training configuration for scaling. It may retain 2B-Rich, simplify it to 2B-Pooled, favor the compact selector, or preserve different configurations for different quality/cost operating points.

### 10.7 Stage 3C — preference versus numerical learning

**Question.** Does discarding magnitude information help or hurt?

Compare numerical difference regression against ranking on the same masks and representation. Keep G-only versus paired supervision distinguishable within the comparison. Evaluate actual decisions, not losses on differently scaled targets.

A numerical model that uses a predicted preservation constraint must receive an adequate anchor and be assessed for constraint prediction error. A preference model must be assessed for differences between its training rankings and its optimized selected set.

**Dependency.** Promote the better formulation before large-scale collection only if the difference is material; otherwise retain the simpler preferred ranking formulation.

### 10.8 Stage 4A — more questions versus more masks

**Question.** Does shared learning benefit more from diversity or per-instance precision?

Use nested question cohorts with matched source and correctness strata. Compare allocations such as

\[
512\times64,\qquad
1{,}024\times32,\qquad
2{,}048\times16,
\]

as an illustrative equal-branch-count design.

Measure actual teacher time rather than equating branch counts with cost. More questions require more baseline generation and setup; longer contexts and targets alter branch cost. Match or report selector training exposure so data volume is not confounded with training steps.

Retain dense diagnostic banks on a small subset. Expand toward 5,000–10,000 questions only while held-out quality or precision of the scientific comparison justifies it.

**Dependency.** This study determines scale. Ten thousand questions is not a mandatory milestone.

### 10.9 Stage 4B — bounded adaptive acquisition

**Question.** Can the next measurements be chosen better than static expansion?

Freeze a selector checkpoint. On training instances, measure its actual selected masks, local alternatives, and continued exploratory proposals. Keep the original self answer and reference fixed.

Compare against an equal-cost static expansion, then retrain with the same effort. Evaluate on untouched documents.

Additional proposal sources can be compared on a small subset:

**ContextCite/surrogate proposals:** fitted from the seed bank, then verified through fresh measurements.

**External-model proposals:** grounded region sets or evidence bundles, converted into valid masks and verified by the frozen reader.

**Selector-guided proposals:** current policy choices and alternatives.

Proposals decide what to measure; measured G/S outcomes decide what to learn. Failed and tied proposals remain data. External-model rationales and gold-conditioned proposal inputs must not leak into the deployed selector.

A common bank is essential when isolating supervision. If S-informed acquisition is used, a G-only model trained on that shared bank is a comparison of labels **conditional on S-informed acquisition**, not a completely S-free pipeline. 

One bounded refresh is the initial experiment. Continuous online adaptation, calibrated active-learning uncertainty, or a full reinforcement-learning loop is not required.

### 10.10 Stage 5 — confirmatory evaluation and transfer

Freeze selected configurations and evaluate natural retrieved contexts, the evidence-complete stratum, and reserved source distributions.

Only a few survivors should enter wider-\(K\), alternative-retriever, alternate-budget, and hierarchical-admission evaluations. Separate checkpoint/feature transfer with fixed pages from a new retrieval pipeline that changes the admitted evidence.

Use validated query perturbations and page-order stress tests to assess whether the model follows decisive constraints rather than memorizing source or layout cues. Keep these separate from official benchmark scores.

If H2 is supported, compare the reader’s actual wrong answer against other verified plausible foils, controlling answer type and length where possible. This tests whether the specific error identity matters beyond generic contrastive supervision.

---

## 11. Baselines and fairness rules

| Baseline | Role and controls |
|---|---|
| **Unpruned reader on the same admitted pages** | Measures net answer correction versus the actual starting context. |
| **Fewer whole pages / ordinary page reranking** | Tests whether simpler admission achieves similar benefit. Report actual cost when whole pages cannot match region budgets exactly. |
| **Repeated regional random selection** | Quantifies removal-only effects. Average repetitions; do not choose the best seed using answers. |
| **Simple Col-style region selection** | Tests ordinary matching. Include a sensible coverage/diversity variant where cheap, not only a deliberately weak summed score. |
| **Matched learned G-only policy** | Primary scientific control for S. |
| **Pure contrast policy** | Tests whether gold awareness prevents harmful suppression. |
| **Relevance/evidence-supervised policy** | Compares intervention feedback with defensible evidence supervision on a matched annotated subset. |
| **Native DocPrune or another validated pruning baseline** | Maintains continuity with the original motivation under a compatible reader/execution. |
| **Privileged measured-mask selection** | Estimates observed headroom within a fixed candidate bank. It is not a global oracle upper bound or a deployable system. |

Use the same pages, rendering, answerer, region mapping, and budget where the comparison permits. If token-level pruning and region-level selection use different populations, make the mismatch explicit.

Do not run native DocPrune secretly to determine every other method’s budget without counting that computation.

Evidence-supervised controls must respect partial labels. An evidence page does not make every region on it positive, and an unannotated region is not automatically negative. Any external-model warm-up should be shared by matched supervision arms.

A small reader prompt/completeness control can be conducted during teacher validation. If it changes the reader contract, regenerate affected labels; do not attribute an inexpensive prompt fix to the selector.

---

## 12. Evaluation, statistical interpretation, and cost

### 12.1 Primary outcomes

The primary endpoint is **complete-answer quality on held-out documents**. Use benchmark metrics for comparability, supplemented where necessary by complete-set correctness, numerical units/tolerance, and blinded semantic adjudication.

Report:

\[
\Delta\mathrm{Accuracy}=e\,r-(1-e)\,h,
\]

where \(e\) is unpruned error rate, \(r\) rescue rate among errors, and \(h\) damage rate among initially correct cases.

Do not report rescue alone on a wrong-enriched population as overall improvement. Partial F1 remains secondary for questions requiring a complete list.

### 12.2 Failure decomposition

Where evidence permits, distinguish:

\[
\text{retrieval/admission failure}
\rightarrow
\text{action or budget infeasibility}
\rightarrow
\text{selection failure}
\rightarrow
\text{reader failure despite retained evidence}.
\]

Allow unknown or overlapping categories. Success after token deletion does not automatically prove that the retained tokens contain a complete independently sufficient semantic evidence set.

### 12.3 Diagnostic outcomes

Supporting diagnostics include held-out mask/preference accuracy, selected-set prediction error, gold-admissibility violations, retrieval-priority coverage, conditional preference reversals, evidence retention, and policy sensitivity to question constraints.

They do not substitute for answer evaluation.

No-context and evidence-only controls on a subset help detect parametric shortcuts and reader limitations. They should not automatically filter every easy or prior-knowledge answer from the population.

### 12.4 Statistical plan

Use paired method comparisons on identical questions and document-grouped uncertainty estimates. Include multiple training seeds for the central comparisons and finalists.

Before confirmatory evaluation, choose a minimum practically important answer improvement, an acceptable preservation tradeoff, and the cost criterion supporting an efficiency claim. These thresholds should be motivated by the application and observed variance, not chosen after seeing test results.

Small stages are screening studies. Failure to achieve statistical significance in a small cohort is not evidence of equivalence. Conversely, repeatedly examining the same held-out set turns it into development data; reserve a final test accordingly.

### 12.5 Whole-system cost

Measure separately:

\[
C_{\mathrm{teacher}}
=
C_{\mathrm{index/preprocess}}
+
\sum_q
\left(
C_{\mathrm{baseline},q}
+
\sum_m C_{\mathrm{branch},qm}
\right)
+
C_{\mathrm{proposal}},
\]

and

\[
C_{\mathrm{inference}}
=
C_{\mathrm{retrieval}}
+
C_{\mathrm{online\ preprocessing}}
+
C_{\mathrm{vision}}
+
C_{\mathrm{selector}}
+
C_{\mathrm{allocation/reranking}}
+
C_{\mathrm{reader}}.
\]

Count all admitted-page vision processing, even when most tokens are later removed. Distinguish offline indexing from query-time work and cold from cached execution. Report latency, throughput, peak memory, and actual token counts on common hardware and precision.

A single 2B prefill is not automatically cheap enough. Likewise, cached retrieval embeddings or available API credits do not make their creation or use costless.

---

## 13. Reuse, dependencies, and optional extensions

### 13.1 What can be reused

| Change | Reusable material | New work required |
|---|---|---|
| Supervision rule or loss | Same G/S bank and page features. | New training; possibly label construction. |
| Compact versus 2B encoder | Same teacher bank and actions. | Selector representations/training and interface validation. |
| Retrieval features with fixed pages | Same answerer measurements. | New feature extraction and spatial mapping. |
| New proposed masks | Existing anchors and unchanged-instance scores. | Actual reader evaluation of new masks. |
| New \(K\), page order, rendering, or answerer | Source assets and annotations. | New baseline and affected measurements. |
| New segmentation | Token-identical masks only. | New feasible actions and generally new mask evaluations. |
| Trainable encoder update | Frozen vision/retrieval features where applicable. | Recompute trainable representations; stale caches are not valid. |

### 13.2 Disposition of additional ideas

| Idea | Program position |
|---|---|
| **Read–compare–re-read** | Defining component of the **2B-Rich primary early architecture arm** in §9.3, evaluated in Stage 2 rather than deferred until simpler models succeed. Compare against 2B-Pooled under matched conditions. Use Stage 3B to test whether re-reading, multiple slots, and cross-page comparison explain any benefit. Related compact readouts remain part of the compact architecture family. |
| **Appended learned region queries** | Alternative readout if external region reading is a bottleneck; account for causal ordering and added tokens. |
| **LAST-style attention guidance** | Low-cost representation/pruning control with a valid causal input arrangement. |
| **Smaller pretrained proxy** | Conditional capacity/cost alternative if the compact reader lacks semantic capability and 2B is expensive. |
| **Intermediate answerer-state head** | Later placement study only; count full prefix computation and measure the correct intervention surface. |
| **Hierarchical page/region routing** | Wider-context extension after evidence-recall and capacity audits; not the starting system. HierDoc is relevant precedent but uses a different evidence-routing/answerer interface.  |
| **Explicit correction to retrieval scores** | Optional parameterization. Do not add raw retrieval scores to \(G-S\) without training the combined quantity and handling scale. |
| **Direct coefficient distillation** | Diagnostic comparator at most; not the principal training labels. |
| **External-model direct distillation** | Separate warm-up/control if oracle-verified proposals alone are insufficient. |
| **Offline policy distillation from Head 2** | Triggered when the richer scorer is substantially better but inference reranking is undesirable. Verify learned-teacher preferences against measurements. |
| **Online self-answer interventions** | Deprioritized. They add a draft and branches, and S alone does not reveal G without validated additional assumptions. |
| **MaxSim redesign or ordinary retriever fine-tuning** | Separate later direction only if evidence shows retrieval aggregation/admission is the dominant bottleneck. |
| **Pure interaction identification** | Not required for selection. Add constraints only if interpretability becomes a separate research claim. |
| **Mandatory pair-factorial masking, giant catch-all regions, fixed token-position mixing, automatic area division** | Deprioritized defaults because of brittleness, information loss, or unsupported assumptions. |

---

## 14. How the program should change in response to results

| Result | Supported conclusion and next action |
|---|---|
| Learned selection improves complete answers and cost | Supports H1; proceed to attribution of gains and transfer. |
| Paired supervision beats matched G-only | Supports H2; examine data efficiency and actual-error specificity. |
| G-only matches paired supervision | Retain the useful selector if warranted; weaken or remove the S-specific claim. |
| Retrieval guidance improves proposals but not learned generalization | Better offline search, not yet a better training method. |
| Retrieval features help while guided masks do not | Keep features; use a simpler sampler. |
| Head 2 fits measurements better but Head 1 does not improve | Do not credit the direct-deployment method; test limited reranking or distillation separately. |
| Compact fails while 2B succeeds | Evidence for richer computation or initialization, not proof that parameter count alone caused the difference. |
| Both fail despite validated useful actions | Revisit teacher alignment, representation accessibility, comparison distribution, and data diversity before scaling. |
| Gains disappear against fewer-page or stronger retrieval baselines | Ordinary admission/localization may explain the benefit. |
| Gains require repaired contexts | Restrict the claim to controlled headroom until natural-context generalization appears. |
| Quality improves but total cost worsens | Report a quality–compute tradeoff, not an efficiency improvement. |
| Finer regions rescue persistent failures | Action granularity is a justified next component; recollect affected labels under the new contract. |

### Program completion criterion

The program should end with a compact explanation of which components were necessary, which were unnecessary, and which hypotheses were weakened—not with every proposed component attached to one large system.

The strongest successful outcome is a selector that:

1. improves complete answers on unseen documents while preserving initially correct answers;
2. demonstrates a defensible total quality/cost advantage;
3. benefits from its chosen supervision under matched controls;
4. and retains that benefit outside the curated pilot and a single retrieval setting.

A valid outcome may also be narrower: gold-only utility works as well as paired supervision; simple pooling suffices; retrieval helps features but not measurement acquisition; or a compact policy is preferable to the 2B system.

**The experimental sequence is designed to discover those outcomes before committing to large teacher collection or elaborate secondary mechanisms.** Each stage produces a decision, a reusable data artifact, and a clearly scoped next question—enough for a subsequent implementation plan without silently turning provisional choices into scientific assumptions.