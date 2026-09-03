# Executive verdict

The project is **scientifically plausible and worth testing**, but the strongest defensible contribution is narrower than “distilling causal attribution into a learned probe.”

Prior work has already demonstrated nearly every ingredient separately:

- hidden properties can be decoded from frozen intermediate representations;
- lightweight probes can sometimes recover information that is not reflected in the model’s output;
- attribution can be amortized across examples instead of refitting a local surrogate;
- hidden-state-dependent masks and routers can be learned while the backbone remains frozen;
- learned VLM token selectors can outperform attention heuristics;
- intervention-derived evidence labels can train routers that repair some model errors.

Most importantly, **vSTREAM is very close**: it trains a shared linear visual attribution model from random semantic-region ablations using attention features, and applies it without new ablations at test time. It also includes DocVQA. That substantially weakens any broad novelty claim about amortized causal visual attribution. 

Your remaining unresolved leap is harder and more specific:

> **Can information available before answer generation in a frozen document VLM predict the gold-conditioned, post-boundary physical-deletion effect of individual semantic regions—especially when the unpruned model is wrong—well enough to support zero-intervention corrective pruning on unseen documents?**

I did not find prior work establishing that full combination.

The best first experiment is **not yet a general learned router**. It should be a two-gate test:

1. **Target-structure gate:** Is your post-boundary intervention response sufficiently additive and predictable for a static per-region utility to make sense?
2. **Representation gate:** Conditional on that ceiling, can a small shared probe recover the deletion response from pre-answer frozen states on document-held-out questions?

Without Gate 1, a failed linear probe would be uninterpretable: it could mean the hidden state lacks the information, or merely that “one static utility per region” is the wrong label model.

---

# 1. What the literature actually establishes

## 1.1 Classical probing: hidden-state information can emerge before behavior does

The classical linear-probing program attaches low-capacity classifiers to frozen layers and asks where a target becomes linearly separable. Layerwise diagnostic classifiers and structural probes established this as a standard way to study representation emergence across depth. 

That supports your proposed layer sweep, but the standard interpretation is limited:

> Probe success means that a property is **decodable from the representation under the probe and data distribution**. It does not establish that the backbone uses that property in its normal computation.

Several methodological results are directly relevant:

- Probe accuracy can reflect the probe learning the task or exploiting memorized regularities rather than a clean property of the representation; control tasks and selectivity were introduced to expose this. 
- Minimum-description-length probing asks how sample-efficiently the property can be extracted, which is often more informative than final accuracy alone. 
- Amnesic and causal probing explicitly distinguish “information exists in the state” from “the model behavior depends on it.” 

That means your study must report three different things:

1. **Decodability:** can the sidecar predict deletion outcomes?
2. **Counterfactual fidelity:** can it predict unseen mask responses?
3. **Behavioral utility:** does pruning according to it actually improve or preserve answers?

A single probe correlation cannot stand in for all three.

## 1.2 Latent knowledge and error probes provide positive—but much easier—precedents

Latent-knowledge work shows that frozen activations can contain information that is not faithfully expressed in generation. Contrast-Consistent Search recovered truth-related directions even under prompts encouraging incorrect answers, while Inference-Time Intervention used selected activation directions to improve TruthfulQA behavior. 

This is evidence against the simplistic objection:

> “If the model answered incorrectly, its hidden states cannot contain anything useful about the correct answer.”

That objection is false in general.

But this literature does **not** establish your premise. Usually the target is a global binary fact, answer correctness, or a low-dimensional truth direction. You are asking for a substantially richer object:

\[
(q,\text{document state},i)
\longmapsto
\text{counterfactual effect of deleting region }i\text{ on a gold sequence}.
\]

The probe must implicitly recover:

- something about the correct answer;
- which region bears on it;
- how that region’s information has propagated by the chosen boundary;
- how removal will affect the remaining computation;
- and whether redundant alternatives remain.

Recent negative evidence is also important: some apparent “truthfulness probes” seem to distinguish whether the model is confidently recalling familiar parametric knowledge rather than whether the answer is actually true. In other words, an internal confidence or familiarity direction can look like latent correctness without representing correctness itself. 

Your equivalent danger is that the probe learns:

- answer regions are usually tables;
- answers usually occur on earlier pages;
- large OCR-heavy regions are usually useful;
- DocPrune’s score is usually adequate;
- certain templates usually place answers in the same location.

Those shortcuts could produce respectable mask correlation without demonstrating latent gold-conditioned reasoning. Document- and template-held-out evaluation is therefore essential.

## 1.3 VLM and document probes make the proposal more plausible

Pre-generation VLM probes have shown that hidden states can predict whether a forthcoming answer will be reliable before generation, often with the final prompt or query states being more informative than visual-token pooling. The best layer and state type are architecture dependent. 

Work on visual-document understanding has similarly found representation–response gaps: information required for a task can be more recoverable from intermediate representations than from the model’s actual output. But it also reports that simple mean-pooling of visual tokens can be weak, while final prompt states are often substantially stronger; relational document questions remain harder. 

Other VLM analyses have separately probed the ground-truth property, the model’s predicted property, and whether those two disagree. These results support your proposed distinction between a latent gold signal and the model’s intended answer, rather than treating correctness as one scalar. 

However, failure-mode probes also indicate that detecting “something is likely wrong” is easier than diagnosing exactly whether the failure is visual access, reasoning, or answer production. Fine-grained attribution of the error source can remain weak even when global error prediction works. 

So the literature gives a clear plausibility ordering:

\[
\text{global error detection}
<
\text{correct-answer decodability}
<
\text{region-specific gold counterfactual prediction}.
\]

Your target is at the difficult end.

---

# 2. Attribution and amortization precedents

## 2.1 ContextCite: the direct teacher precedent

ContextCite samples context masks, evaluates the likelihood of a fixed generated response, and fits a sparse linear surrogate whose coefficients act as context attributions. Its held-out subset prediction and deletion experiments support the use of mask-response regression as a behaviorally grounded local explanation. 

Your method differs in three material ways:

1. You intervene on intermediate visual states rather than ordinary input passages.
2. You physically shorten the sequence at a decoder boundary.
3. Your main target is a **gold answer the model may not generate**, not merely support for the model’s observed generation.

That third difference is the difficult one. ContextCite is largely asking:

> Which context supports the answer the model actually produced?

You are asking:

> Which surviving region would improve the model’s support for the answer it should have produced, even when the current computation favors another answer?

That is closer to latent corrective supervision than ordinary attribution.

## 2.2 AT2: shared attribution learned from intervention outcomes

AT2 is conceptually very close to your amortization argument. Rather than fitting an attribution model independently for each example, it learns a generalizable mapping from model features—particularly headwise attention patterns—to intervention-derived context importance, and uses those scores for context pruning. 

But AT2 has an easier information setting: the generated target response is part of the attribution problem. Its features can describe how that particular answer attended to sources. Your proposed router acts before answer generation and cannot use gold or generated-answer token states.

That distinction must be explicit in the paper. Otherwise a reviewer can reasonably say the core method is AT2 applied to visual regions.

## 2.3 FastSHAP and CXPlain: the broad claim is already established

FastSHAP learns a shared explainer that predicts Shapley-style attributions in one forward pass rather than recomputing expensive coalitional evaluations for every test example. CXPlain learns removal-based explanations for a frozen predictor and uses uncertainty over the learned explainer. 

Therefore, you should not claim novelty for:

> “Train a network from expensive intervention outcomes so that attribution becomes cheap at inference.”

That is an established strategy.

What can still be novel is the **particular counterfactual target and deployment setting**.

## 2.4 DiffMask: amortized hidden-state deletion is not new either

DiffMask may be the closest older conceptual precedent. It learns an amortized mask predictor from hidden states, can mask representations at selected layers and continue the frozen model from there, and performs layerwise analysis of when information becomes discardable. 

Its objective is fundamentally different, however. DiffMask aims to remove information while preserving the model’s existing prediction or output distribution. It is therefore analogous to your \(S\) target:

\[
\text{preserve what the model already wants to say},
\]

not your corrective \(G\) target:

\[
\text{increase support for what the model should say}.
\]

A successful self-utility probe would mostly reproduce the DiffMask-style result in a document-VLM setting. The scientifically important result is whether the **gold head works on baseline-wrong examples**.

## 2.5 vSTREAM is the largest novelty threat

vSTREAM formalizes visual attribution through counterfactual semantic-region ablations, trains a shared linear estimator on those intervention outcomes, uses compact per-layer/per-head attention features, and performs single-pass attribution at inference. It includes document VQA evaluation and studies cross-task transfer. 

The overlap is substantial:

| Component | vSTREAM | Your proposal |
|---|---|---|
| Semantic visual regions | Yes | Yes |
| Random region ablations | Yes | Yes |
| Frozen VLM | Yes | Yes |
| Shared learned attribution model | Yes | Yes |
| Linear estimator | Yes | Candidate |
| Attention/head features | Yes | Candidate |
| No ablations at inference | Yes | Yes |
| Document VQA | Included | Central |
| Pre-answer features only | No; target-span information is used | Yes |
| Gold target on baseline-wrong questions | Not the central setup | Central |
| Intermediate physical sequence deletion | No | Yes |
| Corrective answer rescue | Not the central objective | Central |
| Gold/self decomposition | No | Proposed |
| Multi-page fixed-budget routing | Not the central setting | Central |

This means the project should **cite and build against vSTREAM directly**, not present it as merely adjacent work.

The key comparative experiment should be:

- a vSTREAM-like post-generation/self-support estimator;
- a pre-answer self-support estimator;
- a pre-answer gold-support estimator;
- and the pre-answer gold/self joint model.

That comparison isolates exactly which new information burden you are introducing.

---

# 3. Learned causal routing and token-pruning precedents

## 3.1 Intervention-supervised routers can repair errors

“Learning What Matters” or causal-evidence routing is particularly relevant because it uses interventions to create evidence labels, trains a cheap selector, and shows that causal supervision can outperform attention-based selection and remove stale or conflicting context that would otherwise induce wrong answers. It also highlights redundancy: multiple alternative sufficient evidence families can make singleton importance misleading. 

This is strong positive evidence for the high-level idea:

> A frozen model’s expensive intervention behavior can supervise a learned router that later repairs some errors.

But its controlled text-record setting is materially easier:

- records are discrete and semantically clean;
- evidence relations can often be constructed or verified;
- answer support is less entangled with OCR and layout;
- labels can represent complete alternative evidence families;
- the router is not required to decode a post-boundary physical-deletion effect from internal VLM states.

Still, reviewers will correctly view it as a direct conceptual precursor.

## 3.2 VLM pruning has rapidly moved beyond heuristic attention

DocPrune is designed for document-specific sparsity and structured layout and combines background, question-aware, and comprehension-stage pruning. 

But the broader pruning literature has weakened the assumption that high attention is synonymous with utility:

- DART reports that importance-ranking methods can behave no better than—or worse than—random token selection when they retain redundant tokens, and prioritizes representational diversity instead. 
- Other analyses find that deep visual representations can become sufficiently homogeneous that common importance scores lose discriminative value. 
- Spatial and holistic-retention work warns that individually “unimportant” visual tokens can collectively preserve layout or relational structure. 
- Cross-modal pruning analyses have identified causal, semantic, and spatial misalignment between text-guided scores and actual visual information needs. 

This supports your decision to treat attention as a feature rather than a causal label.

It also supplies the strongest objection to an independent per-region score: document evidence is frequently redundant, relational, and spatially structured.

## 3.3 Learned sidecars are already common

Recent methods train lightweight modules attached to frozen or mostly frozen VLMs:

- MAP distills middle-layer attention behavior into an earlier lightweight predictor. 
- GlimpsePrune learns dynamic visual-token selection rather than relying on fixed attention heuristics. 
- CROP uses query-conditioned localization to guide visual pruning. 
- AutoSelect and DiffPrune attach learned scorers to frozen VLMs and train token retention through differentiable information suppression or noise gating. 
- StepPrune explicitly rejects independent token scores and uses a sequential pointer policy conditioned on previously selected tokens, while keeping the backbone frozen. 

Therefore, “parametrically decoupled but informationally integrated” is a sound architecture principle, but not by itself a contribution.

The contribution has to come from the **causal teacher, pre-answer corrective target, and layerwise scientific analysis**.

---

# 4. What remains genuinely unresolved

## Already demonstrated

Prior work supports all of the following:

- Frozen hidden states can linearly expose latent properties.
- Model states can contain useful information not reflected in the emitted answer.
- Attribution can be amortized across examples.
- Learned masks can be predicted from hidden states.
- Visual-region ablation outcomes can train a shared linear estimator.
- Small learned VLM routers can operate with frozen backbones.
- Intervention-supervised routers can sometimes improve incorrect answers.
- Static independent token scores can fail because of redundancy and interaction.

## Close, but materially easier or different

The closest work generally predicts one of these:

- support for the model’s own generated answer;
- preservation of the current output distribution;
- task relevance on correctly answered examples;
- ordinary visual salience or localization;
- attention-derived importance;
- synthetic or constructed evidence relationships;
- end-to-end average task performance.

Those are all easier than predicting **gold-conditioned corrective utility on examples where the model is already wrong**.

## The actual leap

Your sidecar must infer, from a pre-answer state:

1. what answer is likely correct;
2. which region contains evidence for that answer;
3. whether that evidence is already preserved elsewhere;
4. how much influence from that region has already propagated into surviving states;
5. whether deleting it at this particular layer would help or hurt;
6. and how that effect changes when other regions are also removed.

That is a real open question.

It is also why the project is interesting even if the answer is negative.

---

# 5. Stronger formulation of the research idea

I would stop calling the target simply “region importance.”

Define the deletion operator at boundary \(d\) as \(\mathcal{D}_{d}(D)\), which removes the hidden-state positions belonging to region set \(D\), preserves the surviving tokens’ original positional identifiers, and continues the remaining layers.

Then define:

\[
\Delta^G_{q,d}(D)
=
G_q(\mathcal{D}_{d}(D))
-
G_q(\mathcal{D}_{d}(\varnothing)),
\]

\[
\Delta^S_{q,d}(D)
=
S_q(\mathcal{D}_{d}(D))
-
S_q(\mathcal{D}_{d}(\varnothing)).
\]

For a singleton:

\[
\delta^G_{qi,d}=\Delta^G_{q,d}(\{i\}),
\qquad
\delta^S_{qi,d}=\Delta^S_{q,d}(\{i\}).
\]

These are not intrinsic evidence scores. They are:

> **Layer-, model-, prompt-, target-, and deletion-operator-specific post-boundary deletion responses.**

That terminology avoids several overclaims:

- It does not imply you recovered the model’s true computation graph.
- It does not imply the region was originally used.
- It does not imply the same region would matter at another layer.
- It does not imply the effect is stable under deletion of other regions.
- It acknowledges that information may already have propagated into question or text states before deletion.

A strong project formulation is:

> **Pre-answer causal deletion-risk distillation for corrective document pruning:** train a low-capacity sidecar to predict the gold- and self-conditioned post-boundary deletion responses of semantic document regions from frozen VLM states, and test whether those predictions support safe or corrective region removal without inference-time interventions.

That is substantially sharper than “learn an importance head.”

---

# 6. One important architectural correction

The proposed scalar head

\[
u_i=w^\top r_i
\]

is a useful control, but probably not the right primary model.

In decoder-style VLM prompting, visual tokens are commonly placed before the textual question in an autoregressive sequence. Earlier visual positions therefore cannot ordinarily incorporate later question tokens through causal attention. Qwen-style models process visual tokens as a prefix inside an autoregressive language model. 

Consequently, a visual-region representation \(r_i\) may be semantically rich but largely **question independent**.

Merely concatenating a question state does not fix this:

\[
u_i=w_r^\top r_i+w_q^\top q.
\]

For a fixed question, \(w_q^\top q\) is the same for every region. It changes the score offset but cannot create a question-dependent relative ranking among regions.

You need at least one genuine question–region interaction:

\[
r_i^\top Wq,
\]

or model-derived cross-modal features such as:

\[
Q_{\text{final-prompt}}K_{\text{region}}^\top.
\]

This is not unnecessary architectural complexity. It is the minimum required to represent query-conditioned region ranking unless your region state is already question-conditioned through some noncausal or reordered computation.

---

# 7. The recommended initial feasibility experiment

## Phase 0: first test whether a scalar per-region target is viable

Use the existing 48-question, 256-mask pilot before collecting new data.

For every question and target \(T\in\{G,S,G-S\}\):

1. Split the 256 masks into fitting and held-out sets, for example 192/64.
2. Fit the best permissible per-question additive surrogate on the fitting masks:

\[
\widehat{\Delta T}_q(D)=\sum_{i\in D}c_{qi}.
\]

3. Evaluate it on unseen masks, especially masks near the actual deployment budget.
4. Repeat across several splits.
5. Compare against:
   - a mask-size-only model;
   - an additive ridge model;
   - LASSO;
   - a low-rank pairwise model.

For sampled region pairs, estimate:

\[
I^T_{ij}
=
\Delta T(\{i,j\})
-\Delta T(\{i\})
-\Delta T(\{j\}).
\]

Large \(I_{ij}\) reveals interaction:

- negative or positive redundancy;
- complementarity;
- alternative evidence chains;
- joint distractor effects.

### How to interpret Gate 0

**High held-out additive fidelity**

If a free per-question additive model predicts unseen mask outcomes reasonably well, then a shared additive probe has a meaningful target. Its remaining problem is to infer those coefficients from the state.

**Low held-out additive fidelity**

Then “one utility per region” is falsified before probing begins. A global linear probe cannot reasonably solve a target that even a separate per-question coefficient vector cannot represent.

This failure should lead directly to a pairwise or set-aware model—not to a larger per-region MLP.

I would require at least:

- positive held-out \(R^2\) for a clear majority of questions;
- moderate rank correlation near the deployment budget;
- no systematic collapse on baseline-wrong examples;
- and materially better prediction than mask cardinality alone.

The exact numerical gate should be set on the 48-question development set and frozen before the new cohort.

## Phase 1: redistribute oracle compute toward more questions

For learning a shared probe, **48 questions with 256 masks each is the wrong allocation**. The effective independent units are questions and documents, not mask rows.

For perspective:

\[
48\times256=12{,}288
\]

masked continuations.

The same continuation count could instead support:

\[
384\times32=12{,}288
\]

questions-by-masks.

That does not mean 32 is automatically optimal, but it illustrates the central point: once Gate 0 characterizes the within-question response surface, probe training needs breadth across questions, documents, layouts, and failure types.

A sensible initial cohort would contain approximately:

- **400–600 questions**;
- roughly balanced baseline-correct and baseline-wrong cases for training;
- at least several hundred distinct source documents;
- document-disjoint train, validation, and test splits;
- a secondary test set sampled at the natural correct/wrong prevalence;
- a template- or vendor-held-out split where metadata permits it.

The 48-question pilot should remain development data. A separate oracle-headroom confirmation cohort should remain locked rather than becoming probe-training data.

### A 32-mask design per training question

One workable allocation is:

- 8 sampled singleton deletions;
- 16 masks concentrated around the intended retention budget;
- 4 masks at a less aggressive retention rate;
- 4 pair- or small-set-focused masks for interaction coverage.

The design should balance each region’s inclusion frequency and avoid strong collinearity among region masks.

The held-out test should include fresh:

- singleton deletions;
- budget-matched random subsets;
- and policy-like subsets resembling those the learned router will actually select.

That last category matters because random-mask fidelity need not transfer to top-ranked or sequentially selected masks.

---

# 8. Train directly on mask outcomes, not LASSO coefficients

Let \(d_{qi}=1\) mean region \(i\) was deleted. Center the observed target on the full context:

\[
\Delta G_q(d)=G_q(1-d)-G_q(\mathbf 1).
\]

The probe produces one predicted singleton deletion response:

\[
\hat\delta^G_{qi}=f_\theta(x_{qi},q_q).
\]

Its predicted set response is:

\[
\widehat{\Delta G}_q(d)
=
\sum_i d_{qi}\hat\delta^G_{qi}.
\]

Train against the actual mask outcomes:

\[
\mathcal L_G
=
\frac{1}{|\mathcal Q|}
\sum_q
\frac{1}{M_q}
\sum_m
\rho\left(
\widehat{\Delta G}_q(d_{qm})
-
\Delta G_q(d_{qm})
\right),
\]

where \(\rho\) can be squared error or Huber loss.

This is preferable to supervising on per-question LASSO coefficients because those coefficients are:

- regularization dependent;
- unstable under collinearity;
- dependent on the sampled mask distribution;
- nonunique under redundant evidence;
- and an unnecessary intermediate pseudo-label.

The direct mask loss still imposes additivity, but it estimates the shared probe parameters from the actual behavior you care about.

Weight questions equally. Do not allow a question with more masks to dominate simply because it generated more rows.

---

# 9. The smallest appropriate architecture ladder

The experiment should distinguish strict linear decodability from cheap predictability using frozen cross-modal features.

## Model 0: nonsemantic controls

Use:

- page index;
- bounding box and region area;
- region token count;
- semantic region type;
- OCR density;
- page and region counts;
- DocPrune’s native score;
- pruning layer;
- retention budget.

This is essential. A hidden-state probe is not interesting if layout and DocPrune metadata explain the same effect.

## Model 1: hidden-region linear control

\[
\hat\delta_i
=
b_\ell+w_r^\top \widetilde r_i+\gamma^\top g_i.
\]

Here \(\widetilde r_i\) is a standardized or low-dimensional pooled region state.

This tests whether average semantic properties of regions predict deletion effects. It is not a sufficient query-conditioned model when visual states precede the question.

## Model 2: pre-answer QK-linear probe — recommended primary model

At the answer-start or final-prompt position, obtain the already-computed query vector for each head. For region \(i\), summarize interactions with its keys:

\[
a_{ih}^{\text{mean}}
=
\frac{1}{|R_i|}
\sum_{t\in R_i}
\frac{q_h^\top k_{th}}{\sqrt{d_h}},
\]

with compact variants such as:

- mean;
- maximum;
- log-sum-exp;
- normalized attention mass;
- value-contribution norm;
- question-token-query versus final-prompt-query summaries.

Then fit:

\[
\hat\delta_i
=
b_\ell+\beta^\top a_i+\gamma^\top g_i.
\]

This remains a linear estimator over compact frozen-model features, but unlike \(w^\top r_i\), the features are genuinely question–region dependent.

It also avoids storing or materializing the full quadratic attention matrix. If FlashAttention does not expose the weights, computing only one or a few prompt-query vectors against the visual-region keys is \(O(N)\) in the visual sequence rather than \(O(N^2)\).

## Model 3: low-rank bilinear probe — one prespecified escalation

\[
\hat\delta_i
=
b_\ell
+w_r^\top \widetilde r_i
+\beta^\top a_i
+\gamma^\top g_i
+
\sum_{k=1}^{R}
(u_k^\top\widetilde r_i)(v_k^\top\widetilde q),
\]

with \(R=4\) or \(8\).

This explicitly asks whether a low-dimensional question–region compatibility relation is decodable.

It is still tiny relative to the VLM. For a hidden width of a few thousand, a rank-4 bilinear term is tens of thousands of parameters, not millions.

## Do not begin with an MLP

Starting with an MLP would blur several outcomes:

- the information is linearly accessible;
- the information is present but requires nonlinear extraction;
- the model is learning dataset-level region heuristics;
- the model is compensating for an invalid additive target.

The correct sequence is:

\[
\text{controls}
\rightarrow
\text{QK-linear}
\rightarrow
\text{rank-4 bilinear}
\rightarrow
\text{tiny MLP only after diagnosis}.
\]

---

# 10. Gold and self should initially be separate heads

Train:

\[
\hat\delta_i^G=f_G(x_i,q),
\qquad
\hat\delta_i^S=f_S(x_i,q).
\]

The self target is scientifically valuable because it provides an internal positive control:

- If \(S\) is predictable but \(G\) is not, the state exposes support for the model’s intended answer but not latent corrective evidence.
- If both are predictable, their mismatch may identify incorrect-answer drivers.
- If neither is predictable, the chosen state/features or deletion model are inadequate.
- If \(G\) is predictable only on baseline-correct examples, the sidecar is learning ordinary relevance rather than correction.

Under your deletion-delta convention, the strongest corrective score is:

\[
c_i=\hat\delta_i^G-\hat\delta_i^S.
\]

But **do not select regions using \(c_i\) alone**.

A region could have a large positive difference while deletion still harms gold:

\[
\hat\delta_i^G=-0.2,\qquad
\hat\delta_i^S=-1.0,\qquad
c_i=0.8.
\]

Deleting it damages both answers, merely damaging the self answer more.

The correct constrained policy is:

1. establish gold safety;
2. only among gold-safe regions, use \(c_i\) to prioritize corrective removal.

The two-dimensional outputs are therefore more useful than their difference.

---

# 11. The deletion-auditor framing is stronger—but only under abstention

Your intuition is right with an important qualification.

A deletion auditor is genuinely easier than global importance ranking when it is allowed to say:

> “I am confident these regions are safe or beneficial to remove, but I abstain on the others.”

Then it needs high precision, not a perfect total ordering.

But if the auditor must always hit an aggressive fixed budget entirely from its own scores, it eventually has to rank uncertain regions. At that point, much of the original importance problem returns.

Therefore evaluate two distinct deployment modes.

## Mode A: selective deletion auditor

Estimate uncertainty using a bootstrap ensemble or several independently trained probes.

For a practical equivalence margin \(\epsilon\):

- **Protect:** upper confidence bound on \(\delta_i^G\) is below \(-\epsilon\).
- **Certified safe:** lower confidence bound is above \(-\epsilon\).
- **Certified beneficial:** lower confidence bound is above a positive threshold \(\tau\).
- **Uncertain:** confidence interval crosses the relevant boundary.

Evaluate a risk–coverage curve:

\[
\text{coverage}
=
\text{fraction of tokens certified for deletion},
\]

\[
\text{risk}
=
P(\delta_i^G<-\epsilon\mid \text{certified deleted}).
\]

This is the cleanest test of the deletion-auditor hypothesis.

## Mode B: fixed-budget hybrid router

To reach the exact DocPrune token budget:

1. veto deletion of probe-protected regions;
2. remove certified beneficial regions first;
3. remove certified safe regions next;
4. fill any remaining budget using DocPrune or a diversity-aware baseline among uncertain regions.

Report separately:

- tokens removed by probe certification;
- tokens removed by fallback;
- errors attributable to certified versus forced deletions.

Otherwise a fixed-budget failure could be blamed on the probe even though the harmful deletion came from budget-forced fallback.

Because regions have unequal token costs, selection should be cost aware. Use a small knapsack or a risk-per-token constrained optimization rather than simply deleting a fixed number of regions.

---

# 12. Evaluation should have three levels

## Level A: region-level decodability

On held-out singleton deletions, measure:

- \(R^2\), MAE, and Spearman correlation for \(\delta^G\);
- the same for \(\delta^S\);
- AUPRC for protect/safe/beneficial classes;
- calibration of predicted deletion risk;
- risk–coverage curves;
- performance separately on baseline-correct and baseline-wrong questions.

AUPRC and calibration matter more than raw accuracy because beneficial and harmful deletions may be rare.

## Level B: unseen set-response prediction

On held-out multi-region masks:

- \(R^2\), MAE, and rank correlation;
- metrics restricted to masks near the deployment budget;
- sign accuracy for whether a deletion set helps or harms gold;
- performance on random masks versus learned-policy masks;
- gap to the per-question additive oracle ceiling.

This determines whether singleton-like scores compose into reliable set predictions.

## Level C: actual routed document VQA

Run the VLM once with the learned router, physically prune at the same boundary and budget, and regenerate.

Compare:

- unpruned model;
- native DocPrune;
- regional random;
- metadata-only router;
- attention/QK-only router;
- hidden-only probe;
- combined probe;
- diversity-aware fallback;
- ContextCite gold oracle.

Report:

- paired token-F1;
- exact match;
- exact rescues among baseline-wrong;
- correct-to-wrong flips among baseline-correct;
- fraction of oracle–DocPrune headroom recovered;
- retained token count;
- end-to-end latency and remaining FLOPs.

A useful aggregate is:

\[
\text{gap recovery}
=
\frac{
\overline{\mathrm{F1}}_{\text{probe}}
-
\overline{\mathrm{F1}}_{\text{DocPrune}}
}{
\overline{\mathrm{F1}}_{\text{oracle}}
-
\overline{\mathrm{F1}}_{\text{DocPrune}}
}.
\]

Report the numerator and denominator as paired document-clustered estimates rather than averaging unstable per-question ratios.

---

# 13. Critical controls

The following are necessary for a convincing result.

## No answer-token leakage

Probe features must come only from the full unpruned prompt forward pass **before any gold or generated answer tokens are teacher-forced**.

This is stricter than AT2 and vSTREAM and is central to the claimed contribution.

## Document and template splitting

No questions from the same source document should cross splits. Where possible, hold out entire document templates, vendors, or layout families.

Otherwise the probe may learn that a given form’s answer is always in the same box.

## Question-only and region-only controls

Compare:

- question state alone;
- region representation alone;
- geometry alone;
- DocPrune score alone;
- QK interaction alone;
- combined features.

The question-only state can predict global answerability but cannot identify a region without some region interaction. Region-only performance reveals structural shortcuts.

## Label-shuffle controls

Shuffle region assignments or mask outcomes while preserving:

- region count;
- deletion rate;
- page;
- region type;
- token cost.

A high-capacity probe should not fit these controls beyond chance on held-out documents.

## Gold-answer normalization

Use all accepted normalized gold variants. For multiple valid strings, score the best or a predefined aggregate across variants.

Otherwise a deletion can appear to hurt “gold” only because the selected reference wording differs from an equally valid answer.

## Preserve positional identity

When deleting hidden-state positions, preserve surviving tokens’ original position IDs and M-RoPE coordinates.

Renumbering survivors would mix two interventions:

1. removing region content;
2. changing every later token’s positional encoding.

## State exactly what the intervention means

At an intermediate boundary, later question/prompt states may already contain information propagated from a visual region. Removing the region then does not erase that information.

This is not a defect. It means the target is:

> the contribution still recoverable by deleting the original region at boundary \(d\),

not total historical causal influence.

---

# 14. Layerwise design

The layer study should distinguish two variables that are easy to conflate:

- **read layer \(r\):** where the sidecar reads features;
- **deletion boundary \(d\):** where region positions are removed.

Deployment requires:

\[
r\le d.
\]

## First layer experiment: fixed deletion target, sweep read layer

Choose one fixed deletion boundary \(d^\star\), ideally a layer where your pilot already demonstrates oracle headroom.

Use the same deletion-response labels for every probe, but extract pre-answer features at:

\[
r=0,1,\ldots,d^\star.
\]

This answers:

> When does information predictive of the eventual deletion response first become decodable?

It does not require new masked continuations if the full-forward layer states are cached.

This should be the first layerwise experiment because it isolates representation emergence.

## Second layer experiment: sweep actual deletion boundaries

On a smaller subset, repeat the causal oracle at several boundaries:

\[
d_1<d_2<\cdots<d_k.
\]

For each boundary measure:

- oracle headroom;
- additive ceiling;
- probe decodability;
- routed recovery;
- remaining compute.

This produces the actual three-way curve:

\[
\text{semantic decodability}
\quad\times\quad
\text{deletion recoverability}
\quad\times\quad
\text{compute savings}.
\]

Possible outcomes are scientifically meaningful:

- Early decodability and high recoverability: ideal pruning point.
- Early decodability but low headroom: the signal exists but deletion is not corrective.
- Late decodability and high headroom: scientifically positive, computationally limited.
- Late decodability but low recoverability: distractor influence has already propagated.
- No layer decodes gold utility: the simple premise fails.

For the initial feasibility study, avoid mixing many dynamically selected layers into one regression unless sample sizes are sufficient for layer-specific calibration. A single fixed boundary gives a cleaner answer than a probe simultaneously learning cross-layer scale differences.

---

# 15. What should count as success?

There are three levels of success.

## Minimal scientific success

The premise “gold deletion risk is decodable” is supported when, on document-held-out questions:

1. the combined pre-answer probe predicts \(\delta^G\) significantly better than:
   - geometry;
   - DocPrune;
   - region-only hidden states;
   - and attention/QK-only controls;
2. the gain remains present on baseline-wrong questions;
3. mask-level prediction is nontrivial near the deployment budget;
4. performance survives template or layout shift;
5. the probe can certify safe deletions at useful coverage.

A reasonable initial operational target would be approximately:

- at least 95% precision for safe deletions;
- at least 25% token or region coverage at that precision;
- positive and practically meaningful mask-level \(R^2\);
- Spearman correlation around 0.3 or higher near the deployment budget.

Those are proposed go/no-go targets, not universal probing standards.

## Practical routing success

A useful router should additionally:

- recover at least roughly 25–30% of the gold-oracle versus DocPrune F1 headroom on baseline-wrong questions;
- produce exact rescues above DocPrune;
- limit baseline-correct degradation to a predeclared noninferiority margin, such as one percentage point;
- and produce real end-to-end savings after sidecar overhead.

## Strong success

The strongest result would show:

- predictive gold signal at an early or middle layer;
- additional value from hidden/QK features beyond layout and attention;
- successful baseline-wrong correction;
- correct-case safety;
- generalization to unseen document templates;
- and interpretable gold/self mismatch behavior.

That would justify expanding from a feasibility probe into a publishable router.

---

# 16. What would falsify the simple hypothesis?

| Result | Correct interpretation | Next step |
|---|---|---|
| Per-question additive oracle has poor held-out fidelity | Static per-region utility is an invalid target | Pairwise or set-aware model |
| Additive ceiling is high, but all probes match controls | No evidence that the chosen pre-answer states decode the target | Try different state locations or accept a negative result |
| \(S\) succeeds but \(G\) fails | The model exposes support for its intended answer, not latent corrective evidence | Self-preserving pruning only; external evidence may be needed |
| \(G\) succeeds on correct cases but fails on wrong cases | Probe learns ordinary relevance, not correction | Main corrective claim fails |
| Singleton prediction succeeds but multi-mask prediction fails | Redundancy/complementarity dominates | Low-rank pairwise model, evidence-family objective, or sequential router |
| Mask prediction succeeds but routed QA does not improve | Gold log-likelihood response is misaligned with discrete generation | Predict gold–self margin, rescue probability, or sequence outcome |
| Metadata and DocPrune match hidden features | Internal representations add no value | Use the simpler router and weaken the representational claim |
| Only late-layer probes work | Decodability exists but acceleration headroom is limited | Late KV pruning or late-to-early distillation |
| Random/template-held-out controls collapse performance | Probe learned shortcuts | Broader documents, stronger splits, redesigned features |
| Gold-safe certification works only at tiny coverage | Auditor is accurate but not useful at the desired budget | Hybrid routing or variable-budget deployment |

A particularly valuable negative result would be:

> The per-question additive target is predictable, self utility is decodable, but gold utility on baseline-wrong cases is not.

That would directly show that the frozen model’s state represents its own answer-support structure but does not linearly expose the missing corrective signal.

---

# 17. Escalation path after the initial experiment

## Outcome A: QK-linear works

Keep it.

Do not replace a successful linear model with an MLP merely because a larger model is possible. Add:

- calibration;
- uncertainty;
- selective deletion;
- cost-aware budget allocation;
- and the layer sweep.

## Outcome B: linear fails, low-rank bilinear works

The property is cheaply predictable but not strictly linearly separable from independent question and region representations.

That is still a strong result:

> Corrective utility requires explicit question–region matching.

## Outcome C: additive ceiling is high, but linear and bilinear probes fail

Then try one small two-layer MLP over the same features, with matched controls and sample-efficiency curves.

This tests nonlinear readout while preserving the same additive output structure.

## Outcome D: singleton effects work but multi-region masks fail

Use the smallest interaction model:

\[
\widehat{\Delta G}(D)
=
\sum_{i\in D}\hat\delta_i
+
\sum_{i<j,\;i,j\in D}z_i^\top z_j,
\]

where \(z_i\) is low dimensional.

This can model:

- redundant evidence;
- complementary header/cell pairs;
- alternative copies;
- interacting distractors.

If pairwise structure remains inadequate, use a small DeepSets-style set predictor:

\[
\widehat{\Delta G}(D)
=
\rho\left(
q,\;
\sum_{i\in D}\phi(x_i),\;
\sum_{i\notin D}\phi(x_i)
\right).
\]

## Outcome E: alternative sufficient evidence families dominate

Adopt the causal-family or coverage framing from “Learning What Matters” rather than learning one global score. The router should preserve at least one complete evidence route rather than rank every region independently. 

## Outcome F: selection depends strongly on previously selected regions

Move to a small sequential pointer or autoregressive set router, conceptually similar to StepPrune but trained on your causal deletion teacher rather than only end-to-end language-model loss. 

This should be the last escalation, not the first. It is more expressive but makes the core decodability question harder to interpret.

---

# 18. The novelty claim I would use

Avoid these claims:

- first amortized causal attribution;
- first learned explainer from ablations;
- first generalizable context attribution;
- first visual semantic-region ablation estimator;
- first hidden-state-derived mask;
- first frozen-VLM learned router;
- first intervention-supervised evidence router;
- first pre-generation VLM probe;
- first learned token pruner that can improve performance.

A defensible formulation is:

> We study whether a frozen document VLM’s pre-answer internal representations encode the **gold-conditioned post-boundary deletion response** of semantic regions, including on questions the unpruned model answers incorrectly. We distill that intervention-defined signal into a low-capacity, zero-intervention deletion-risk probe and separate gold support from support for the model’s own intended answer. We then characterize the layerwise relationship between decodability, causal recoverability, and remaining computation.

The strongest possible empirical claim would be:

> A low-capacity pre-answer sidecar recovers a significant portion of privileged gold-oracle pruning headroom on unseen baseline-wrong multi-page document questions while preserving baseline-correct answers.

That claim is narrow, measurable, and meaningfully beyond the closest attribution and pruning precedents.

# Bottom line

The project should proceed, but with a stricter first experiment than a normal router-training study.

The recommended sequence is:

1. **Use the existing 48×256 pilot to estimate the additive ceiling and interaction structure.**
2. **Fix one deletion boundary for the cleanest first decodability test.**
3. **Collect fewer masks over many more document-disjoint questions.**
4. **Train directly from centered mask outcomes, not ContextCite coefficients.**
5. **Use a pre-answer QK-linear probe as the smallest credible query-conditioned model.**
6. **Use raw hidden-state linear probing as a control and rank-4 bilinear probing as the only prespecified escalation.**
7. **Make gold utility primary and self utility a separate diagnostic head.**
8. **Evaluate selective deletion certification separately from forced fixed-budget routing.**
9. **Require success specifically on baseline-wrong questions.**
10. **Only move to pairwise, set-aware, or sequential routing after identifying which simple assumption failed.**

The deepest risk is not that a linear probe will be too small. It is that the phrase “utility of region \(i\)” hides a fundamentally conditional set function. Gate that assumption first. If it survives, your proposed probe is both plausible and cleanly testable.

---

# Task 9 initial-implementation addendum — 2026-09-03

The research review above is preserved verbatim. This addendum records the
approved operational scope for the first implementation and does not rewrite
or reinterpret the review.

- Treat this as an appended Task 9 experiment, separate from the preliminary
  48-question development pilot and the locked 100-question baseline-wrong
  confirmation cohort.
- Target a new 600-question cohort drawn from the existing authenticated
  corpus and cached retrieval artifacts. Keep questions document-disjoint
  across train, validation, and test partitions; preserve balanced
  baseline-correct/baseline-wrong learning and primary-test partitions plus a
  separate natural-prevalence evaluation partition.
- Use the existing 48×256 pilot only for the target-structure/additivity gate
  and development decisions. Do not train on or absorb the locked
  100-question oracle-headroom confirmation cohort.
- The first implementation targets direct centered mask outcomes with the
  pre-answer QK-linear probe described as Model 2. Model 3 remains the single
  prespecified escalation if needed.
- Freeze the physical-deletion teacher target at decoder boundary `B13` for
  every question. This reuses the already validated B13 intervention and avoids
  mixing question-specific deletion targets. The pre-answer QK **read layer**
  may be swept across prespecified layers while the deletion boundary remains
  fixed at B13; read layer and deletion boundary must never be conflated.
- Models 0 and 1 are deliberately deferred for the initial implementation.
  They are not rejected as scientific controls; their implementation and
  evaluation may be restored in a later approved phase.
- Maximize parallelism by separating CPU cohort/scaffolding work on SOL from
  all required model/GPU data-generation jobs on CoRAL H200. No new GPU job
  for this experiment runs on SOL.
- This file is scientific/design authority. Machine-specific sequencing and
  transfer ownership are governed by the paired SOL and H200 handoffs.

The initial model staging in this addendum is superseded, before cohort
sealing or new outcomes, by
[`TASK9_SHARED_PROBE_LAYER_TRAJECTORY_AMENDMENT_2026-09-03.md`](TASK9_SHARED_PROBE_LAYER_TRAJECTORY_AMENDMENT_2026-09-03.md).
The fixed B13 target, cohort, teacher schedule, and direct gold/self outcome
contract remain unchanged.
