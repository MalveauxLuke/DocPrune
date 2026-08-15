# Segment-Level VLM Evidence Scoring: Experimentation Plan

Date: 2026-07-02.
Scope: `known candidate page + query → fine-grained evidence localization` on the SciEGQA 4K DeepSeek-OCR2 evidence dataset. This plan designs experiments; it does not train models.

---

## 1. Executive recommendation

**Arm A (primary):** LoRA-fine-tune **Qwen3-VL-2B-Instruct** as a pointwise binary evidence scorer using the exact Qwen3-VL-Reranker input template and decision-token loss: `s = z_yes − z_no`, `loss = BCEWithLogits(s, label)`, loss applied only at the single decision position. LoRA r=16, α=32, dropout 0.05, all-linear targets in the **language model only** (vision tower and merger frozen), lr 1e-4 cosine with 3% warmup, bf16, gradient checkpointing, effective batch 32, 2 epochs with early stopping on validation PR-AUC, 3 seeds.

**Arm R (new reranker-initialized VLM arm):** LoRA-fine-tune **Qwen/Qwen3-VL-Reranker-2B** as a task-adapted multimodal evidence scorer using the official Qwen3-VL-Reranker template and the same `render_v2` payloads as Arm A. Score with `s = z_yes − z_no`, with logits gathered at the last non-padding decision position. Primary loss is the Qwen3-VL-Reranker label-token objective: full-vocabulary CE on the target token `"yes"`/`"no"` at the single decision position; `R1-BCE` is the matched-objective ablation using `BCEWithLogits(s, label)`. LoRA follows the released Qwen3-VL-Reranker config where specified: r=32, α=32, target modules `q_proj`, `k_proj`, `v_proj`, `gate_proj`, `up_proj`, `down_proj` in the **language model only**; vision tower, merger, embeddings, and lm_head frozen. For unspecified knobs, inherit the project defaults: dropout 0.05, bias none, bf16, gradient checkpointing, effective batch 32, AdamW, weight decay 0.01, gradient clip 1.0, cosine schedule with 3% warmup, max 4 epochs with early stopping on validation PR-AUC, 3 seeds. Peak lr defaults to 5e-5 with a pilot at 1e-4. Arm R tests whether a released general multimodal reranker can be domain-adapted into a calibrated segment-level scientific evidence gate.

**Arm B (mandatory comparison):** LoRA-fine-tune `vidore/colqwen2-v1.0` mirroring its published adapter recipe (r=32, α=32, dropout 0.1, LM q/k/v/o/gate/up/down + `custom_text_proj`), scored by length-normalized MaxSim passed through a learned affine + sigmoid, trained with the same BCE objective plus an in-group softmax auxiliary (λ=0.5).

**Shared input rendering (render_v2, per user direction 2026-07-03):** the section content is delivered in its native modality using the DeepSeek-OCR2 preprocessing already in the dataset. Text-kind sections (headed_text, document_preamble, deepseek_paragraph) enter as their OCR markdown text. Visual bundles enter as pixel crops of their image/table member boxes (one image per member box, native aspect, from the full-resolution page PNG) plus the caption/`figure_title` text as OCR text. Tables additionally include DeepSeek's table markdown (image+text; text-only and image-only are ablations). Full-page-with-highlight rendering (render_v1) is retained as an ablation arm. Arm B receives the same pixel crops; since ColQwen2 cannot ingest document-side text, text sections are supplied to Arm B as region crops of the same member boxes — a documented modality asymmetry with a bridging ablation (E-R5).

**Primary baseline:** frozen ColQwen2 MaxSim with Platt calibration fitted on validation (the "retrieval similarity is enough" null hypothesis). Second load-bearing baseline: a fine-tuned text-only cross-encoder (Qwen3-Reranker-0.6B) over query + OCR section text (the "pixels don't matter" null hypothesis). Additional load-bearing baseline: frozen **Qwen/Qwen3-VL-Reranker-2B** (`B6`), using its official reranker template over the same `render_v2` payloads with validation-only calibration; B6 is the direct null for Arm R.

**Why this is the best-supported design under the project's constraints:** the Qwen3-VL-Reranker report ([arXiv:2601.04720](https://arxiv.org/abs/2601.04720)) demonstrates that exactly this yes/no decision-token formulation, trained with LoRA on a compact Qwen-VL backbone, beats ColPali-style late-interaction models of similar size on visual-document relevance (ViDoRe-v3: Reranker-2B 60.8 vs colqwen2.5 52.4); Dai et al. ([arXiv:2510.14824](https://arxiv.org/abs/2510.14824)) show SFT-style yes/no training dominates contrastive training for MLLM reranking; and the project's own MMDocIR verifier pilot showed frozen ColQwen embeddings carry no extractable independent evidence signal (content-only 0.546 vs MaxSim 0.699 Rank@1). Direct pixel inspection with a generative scorer is the direction all three lines of evidence point to. The design is not claimed universally optimal; conditions that would flip each choice are listed per decision.

**Dataset status (updated 2026-07-07, evening): combined Option 4 dataset COMPLETE and verified.** Run `/scratch/lmalveau/sciegqa_train_4k/evidence_option4_10431/combined_evidence/20260707T210746Z` (jobs 58687162/58687220/58687258; commit `d1fedc9`). Results: 10,431 source queries / 6,630 source pages; 6,591 OCR-valid pages (39 terminal OCR failures; 0/19 expansion retries became valid, consistent with the 4K run's 0/20); **4,463 pages / 6,918 queries retained; 2,167 pages / 3,513 queries quarantined (66.3% query retention — above the ~6,300 projection)**; fresh 70/15/15 split (seed 20260707): **train 4,842 / validation 1,038 / test 1,038 queries**, no leakage, byte-identical determinism audit, shared sectioning implementation confirmed; 99 tests passed. The original 4K run (`20260703T090303Z`) is retired to provenance. Derived exact constants: steps/epoch = ceil(4,842/8) = **606**; max_steps = **2,424**; warmup = **73**. Noise floor at 1,038 test queries ≈ **±2.8 pts**; projected visual-bundle test stratum ≈ 290 (28.6% rate from the 4K audit — confirm from combined by-kind audit). Pair counts (train pairs ≈ 4,842 × ~4.1 ≈ 20k) to be read from `labeling_audit.json`. **Quarantined pages/queries are excluded, fully audited, and never manually adjudicated in this POC** (pinned dataset contract); they are earmarked as raw material for the future multi-evidence dataset.

---

## 2. Project and dataset summary

### 2.1 Verified repository state (inspected 2026-07-02, local checkout)

| Item | Status |
|---|---|
| Main branch `COLQWEN_binary_classification`, HEAD `a88a8bc` | present |
| `codex/sciegqa-final-labeling` worktree, HEAD `3947d2e` ("docs: hand off SciEGQA evidence dataset") | present; contains `scripts/document_parsing/semantic_sections.py` (1,043 lines), `sol/SCIEGQA_4K_DEEPSEEK_EVIDENCE_HANDOFF.md` (643 lines), `tests/test_sciegqa_final_labeling.py` (610 lines) |
| 4K selection + page extraction on SOL scratch | reported complete per `sol/CURRENT_SOL_TASK.md` and `SCIEGQA_4K_HANDOFF.md` (4,000 queries / 3,711 pages, seed 20260630) |
| 32-page parser-comparison run (MinerU 3.4, DeepSeek-OCR, DeepSeek-OCR-2), viewer, visual review | complete (jobs 58092005–58092576, run `20260701T095214Z`); DeepSeek-OCR2 ≈ 26.9 s/page |
| Original 4K evidence dataset (run `20260703T090303Z`) | Generated and verified (2,644 pages / 2,818 queries, 10,774 sections, 11,475 pairs; by-kind positives: headed 1,261, preamble 519, paragraph 231, visual 807). **Superseded** — its labels/splits are retired to provenance; its OCR artifacts are reused by hash in the combined build |
| **Combined Option 4 dataset (10,431 queries / 6,630 pages)** | **Complete and verified** (run `20260707T210746Z`): 6,591 OCR-valid pages, 39 terminal OCR failures, 4,463 retained pages, 6,918 retained queries, 2,167 quarantined pages, 3,513 quarantined queries, fresh split seed `20260707` with train 4,842 / validation 1,038 / test 1,038 queries, byte-identical determinism audit, shared sectioning implementation confirmed |
| Page-level MMDocIR verifier experiments | complete and frozen; results as in §2.3 |

Confirmed source-side statistics: 30,780 rows, 11,668 strict rows, original 4,000-query / 3,711-page selection, and combined 10,431-query / 6,630-page Option 4 dataset. Pair counts and section-kind breakdowns should still be read from the combined run's generated audits before reporting final tables.

### 2.1a Option 4 combined expansion (decided 2026-07-07; supersedes the 2026-07-03 eval-only design, which was never executed)

Because no model had trained on the old split, the eval-only ("new pages → val/test, train untouched") design was superseded by the strictly better full merge:

- **Query universe:** all eligible queries (uncapped; +15 vs the 5-per-page cap, recorded) on the original 3,711 pages **plus** all eligible queries on 2,919 new pages = **10,431 queries / 6,630 pages**. The final 814-page pool (Options 3/5, q-fin/q-bio-heavy remainder) is deliberately excluded as poor marginal value.
- **Cost:** only the 2,919 new pages need OCR (~22 GPU-h, spent); the original pages' OCR artifacts are **reused after SHA-256 verification** in a self-contained combined run tree (no references into mutable old run directories).
- **Cannot append:** densification adds queries to original pages, and any bad relation quarantines the whole page — so original pages can flip from retained to quarantined. Combined labeling, quarantine, and a **fresh page-grouped 70/15/15 split (seed 20260707)** are mandatory; the old split is retired to provenance and nothing evaluates against it.
- **Balance policy:** the merged pool is availability-skewed (math capped at source ceiling; expansion lacks math). Headline metrics use **domain macro-averaging**; balanced training subsampling remains available as a config flag if imbalance proves harmful (testable, not presumed).
- **POC-scoped audits** (heavier variants deliberately dropped): pipeline-native hash validation of reused artifacts; existing unit/golden sectioning tests (no new full-corpus identity gate); one densification sanity number — quarantine rate among original pages still carrying exactly 1 query must match the 4K run's rate for such pages (identical inputs → identical rate; deviation = bug); raw per-batch counts dumped in the audit JSON without a comparison framework.
- Full-pool expansion (the remaining 814 pages / ~1.2k queries) stays a documented contingency only if E-D1 shows the model still training-data-limited on the combined training set.

### 2.2 Dataset properties that drive the design

- Labels are **relations** `(query_id, section_id) → {positive, hard_negative}`; exactly one positive per retained query; all other same-page sections are hard negatives; negatives are same-page only.
- Sections are semantic (headed-text spans kept intact regardless of length; visual bundles with member boxes and linked captions). Members can be **discontiguous**; geometry is the union of member boxes, not an envelope.
- No stitched crops exist; only the full page image + member boxes. The rendering policy is therefore an experimental decision (§9).
- SciEGQA queries presuppose the gold page → all claims stay scoped to page-conditional evidence localization. This also means the query rarely disambiguates *which document*, but must disambiguate *which section* — precisely the regime where same-page hard negatives are informative.
- Class imbalance is structural but **mild**: mean 3.1 negatives per positive (24.6% positive rate), mean 4.1 sections/page — measured on the 4K run; the group structure is page-driven, so these carry over to the combined dataset essentially unchanged (recheck at audit). Group-aware batching remains (§10); heavy imbalance remedies (pos_weight, focal) are unnecessary at this ratio.
- The answer field is provenance-only. A leakage check must prove it never reaches model input (§16).

### 2.3 Why the pivot is justified (prior evidence)

Frozen ColQwen2 embeddings + MaxSim features, page-level MMDocIR, 3 seeds: MaxSim baseline strict Rank@1 0.6988; full verifier 0.7008 ± 0.0086 (gain < seed std); alignment-features baseline 0.6734 ± 0.0030; content-only 0.5460 ± 0.0068; MaxSim-ablated 0.2460 ± 0.0168. Conclusion carried forward: frozen late-interaction embeddings are a retrieval signal, not a calibrated evidence signal; a scorer must be allowed to *look at the pixels* and adapt its representation. This does not prove segment-level ColQwen LoRA fails — hence mandatory Arm B.

---

## 3. Findings from the attached papers

### 3.1 Qwen3-VL-Embedding & Qwen3-VL-Reranker (arXiv:2601.04720; also attached as `Qwen_rerank.md`)

- **Task formulation (reranker):** pointwise binary relevance; given (Instruction, Query, Document) predict "yes"/"no". Score at inference: `s = sigmoid(logit(yes) − logit(no))` (their Eq. 5) — identical to this project's mandated score.
- **Architecture:** cross-encoder on Qwen3-VL backbone (causal attention), 2B (28 layers) and 8B (36 layers), 32K context; LM head reused, no new head.
- **Template (verbatim):** system: `Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".`; user: `<Instruct>: {Instruction}\n<Query>: {Query}\n<Document>: {Document}`; then `<|im_start|>assistant`. Multimodal content is embedded in the user message.
- **Loss:** `L = −log p(l | I, q, d)` on the yes/no label token (their Eq. 4), citing Dai et al. 2025 that this beats contrastive for LLM reranking.
- **Negatives:** two-stage mining — recall by embedding, then *positive refinement* (keep query only if a positive scores > t⁺) and *hard-negative selection* (keep negative only if score < s̄⁺ + δ⁻ margin, guarding against false negatives). Reranker trained on ~4M retrieval pairs.
- **PEFT:** trained **with LoRA** from Qwen3-VL-Instruct; motivations: memory, better generalization, cheap merging. (Placement/rank not disclosed — a gap our LoRA ablation covers.)
- **Released LoRA config detail:** Qwen's official reranker LoRA config lists r=32, α=32, and target modules `q_proj`, `v_proj`, `k_proj`, `up_proj`, `down_proj`, `gate_proj`. It does not list `o_proj`; therefore Arm R excludes `o_proj` in the primary target list, while Arm A keeps its existing all-linear LM-only target list.
- **Resolution:** dynamic, image budget capped at 1,280 visual tokens (~1.3 MP); performance vs visual-token budget shows steep gains to ~1,000 tokens then plateau/slight regression (their Fig. 7) — grounds our 1,288 px / ≤1,280-token rendering budget.
- **Results relevant here:** Reranker-2B beats colqwen2.5-v0.2 (3B) on ViDoRe-v3 (60.8 vs 52.4) and JinaVDR (80.9 vs 75.6). Transfers: the yes/no cross-encoder formulation on a 2B Qwen-VL is state-of-practice for visual-document relevance at this scale.
- **Does not transfer:** their 4M-pair scale, instruction diversity, and mined-negative pipeline. Our negatives are structurally given (same-page), our data is ~4 orders smaller — hence LoRA-only, few epochs, heavy regularization, and staged ablations rather than their multi-stage pipeline.

### 3.2 ColPali (arXiv:2407.01449, ICLR 2025)

- **Task:** page-level visual document retrieval; ViDoRe benchmark; nDCG@5 headline.
- **Architecture:** PaliGemma-3B → 128-d projection per output token; late interaction `LI(q,d) = Σᵢ maxⱼ ⟨q⁽ⁱ⁾, d⁽ʲ⁾⟩`.
- **Loss:** pairwise softplus CE on the *hardest in-batch negative*: `L = log(1 + exp(s⁻ − s⁺))`; ablation: full in-batch contrastive is *worse* by 1.6 nDCG@5. Batch 32, 1 epoch, bf16.
- **LoRA:** r=32, α=32 on LM transformer layers + trained projection layer; `paged_adamw_8bit`; lr 5e-5 linear decay, 2.5% warmup.
- **Key ablations that transfer:** (i) unfreezing the vision encoder *hurt* (−0.7 nDCG@5) at ~118K-pair scale — supports freezing the vision tower in both our arms at 4K-query scale; (ii) ColQwen2-VL (+5.3 over ColPali, 768-patch budget) — the model we use in Arm B; (iii) query augmentation tokens matter mostly cross-lingually.
- **Verified adapter of `vidore/colqwen2-v1.0` (fetched from HF):** r=32, α=32, dropout 0.1, bias none, `init_lora_weights="gaussian"`, `target_modules` regex `(.*(model).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$|.*(custom_text_proj).*$)` — i.e., LM all-linear + retrieval projection; **vision tower excluded** (its modules live under `visual.*`, which the regex does not match).
- **Assumption that does not transfer:** ColPali optimizes corpus-level ranking with in-batch negatives across *different* documents; our task is within-page discrimination with a calibrated absolute output. Raw MaxSim is query-length dependent and uncalibrated — motivating normalization + learned calibration in Arm B.

### 3.3 `DocVQA.pdf` (attached slide deck — project design document, not the Mathew et al. 2021 benchmark paper)

The attachment is the project's own 8-slide deck. Extracted commitments: document VQA decomposes into evidence retrieval + answer inference; fixed top-k retrieval either misses evidence or floods the reader (SimpleDoc table: retaining more pages raises All-Hit% but collapses F1); ColPali/ColQwen MaxSim "rewards token presence," motivating a query-conditioned gate; the intended scorer is a 2–4B VLM (deck names Qwen2.5-VL-3B/7B, InternVL2-4B/8B as candidates) with LoRA/PEFT and segment-level TRUE/FALSE supervision; the planner–expert loop is future work. This plan supersedes the deck's model shortlist with the current strongest compact models (§5) while honoring its architecture. *Note: the actual DocVQA benchmark paper (Mathew et al., arXiv:2007.00398) was not among the attachments; no claims below depend on it.*

---

## 4. Iterative literature-review log

### Pass 1 — landscape mapping

| Question | Finding | Source class |
|---|---|---|
| Generative yes/no relevance classification | Standard for LLM rerankers (Qwen3-Reranker text + VL; RankLLaMA pointwise; MonoT5 lineage). Qwen3 official scoring: logits at last position, log-softmax over {no,yes}, take p(yes) | Official papers/code |
| SFT vs contrastive for MLLM reranking | SFT (yes/no CE) > CL; decomposition into update weight vs direction; SoTA on MRB via large-scale SFT — [Dai et al. 2510.14824](https://arxiv.org/abs/2510.14824), [code](https://github.com/vec-ai/lychee-rerank-mm) | Peer-reviewed/preprint |
| Multimodal rerankers on VLM backbones | Qwen3-VL-Reranker (2601.04720); [jina-reranker-m0](https://huggingface.co/jinaai/jina-reranker-m0): Qwen2-VL-2B, **LoRA on LM only**, frozen vision encoder+projector, MLP scoring head, 2.4B params | Official model cards |
| ColPali/ColQwen fine-tuning | Official configs verified (§3.2); community fine-tunes follow the same LM-only LoRA pattern | Official implementation |
| LoRA placement | QLoRA finding: covering **all linear layers** matters more than rank; attention-only underperforms; α≈2r heuristic; lr 1e-4–3e-4 stable ([PEFT docs](https://huggingface.co/docs/peft/developer_guides/lora), [Unsloth guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide)) | Official docs + community |
| Calibration under imbalance | Temperature/Platt scaling on val standard; BCE on the decision logit is already a proper scoring rule; post-hoc temperature preserves ranking ([overview](https://www.kdnuggets.com/a-deep-dive-into-calibration-of-language-models-platt-scaling-isotonic-regression-temperature-scaling)) | Textbook + surveys |
| Same-page hard negatives | [DocReRank (2505.22584)](https://arxiv.org/abs/2505.22584): single-page hard-negative generation specifically to train multimodal RAG rerankers — independent validation that same-page negatives are the right difficulty distribution | Preprint |

### Pass 2 — targeted design research

| Open choice | Evidence found | Resolution |
|---|---|---|
| Vision-encoder adaptation | ColPali: unfreezing vision hurt (−0.7). Community Qwen-VL recipes: freeze ViT+merger, LoRA the LLM; vision unfreezing "helps a little on tiny-text OCR but is easy to destabilize" ([guide](https://medium.com/@aminfadaeinejad.edu/fine-tuning-qwen3-vl-a-practical-guide-for-vision-language-model-adaptation-d66d3f61e888), [Datature](https://datature.io/blog/how-to-fine-tune-qwen2-5-vl)); one ablation shows joint ViT+LLM tuning best *at scale* with full fine-tuning | Freeze vision in primary; single ablation arm adds merger-only LoRA, second adds last-6 vision blocks |
| Region conditioning: crop vs highlight | Visual-prompting literature: red-circle/box marks direct VLM attention while *preserving global context*; cropping "discards valuable global information" ([FGVP NeurIPS'23](https://proceedings.neurips.cc/paper_files/paper/2023/file/4e9fa6e716940a7cfc60c46e6f702f52-Paper-Conference.pdf), [survey 2409.15310](https://arxiv.org/html/2409.15310)); Set-of-Mark for multi-region | Primary = full page + red member-box outlines; crop variants as ablations |
| Group supervision with 1 positive | Listwise softmax over {1 positive + k hard negatives} is standard and effective (RankLLaMA-style pipelines, SweRank); ColPali's own hardest-negative pairwise beat full in-batch CE | Keep BCE primary (mandated, calibrated); add group-softmax as *auxiliary* (λ ablation), never replacing sigmoid output |
| Resolution budget | Qwen3-VL report Fig. 7: visual-doc performance saturates ~1,000–1,280 tokens, slight regression beyond | 1,280-token cap primary; 768-token ablation for parity with ColQwen2 budget |
| Class weighting | Query-balanced sampling preferred over global pos_weight when group sizes vary (keeps per-query calibration); focal loss mainly helps extreme imbalance with abundant data | Query-balanced batches + capped negatives primary; pos_weight and focal as pilot-stage checks |

### Pass 3 — adversarial validation

- **Tokenizer check (verified from official Qwen3-Reranker usage code):** score tokens are lowercase `"yes"`/`"no"` via `convert_tokens_to_ids` — single tokens in the Qwen 151,936 vocab. Capitalized `"Yes"`/`"No"` are also single tokens but are *not* what the sibling reranker used; we adopt lowercase for template compatibility. **Must-run local assert:** `len(tok("yes").input_ids)==1` and ditto for `"no"` on the exact pinned tokenizer revision, plus assert the chat template inserts no `<think>` block for Qwen3-VL-Instruct (the *text* Qwen3-Reranker template includes an empty `<think></think>`; Qwen3-VL-Reranker's does not — copying the wrong template silently shifts the decision position).
- **Tied embeddings (verified from Qwen3-VL-2B config):** `tie_word_embeddings: true` — do **not** add LoRA to `lm_head`/`embed_tokens` (gradients would leak into input embeddings; also PEFT's handling of tied weights + `modules_to_save` is a known footgun). All-linear targets must therefore be enumerated explicitly, not `"all-linear"` shorthand, if the PEFT version would sweep `lm_head`. Verify with a printout of matched modules before training.
- **Vision + quantization:** community-maintained Qwen-VL fine-tuning repo warns **do not combine 4/8-bit QLoRA with training vision-related modules** ([2U1/Qwen-VL-Series-Finetune](https://github.com/2U1/Qwen-VL-Series-Finetune)); FA2 has raised CUDA errors on some stacks while SDPA was stable — smoke-test both, fall back to SDPA. QLoRA unnecessary anyway on 80 GB A100 for a 2B model → use plain LoRA bf16 (drops a confound).
- **Contradictory evidence honestly logged:** (i) one ablation line shows vision-encoder tuning *helps* perception-heavy tasks — if pilot error analysis shows failures concentrated in fine-print/plot-reading, the vision-LoRA ablation (E-A3b) becomes decision-relevant rather than a formality. (ii) Dai et al. optimize *ranking* metrics; our pair-level calibrated-probability requirement is stricter — BCE stays primary regardless. (iii) Zero-shot VLMs exhibit yes-bias / poor calibration on binary probes — a reason the zero-shot baseline must be threshold-tuned on validation before being declared weak.
- **Model code verification:** ColQwen2 adapter regex confirmed (§3.2). Qwen3-VL-2B: text 28 layers, hidden 2048, vision depth 24, deepstack indexes {5,11,17}; LM module names `model.language_model.layers.{i}.{self_attn.{q,k,v,o}_proj, mlp.{gate,up,down}_proj}`, vision under `model.visual.blocks.*`, merger `model.visual.merger` (+ `deepstack_merger_list`) — final names to be asserted by printing `named_modules()` at smoke time, since transformers refactors move prefixes.
- **SOL hardware (public docs, to be confirmed on-cluster):** 56 nodes × 4× A100 80 GB, 4 nodes × 3× A30 24 GB, a few H100s ([ASU RC docs](https://docs.rc.asu.edu/supercomputer-hardware/), [PEARC23 paper](https://dl.acm.org/doi/fullHtml/10.1145/3569951.3597573)). Plan assumes **1× A100 80 GB per job**; A30 fallback requires halving batch and is not planned. Confirm with `sinfo -p public -o "%n %G"` and per-job `nvidia-smi` (already standard in repo sbatch scripts). Request explicitly: `--gres=gpu:a100:1`.

---

## 5. Candidate model comparison (Arm A backbone)

| Model | Size | Pros | Cons | Verdict |
|---|---|---|---|---|
| **Qwen3-VL-2B-Instruct** (2025-10-21) | 2B | Same backbone family as the SoTA multimodal reranker built with the identical yes/no objective; strong doc/OCR skills; 1,280-token dynamic-res budget validated in-family; LoRA recipes mature; fits 80 GB with huge headroom; tokenizer/template verified | Newest stack → verify transformers pin supports `qwen3_vl` on SOL | **Primary** |
| **Qwen3-VL-Reranker-2B** | 2B | Purpose-built multimodal reranker; same Qwen3-VL family; 28-layer, 32K-context, yes/no relevance scoring; strongest relevance-tuned initialization for this task | Less novel as a new-model story; must compare frozen B6 vs tuned Arm R to isolate adaptation gain | **New Arm R; not a replacement for Arm A unless validation supports it** |
| Qwen3-VL-4B-Instruct | 4B | tomoro-colqwen3-4b results suggest strong doc capability per FLOP | ~2× train/infer cost; 2B suffices for a gate; scale ablation only if 2B underperforms | Contingency (E-A8) |
| Qwen2.5-VL-3B-Instruct | 3B | Deck's original candidate; mature | Superseded by Qwen3-VL at equal/lower size on doc benchmarks; no in-family reranker precedent | Rejected as primary |
| InternVL3.5-2B | 2B | Competitive general VLM | Different template ecosystem; no yes/no-reranker precedent; weaker evidence on scientific-doc OCR at 2B | Rejected |
| jina-reranker-m0 (Qwen2-VL-2B) | 2.4B | Purpose-built visual reranker | Older backbone; MLP-head scoring (not yes/no) conflicts with mandated score; license check needed | Zero-shot reference only (optional) |

Selection basis: not familiarity — the decisive evidence is (a) Table 3 of 2601.04720 (2B yes/no reranker > 3B ColQwen-style embedder on visual-doc relevance), (b) exact-objective precedent in-family, (c) verified tokenizer/template behavior. Risk condition: if the pinned SOL transformers version cannot load `qwen3_vl`, fallback is Qwen2.5-VL-3B-Instruct with the same template (its tokenizer also has single-token yes/no — assert at smoke).

Arm R is included as a separate trained arm because it tests a different question from Arm A: whether task-specific supervision improves a released relevance-tuned multimodal reranker, rather than whether an instruction VLM can be converted into a reranker.

---

## 6. Final Arm A specification (VLM binary evidence classifier)

### 6.1 Input/prompt (frozen, versioned as `prompt_v1`)

```
<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: Determine whether this document section contains sufficient evidence to answer the query. The section content below was extracted from a document page and may be text, a figure or table image, and/or a caption.
<Query>: {query}
<Document>: {section content per §9 render_v2 — OCR markdown for text sections; member crop image(s) + "Caption: {caption}" (+ table markdown) for visual bundles}<|im_end|>
<|im_start|>assistant
```

- Decision position: the first generated token (index = last prompt position's next-token logits). Extract `z_yes`/`z_no` from logits gathered at the last non-padding real token using `attention_mask.sum(dim=1) - 1` in a single forward pass (no generation loop); `logits[:, -1, :]` is forbidden under padding.
- Token IDs: `id_yes = tokenizer.convert_tokens_to_ids("yes")`, `id_no = ...("no")`; assert both ≠ unk and single-token; record IDs in the run manifest. If either were multi-token (not expected), fallback policy: use the first subtoken and log it — but treat as a stop condition for template review.
- Loss: `BCEWithLogitsLoss(z_yes − z_no, y)` — mathematically the 2-way softmax CE restricted to {yes,no}, and exactly Eq. 4/5 of 2601.04720. Loss on the decision token only; no LM loss on prompt tokens; no free-form generation objective.
- The answer string, gold box, coverage, and candidate precision never appear in the prompt.

### 6.2 LoRA / trainable modules

- Adapt: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` in all 28 LM decoder layers. Frozen: vision tower (24 blocks), patch merger(s), embeddings, lm_head (tied).
- r=16, α=32, dropout 0.05, bias "none", default (Kaiming/He) PEFT init for A and zeros for B. Estimated trainable params ≈ 17 M (~0.8% of backbone); exact count printed and logged at smoke.
- LoRA vs QLoRA: **LoRA in bf16.** QLoRA rejected: unnecessary at 2B on 80 GB, adds quantization noise as a confound, and community guidance forbids mixing 4/8-bit with any vision-module training, which would poison ablations E-A3a/b.
- Placement ablation (compact, §13): (a) attention-only r=16; (b) all-linear r=16 [primary]; (c) all-linear r=16 + merger LoRA; (d) all-linear r=16 + vision last-6-blocks LoRA (bf16). Ranks {8, 32} tested only for the winning placement.

### 6.3 Optimization

| Item | Value | Rationale / alternatives |
|---|---|---|
| Optimizer | AdamW (β 0.9/0.999, ε 1e-8) | paged_adamw_8bit unnecessary at this footprint; keep plain for determinism |
| LR (LoRA params) | 1e-4, cosine to 0.1×, warmup 3% | Stable LoRA band 1e-4–3e-4; search {5e-5, 1e-4, 2e-4} in pilot; single LR group (no vision/projector groups since frozen) |
| Weight decay | 0.01 on LoRA weights | {0, 0.01} pilot check |
| Precision | bf16 (A100-verified in repo sbatch preflight) | fp16 rejected (loss-scale instability with VLMs) |
| Grad accumulation | per-device batch 4 × accum 8 = effective 32 | matches ColPali/ColQwen batch scale; adjust per measured memory |
| Grad clipping | 1.0 (global norm) | standard |
| Gradient checkpointing | on (LM only) | ~1,300 image + ~100 text tokens/sample; large headroom but batch-4 safety |
| Attention impl | flash_attention_2 if smoke passes, else sdpa | pass-3 finding: FA2 flaky on some stacks |
| Epochs / stopping | max 4 (606 query-group steps/epoch on the combined dataset; 2,424 max steps); evaluate val PR-AUC every 250 steps; early stop patience 3 evals; select checkpoint by val PR-AUC | 2.2× more training data than the 4K run → fewer epochs needed than the interim ≤6 rule; early stopping still the main overfit guard |
| Seeds | 13, 17, 23 (repo convention) | 3 seeds mandated |

### 6.4 Batching, weighting, negatives — see §10.

### 6.5 Calibration & threshold

Fit temperature T on validation logits (NLL-minimizing); report pre/post ECE. Select operating threshold on validation only, at two operating points: (i) max-F1, (ii) high-recall point (recall ≥ 0.95 of positives, report precision). Test set touched once, at the end, with frozen T and thresholds.

---

## 6R. Final Arm R specification (LoRA-tuned Qwen3-VL-Reranker-2B)

Arm R is the reranker-initialized counterpart to Arm A. It starts from `Qwen/Qwen3-VL-Reranker-2B`, keeps the official reranker template, and adapts the released reranker to the project’s segment-level scientific evidence labels. It is interpreted primarily against frozen B6.

### 6R.1 Input/prompt

- Base model: `Qwen/Qwen3-VL-Reranker-2B`; pin HF revision at environment build and record it in every manifest.
- Prompt/template: use the official Qwen3-VL-Reranker model-card template and usage path. Do not approximate it with the Arm A prompt unless byte-equivalence of the rendered dummy prompt is verified and logged.
- System text remains the official reranker system string:
  `Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".`
- The task instruction is the same project instruction used by Arm A:
  `Determine whether this document section contains sufficient evidence to answer the query. The section content below was extracted from a document page and may be text, a figure or table image, and/or a caption.`
- User content follows the official reranker structure:
  `<Instruct>: {Instruction}\n<Query>: {Query}\n<Document>: {Document}`
- `<Document>` receives the same `render_v2` payloads as Arm A: OCR markdown text for text sections; member crop image(s) plus caption text for visual bundles; table markdown plus caption text for table bundles.
- Assert the rendered dummy prompt contains no `<think>` block.
- Assert `"yes"` and `"no"` are single tokens under the pinned reranker tokenizer.
- Record the rendered dummy prompt hash and token IDs in the manifest.

### 6R.2 Score and loss

Use a single forward pass, no generation loop. Gather logits at the last non-padding real token exactly as in the implementation spec. `logits[:, -1, :]` is forbidden.

```python
out  = model(**inputs)
mask = inputs["attention_mask"]
last = mask.sum(dim=1) - 1
bidx = torch.arange(out.logits.size(0), device=out.logits.device)
z    = out.logits[bidx, last, :]
s    = z[:, ID_YES] - z[:, ID_NO]
```

Primary Arm R training loss:

```python
target = torch.where(y.bool(), ID_YES, ID_NO)
loss = F.cross_entropy(z, target)
```

Scoring, ranking, calibration, and thresholds still use `s = z_yes − z_no`.

Add one matched-objective ablation:

```text
R1-BCE:
  loss = BCEWithLogits(z_yes − z_no, label)
  scoring unchanged
```

Rationale: full-vocabulary CE is faithful to the released Qwen3-VL-Reranker objective; `R1-BCE` isolates whether the project’s restricted BCE objective helps calibration or stability on this smaller dataset.

### 6R.3 LoRA / trainable modules

Primary Arm R LoRA follows the official Qwen3-VL-Reranker LoRA config where it is explicit:

- r=32
- α=32
- target modules: `q_proj`, `k_proj`, `v_proj`, `gate_proj`, `up_proj`, `down_proj`
- no `o_proj` in the primary Arm R target list
- dropout 0.05
- bias `"none"`
- PEFT default init

Frozen:
- vision tower
- merger
- embeddings
- lm_head

Target-module construction:

```python
target_modules = [f"{P}.layers.{i}.{m}"
                  for i in range(28)
                  for m in ("self_attn.q_proj","self_attn.k_proj","self_attn.v_proj",
                            "mlp.gate_proj","mlp.up_proj","mlp.down_proj")]
```

`P` is the language-model module prefix discovered at runtime, using the same resolver policy as Arm A. After PEFT wrapping, assert that the trainable-parameter name set equals exactly the intended LoRA A/B weights and nothing else. Abort if any trainable parameter appears under `lm_head`, `embed_tokens`, the vision tower, or any merger module.

### 6R.4 Optimization

| Item | Value |
|---|---|
| Optimizer | AdamW |
| LR | 5e-5 primary; pilot `R1-lr1e4` at 1e-4 |
| Schedule | cosine decay to 0.1×, 3% warmup |
| Weight decay | 0.01 |
| Precision | bf16 |
| Grad clipping | 1.0 global norm |
| Gradient checkpointing | on, language model only |
| Batch/sampler | identical query-balanced sampler as §10 |
| Negatives | all same-page negatives |
| Class weighting | none |
| Epochs/stopping | max 4 epochs; evaluate validation PR-AUC every 250 steps; early stop patience 3 |
| Seeds | 13, 17, 23; optional 29, 31 at confirmation |
| Calibration | validation-only temperature scaling |
| Thresholds | validation max-F1 and validation recall≥0.95 point |
| Test discipline | test touched once only, after all validation decisions are frozen |

Do not include a 2e-4 Arm R pilot by default. Arm R starts from a relevance-tuned checkpoint, so the initial LR sweep is deliberately narrower than Arm A’s.

### 6R.5 Interpretation

- `R1 > B6`: task-specific segment-level evidence supervision improves the released reranker.
- `R1 ≈ B6`: the released reranker already transfers well; contribution shifts toward evaluation/calibration.
- `R1 > B4-ft`: multimodal input adds value beyond OCR-text reranking.
- `B4-ft ≈ R1`: OCR text carries most of the signal; multimodal claim weakens.
- `R1 > B2/B2b`: cross-encoder evidence scoring beats retrieval similarity.
- `B2b ≈ R1` or `B2b > R1`: central reranker motivation is in trouble; audit rendering, labels, and evaluation before more training.
- `R1 > A1`: reranker initialization is better than instruct initialization.
- `A1 > R1`: surprising; inspect template, loss, LR, and whether reranker pretraining mismatches segment-level sufficiency.

---

## 7. Final Arm B specification (LoRA-tuned ColQwen2)

- Base: `vidore/colqwen2-v1.0` (adapter on `vidore/colqwen2-base` = Qwen2-VL-2B). Load the published adapter, continue training it (do not re-init): the published weights are the validated retrieval state; re-initializing a second adapter stack risks the frozen-embedding failure mode returning.
- Input: the same section pixel crops as Arm A's visual bundles; for text sections, region crops of the same member boxes (ColQwen2 cannot ingest document-side text — documented asymmetry, bridged by E-R5). Query text through the official `ColQwen2Processor` unchanged; image budget 768 patches per its validated config. Multi-crop sections: embed each crop, score query against the concatenated token set (MaxSim is order-invariant over document tokens). Caveat: small region crops are out-of-distribution vs ColQwen2's page-level pretraining; E-R1 (full-page highlight input) bounds this.
- Score: `maxsim(q, page) / N_q` (query-length normalized; raw-vs-normalized is ablation E-B2), then learned affine `score = a·m + b` → `P = sigmoid(score)`. Rationale: raw MaxSim scale grows with query length (sum over tokens) and is uncalibrated; a 2-parameter affine is the minimal calibrated transformation and reduces to Platt scaling when LoRA is frozen — making baseline B2 and Arm B nested models.
- Loss: `BCEWithLogits(score, y)` primary + auxiliary in-group softmax over each query's {1 positive + sampled negatives} with learnable temperature, total `L = L_BCE + λ·L_group`, λ ∈ {0, 0.5, 1.0} (E-B3). The sigmoid output is preserved regardless of λ.
- LoRA: exactly the published recipe — r=32, α=32, dropout 0.1, gaussian init, bias none, LM q/k/v/o/gate/up/down + `custom_text_proj` trainable. Vision excluded (matches both the official adapter and ColPali's negative result on vision unfreezing).
- Optimization: lr 5e-5 (published), linear decay, 2.5% warmup, bf16, effective batch 32 (device 8 × accum 4 — cheaper per sample than Arm A since no 28-layer generative pass over 1,280 image tokens... measured at smoke), max 3 epochs, early stop on val PR-AUC, seeds 13/17/23. Affine calibrator lr 1e-3 (separate param group; it has 2 params).
- Checkpoint/calibration/thresholds: same protocol as §6.5.
- Fairness contract: same pairs, same rendered pixels, same split, same seeds, same metric code, same calibration protocol. Differences (patch budget, query-side processing) are inherent to the family and documented; E-R1 bounds the budget effect.

---

## 8. Baseline suite

| ID | Baseline | Trainable? | Purpose |
|---|---|---|---|
| B0 | Class prior + random scores | no | metric floor; PR-AUC floor = positive rate |
| B0k | Section-kind-conditional class prior | no | guards against render_v2's kind-correlated input formats being mistaken for evidence judgment (§9) |
| B1 | Zero-shot Qwen3-VL-2B-Instruct, identical prompt/score | no | value of LoRA (mandated ablation pair with Arm A) |
| **B2 (primary)** | Frozen ColQwen2 MaxSim (+ length norm) + Platt on val | 2 params | the null hypothesis "retrieval similarity already solves within-page evidence selection." The project's central claim is precisely that it does not; every trained model must beat B2 to matter |
| B2b | Frozen ColQwen2.5 MaxSim (+ length norm) + Platt on val | 2 params | strongest retrieval-similarity null; report alongside B2; headline comparator remains B2 because B2 is nested with Arm B |
| B3 | BM25 (query vs OCR section text) | no | lexical floor; quantifies lexical-overlap shortcut in SciEGQA phrasing |
| **B4 (second load-bearing)** | Qwen3-Reranker-0.6B text cross-encoder over query + OCR markdown, zero-shot and LoRA-fine-tuned (same BCE objective, same splits) | yes | the null hypothesis "pixels don't matter — DeepSeek OCR text suffices." If B4-ft ≈ Arm A, the finding is that OCR text is the evidence carrier; visual sections stratum will adjudicate |
| B5 | Frozen-VLM linear probe: logistic regression on last-hidden-state at decision position of B1 | tiny | separates "representation already knows" from "LoRA teaches the decision" |
| **B6** | Qwen3-VL-Reranker-2B multimodal cross-encoder over query + `render_v2` section payload, frozen, official template, temperature+thresholds on validation | no | off-the-shelf multimodal reranker null; direct comparator for Arm R |

Rejected as unnecessary: jina-reranker-m0 zero-shot (different scoring head, adds license/eng cost without changing any decision); a from-scratch cross-encoder (dominated by B4).

**B6 — Qwen3-VL-Reranker-2B zero-shot.** The released reranker sibling of Arm A's backbone, scored with its own official model-card template and usage code over the same `render_v2` payloads. Inference-only; temperature and thresholds fitted on validation only. B6 is the required null for Arm R: if B6 ≈ R1, task-specific LoRA adds little; if R1 ≫ B6, segment-level adaptation is justified.

Primary-baseline justification: B2 is the correct comparator because the pivot narrative ("MaxSim is high-recall but not an evidence gate") is only established if calibrated MaxSim demonstrably fails at pair classification while pixel-inspecting models succeed. B4 guards the complementary over-claim ("VLM needed" when OCR text suffices).

### 8.1 Abstracted-query evaluation set (lexical-shortcut diagnostic)

Motivation: SciEGQA queries are written with the gold page open and embed the evidence tokens verbatim ("For PT, α=0.65, what is C_tot when i_r=0?"), so lexical-overlap models (B3, B4) can partly solve the text strata by string match, and — because text sections reach Arm A as OCR text (§9) — Arm A has no structural advantage to demonstrate on those strata. The abstracted set removes the verbatim evidence tokens while preserving the answer target and the page-locating context, converting the biggest validity threat (§17 risk 2) into a direct measurement.

- **Scope of abstraction:** strip tokens that also appear in the gold section's OCR text (values, symbols, exact settings); **keep** the page anchor (which figure/table/system the question concerns) so it remains a single-page question, not an open-corpus one (§2.2 forbids the latter). Example: "For PT, α=0.65, what is C_tot when i_r=0?" → "For the PT configuration, what is the total capacitance under the specified reverse-current condition?"
- **Provenance:** pinned rewriter model + revision, temperature 0, deterministic prompt, every raw response retained and hashed — same discipline as the OCR model. This is the only LLM in the query path; it touches evaluation only.
- **Eval-only discipline:** the abstracted set is **held out from all primary-arm training**. Every model is trained on original queries and evaluated on **both** original and abstracted (the paired original→abstracted gap is the diagnostic; dropping the original eval loses the reference point). Because the coverage label is geometric (`gold_box ∩ union(member_boxes) / area(gold_box)`, query-wording-independent), positive labels are preserved for free under rewording.
- **Uniqueness re-audit (required):** after abstraction, recompute whether the gold section still *uniquely* answers each query; **drop any query where abstraction makes a former hard-negative section also answer it** (would inject false negatives into the metric). Report the drop count.
- **Primary reading:** differential degradation. Expect B3/B4 to fall sharply on abstracted (crutch removed) and Arm A to fall little (pretrained semantics); if Arm A's advantage over B4 *grows* on abstracted queries, that is the cleanest evidence for the project's thesis on the text strata.
- **Optional, fenced-off robustness run (not part of any primary comparison):** train on an original+abstracted mix, evaluate on held-out **page-disjoint** abstracted queries — answers the separate question "*can* the model learn the harder task when given the signal?" Run only if the eval-only diagnostic shows a large shortcut effect and deployment realism matters. Never shares a comparison with the primary result.

---

## 9. Shared input-rendering specification (`render_v2`)

Primary policy (set by user direction 2026-07-03): **modality-native section content using the DeepSeek-OCR2 preprocessing already stored in the dataset.**

| Section kind | Arm A input (`<Document>` field) | Arm B input (document side) |
|---|---|---|
| headed_text / document_preamble / deepseek_paragraph | OCR markdown text of the section, verbatim from `labels/semantic_sections.jsonl` | Region crop image of the section's member-box union area (one crop per member box if discontiguous) |
| visual_bundle — image run | One pixel crop per image member box (native aspect, cropped from the full-resolution page PNG, boxes scaled from [0,1000] with round-half-even) + caption/`figure_title` OCR text appended as `Caption: {text}` | Same crops |
| visual_bundle — table run | Table member crops + DeepSeek table markdown + caption OCR text | Same crops |

Rules and rationale:

- **Discontiguous members → multiple images**, in reading order, never a stitched composite or envelope (envelope swallows sibling sections; stitching violates dataset provenance rules). Qwen3-VL natively supports multi-image inputs; assert per-sample total visual tokens ≤ 2,048 and log the distribution. Sections exceeding the budget are downscaled proportionally, never dropped.
- **Caption text as text** (user direction). Captions are also present in the crop pixels when their boxes are members; that redundancy is accepted and identical for both classes.
- **Long text sections:** rendered as-is up to 6,144 text tokens (fits 32K context with margin); if any section exceeds it, truncate tail, set a `truncated` flag in the manifest, and report the affected count from audits (expected rare; headed sections are page-bounded).
- **Crop margin:** 8 px context padding at page scale around each member box (avoids clipping glyph edges from box quantization); padding value is part of render_v2 and identical across arms.
- **Determinism:** every rendered artifact is a pure function of (page PNG SHA-256, member boxes, render_v2 params); crop PNGs and text payloads are SHA-256-hashed into training manifests.
- **Base-rate shortcut check:** input format now correlates with section kind. Kind is a legitimate observable, not leakage, but the stratified analysis must verify the model outperforms a kind-conditional class-prior baseline (added as B0k) so that "image present → answer no" style priors are not mistaken for evidence judgment.
- **Arm R input:** Arm R receives the exact same `render_v2` `<Document>` payloads as Arm A and uses the official Qwen3-VL-Reranker template around those payloads. This keeps `R1` vs `B6` and `R1` vs `A1` comparisons focused on initialization/training rather than rendering.
- **Arm asymmetry (documented):** Arm A sees OCR text where Arm B sees text-region pixels; Arm A additionally sees table markdown. This mirrors each family's native input channel and the deployment design (VLM judges preprocessed content). Bridging ablation **E-R5**: Arm A with text sections given as region crops instead of OCR text — isolates how much of any Arm A advantage is the OCR text channel rather than the backbone.
- **Retained ablations:** E-R1 (full-page + red member-box highlight, the former render_v1 primary — tests whether page context beats extracted content), E-R3 (image-only visual bundles, no caption/table text), E-R4 (crops + OCR text for *all* kinds — the "give everything" variant).

Risks: (1) region crops are out-of-distribution for ColQwen2's page-level pretraining — Arm B results must be read with this caveat, partially bounded by E-R1 giving Arm B a full-page input; (2) OCR text errors now propagate directly into Arm A's text-section inputs (label noise and input noise become correlated since both derive from DeepSeek-OCR2 — flag in limitations); (3) text sections make Arm A vs B4 a same-input comparison on the text strata, which sharpens interpretation: any Arm A edge there is backbone quality, and the visual-bundle stratum alone carries the multimodality claim.

---

## 10. Loss, batching, and negative-sampling design

- **Primary loss (Arm A and Arm B):** pair-level BCE on the mandated logit. Grounds: proper scoring rule (calibration requirement), SFT>CL evidence (Dai et al.), Qwen3-VL-Reranker precedent.
- **Arm R loss:** full-vocabulary CE on the `"yes"`/`"no"` label token at the decision position, matching the released Qwen3-VL-Reranker objective; scoring and calibration still use `s = z_yes − z_no`. `R1-BCE` tests the restricted BCE variant for apples-to-apples comparison with Arm A.
- **Group structure:** each retained query contributes 1 positive + n_q same-page negatives, n_q variable. Unweighted pair-iid training lets many-section pages dominate gradients and skews the base rate. Design:
  - **Query-balanced batching:** sample queries uniformly; per sampled query take its positive + **all** its negatives (mean n_q ≈ 3.1 per the 4K run, expected stable in the combined set — the planned K=7 cap is moot since most queries have ≤3 negatives; ablation E-L2 deleted). Per-device batch **8 queries × ~4 pairs ≈ 32 pairs/step**; base rate per batch ≈ 1/4.
  - **Negative resampling:** unnecessary — all negatives fit in every epoch. Long-page domination is bounded by the query-uniform sampler.
  - **Class weighting:** none (3.1:1 is mild; global pos_weight would distort calibration). E-L3 (pos_weight/focal) deleted as moot at this ratio; revisit only if a future dataset changes the ratio materially.
  - **Auxiliary ranking loss:** Arm B gets group-softmax (§7). Arm A primary stays pure BCE; E-L4 tests +λ·group-softmax over the same K+1 logits (z_yes−z_no per pair) at λ=0.5 — hypothesis: helps Rank@1, risks calibration; adopt only if Rank@1 +>1.5 pts and ECE within +1 pt.
  - **Hard-negative curriculum:** rejected for primary — all negatives are already same-page hard negatives by construction; a curriculum needs a difficulty signal we'd have to invent (B2 score could serve; noted as future work).
  - **Difficulty/type-stratified negative sampling:** rejected (would distort the natural section-type mix the gate faces at deployment); the stratified *evaluation* (§14) covers the diagnostic need.
- **Sampling determinism:** sampler seeded per (seed, epoch); manifests record every (pair_id, step).

---

## 11. Hyperparameter table (exact values / search ranges)

Legend: [P] pilot-searched, [F] fixed by evidence, [A] audit-dependent rule.

| # | Hyperparameter | Arm A | Arm B | Basis |
|---|---|---|---|---|
| 1 | Backbone | Qwen3-VL-2B-Instruct (pin HF revision at env build) | vidore/colqwen2-v1.0 (pin revision) | §5, mandate |
| 2 | Prompt/template | prompt_v1 (§6.1) [F] | official ColQwen2 processors [F] | 2601.04720 |
| 3 | Decision tokens | "yes"/"no" single-token IDs, asserted+logged [F] | n/a | verified usage code |
| 4 | Loss | BCEWithLogits(z_yes−z_no) [F] | BCE(sigmoid-affine(maxsim/N_q)) + λ·group-softmax [P λ∈{0,0.5,1}] | §10 |
| 5 | LoRA targets | LM all-linear q,k,v,o,gate,up,down ×28 [F; placement ablation §13] | published regex incl. custom_text_proj [F] | QLoRA finding; HF adapter |
| 6 | LoRA r / α / dropout / bias / init | 16 / 32 / 0.05 / none / PEFT default [P r∈{8,16,32}] | 32 / 32 / 0.1 / none / gaussian [F] | §6.2; §3.2 |
| 7 | Vision tower / merger | frozen [F; ablations c,d §6.2] | frozen (excluded by regex) [F] | ColPali −0.7; community |
| 8 | Visual/text budget | ≤2,048 visual tokens per sample (multi-crop, native aspect); text sections ≤6,144 tokens, truncation logged [F] | 768 patches per crop set [F] | §9 render_v2; Fig. 7 2601.04720 |
| 9 | Optimizer | AdamW [F] | AdamW [F] | §6.3 |
| 10 | LR | 1e-4 [P {5e-5,1e-4,2e-4}] | 5e-5 [F]; calibrator 1e-3 [F] | LoRA band; published |
| 11 | Schedule / warmup | cosine→0.1×, 3% [F] | linear→0, 2.5% [F] | §6.3; published |
| 12 | Weight decay | 0.01 [P {0,0.01}] | 0.01 [F] | §6.3 |
| 13 | Precision / ckpting / clip | bf16 / on / 1.0 [F] | bf16 / on / 1.0 [F] | §6.3 |
| 14 | Batch | 8 queries × ~4 pairs ≈ 32 eff. (accum to fit) [instantiated from audits] | 32 pairs eff. same sampler [F] | §10 |
| 15 | Negatives per query | all (mean 3.1; K cap moot, E-L2 deleted) | same | §10; audits |
| 16 | Epochs / eval cadence / patience | ≤4 / every 250 steps / 3 [combined dataset: 606 query-group steps/epoch] | same | §6.3; combined audit |
| 17 | Checkpoint selection | best val PR-AUC [F] | same | mandate |
| 18 | Calibration | temperature on val [F] | affine is in-model; optional extra T [F] | §6.5 |
| 19 | Thresholds | val max-F1 + recall≥0.95 point [F] | same | §6.5 |
| 20 | Seeds | 13, 17, 23; extend to 5 seeds (+29, 31) at Stage 4 if budget allows — runs are now cheap (~1–2 GPU-h) and eval noise is the binding constraint | same | repo convention; audits |
| 21 | Attention impl | FA2→sdpa fallback [F at smoke] | per colpali-engine default [F] | pass 3 |
| 22 | Steps/epoch | 606 query-group steps/epoch (`ceil(4,842 train queries / 8 query-groups)`) | same | combined audit |

### 11.1 Arm R hyperparameters

| # | Hyperparameter | Arm R | Basis |
|---|---|---|---|
| 1 | Backbone | Qwen/Qwen3-VL-Reranker-2B (pin HF revision at env build) | released multimodal reranker |
| 2 | Prompt/template | official Qwen3-VL-Reranker template and usage path [F] | B6/R1 fairness |
| 3 | Decision tokens | `"yes"`/`"no"` single-token IDs, asserted+logged [F] | official reranker scoring |
| 4 | Score | `z_yes − z_no`, gathered at last non-padding decision position [F] | shared scorer API |
| 5 | Loss | full-vocabulary CE on the yes/no label token [F]; `R1-BCE` uses `BCEWithLogits(z_yes−z_no)` | official reranker objective; matched ablation |
| 6 | LoRA targets | LM `q,k,v,gate,up,down` ×28; no `o_proj` [F] | official Qwen3-VL-Reranker LoRA config |
| 7 | LoRA r / α / dropout / bias / init | 32 / 32 / 0.05 / none / PEFT default [F except dropout project-default] | official r/α; project regularization |
| 8 | Vision tower / merger | frozen [F] | project primary policy |
| 9 | Visual/text budget | same as Arm A/render_v2 [F] | fairness with A/B6 |
| 10 | Optimizer | AdamW [F] | project default |
| 11 | LR | 5e-5 [P {5e-5,1e-4}] | safer adaptation from relevance-tuned checkpoint |
| 12 | Schedule / warmup | cosine→0.1×, 3% [F] | project default |
| 13 | Weight decay | 0.01 [F] | project default |
| 14 | Precision / ckpting / clip | bf16 / on / 1.0 [F] | project default |
| 15 | Batch | 8 queries × ~4 pairs ≈ 32 effective [F] | same sampler |
| 16 | Negatives per query | all [F] | same sampler |
| 17 | Epochs / eval cadence / patience | ≤4 / every 250 steps / 3 [F] | same training budget |
| 18 | Checkpoint selection | best validation PR-AUC [F] | mandate |
| 19 | Calibration | temperature on validation [F] | shared protocol |
| 20 | Thresholds | validation max-F1 + recall≥0.95 point [F] | shared protocol |
| 21 | Seeds | 13, 17, 23; optional +29, +31 at Stage 4 [F] | repo convention |
| 22 | Attention impl | FA2→SDPA fallback [F at smoke] | same as Arm A |
| 23 | Steps/epoch | 606 query-group steps/epoch | combined audit |

---

## 12. LoRA target-module maps (verified)

**Arm B — verified from `vidore/colqwen2-v1.0/adapter_config.json` (fetched):**
```json
r: 32, lora_alpha: 32, lora_dropout: 0.1, bias: "none", init_lora_weights: "gaussian",
target_modules: "(.*(model).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$|.*(custom_text_proj).*$)"
```
Matches Qwen2-VL LM decoder projections + retrieval head; vision (`visual.*`) unmatched → frozen.

**Arm A — Qwen3-VL-2B (from fetched config.json: 28 text layers, hidden 2048, tied embeddings; vision depth 24, deepstack {5,11,17}):**
```python
target_modules = [f"model.language_model.layers.{i}.{m}"
                  for i in range(28)
                  for m in ("self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj",
                            "mlp.gate_proj","mlp.up_proj","mlp.down_proj")]
# EXCLUDE: lm_head / embed_tokens (tied), model.visual.* (frozen), merger & deepstack_merger_list (frozen; ablation c adds them)
```
**Mandatory smoke assertion:** print `[n for n,_ in model.named_modules()]`, assert every target resolves and that the PEFT-wrapped model's trainable set equals exactly the intended list (guards against transformers prefix refactors, e.g. `model.model.language_model...`). Record trainable-parameter count in the manifest.

**Arm R — Qwen3-VL-Reranker-2B (official reranker LoRA target family):**
```python
target_modules = [f"model.language_model.layers.{i}.{m}"
                  for i in range(28)
                  for m in ("self_attn.q_proj","self_attn.k_proj","self_attn.v_proj",
                            "mlp.gate_proj","mlp.up_proj","mlp.down_proj")]
# EXCLUDE in primary Arm R: self_attn.o_proj, lm_head / embed_tokens (tied),
# model.visual.* (frozen), merger & deepstack_merger_list (frozen)
# r=32, lora_alpha=32, lora_dropout=0.05, bias="none"
```

If the actual prefix differs from `model.language_model`, the runtime resolver must fill `P`; the schematic above is not permission to guess the prefix.

**Mandatory Arm R smoke assertion:** discover the actual language-model prefix under the pinned transformers revision; assert every target resolves; assert the trainable set contains exactly the intended LoRA A/B weights and nothing else. If the resolver accidentally includes `o_proj`, `lm_head`, embeddings, vision modules, or merger modules, abort.

---

## 13. Staged experiment matrix

Stage gates prevent a Cartesian grid (testing every combination of every dial — ~800+ runs; instead: fixed defaults, one-dial-at-a-time ablations, gated progression). All selection on validation; seeds 13/17/23 (optionally +2) only at Stage 4; single seed 13 elsewhere. Compute per run from the combined dataset: **4,842 train queries, 606 query-group steps/epoch, ≤4 epochs → ~2–4 GPU-h per Arm A run** (Stage 0 measurements remain authoritative). Decision-threshold caveat: with 1,038 validation queries, differences below **~±3 pts** are within eval noise — thresholds at or below that are directional, not decisive.

**Stage 0 — Smoke (½ GPU-h).** 64 pairs, 20 steps, Arm A, Arm R, and Arm B. Asserts: tokenizer/decision-position, LoRA target resolution, render determinism (hash-stable), loss decreases, memory + s/step measured. Stage 0 must load Qwen3-VL-Reranker-2B, run frozen B6 scoring on the same 64 validation groups, run a 20-step R1 microtrain on 64 train groups, and verify the official reranker template, no `<think>` block, yes/no single-token IDs, last-non-padding logit gather, and trainable-module assertions. Failure → fix before anything else.

**Arm A Stage 0.5 diagnostic (diagnostic only).** After the Stage 0.5 hard certification checks are in place, score frozen Arm A/B1 on the same certification slice and compare it to B6 and R1-step0. This is a lightweight diagnostic for how much Qwen's dedicated reranker training helps before our task-specific training; it reports metrics and deltas only, does not change the certification pass/fail decision, and is not a new research track.

### 13.1 Insights

- Initial Stage 0.5 diagnostic: frozen Arm A scored similarly to R1-step0/B6 on the small certification slice despite R1/B6 starting from a dedicated reranker-trained checkpoint. Arm A's Rank@1 was slightly higher on that slice, while B6/R1-step0 had better PR-AUC; treat this as a small-slice signal, not a conclusion.
- Stage 1 B5 vs B6 interpretation: B5 exceeded B6 on full validation (B5 PR-AUC 0.8565, Rank@1 0.8775; B6 PR-AUC 0.8304, Rank@1 0.8695), but this is not a clean zero-shot model-strength comparison. B5 trains a supervised logistic probe on our train labels over frozen B1 hidden states, while B6 is a frozen zero-shot reranker using its released yes/no head. Read B5 as evidence that the frozen VLM representation contains useful segment-evidence signal; do not treat the B5>B6 result alone as suspicious enough to block Stage 2, nor as proof that the instruct backbone is inherently stronger than the reranker.
- Stage 2 Arm R pilot result (2026-07-10): both completed R1 pilots substantially exceeded frozen B6 on validation without touching test. B6/step-0 validation PR-AUC was 0.8304. `R1` (lr 5e-5) selected step 1750 with validation PR-AUC 0.9393 and Rank@1 0.9247; `R1-lr1e4` selected step 1250 with validation PR-AUC 0.9396 and Rank@1 0.9378. Train/validation score-file integrity checks passed for both pilots (19,474 train rows, 4,155 validation rows, no duplicate pair IDs, no label/split mismatches, zero test IDs). The earlier 8h jobs timed out after already reaching the same performance region; the completed 16h jobs showed a plateau/early-overfit pattern rather than continued late improvement: R1 peaked at step 1750 and drifted down through step 2424, while `R1-lr1e4` peaked at step 1250 and then drifted down by step 2000. Treat the gain as strong evidence of task-adaptation signal, but continue Stage 2 comparisons against B4-ft and Arm B before claiming multimodal specificity.

**Stage 1 — Frozen baselines (≈8 GPU-h).** B0–B6 zero-shot/frozen parts, B2/B2b/B3 Platt fits. Decision: record B2/B2b, B4-zs, and B6 as the bars; if B6 zero-shot PR-AUC is already near the trained models, the project must be framed as reranker transfer/adaptation rather than a new-model result.

**Stage 1b — Arm A full-validation diagnostic (optional, diagnostic only).** If compute/time is reasonable after Stage 1 frozen baselines, score frozen Arm A/B1 on the full validation split with no training and no test access. Question: **Does Arm A's Rank@1 advantage survive full validation, or was it a small-slice artifact?** Report PR-AUC, Rank@1, MRR, Recall@2, calibration, and deltas versus B6/R1-step0. This diagnostic may inform interpretation and Stage 2 prioritization, but it is not a new training run and does not replace the Stage 1 baseline report.

**Stage 2 — Pilots, 1 seed (≈15–28 GPU-h at combined dataset size).**
- E-A1 Arm A default (r16 all-linear, K=7, lr 1e-4) + LR mini-sweep {5e-5, 2e-4} (3 runs).
- `R1` Arm R default (r32 official targets, full-vocabulary CE, lr 5e-5) + `R1-lr1e4` (2 runs). Run `B6` before `R1`.
- E-B1 Arm B default (λ=0.5) + λ∈{0,1} (3 runs, cheaper).
- B4-ft text cross-encoder (1 run).
- If compute is constrained, prioritize `R1`, `B4-ft`, and Arm B over the old A1 LR sweep; keep A1 as the instruct-initialization comparison.
- Decision rules: Arm A pilot must beat B2 on val PR-AUC by >4 pts (comfortably above the ~±3 pt noise floor at 1,038 validation queries) to proceed to full ablations, else run E-R1/E-R4 first (input policy may be the bottleneck). Arm B compared to B2 tests whether LoRA rescues MaxSim. `R1` must beat `B6` by >3 validation PR-AUC points to support a task-adaptation claim, must beat or clearly match `B2b` to keep the cross-encoder claim alive, and must beat `B4-ft`, especially on visual bundles and abstracted queries, to support the multimodal claim. If `B6 ≈ R1`, continue only as an evaluation/calibration study or reduce Arm R ablations. If `R1 < B6`, inspect LR/loss/template before any further Arm R ablation.

**Stage 3 — Ablations, 1 seed each on the Stage-2 winner (≈30–50 GPU-h).**

| ID | Ablation | Hypothesis | Adopt if |
|---|---|---|---|
| E-A2 | LoRA placement (a) attn-only, (c) +merger, (d) +vision-last-6 | all-linear ≥ attn-only; vision adaptation ≤ +1 pt | val PR-AUC ± seed-13 noise band |
| E-A2r | rank {8, 32} at winning placement | flat in r | pick smallest within 1 pt |
| `R1-BCE` | Arm R | restricted `BCEWithLogits(z_yes−z_no)` instead of full-vocabulary CE | adopt only if validation PR-AUC or calibration improves beyond noise |
| `R1-r16` | Arm R | r=16, α=32 with the same official target modules | run only if R1 overfits or Stage 0 shows r32 is unnecessarily costly; pick smallest within 1 pt |
| `R1-o_proj` | Arm R | add `self_attn.o_proj` to the official target list | diagnostic only; adopt only if >3 pts val PR-AUC and no instability |
| ~~`E-A9`~~ | ~~Arm A~~ | ~~initialize LoRA training from Qwen3-VL-Reranker-2B weights instead of Instruct~~ | **promoted to Arm R / `R1`; no longer a late ablation** |
| E-R1 | full-page image + red member-box highlight (both arms) vs render_v2 | page context ≤ +2 pts over extracted content; also gives Arm B an in-distribution page input | if highlight > +2 pts, revisit render_v2 for confirmation stage |
| E-R3 | visual bundles image-only (drop caption/table text) | caption text carries real signal (drop hurts) | reporting; quantifies text-channel value |
| E-R4 | crops + OCR text for all kinds ("give everything") | ≈ render_v2 | adopt only if > +2 pts |
| E-R5 | Arm A text sections as region crops (no OCR text) | OCR text ≥ pixels for text kinds | bridges Arm A/B asymmetry; reporting |
| ~~E-L2~~ | ~~K ∈ {3, all} vs 7~~ | **deleted** — mean 3.1 negatives/query makes the cap moot; "all" is the only sensible setting | — |
| ~~E-L3~~ | ~~pos_weight / focal γ=2~~ | **deleted** — 3.1:1 imbalance is mild; remedies unjustified | — |
| E-L4 | +group-softmax aux (Arm A) | +Rank@1, −calibration | adopt only if Rank@1 gain exceeds noise floor (~+3 pts at n=1,038 validation queries), ΔECE<1 |
| E-B2 | Arm B raw vs length-normed MaxSim | normed better calibrated | keep winner |
| E-B4 | Arm B frozen-adapter (train affine only) ≡ B2 | nested-model sanity | reporting |
| E-Q1 | **Abstracted-query eval** (§8.1): all trained models + B3/B4, evaluated on abstracted set (trained on originals only) | B3/B4 drop sharply; Arm A drops little; Arm A−B4 gap grows | reporting; primary evidence for thesis on text strata |
| E-D1 | **Learning curve**: Arm A pilot config on 25% / 50% / 100% of training pages (page-grouped subsets, seed 13), val PR-AUC vs train size, overall + per-stratum | curve flattens by 50→100% (data-saturated on the combined training set) | if 50→100% gain ≥ 3 pts, model is training-data-limited → only remaining headroom is the 814-page remainder pool (§2.1a); otherwise scope the claim; ~5–8 GPU-h |

**Stage 4 — Confirmation, 3–5 seeds (≈20–40 GPU-h).** Final Arm R config if selected, final Arm A config if still active, final Arm B config, B4-ft; label-shuffle control (1 seed, expect collapse to B0); leakage checks (§16). Paired stats per §14. Prefer 5 seeds (13/17/23/29/31) if Stage 0 confirms ~2–4 GPU-h/run: seed variance is a large fraction of the resolvable effect size.

**Stage 5 — Test (≈4 GPU-h).** One pass for final Arm R if selected and the other selected final models, frozen thresholds/temperatures. Also: document-disjoint sensitivity re-evaluation (below), abstracted-query test evaluation (§8.1; paired original-vs-abstracted, reported per stratum), and stratified reporting.

**Document-disjoint sensitivity (analysis, not a new split):** within the canonical test set, partition queries into (i) pages whose source document also contributes ≥1 train page, (ii) documents unseen in train. Report metrics on both; a gap > 3 pts PR-AUC flags template/authorship leakage and mandates a document-grouped split for any follow-up work. The canonical page-grouped split is never replaced silently (mandate).

For every experiment the run manifest records: hypothesis, trainable modules, exact input hash, loss, data subset, all hyperparameters, seed(s), metrics, GPU-hours, and the decision taken.

---

## 14. Evaluation, calibration, statistics

**Pair-level (primary: PR-AUC):** PR-AUC, ROC-AUC, precision/recall/F1 and balanced accuracy at the two val-chosen thresholds, MCC, NLL, Brier, ECE (15 equal-mass bins) pre/post temperature.
**Query-group:** positive-section Rank@1, MRR, Recall@2, accepted-sections/query at each threshold, evidence recall at fixed budget of 1 and 2 accepted sections, precision at the recall≥0.95 operating point. **Instantiated floors (mean 4.1 sections/page):** random Rank@1 ≈ 0.25 and Recall@3+ is near-trivial — report Rank@1 and Recall@2 only, always alongside the random floor; PR-AUC floor = 0.246 (pair positive rate).
**Stratified:** on the combined dataset, test has 1,038 queries (noise floor ≈ ±2.8 pts; projected visual-bundle test stratum ≈ 290, confirm from combined by-kind audit) — the headline strata (text vs visual bundle; domain macro-average; collapsed intent) are adequately powered. Headline metrics use **domain macro-averaging** (§2.1a balance policy). Finer cells (reasoning op × kind, member-count, coverage bands) remain descriptive appendix under the <30-query suppression rule. Report **coarse strata as primary** (text vs visual bundle; domain; query intent collapsed to {table, plot/figure, formula, textual}); treat finer cells (reasoning op × kind, member-count, coverage bands) as descriptive appendix only. Small-cell rule: suppress cells with <30 queries into "other".
**Query-set condition:** every headline table is reported twice — on **original** and on **abstracted** queries (§8.1) — for all trained models and B3/B4. The primary lexical-shortcut metric is the paired per-model drop `metric(original) − metric(abstracted)` and the change in the Arm A−B4 gap between conditions, on the text strata. Models are trained on originals only; abstracted is eval-only.
**Statistics:** per-query bootstrap (10,000 resamples, clustered by page) for 95% CIs on all headline metrics; paired comparisons (Arm R vs B6, Arm R vs B2/B2b, Arm R vs B4-ft, Arm R vs Arm A, Arm A vs B2, Arm A vs B4-ft, Arm A vs Arm B, and original vs abstracted within each model) via paired bootstrap on per-query Rank@1/AP differences + Wilcoxon signed-rank; seed variability reported as mean ± sd over 3 seeds; a difference counts only if the paired 95% CI excludes 0 *and* exceeds seed sd.
**Discipline:** thresholds, temperature, checkpoint selection, and all model decisions on validation; test evaluated once per final model.

---

## 15. SOL resources and execution order

**Hardware (to confirm on cluster before finalizing batch sizes):** target 1× A100 80 GB (`--gres=gpu:a100:1 -p public`); Sol has 56×4 A100-80GB nodes + a few H100s and 15 A30s (public docs). Confirm live: `sinfo -p public -o "%n %G %m"`; per-job `nvidia-smi` preflight is already repo standard. **Must be measured at Stage 0 (do not trust estimates):** s/step and peak VRAM for all trained arms at per-device batch 4; rendered-page visual-token count under each processor.

**Planning estimates (combined dataset; Stage 0 measurements still authoritative):** 4,842 train queries → 606 query-group steps/epoch; Arm A at ~1.5–3 s/optimizer-step → **~2–4 GPU-h per run (≤4 epochs with early stop)**; Arm B ~0.5× that; text sections as OCR text (render_v2) make the average sequence far cheaper than the old full-page estimate. Totals: Stage 0 ≈ 0.5; Stage 1 ≈ 5–8; Stage 2 ≈ 15–28; Stage 3 ≈ 30–50 (incl. E-D1 ≈ 5–8); Stage 4 ≈ 20–40 (3–5 seeds); Stage 5 ≈ 3–5; expansion OCR ≈ 22 (**spent**). **Grand total ≈ 95–155 A100-hours**, ~27–38 distinct runs. Arm R is expected to be the same order of cost as Arm A, but Stage 0 measurements are authoritative; add approximately two Stage-2 pilot runs (`R1`, `R1-lr1e4`) unless B6 or smoke tests make R1 unnecessary. Arm A confirmation runs (~2–4 GPU-h) still fit 4 h `htc` jobs; if Stage 0 measures the high end, use ≤8 h `public` jobs for Stage 4. Checkpoint storage: keep best + last per run; LoRA adapters ~70 MB → <10 GB total on scratch; rendered-crop cache well under 4 GB (render once, reuse across arms; keyed by SHA-256).

**Execution order:** (1) merge `codex/sciegqa-final-labeling` → main, push, pull on SOL, verify `git merge-base --is-ancestor 546d9c8 HEAD`; (2) run the evidence-dataset handoff phases 1–10; (3) freeze audits and instantiate all [A]-tagged hyperparameters; (4) build the training env (pin transformers version supporting `qwen3_vl` + `colpali-engine>0.3.4`; freeze `pip list` into the manifest); (5) Stages 0→5 with the gates in §13. Heavy work on compute nodes only; caches/outputs on `/scratch/$USER`; code in home/repo.

---

## 16. Reproducibility and leakage checklist

- Pins: dataset `Yuwh07/SciEGQA-Train@4ffb867c...`; DeepSeek-OCR-2 `@aaa02f38...`; Qwen3-VL-2B-Instruct revision (resolve at env build, record SHA); `Qwen/Qwen3-VL-Reranker-2B` revision; official Qwen3-VL-Reranker template hash; Arm R yes/no token IDs; Arm R LoRA target list hash; `vidore/colqwen2-v1.0` revision; tokenizer/processor revisions; repo commit; full `environment freeze` per job (existing repo pattern).
- Record: seeds (selection 20260630; expansion selection seed; **combined split seed 20260707**; training 13/17/23[/29/31]), prompt_v1 text hash, yes/no token IDs, render_v2 params + per-artifact output hashes, split assignment hashes, sampler traces, all metric code versions.
- **Leakage battery (Stage 4):** (a) label-shuffle → expect ≈ B0; (b) answer-field grep over every serialized model input across Arm A, Arm R, B6, and other serialized model inputs (assert 0 matches of answer strings, modulo whitespace normalization); (c) gold-box/coverage exclusion audit: assert prompt builder consumes only (query, page image, member boxes of the *candidate*), never gold geometry — the red boxes must be the candidate's members for negatives too (identical rendering rule for both classes, so the highlight itself carries no label signal); (d) page-split integrity re-assert at load time; (e) document-disjoint sensitivity analysis (§13).
- Determinism: two independent evaluation runs of the final models must produce byte-identical metric JSONs (mirrors the dataset `determinism.json` contract).

---

## 17. Risks, limitations, non-claims

- **Cannot claim:** open-corpus retrieval performance; page-level retrieval gains; answer-generation quality; planner behavior; generality beyond arXiv-style scientific pages or beyond DeepSeek-OCR2 segmentations (labels are parser-relative); human-verified ground truth (labels derive from a 0.70 geometric-coverage rule).
- **Known risks:** (1) combined quarantine removed 2,167 / 6,630 pages and 3,513 / 10,431 queries — whether it disproportionately removed visually complex pages is still unknown until the **by-kind breakdown** is pulled from `labeling_audit.json`; if visual-bundle positives land materially below the projected stratum, the multimodality claim downgrades from stratified result to suggestive observation; (2) SciEGQA lexical phrasing may let B3/B4 solve text sections — the visual-bundle stratum is the claim-bearing subset for "pixels matter," and the abstracted-query eval (§8.1) is the counter-measurement; (3) **evaluation power**: the combined dataset's 1,038 test queries resolve ≈ ±2.8 pt differences — adequate for the POC's go/no-go and coarse strata, still not 1–2 pt resolution; (4) one-positive-per-query construction means "multiple sufficient sections" cases were quarantined — deployment will see them; the calibrated threshold policy (not argmax) partially covers this; (5) `qwen3_vl` support on the SOL software stack unverified — fallback Qwen2.5-VL-3B documented in §5; (6) remaining compute estimates in §15 are planning numbers; Stage 0 measurements are authoritative.
- **Arm R-specific risks:** if frozen B6 matches tuned Arm R, the modeling novelty downgrades; the contribution becomes evidence-gating evaluation, transfer, and calibration of an existing multimodal reranker. If B4-ft matches Arm R, the multimodal claim weakens; OCR text may be the dominant evidence carrier on the retained dataset.

## 18. Implementation sequence (condensed)

1. ~~Original 4K dataset build~~ — **DONE** (run `20260703T090303Z`); ~~expansion selection + new-page OCR~~ — **DONE** (3,700 queries / 2,919 pages, 32/32 shards); ~~combined Option 4 finalization~~ — **DONE** (run `20260707T210746Z`).
2. Pull combined audit details needed for reporting (pair counts by split, section-kind breakdowns, negatives/query distribution, sections/page p50/p95, 1-query-page quarantine sanity number).
3. **Post-audit, never in the build pipeline:** generate the abstracted-query eval variant (§8.1) with a pinned rewriter — strip gold-section-overlapping tokens, keep page anchor, temperature 0, retain+hash all responses; run the uniqueness re-audit and drop queries whose gold section is no longer unique; emit an `abstracted_query` field + `abstraction_audit.json`. Runs in parallel with Stages 0–2 and never gates the main pipeline.
4. Render+cache all section crops/text payloads (render_v2); audit-dependent hyperparameters are already instantiated in §10/§11.
5. Build training env; pin + freeze; Stage 0 smoke (assertions in §6.1/§12; measure memory & throughput), including B6 frozen scoring and R1 20-step microtrain.
6. Stage 1 frozen baselines including B6 → Stage 2 pilots including R1 and R1-lr1e4 → Stage 3 ablations → Stage 4 three-seed confirmation + leakage battery → Stage 5 single test pass. Do not touch test before Stage 5.
7. Write results with stratified analysis and paired statistics; archive manifests off scratch.

---

### Missing facts required before final numbers (explicit)

| Missing fact | How to obtain |
|---|---|
| **Combined-dataset audit details** (section counts, pair counts, by-kind positives, negatives/query, sections-per-page p50/p95, 1-query-page quarantine sanity number) | combined run's `labeling_audit.json` / `split_audit.json` — pull before Stage 1 reporting |
| Live GPU availability / model on public partition | `sinfo -p public -o "%n %G %m"`; job preflight `nvidia-smi` |
| transformers version supporting `qwen3_vl` on SOL; FA2 vs sdpa stability | env build + Stage 0 |
| yes/no single-token assertion on pinned tokenizer | Stage 0 assert (command in §6.1) |
| Actual s/step, VRAM, visual-token counts per arm | Stage 0 measurements |
| Exact Qwen3-VL module prefixes under pinned transformers | `named_modules()` printout at Stage 0 |
