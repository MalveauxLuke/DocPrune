# Segment Evidence Classifier — Implementation Architecture Specification

Version 1.2 (2026-07-08; v1.2 adds Arm R / R1 reranker-initialized training path and promotes E-A9 into a first-class trained arm). Companion to `sol/task_spec/segment_evidence_experiment_plan.md` (the *why*); this document is the *how*. It is written for a builder with **zero project context**. Follow it literally; nothing here is advisory. Where a value depends on the final dataset audit, it is marked `[RECOMPUTE]` with the exact formula; every other value is final.

The task: given a query and one semantic section extracted from a scientific-document page, output a calibrated probability that the section contains sufficient evidence to answer the query. Labels are binary. Each query has exactly 1 positive section and all other sections from the same page as negatives.

---

## 1. Pinned assets

| Asset | Identifier | Notes |
|---|---|---|
| Dataset (combined) | `/scratch/$USER/sciegqa_train_4k/evidence_option4_10431/combined_evidence/20260707T210746Z/` | Self-contained combined Option 4 run (10,431 source queries / 6,630 source pages → 6,918 retained queries / 4,463 retained pages; commit `d1fedc9`) |
| Source dataset | `Yuwh07/SciEGQA-Train` rev `4ffb867c88e3264161920b4b2446d5ac6352269e` | provenance only; never re-read during training |
| OCR model | `deepseek-ai/DeepSeek-OCR-2` rev `aaa02f3811945a91062062994c5c4a3f4c0af2b0` | already applied; not used at training time |
| Arm A backbone | `Qwen/Qwen3-VL-2B-Instruct` | resolve and pin the HF revision SHA at environment build; record in every manifest |
| Arm R / B6 model | `Qwen/Qwen3-VL-Reranker-2B` | resolve and pin the HF revision SHA at environment build; record official reranker template hash, Arm R yes/no token IDs, and Arm R LoRA target list hash in every relevant manifest |
| Arm B model | `vidore/colqwen2-v1.0` (adapter) on `vidore/colqwen2-base` | pin revision SHA at env build |
| B4 model | `Qwen/Qwen3-Reranker-0.6B` | pin revision SHA at env build |
| Selection seed | `20260630` | fixed, historical |
| Split seed | `20260707` | fixed |
| Training seeds | `13, 17, 23` (confirmation may extend `29, 31`) | see §14 |
| Diagnostic seed | `101` | label-shuffle control |

Software: Python ≥3.11; `torch` (CUDA build for the cluster), `transformers` (version must support model type `qwen3_vl` — verify by loading the model class before anything else), `peft`, `colpali-engine>0.3.4`, `scikit-learn`, `scipy`, `rank_bm25`, `Pillow`. Freeze `pip list --format=freeze` into every run manifest. Hardware: 1× NVIDIA A100 80 GB per job (Slurm: `--gres=gpu:1 --partition=public`, 4 CPU, 64 GB RAM). All heavy work on compute nodes; caches (`HF_HOME`) and outputs on `/scratch/$USER`.

---

## 2. Data interface

### 2.1 Input files (inside the combined run directory)

- `labels/semantic_sections.jsonl` — one row per section: `section_id`, `page_id`, `kind` ∈ {`headed_text`, `document_preamble`, `deepseek_paragraph`, `visual_bundle`}, `heading`, `text`, `markdown`, `member_segment_ids`, `member_bboxes_norm_1000` (list of `[x0,y0,x1,y1]`, coordinates in [0,1000] normalized to page width/height), `member_reading_orders`, `member_types`, `member_raw_types` (raw DeepSeek types; visual members are `image` or `table`; captions are `figure_title`), `caption_link`, `word_count`, page image path + SHA-256, provenance fields.
- `labels/queries.jsonl` — one row per retained query: `query_id`, `query` (text), `answer` (**provenance only — must never enter any model input; see §13 leakage checks**), `page_id`, `positive_section_id`, `split` ∈ {`train`,`validation`,`test`}, domain, `query_intent`, `reasoning_operation`.
- `labels/query_section_labels.jsonl` — one row per pair: `pair_id`, `query_id`, `page_id`, `section_id`, `split`, `label` ∈ {`positive`,`hard_negative`}, `gold_coverage`, `candidate_precision`, gold box, member boxes.
- `splits/{train,validation,test}.jsonl`, `splits/page_assignments.jsonl`.
- Page images: PNG per page at recorded path; verify SHA-256 on first read.
- `audits/labeling_audit.json`, `audits/split_audit.json` — source of all `[RECOMPUTE]` values.

### 2.2 Dataset-derived constants

| Symbol | Formula | Value (combined run `20260707T210746Z`) |
|---|---|---|
| `N_train_queries` | count of train queries | **4,842** (final) |
| `N_val_q` / `N_test_q` | val/test query counts | **1,038 / 1,038** (final) |
| `N_train_pairs` | count of pairs with split=train | ~20,000 `[RECOMPUTE from labeling_audit.json]` |
| `steps_per_epoch` | `ceil(N_train_queries / 8)` (8 query-groups per optimizer step, §5.3) | **606** (final) |
| `max_steps` | `4 × steps_per_epoch` | **2,424** (final) |
| `warmup_steps` | `ceil(0.03 × max_steps)` | **73** (final) |
| `eval_every` | 250 optimizer steps | fixed |

Retained dataset: 4,463 pages / 6,918 queries (2,167 pages / 3,513 queries quarantined; quarantined material is excluded and never manually adjudicated). Dataset run directory: `/scratch/$USER/sciegqa_train_4k/evidence_option4_10431/combined_evidence/20260707T210746Z`.

### 2.3 Frozen schema keys, document identity, abstracted-query interface

**Schema validation (Stage 0, fail closed):** before anything else, validate the actual JSONL files against a required-key list and write the discovered schema to `constants.json`. Required keys per file (if any name below differs in the actual files, the builder must NOT guess: record the actual key mapping in `constants.json` after human confirmation, then proceed):

```
semantic_sections: section_id, page_id, kind, heading, text, markdown,
  member_segment_ids, member_bboxes_norm_1000, member_reading_orders,
  member_types, member_raw_types, caption_link, word_count,
  <page image path key>, <page image sha256 key>
queries: query_id, query, answer, page_id, positive_section_id, split,
  <domain key>, query_intent, reasoning_operation, category, doc_name
query_section_labels: pair_id, query_id, page_id, section_id, split, label,
  gold_coverage, candidate_precision
```

**Caption text source (exact rule for §3.3):** caption text = the `text` content of the member segments whose raw type is `figure_title`, joined with `" "` in ascending reading order. If the section rows carry a parallel member-text array, use it; otherwise resolve `member_segment_ids` against the combined run's `deepseek_ocr2/segments.jsonl` (segment records carry their text). If neither path yields text for a `figure_title` member, abort with the offending `section_id` — never silently substitute.

**Document identity:** `source_document_id = (category, doc_name)` from `queries.jsonl`. This is the key for the document-disjoint sensitivity analysis (§13.5). Do not infer document identity from `page_id` string structure. If either field is absent, abort.

**Abstracted-query file** (produced by the separate abstraction pipeline; consumed at Stages 3/5 only):

```
abstracted_queries.jsonl rows:
{ "query_id": str, "abstracted_query": str, "abstraction_version": str,
  "source_query_sha256": str }
```

Rules: every **validation and test** `query_id` must map to exactly one row (queries dropped by the abstraction uniqueness audit are listed in a sibling `abstraction_dropped.jsonl` and are excluded from *abstracted-condition* metrics only); train queries are never evaluated under abstraction except in explicitly-marked diagnostics; if the file is missing at Stage 5, run the original-query evaluation and emit a `WARN` — never improvise a substitute.

### 2.3a Split discipline

Train on `train` only. All model selection (checkpoints, thresholds, temperatures, ablation winners) on `validation` only. `test` is evaluated exactly once per final model at Stage 5. Pages never cross splits; assert this at data-load time by checking `page_id → split` is a function.

---

## 3. Input rendering (`render_v2`) — deterministic, shared across all arms

Rendering is a pure function of `(page_png_sha256, section row, render_v2 constants)`. Render once into a cache directory; key each artifact by `sha256(page_sha ∥ section_id ∥ "render_v2")`; store the artifact hash in every training manifest.

### 3.1 Per-kind payloads

| Section kind | Text payload | Image payload |
|---|---|---|
| `headed_text`, `document_preamble`, `deepseek_paragraph` | the section's `markdown` field verbatim | none (Arm A/Arm R); region crops (Arm B and ablation E-R5 only) |
| `visual_bundle`, image run | caption text (see 3.3) | one crop per member box with `member_raw_type == "image"` |
| `visual_bundle`, table run | DeepSeek table markdown (from the section `markdown` field) + caption text | one crop per member box with `member_raw_type == "table"` |

Caption/`figure_title` member boxes are **not** cropped separately (their text is supplied; their pixels also appear inside member crops only if geometrically contained — do not engineer this either way).

### 3.2 Crop algorithm (exact)

For each member box `[x0,y0,x1,y1]` in `member_bboxes_norm_1000`, on the full-resolution page PNG of size `(W,H)` pixels:

1. Pixel box: `px0 = round_half_even(x0/1000*W)`, similarly `py0, px1, py1` with H for y.
2. Pad by 8 px: `px0 = max(0, px0-8)`, `py0 = max(0, py0-8)`, `px1 = min(W, px1+8)`, `py1 = min(H, py1+8)`.
3. Crop with PIL `Image.crop((px0,py0,px1,py1))`; save PNG, no resampling at this stage (native resolution preserved; each model's processor handles resizing).
4. Multi-member sections: emit crops in ascending `member_reading_orders`; feed them to the model as multiple images in that order.
5. Visual-token budget (Arm A and Arm R): after processing, if the section's total visual tokens exceed **2,048**, downscale all its crops by a common factor `f = sqrt(2048 / measured_tokens)` (PIL LANCZOS), re-process, repeat once at most; log `downscaled=true` and the factor. Never drop a crop. **Visual-token measurement (exact):** `visual_tokens = count of positions in input_ids equal to the image-placeholder token id` (read the id from the model config / tokenizer special tokens, log it); at Stage 0, cross-check on 10 samples that this equals `sum over images of prod(image_grid_thw_i) / merge_size²` from the processor metadata and abort on mismatch. Additionally log, over the full dataset, the fraction of samples whose visual tokens exceed **1,280** (the budget the Qwen3-VL family validated for single documents); if p95 exceeds 1,280, run the E-R8 cap-1280 variant before trusting multi-crop results.

### 3.3 Text assembly (exact strings)

- Text sections: `doc_text = markdown` (verbatim; no cleaning, no truncation below 6,144 tokens). **Truncation procedure (never truncate the serialized chat template — that can destroy assistant markers or image placeholders):** tokenize `doc_text` *alone* (`add_special_tokens=False`); if its length exceeds **6,144**, truncate the token list from the tail, decode back to text, and build the chat template from the truncated text; after final templating, re-assert the prompt ends at the assistant generation position (§4.2.2). Set manifest flag `truncated=true`.
- Visual bundles: `doc_text = "Caption: " + caption_text` where `caption_text` joins all linked `figure_title` member texts with `" "`; if none exists, `doc_text = "Caption: (none)"`. Table runs prepend the table markdown: `doc_text = table_markdown + "\n" + "Caption: " + ...`.
- Empty-text guard (used by B3/B4 on visual bundles): if a section's assembled text is empty after the above, use the literal string `"(no extractable text)"`.

Arm R receives the exact same `render_v2` payloads as Arm A: OCR markdown text for text sections, member crop images plus caption text for visual bundles, and table markdown plus caption text for table bundles. Arm R wraps those payloads in the official Qwen3-VL-Reranker template and usage path; do not approximate it with Arm A's prompt unless byte-equivalence of a rendered dummy prompt is verified and logged.

---

## 4. Arm A prompt (`prompt_v1`) and tokenizer contract

### 4.1 Message construction

Build with the HF chat API — do **not** hand-concatenate special tokens:

```python
messages = [
  {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
  {"role": "user",   "content": [
      {"type": "text", "text": f"<Instruct>: {INSTRUCT}\n<Query>: {query}\n<Document>: "},
      # for visual bundles: one {"type": "image", "image": crop_i} per crop, in reading order
      {"type": "text", "text": doc_text},
  ]},
]
inputs = processor.apply_chat_template(messages, add_generation_prompt=True,
                                       tokenize=True, return_dict=True, return_tensors="pt")
```

Exact constant strings (hash them into the manifest):

- `SYSTEM` = `Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".`
- `INSTRUCT` = `Determine whether this document section contains sufficient evidence to answer the query. The section content below was extracted from a document page and may be text, a figure or table image, and/or a caption.`

Text sections have no image items; the `<Document>: ` text item is followed directly by `doc_text`.

### 4.2 Mandatory tokenizer assertions (run once at startup; abort on failure)

1. `ids_yes = tokenizer("yes", add_special_tokens=False).input_ids`; assert `len(ids_yes) == 1`; same for `"no"`. Set `ID_YES = ids_yes[0]`, `ID_NO = ids_no[0]`; log both integers.
2. Render one dummy example; decode the tokenized prompt; assert it ends with the assistant-turn opener (i.e., the string ends with `<|im_start|>assistant` plus at most a newline) and contains **no** `<think>` block. (The *text* Qwen3-Reranker template contains `<think></think>`; the VL template must not.)
3. Assert the decoded prompt contains the literal substrings `<Instruct>: `, `<Query>: `, `<Document>: ` exactly once each.

### 4.3 Score extraction (identical for training and inference)

Single forward pass, no generation loop. **`logits[:, -1, :]` is forbidden** — under right-padding with variable-length prompts, index `-1` points at padding. Gather at the last non-padding position:

```python
out  = model(**inputs)
mask = inputs["attention_mask"]                       # (B, L)
last = mask.sum(dim=1) - 1                            # index of final real token per row
bidx = torch.arange(out.logits.size(0), device=out.logits.device)
z    = out.logits[bidx, last, :]                      # (B, V) next-token logits at the generation position
s    = z[:, ID_YES] - z[:, ID_NO]                     # decision logit, one scalar per pair
p    = sigmoid(s / T)                                 # T = 1.0 in training; calibrated T at final eval (§5.4)
```

The same `(bidx, last)` gather applies everywhere a "decision-position" quantity is read: B4 score extraction, B5 hidden-state features, and all §8 diagnostic hidden-state/logit captures (`hidden_states[i][bidx, last, :]`). Stage-0 assertion: for a single-example batch, the gathered position must equal the final token and the decoded suffix must be the assistant-turn opener (§4.2.2).

Loss: `F.binary_cross_entropy_with_logits(s, y)` with `y ∈ {0.,1.}`. No LM loss on any other position. Never train `lm_head` or embeddings.

---

## 5. Arm A (PRIMARY): LoRA-tuned Qwen3-VL-2B binary evidence classifier

### 5.1 Model preparation

1. Load `Qwen3VLForConditionalGeneration.from_pretrained(..., torch_dtype=torch.bfloat16, attn_implementation=ATTN)` where `ATTN = "flash_attention_2"` if the Stage-0 smoke passes with it, else `"sdpa"` (record which).
2. Enable gradient checkpointing on the language model; call `model.config.use_cache = False` for training.
3. PEFT config (exact):

```python
LoraConfig(
  r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
  task_type="CAUSAL_LM",
  target_modules=[f"{P}.layers.{i}.{m}" for i in range(28) for m in (
     "self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj",
     "mlp.gate_proj","mlp.up_proj","mlp.down_proj")]
)
```

`P` is the language-model module prefix discovered at runtime: print `sorted({n for n,_ in model.named_modules()})`, find the prefix containing `.layers.0.self_attn.q_proj` (expected `model.language_model` under current transformers; do not hardcode without checking). **PEFT name-matching caveat:** PEFT versions differ in whether `target_modules` accepts fully-qualified names, suffixes, or regexes — if exact-name matching fails to attach adapters, the builder may switch to a suffix/regex resolver, *provided the final assertion below still passes unchanged*. After wrapping: assert the trainable-parameter name set equals exactly the 196 intended modules' `lora_A/lora_B` weights and nothing else (no `lm_head`, no `embed_tokens`, nothing under the vision tower or any merger module); log `sum(p.numel() for p in model.parameters() if p.requires_grad)` (expected ≈ 17 M). The assertion is the scientific contract; the matching mechanism is an implementation detail.

### 5.2 Optimization (exact)

| Item | Value |
|---|---|
| Optimizer | `torch.optim.AdamW(trainable_params, lr=1e-4, betas=(0.9,0.999), eps=1e-8, weight_decay=0.01)` |
| Schedule | linear warmup for `warmup_steps`, then cosine decay from `1e-4` to `1e-5` at `max_steps` |
| Grad clip | global norm 1.0, applied per optimizer step after accumulation |
| Precision | bf16 weights/activations; fp32 optimizer states (AdamW default) |
| Epoch cap | 4 epochs (`max_steps`); early stop: evaluate val PR-AUC every 250 optimizer steps; stop after 3 consecutive non-improving evals; restore best checkpoint |
| Checkpoint | save LoRA adapter + optimizer state at every eval; keep `best` (val PR-AUC) and `last` only |

LR pilot sweep (Stage 2 only): repeat the default run with peak lr `5e-5` and `2e-4`; pick by val PR-AUC; ties (within 1 pt) → keep `1e-4`.

### 5.3 Batch construction (deterministic sampler — identical for Arm A, Arm B, B4-ft)

One **query-group** = the query's positive pair plus **all** its same-page negative pairs (mean group ≈ 4.1 pairs). One **optimizer step** = 8 query-groups, processed as 8 gradient-accumulation microbatches of one group each (variable pair count per microbatch).

Algorithm, per epoch `e` with training seed `s`:

1. `rng = random.Random(f"{s}:{e}")`; take the train query list sorted by `query_id`; `rng.shuffle(list)`.
2. Iterate in shuffled order; each query emits its group with pairs ordered `[positive] + negatives sorted by section_id`.
3. Consume 8 groups per optimizer step; the final short step of an epoch is executed with however many groups remain.
4. Loss normalization per optimizer step: `loss_step = Σ_groups Σ_pairs BCE(s_pair, y_pair) / (total pairs across the step's groups)`. Scale each microbatch's backward by `pairs_in_group / total_pairs_step` (compute `total_pairs_step` before the microbatch loop — group sizes are known from the index).
5. Log `(step, [pair_ids])` to `sampler_trace.jsonl`.

No class weights. No negative subsampling. No curriculum.

**Degenerate groups (zero-negative queries — real, not theoretical: a retained page with exactly one section yields a query with a positive and no negatives).** Handling (do not abort): such groups ARE included in BCE training (their positive still teaches); they are SKIPPED by any group-softmax term (undefined over one element) and by the within-group label shuffle (no-op); they are EXCLUDED from Rank@1/MRR/Recall@k denominators (trivially perfect) but INCLUDED in all pair-level metrics and in accepted-sections/query. Count them once at data load and record `n_degenerate_groups` per split in `constants.json`. Assert every group has exactly one positive; abort only on zero or multiple positives (that would be a dataset-contract violation).

### 5.4 Calibration and thresholds (val only)

1. Collect val decision logits `s_i` of the best checkpoint. Fit temperature `T*` = argmin over `T ∈ [0.05, 20]` of val NLL of `sigmoid(s/T)` using `scipy.optimize.minimize_scalar(bounded)`.
2. Threshold `t_F1` = the value among all unique val probabilities maximizing F1. Threshold `t_HR` = the largest threshold with positive-recall ≥ 0.95 on val.
3. Freeze `(T*, t_F1, t_HR)`; apply unchanged at test.

---

## 5R. Arm R: LoRA-tuned Qwen3-VL-Reranker-2B

Arm R is the task-adapted reranker-initialized VLM path. It starts from the released `Qwen/Qwen3-VL-Reranker-2B`, uses the official reranker template, consumes the same `render_v2` payloads as Arm A, and is interpreted primarily against frozen B6.

### 5R.1 Model preparation

1. Load `Qwen/Qwen3-VL-Reranker-2B` with bf16 weights. Use `ATTN = "flash_attention_2"` if the Stage-0 smoke passes with it, else `"sdpa"`; record which.
2. Enable gradient checkpointing on the language model and set `model.config.use_cache = False` during training.
3. Use the official Qwen3-VL-Reranker model-card template and usage path. Do not approximate it with Arm A's prompt unless byte-equivalence of a rendered dummy prompt is verified and logged.
4. Stage-0 assertions: rendered dummy prompt hash recorded; decoded prompt contains no `<think>` block; `"yes"` and `"no"` are single tokens under the pinned reranker tokenizer; `<Instruct>: `, `<Query>: `, and `<Document>: ` appear exactly as required by the official reranker template.

### 5R.2 Score and loss

Use a single forward pass, no generation loop. Gather logits at the padding-safe last non-padding position, using the same contract as §4.3:

```python
out  = model(**inputs)
mask = inputs["attention_mask"]
last = mask.sum(dim=1) - 1
bidx = torch.arange(out.logits.size(0), device=out.logits.device)
z    = out.logits[bidx, last, :]
s    = z[:, ID_YES] - z[:, ID_NO]
```

Primary training loss:

```python
target = torch.where(y.bool(), ID_YES, ID_NO)
loss = F.cross_entropy(z, target)
```

Scoring, ranking, calibration, and thresholds use `s = z_yes − z_no`.

Listed ablation:

```python
# R1-BCE
loss = F.binary_cross_entropy_with_logits(s, y)
```

`R1-BCE` scoring is unchanged; it only swaps the full-vocabulary CE objective for the restricted BCE objective.

### 5R.3 LoRA config

Primary Arm R LoRA:

- r=32
- alpha=32
- dropout=0.05
- bias=`"none"`
- PEFT default init
- LM-only target modules:
  - `self_attn.q_proj`
  - `self_attn.k_proj`
  - `self_attn.v_proj`
  - `mlp.gate_proj`
  - `mlp.up_proj`
  - `mlp.down_proj`
- Do not include `self_attn.o_proj` in primary Arm R.
- Freeze vision tower.
- Freeze merger.
- Freeze embeddings.
- Freeze `lm_head`.

Discover the actual language-model prefix at runtime using the same resolver policy as §5.1. After PEFT wrapping, assert the trainable parameters are exactly the intended LoRA A/B weights and nothing else; abort if any trainable parameter appears under `self_attn.o_proj`, `lm_head`, `embed_tokens`, the vision tower, or any merger module.

### 5R.4 Optimization

| Item | Value |
|---|---|
| Optimizer | AdamW |
| LR | 5e-5 primary; pilot `R1-lr1e4` at 1e-4 |
| Schedule | cosine decay to 0.1×, 3% warmup |
| Weight decay | 0.01 |
| Precision | bf16 |
| Grad clipping | 1.0 global norm |
| Gradient checkpointing | on, language model only |
| Batch/sampler | identical deterministic query-group sampler as §5.3 |
| Negatives | all same-page negatives |
| Class weighting | none |
| Epochs/stopping | max 4 epochs; evaluate validation PR-AUC every 250 optimizer steps; early stop patience 3 |
| Checkpoint | best validation PR-AUC |
| Seeds | 13, 17, 23; optional 29, 31 at confirmation |
| Calibration | validation-only temperature scaling |
| Thresholds | validation max-F1 and validation recall≥0.95 point |
| Test discipline | test touched once only, at Stage 5 after all validation decisions are frozen |

Do not include a 2e-4 Arm R pilot by default. Arm R starts from a relevance-tuned checkpoint, so the initial LR sweep is deliberately narrower than Arm A's.

---

## 6. Arm B (SECONDARY): LoRA-tuned ColQwen2 with calibrated MaxSim

### 6.1 Model and inputs

- Load `ColQwen2.from_pretrained("vidore/colqwen2-v1.0", torch_dtype=torch.bfloat16)` and `ColQwen2Processor` from `colpali-engine`. This loads the published adapter; **continue training it** (set the existing LoRA weights trainable; do not add a second adapter, do not re-initialize). Trainable set = the published adapter's modules (LM `q,k,v,o,gate,up,down` + `custom_text_proj`) + the affine calibrator + `log_tau` (§6.2). Verify by printing trainable names.
- Document side: the section's member **crops** (§3.2) for *all* kinds (text sections included — ColQwen2 cannot ingest document-side text). Process each crop with `processor.process_images`; concatenate the resulting document token embeddings along the token axis into one matrix `D ∈ R^{n_d×128}`. ColQwen2's native image budget (768 patches per image) is left at its default. **Doc-token cap:** log `n_d` per pair at render time; if a section's concatenated `n_d` exceeds **4,608** (≈6 full-budget crops), downscale all its crops by `f = sqrt(4608 / n_d)` and re-process (mirror of Arm A's §3.2.5 rule); report the affected `section_id`s and the `n_d` distribution in the Stage-0 output before any training.
- Query side: `processor.process_queries([query])` unchanged → `Q ∈ R^{n_q×128}`.

### 6.2 Scoring head and loss (exact)

```
maxsim   = Σ_{i=1..n_q} max_{j=1..n_d} (Q_i · D_j)        # raw late interaction
m        = maxsim / n_q                                    # n_q = number of query embedding vectors emitted by the processor
s_logit  = a * m + b                                       # affine calibrator; a init 1.0, b init 0.0; fp32
P        = sigmoid(s_logit)
L        = BCEWithLogits(s_logit, y)  +  λ · L_group,  λ = 0.5
L_group  = per query-group: cross_entropy( [s_logit_1..s_logit_G] / exp(log_tau), index_of_positive )
```

`log_tau` is a learnable fp32 scalar, init `0.0`. Ablation E-B3 reruns with `λ ∈ {0, 1.0}`; ablation E-B2 reruns with raw `maxsim` in place of `m` (calibrator absorbs scale).

### 6.3 Optimization

Same sampler as §5.3. AdamW; LoRA params lr `5e-5` with linear decay to 0 and 2.5% warmup; calibrator + `log_tau` in a separate param group at lr `1e-3`, no weight decay. Weight decay 0.01 on LoRA params. bf16, grad clip 1.0, 8 groups per optimizer step, epoch cap 4, eval/early-stop/checkpoint/calibration/threshold protocol identical to §5.2/§5.4 (temperature fitting still applies, on `s_logit`).

---

## 7. Baselines — exact preparation

All baselines are scored on the identical pair list and evaluated with the identical metric code (§10). Each produces a per-pair score file `scores/{baseline}/{split}.jsonl` with rows `{pair_id, score, prob}`.

**B0 — random ranking / prior probability.** `score = rng.random()` with `rng = random.Random(20260707)` seeded once, pairs processed sorted by `pair_id`; `prob = train positive rate` (constant). Intentional split personality: group metrics measure random ranking; pair-level PR-AUC equals the positive-rate floor by construction. Name it exactly this way in reports.

**B0k — kind prior.** `prob = train-split positive rate of the pair's section kind` (4 constants computed from train labels; log them). `score = prob` (ties broken by the Rank@1 tie rule, §10.1, i.e. ties count as failures — this is intended).

**B1 — zero-shot Arm A.** The §5 model with **no LoRA**, same prompt/rendering/score extraction. Fit temperature and thresholds on val exactly as §5.4.

**B2 — frozen ColQwen2 + Platt (PRIMARY BASELINE).** §6 pipeline with all model weights frozen; compute `m` for every pair; fit `(a,b)` on validation by minimizing BCE of `sigmoid(a·m+b)` (LBFGS, 500 iterations, fp64); apply to test. This is exactly Arm B with zero training steps — implement it as that code path.

**B2b — frozen ColQwen2.5 + Platt (strongest retrieval-similarity null).** Identical to B2 but with `vidore/colqwen2.5-v0.2` (pin revision; its own processor). Inference-only. Rationale: B2 uses colqwen2-v1.0 for architectural consistency with Arm B; B2b ensures the "retrieval similarity is enough" null is tested at its strongest available strength. Report both; the headline comparator remains B2 (nested with Arm B), with B2b footnoted whenever it exceeds B2.

**B3 — BM25.** `rank_bm25.BM25Okapi`, `k1=1.5, b=0.75`. Corpus = per page: its sections' assembled text (§3.3, with the empty-text guard), tokenized by lowercasing and splitting on non-alphanumerics (`re.split(r"[^a-z0-9]+")`, drop empties). Score = BM25 score of the query against each same-page section. Group metrics use raw scores; pair metrics use Platt-calibrated scores (fit on val as in B2).

**B4 — text cross-encoder (SECOND LOAD-BEARING BASELINE).** `Qwen/Qwen3-Reranker-0.6B`. **Prompt construction must be copied verbatim from the model's official HF model-card usage snippet** (the `format_instruction`-style prefix/suffix strings and message assembly) — do not approximate. Stage-0 assertions for this model: rendered dummy prompt hashed into the manifest; `<think></think>` appears **exactly once** (this text-family template contains it; Arm A's must not — §4.2); `"yes"`/`"no"` single-token under *its* tokenizer; score = yes/no logit difference gathered at the last non-padding position (§4.3 gather). Document = assembled section text only (§3.3; visual bundles therefore see caption + table markdown only — this degradation is the point of the baseline). Two variants: **B4-zs** (frozen, temperature+thresholds on val) and **B4-ft** (LoRA r=16, α=32, dropout 0.05, all-linear on its decoder layers discovered the same way as §5.1; lr 1e-4 cosine, and the *identical* sampler, loss, schedule, early-stopping, and calibration protocol as Arm A).

**B6 — Qwen3-VL-Reranker-2B zero-shot (purpose-built multimodal reranker).** The frozen `Qwen/Qwen3-VL-Reranker-2B` model (pin revision), scored with **its own official model-card template and usage code** (assert: no `<think>` block, single-token yes/no under its tokenizer, §4.3 gather), over the same `render_v2` payloads as Arm R and Arm A. Inference-only; temperature+thresholds on val. B6 is the direct frozen comparator for R1: if B6 ≈ R1, LoRA adaptation adds little and the contribution shifts to evaluation/calibration/transfer; if R1 ≫ B6, task-specific segment evidence adaptation is justified.

**Arm A Stage 0.5 diagnostic — diagnostic only.** During Stage 0.5 pipeline certification, also score frozen `Qwen/Qwen3-VL-2B-Instruct` with the Arm A prompt/rendering path on the same fixed validation slice and report PR-AUC, Rank@1, MRR, Recall@2, orientation, calibration, and deltas versus B6 and R1-step0. This diagnostic estimates the effect of Qwen's dedicated reranker training relative to the base instruct checkpoint; it is not a certification gate, must not alter the existing Stage 0.5 hard checks, and must not block Stage 1 if the hard checks already pass.

**B5 — frozen linear probe.** Run B1's forward over all pairs; store the last-layer hidden state at the decision position using the §4.3 `(bidx, last)` gather (`hidden_states[-1][bidx, last, :]`, 2048-d, fp32). Train `sklearn.linear_model.LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)` on train features; thresholds on val.

**Label-shuffle control.** Arm A default config, seed 101, with train labels permuted *within each query-group* (i.e., the positive index reassigned uniformly at random per query, `random.Random(101)`). Expected outcome ≈ B0; any material skill is a leakage alarm.

---

## 8. Instrumentation — every run must emit all of the following

Directory per run: `runs/<experiment_id>/<seed>/`. All logs are JSONL (one object per line), flushed at write.

1. `manifest.json` — experiment id; hypothesis (one sentence); git commit; pip freeze; model revisions; prompt hashes; official reranker template hash where applicable; `ID_YES/ID_NO`; Arm R LoRA target list hash where applicable; trainable-parameter assertion output where applicable; ATTN impl; all hyperparameters (explicit, no defaults-by-omission); dataset run id + file SHA-256s; render cache hashes; seeds; Slurm job id; GPU name from `nvidia-smi`; start/end UTC.
2. `train_log.jsonl` — per optimizer step: `step, epoch, loss, lr, grad_norm_preclip, pairs_in_step, wall_ms, cuda_max_mem_bytes` (`torch.cuda.max_memory_allocated()`, reset each step).
3. `eval_log.jsonl` — per eval: step + full §10 metric block on validation + `is_best`.
4. `diagnostics.jsonl` — every 250 steps *and* at every eval, computed on the **fixed diagnostic batch** (the first 64 validation pairs by ascending `pair_id`; constant across all runs and arms):
   - per LM decoder layer `i`: L2 norm of the hidden state at the decision position, batch-mean (`hidden_states[i][bidx, last, :].norm(dim=-1).mean()`), with `output_hidden_states=True`;
   - per LoRA module: `‖B@A‖_F` and `‖B@A‖_F / ‖W_base‖_F` (the effective update magnitude, per layer/projection — this is the "look into any layer" hook: a layer whose update norm is flat at 0 is not learning; a spiking one is unstable);
   - per layer-group (embeddings excluded, layers 0–6 / 7–13 / 14–20 / 21–27): gradient L2 norm on the diagnostic batch;
   - decision-margin histogram: deciles of `s` on the diagnostic batch, split by label;
   - `z_yes, z_no` batch means (drift of the two logits separately, not just their difference).
5. `sampler_trace.jsonl` — §5.3 step→pair_ids mapping.
6. `scores/{split}.jsonl` — final per-pair `{pair_id, s, prob(T*), label}` for the best checkpoint (val always; test only at Stage 5).
7. `compute.json` — §9 measurements.

Arm B logs the analogues (calibrator `a,b`, `log_tau`, per-module LoRA norms on its adapter, `maxsim`/`m` histograms by label). Baselines log manifest + scores + compute only.

---

## 9. Compute-cost measurement protocol (every arm and baseline)

Comparative cost is a first-class result. Protocol:

1. **Training cost:** GPU-hours = Slurm `sacct -j <jobid> --format=Elapsed` × GPUs (1); also report Σ`wall_ms` from `train_log`. Report per run and per final-config total (all seeds).
2. **Inference benchmark set:** a fixed list of 256 test pairs — the first 128 text-kind and first 128 visual-bundle pairs by ascending `pair_id` (constant across all methods; write `benchmark_pairs.json` once and reuse).
3. **Latency:** batch size 8, 3 warmup batches then 10 timed batches (CUDA-synchronized wall clock); report median and p95 **ms/pair**, separately for the text half and visual half, plus pairs/sec/GPU. Same protocol for every scoring method (B0/B0k/B3 run on CPU; report CPU ms/pair and note device).
4. **Memory:** `torch.cuda.max_memory_allocated()` peak during the benchmark and during one training step (per-arm).
5. **Input-size accounting:** distribution (mean/p50/p95) of total tokens per pair — text tokens and visual tokens separately — under each model's processor, over the benchmark set.
6. Report table (one row per method): train GPU-h | ms/pair text | ms/pair visual | pairs/s | peak VRAM | mean tokens/pair. Emit as `compute.json` per run and aggregate into `reports/compute_table.md`.

---

## 10. Evaluation suite (single shared implementation; identical code for every method)

### 10.1 Metric definitions (exact)

Pair-level, on `prob` (post-temperature): **PR-AUC** = `sklearn.metrics.average_precision_score`; **ROC-AUC** = `roc_auc_score`; **NLL** = mean binary log loss; **Brier** = mean squared error of `prob` vs label; **ECE** = 15 equal-mass bins over `prob`, `Σ (n_b/N)·|acc_b − conf_b|`, reported pre- and post-temperature; at each frozen threshold (`t_F1`, `t_HR`): precision, recall, F1, balanced accuracy, MCC.

Query-group, on raw `s` within each query's group: **Rank@1** = fraction of queries where the positive's score is **strictly greater** than every negative's (ties = failure); **MRR** with `rank = 1 + |{negatives with score ≥ positive}|`; **Recall@2** = fraction where `rank ≤ 2`; **accepted/query** = mean count of sections with `prob ≥ t` (both thresholds); **evidence recall@budget k** (k=1,2) = fraction of queries whose positive is among the top-k scored sections; **precision at `t_HR`**.

Always report alongside floors: pair positive rate (PR-AUC floor) and mean `1/group_size` (Rank@1 random floor).

**Per-query AP (used in paired tests):** with exactly one positive per group, define `AP_query = 1 / rank_positive` (identical to reciprocal rank under the §10.1 pessimistic rank rule) — do not implement a general AP.

**Acceptance semantics (the scorer API contract):** the gate's output per query is the *set* `{sections with prob ≥ t}` — zero, one, or several sections are all legal outcomes; nothing in evaluation or downstream code may assume top-1. Rank@1/MRR are diagnostic ranking views, not the operating mode; the operating-mode metrics are accepted-sections/query, evidence recall@budget, and precision at `t_HR`.

**Calibration by stratum:** ECE, NLL, and Brier are reported not only overall but per modality stratum (text kinds vs visual bundles) and per domain — a gate calibrated on text but miscalibrated on figures is a deployment failure that pooled ECE hides.

### 10.2 Reporting structure

Every headline table is produced twice: **original queries** and **abstracted queries** (the abstracted eval set is a drop-in replacement of the `query` string per `query_id`, produced by the separately-specified abstraction pipeline; models are never trained on it). Headline aggregates use **domain macro-averaging** (compute per-domain, average domains equally); micro (pooled) values go to the appendix. Strata reported: text vs visual bundle; domain; collapsed intent {table, plot/figure, formula, textual}; section kind. Suppress any stratum cell with <30 queries.

### 10.3 Statistics

- **CIs:** cluster bootstrap, 10,000 resamples, percentile 95% intervals on every headline metric. Exact mechanics: sample `page_id`s with replacement from the split's page list (same count as original); a page sampled k times contributes all its query-groups k times; compute the metric on the resulting multiset. Bootstrap RNG: `numpy.random.default_rng(20260707)`.
- **Paired comparisons** (R1 vs B6; R1 vs B2/B2b; R1 vs B4-ft; R1 vs A1; Arm A vs B2; Arm A vs B4-ft; Arm A vs Arm B if Arm A remains active; per-model original vs abstracted): paired cluster bootstrap on per-page metric differences + Wilcoxon signed-rank on per-query AP/Rank@1 differences.
- **Seeds:** report mean ± sd across seeds; a difference "counts" only if the paired 95% CI excludes 0 **and** |Δ| > seed sd.
- Two independent executions of the final evaluation must produce byte-identical metric JSONs. To make this achievable: **final scoring always runs with the deterministic eval backend chosen at Stage 0** (SDPA with deterministic settings if FA2 proves non-reproducible — training may still use FA2); eval batch order is ascending `pair_id`; metric JSON floats are rounded to 10 decimal places before serialization; if byte-identity still fails, diff the per-pair score files first (score drift = backend problem; identical scores with differing JSON = serialization problem).

---

## 11. Experiment registry

`experiment_id` naming: as below. Every non-listed hyperparameter inherits its arm's default (§5/§6). One seed (13) unless stated.

| ID | Base | Delta from default | Purpose / decision rule |
|---|---|---|---|
| `A1` | Arm A | none (r16 all-linear, lr 1e-4, λ=0) | Stage-2 pilot; must beat B2 val PR-AUC by >4 pts to unlock Stage 3, else run E-R1/E-R4 first |
| `A1-lr5e5`, `A1-lr2e4` | Arm A | peak lr | LR pick; ties→1e-4 |
| `R1` | Arm R | r32 official targets, full-vocabulary CE, peak lr 5e-5 | primary reranker-initialized pilot; tests adaptation over B6 |
| `R1-lr1e4` | Arm R | peak lr 1e-4 | LR pilot |
| `R1-BCE` | Arm R | restricted `BCEWithLogits(z_yes−z_no)` instead of full-vocabulary CE | objective ablation |
| `R1-r16` | Arm R | r=16, α=32, same official target family | run only if R1 overfits or r32 is unnecessarily costly |
| `R1-o_proj` | Arm R | add `self_attn.o_proj` | diagnostic only; not primary |
| `E-A2a` | Arm A | LoRA attention-only (q,k,v,o) | placement; expect ≤ all-linear |
| `E-A2c` | Arm A | + LoRA on merger modules (bf16) | vision-side adaptation, cheap arm |
| `E-A2d` | Arm A | + LoRA on last 6 vision blocks (bf16) | vision-side adaptation, deep arm |
| `E-A2r8`, `E-A2r32` | winner of A2 | r=8 / r=32 (α always 2r) | pick smallest r within 1 pt |
| `E-R1` | both arms | input = full page image, long side 1288 px, every member box outlined 6 px RGB(255,0,0); no OCR text | page-context vs extracted content; also Arm B's in-distribution input |
| `E-R3` | Arm A | visual bundles: images only, drop caption/table text | value of the text channel |
| `E-R4` | Arm A | crops + OCR text for *all* kinds | "give everything"; adopt if > +2 pts |
| `E-R5` | Arm A | text sections as region crops instead of markdown | bridges Arm A/B input asymmetry |
| `E-R6` | Arm A (eval of A1/final) | visual bundles: caption/table **text only**, images dropped | complement of E-R3; same-backbone pixel-value test — any drop vs full input isolates what the pixels contribute with backbone held constant |
| `E-R7` | Arm A | visual bundles: single **envelope crop** (bounding box of all member boxes + 8 px pad) instead of per-member crops | tests whether inter-member context beats member fidelity; envelope may swallow sibling-section content — that contamination is part of the hypothesis |
| `E-R8` | Arm A | visual-token cap 1,280 instead of 2,048 | run only if p95 visual tokens > 1,280 (§3.2.5); bounds the out-of-validated-regime risk |
| ~~`E-A9`~~ | ~~Arm A~~ | ~~initialize LoRA training from Qwen3-VL-Reranker-2B weights instead of Instruct~~ | promoted to Arm R / `R1`; no longer a late ablation |
| `E-L5` | Arm A | training loss = full-vocabulary CE on the label token (Qwen3-VL-Reranker's Eq. 4) instead of restricted BCE; scoring unchanged | resolves the one deliberate objective deviation from the cited precedent; adopt if it wins by > noise |
| `E-L4` | Arm A | + λ=0.5 group-softmax on `s` (per §6.2 form) | adopt if Rank@1 +>3 pts and ΔECE < 1 pt |
| `E-B2` | Arm B | raw maxsim instead of length-normed | keep winner by val ECE then PR-AUC |
| `E-B3-0`, `E-B3-1` | Arm B | λ = 0 / 1.0 | aux-loss weight |
| `E-B4` | Arm B | freeze everything, train affine only | must reproduce B2 (nested-model sanity) |
| `E-D1-25`, `E-D1-50` | Arm A | train on 25% / 50% of train pages (page-grouped, `random.Random(13)` selection) | learning curve; 50→100 gain ≥3 pts ⇒ training-data-limited |
| `E-Q1` | none | evaluate all trained finals + B3/B4 on abstracted queries | lexical-shortcut diagnostic |
| `SHUF` | Arm A | seed 101, within-group label shuffle | leakage control; expect ≈ B0 |

**Stages:** 0 smoke (schema/split/render smoke; load Qwen3-VL-Reranker-2B; verify official reranker template, no `<think>` block, yes/no tokens, and last-non-padding gather; run B6 frozen scoring on fixed 64 validation groups; run R1 20-step microtrain on 64 train groups; keep Arm A and Arm B smoke if still active) → 1 frozen baselines (B0, B0k, B1, B2, B2b, B3, B4-zs, B5, B6; no test) → 2 pilots (R1, R1-lr1e4, B4-ft, Arm B λ sweep, A1/A1 LR sweep only if compute allows or instruct-initialization comparison is needed) → 3 ablations (table above) → 4 confirmation (final selected configs × seeds 13/17/23 [+29/31 if ≤4 GPU-h/run], SHUF, leakage battery §13) → 5 single test pass with frozen `(T*, t_F1, t_HR)` + abstracted-set test eval + compute table.

Stage-2 decision gates: R1 must beat B6 by >3 validation PR-AUC points to support task-adaptation novelty; R1 must beat or clearly match B2b to keep the cross-encoder claim alive; R1 must beat B4-ft, especially on visual bundles and abstracted queries, to support the multimodal claim. If B6 ≈ R1, reduce Arm R ablations and frame as transfer/calibration. If R1 < B6, inspect LR/loss/template before more Arm R training.

---

## 12. Determinism and seed policy

Per run: `random.seed(S); numpy.random.seed(S); torch.manual_seed(S); torch.cuda.manual_seed_all(S)` where `S` = training seed. Dataloader order comes from §5.3's own RNG (independent of torch). `torch.backends.cudnn.benchmark = False`. Flash-attention backward is nondeterministic; this is accepted for training but **evaluation must be deterministic**: run eval with fixed batch order (ascending `pair_id`) and no dropout (`model.eval()`); byte-identical repeat required (§10.3). Log every seed in the manifest.

## 13. Leakage battery (Stage 4, mandatory, all must pass)

1. **Answer-field isolation (three parts; note the *evidence section legitimately contains the answer* — string presence is NOT leakage; consumption of the `answer` field is):**
   a. *Dataflow check:* renderers, prompt builders, dataset/collator classes, baseline scorers, and caches receive only (`query`, section fields, page image, candidate member boxes). The `answer`, `gold_coverage`, gold box, and `candidate_precision` fields must be structurally unreachable from those code paths (enforce by loading queries for training through a schema that drops those fields).
   b. *Runtime sentinel:* on an in-memory copy of the dataset, replace every `answer` value with the poison string `"__ANSWER_SENTINEL_9f3a__"`; serialize all model inputs for 500 randomly chosen pairs (seed 20260707) across every arm/baseline, explicitly including Arm R and B6; assert zero occurrences of the sentinel. This proves the field is never consumed, regardless of code structure.
   c. *Occurrence diagnostic (reported, never a halt):* count whitespace-normalized occurrences of the real answer string in query text, positive-section text, and hard-negative-section text; write to `audits/answer_occurrence.json`. This is lexical-shortcut analysis data (e.g., answer-in-negative rates), not a leakage gate.
2. **Gold-geometry exclusion:** static check that the rendering/prompt code paths consume only (`query`, page image, the *candidate section's* member boxes, section text) — never `gold_coverage`, gold box, or `candidate_precision`. Negatives are rendered by the identical rule as positives.
3. **Split integrity at load time:** every `page_id` maps to exactly one split; every pair's split equals its page's split.
4. **SHUF control** ≈ B0 (any val PR-AUC > B0 + 5 pts halts the program pending investigation).
5. **Document-disjoint sensitivity:** partition test queries by whether their source document contributes ≥1 train page; report headline metrics on both partitions; a gap > 3 pts PR-AUC is reported as a limitation (not a halt).

## 14. Execution checklist (strict order)

**First-milestone guardrail (binding on any autonomous builder):** the first assignment is scaffolding only — repository layout, config/`constants.json` loader, schema validators (§2.3), split-integrity checks, render cache, sampler, metrics, calibration/Platt/bootstrap utilities, manifests/logging, compute harness, B0/B0k/B3, model-loading smoke tests, and the Stage-0 assertion suite. **No full fine-tuning, no test-set evaluation, no expensive ablation may launch until Stage 0 passes on 64 pairs and every §2.3/§4.2/§5.1 assertion is green.** Every unresolved ambiguity must fail closed with a clear error naming this spec's section — never resolved by inference.

1. Verify combined dataset audits exist; fill every `[RECOMPUTE]`; write `constants.json`.
2. Build env; pin revisions; run §4.2 and §5R assertions; render + cache all sections (log cache manifest).
3. Stage 0 smoke: env/model/template/token check; render/token accounting; B6 frozen scoring on fixed 64 validation groups; R1 20-step microtrain; Arm A/Arm B smokes if still active. Fix anything before proceeding.
4. Stage 1 frozen baselines including B6 → `reports/baselines.md`. No test.
5. Stage 2 pilots: R1, R1-lr1e4, B4-ft, Arm B λ sweep, and A1/A1 LR sweep only if allowed by compute/comparison needs → apply gate rules.
6. Stage 3 ablations → pick final configs (val only).
7. Stage 4 multi-seed confirmation + SHUF + leakage battery.
8. Stage 5: one test pass per final model (frozen calibration), abstracted-set evaluation, compute table, stratified report with cluster-bootstrap CIs.
9. Archive `runs/`, `reports/`, `scores/`, and all manifests off scratch.
