# Overlap-First V1 MiniVGent POC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and evaluate the staged R0/R1/M0/M1 answer-anchor POC over immutable overlap-first V1, while keeping HierDoc/A1 and answer-sufficiency work outside the active program.

**Architecture:** Stage 00 freezes one query-independent DeepSeek-OCR-2 semantic-candidate universe and an audited answer-anchor view. Stage 01 produces stock and hard-negative-tuned Qwen pairwise controls. Stage 02 adds a frozen-Qwen MiniVGent decoder with exact hidden-state, ROI, tensor, loss, and checkpoint contracts. Stages 03-05 run a bounded screen, a registered three-seed comparison, and an optional locked robustness evaluation. Every stage stops for control-plane review and requires a new SOL handoff.

**Tech Stack:** Python 3.11, PyTorch 2.8.*, Torchvision 0.23.*, Transformers 4.57.3, Accelerate 1.12.*, qwen-vl-utils 0.0.14, Pillow, NumPy/SciPy, JSON/JSONL, pytest, Hugging Face immutable revisions, and Slurm on SOL.

## Global Constraints

- Binding specification: `docs/specifications/overlap_v1_evidence_localization/README.md` and its architecture/experiment files.
- Immutable input: `/home/lmalveau/overlap_first_document_corpus/v1`.
- Active arms are only R0, R1, M0, and M1. Do not implement or execute HierDoc H0/H1, A1, answer-sufficiency, page routing, synthetic multi-hop, or answer-feedback optimization.
- Preserve the V1 document-grouped splits and `usable_in_v1=true`; never mutate V1 or flatten unresolved conflicts.
- Keep `infographicsvqa_holdout` sealed until optional Stage 05 and never use it for training, threshold fitting, early stopping, or model choice.
- One frozen candidate revision feeds every arm. A candidate-definition change restarts Stage 00 and invalidates downstream outputs.
- V1 labels are answer-bearing anchors, not complete evidence. Partial/plausible/unresolved candidates are unknown unless independently audited as verified non-anchors.
- Mine negatives from `train` only. Validation, internal test, and holdout never enter optimizer updates.
- Qwen R0/R1 is mandatory. Jina is optional only under a separate registered side-arm config and cannot replace or delay Qwen.
- Keep Qwen frozen for M0/M1. No Qwen LoRA, backbone unfreezing, second vision encoder, answer generation, or box regression.
- Resolve model/source references to 40-character immutable commits; reject branches, tags, `latest`, and mutable package environments.
- Large candidates, crops, overlays, predictions, logs, caches, weights, checkpoints, and Slurm output stay outside Git. Git retains code, configs, hashes, small manifests, reports, and completion records.
- Environment creation, image work, hashing at scale, model loading, inference, mining, and training run only inside an approved allocation. Login nodes are limited to Git, light inspection, and submission.
- Existing outputs are create-once. Resume validates every upstream hash and semantic key; never truncate or overwrite a differing artifact.
- Only the stage named by `sol/CURRENT_SOL_TASK.md` may execute. Completion reports never self-authorize a successor.

## Proposed Execution Layout

These paths become binding only after owner approval of this plan and the workspace registry update:

```text
local worktree: /Users/god/Documents/COLPALI_binary_classification-minivgent-poc
branch: codex/overlap-v1-minivgent-poc
SOL checkout: ~/COLPALI_binary_classification-minivgent-poc
scratch root: /scratch/$USER/overlap_v1_minivgent_poc
HF cache: /scratch/$USER/overlap_v1_minivgent_poc/huggingface
run IDs: <stage_id>-<first 12 config SHA characters>-<Slurm job ID>
run roots: /scratch/$USER/overlap_v1_minivgent_poc/runs/<stage_id>/<registered_run_id>
Git-safe reports: reports/overlap_v1_evidence_localization/
environment name: overlap-v1-minivgent-poc
```

Stage 00 starts with a public CPU job requesting 8 cores, 64 GiB RAM, and 24 hours. GPU partitions, memory, time, accumulation, and candidate caps for Stages 01-05 are frozen in their later handoffs from measured smoke/preflight evidence; they are not guessed here.

## File Structure

```text
src/data/artifact_io.py                         canonical JSON, hashing, create-once writes
src/data/overlap_v1_records.py                  typed V1/candidate/relation/view records
src/candidates/overlap_v1_segments.py           candidate construction and geometry
src/evaluation/evidence_localization.py          ranking, slices, bootstrap, comparisons
src/models/qwen_pairwise.py                     official pairwise Qwen adapter and locks
src/training/hard_negative_mining.py             guarded training-only negative mining
src/training/qwen_pairwise.py                    R1 datasets, trainer, checkpoint/resume
src/models/minivgent/                            config, Qwen memory, ROI, encoder, decoder, model
src/training/minivgent_losses.py                 stable OR/rank/negative loss
src/training/minivgent_trainer.py                online frozen-Qwen M0/M1 training
scripts/evidence_localization/                   one CLI per stage and report builder
configs/evidence_localization/                   immutable stage/model/run configs
environments/overlap-v1-minivgent-poc.yml        starting environment lock
sol/jobs/overlap_v1_evidence_localization/       stage-specific Slurm wrappers
tests/evidence_localization/                     deterministic data/model/eval tests
tests/minivgent/                                  tensor, model, loss, checkpoint tests
reports/overlap_v1_evidence_localization/         Git-safe stage summaries and registrations
```

---

## Stage 00 — Frozen Candidates and Oracle

### Task 1: Add Artifact Primitives and Typed Records

**Files:**
- Create: `src/data/artifact_io.py`
- Create: `src/data/overlap_v1_records.py`
- Create: `tests/evidence_localization/test_artifact_io.py`
- Create: `tests/evidence_localization/test_records.py`
- Modify: `src/data/README.md`

**Interfaces:**
- Produces: `canonical_json_bytes(value: object) -> bytes`.
- Produces: `sha256_file(path: Path) -> str` and `sha256_json(value: object) -> str`.
- Produces: `write_create_once(path: Path, payload: bytes) -> str`.
- Produces immutable `CandidateRecord`, `QuestionRelation`, `EligibilityRecord`, `ExclusionRecord`, and `StageManifest` dataclasses with strict `from_dict`/`to_dict` methods.

- [ ] **Step 1: Write failing canonicalization and overwrite tests**

```python
def test_canonical_json_is_order_independent() -> None:
    assert canonical_json_bytes({"b": 2, "a": 1}) == b'{"a":1,"b":2}\n'


def test_create_once_rejects_different_bytes(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    write_create_once(path, b"first\n")
    with pytest.raises(FileExistsError, match="different contents"):
        write_create_once(path, b"second\n")
```

- [ ] **Step 2: Run the focused tests and confirm missing-module failures**

Run: `python -m pytest -q tests/evidence_localization/test_artifact_io.py tests/evidence_localization/test_records.py --tb=short`  
Expected: FAIL because the two modules do not exist.

- [ ] **Step 3: Implement strict artifact helpers**

Use sorted UTF-8 JSON with separators `(",", ":")`, `allow_nan=False`, and one trailing newline. `write_create_once` creates parents, writes through a temporary sibling, `fsync`s, atomically renames, and returns SHA-256. If the target exists, return its hash only when bytes match exactly; otherwise raise.

- [ ] **Step 4: Implement immutable record schemas**

```python
@dataclass(frozen=True)
class CandidateRecord:
    candidate_id: str
    canonical_page_id: str
    candidate_revision: str
    candidate_type: str
    member_boxes_1000: tuple[tuple[int, int, int, int], ...]
    member_reading_orders: tuple[int, ...]
    member_types: tuple[str, ...]
    text: str
    crop_path: str
    crop_sha256: str
    page_sha256: str


@dataclass(frozen=True)
class QuestionRelation:
    canonical_question_id: str
    candidate_id: str
    split: str
    relation: Literal["positive_anchor", "partial_anchor", "verified_non_anchor", "unverified_context"]
    alternative_ids: tuple[str, ...]
    gold_coverage: float
    candidate_precision: float
```

Reject unknown keys, non-40/64-character hashes where required, empty identifiers, invalid boxes/order, non-finite values, illegal split/relation values, and any gold/query field in `CandidateRecord`.

- [ ] **Step 5: Run focused tests**

Run: `python -m pytest -q tests/evidence_localization/test_artifact_io.py tests/evidence_localization/test_records.py --tb=short`  
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/data/artifact_io.py src/data/overlap_v1_records.py \
  tests/evidence_localization/test_artifact_io.py \
  tests/evidence_localization/test_records.py src/data/README.md
git commit -m "feat: add evidence localization artifact contracts"
```

### Task 2: Implement Query-Independent Semantic Candidates

**Files:**
- Create: `src/candidates/overlap_v1_segments.py`
- Create: `configs/evidence_localization/stage00_candidates.json`
- Create: `tests/evidence_localization/test_candidates.py`
- Modify: `src/candidates/README.md`

**Interfaces:**
- Consumes packaged V1 `pages.jsonl`, `ocr.jsonl`, page images, and existing `scripts.document_parsing.semantic_sections` primitives.
- Produces: `build_candidate_revision(v1_manifest_sha256: str, code_sha: str, config: Mapping[str, object], schema_sha256: str) -> str`.
- Produces: `build_page_candidates(page: V1Page, ocr: OCRRecord, cfg: CandidateConfig) -> tuple[CandidateRecord, ...]`.
- Produces: `union_area(boxes: Sequence[Box]) -> float` and `intersection_area(left: Sequence[Box], right: Sequence[Box]) -> float` without envelope substitution.

- [ ] **Step 1: Write failing geometry and segmentation tests**

```python
def test_disconnected_member_union_excludes_envelope_gap() -> None:
    members = ((0, 0, 50, 100), (75, 0, 100, 100))
    assert union_area(members) == 7_500


def test_heading_section_preserves_actual_members() -> None:
    candidates = build_page_candidates(page_fixture(), heading_ocr_fixture(), CFG)
    section = next(row for row in candidates if row.candidate_type == "headed_text")
    assert section.member_types == ("title", "text", "formula")
    assert len(section.member_boxes_1000) == 3
```

- [ ] **Step 2: Run tests and confirm the implementation is missing**

Run: `python -m pytest -q tests/evidence_localization/test_candidates.py --tb=short`  
Expected: FAIL because `src.candidates.overlap_v1_segments` does not exist.

- [ ] **Step 3: Implement exact semantic candidate rules**

Reuse the reviewed DeepSeek parser. Emit heading-led `headed_text`, `document_preamble`, per-block `deepseek_paragraph` on headingless pages, and connected `visual_bundle` candidates using same-role runs, at most two intervening text blocks, upward-then-downward title search, one primary title claim, and 350-normalized-unit floating-title fallback. Preserve actual member boxes, types, order, content, and link provenance.

- [ ] **Step 4: Freeze the candidate config**

The canonical JSON config records the DeepSeek source revision `aaa02f3811945a91062062994c5c4a3f4c0af2b0`, prompt, base/image sizes, crop/eval flags, raw-type vocabulary, heading rules, visual gap `2`, floating-title distance `350`, coordinate system `0..1000`, rendering size/format, and schema version. It contains no query, answer, source-family, conflict, or gold feature.

- [ ] **Step 5: Add determinism and leakage tests**

Build the same fixture twice and assert byte-identical ordered rows/IDs. Assert candidate IDs change when candidate-defining config changes, remain stable when output root changes, and candidate JSON contains none of `question`, `answer`, `gold`, `relation`, `source_family`, or `mapping_confidence`.

- [ ] **Step 6: Run focused and existing parser tests**

Run: `python -m pytest -q tests/evidence_localization/test_candidates.py tests/test_semantic_sections.py tests/test_deepseek_grounding.py --tb=short`  
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/candidates/overlap_v1_segments.py \
  configs/evidence_localization/stage00_candidates.json \
  tests/evidence_localization/test_candidates.py src/candidates/README.md
git commit -m "feat: build frozen overlap v1 candidates"
```

### Task 3: Map Answer Anchors, Eligibility, Exclusions, and Oracle

**Files:**
- Create: `src/candidates/overlap_v1_relations.py`
- Create: `src/evaluation/candidate_oracle.py`
- Create: `scripts/evidence_localization/stage00_candidates.py`
- Create: `sol/jobs/overlap_v1_evidence_localization/stage00_candidates.sbatch`
- Create: `tests/evidence_localization/test_relations.py`
- Create: `tests/evidence_localization/test_stage00.py`

**Interfaces:**
- Produces: `gold_coverage(gold_boxes: Sequence[Box], member_boxes: Sequence[Box]) -> float`.
- Produces: `candidate_precision(gold_boxes: Sequence[Box], member_boxes: Sequence[Box]) -> float`.
- Produces: `map_question(question, candidates) -> EligibilityRecord | ExclusionRecord` plus relation rows.
- CLI emits all Stage 00 artifacts and `stage_00_completion.json`.

- [ ] **Step 1: Write failing OR-alternative and exclusion tests**

```python
def test_alternative_occurrences_are_or_equivalent() -> None:
    mapped = map_question(question_with_two_alternatives(), candidate_covering_second())
    assert mapped.eligibility.eligible is True
    assert mapped.positive_alternative_ids == ("alt-2",)


def test_partial_only_question_is_excluded_once() -> None:
    mapped = map_question(question_fixture(), partial_candidate_fixture(coverage=0.69))
    assert mapped.exclusion.reason == "partial_anchor_coverage"
    assert len(mapped.exclusions) == 1
```

- [ ] **Step 2: Confirm tests fail**

Run: `python -m pytest -q tests/evidence_localization/test_relations.py tests/evidence_localization/test_stage00.py --tb=short`  
Expected: FAIL because relation/oracle/CLI modules do not exist.

- [ ] **Step 3: Implement member-union anchor mapping**

Assign `positive_anchor` at `gold_coverage >= 0.70`, `partial_anchor` for `(0,0.70)`, and `unverified_context` at zero unless a later audit certifies a negative. Group multi-box members within each accepted answer alternative and preserve OR alternatives. Emit exactly one exclusion row per ineligible question using the binding reason vocabulary.

- [ ] **Step 4: Implement candidate-oracle reporting**

Report source/eligible/excluded counts; oracle hit rate by split, source, candidate type, single/multi-box, answer-string present/absent, conflict history, candidate-count bucket, and gold-area bucket; candidate/member count p50/p90/p95/max; partials; unmapped raw types; overlaps/duplicates; selected-area/token estimates; and representative audit keys. Never substitute planning thresholds for observed values.

- [ ] **Step 5: Implement the Stage 00 CLI and completion contract**

```bash
python scripts/evidence_localization/stage00_candidates.py \
  --v1-root /home/lmalveau/overlap_first_document_corpus/v1 \
  --config configs/evidence_localization/stage00_candidates.json \
  --output-root "$POC_RUN_ROOT/runs/stage_00/stage00-primary-v1" \
  --verify-source-hashes
```

Require a new output root, a clean Git commit lock, a verified V1 manifest, and no holdout materialization unless the command explicitly runs Stage 05. Write `candidates.jsonl`, `relations.jsonl`, `eligible_questions.jsonl`, `excluded_questions.jsonl`, manifest/oracle JSON, render hashes, Markdown summary, and completion JSON.

- [ ] **Step 6: Implement static Slurm policy tests**

The Stage 00 wrapper requests `public`, 24 hours, 8 CPUs, 64 GiB, no GPU, sets `POC_RUN_ROOT=/scratch/$USER/overlap_v1_minivgent_poc`, and refuses to run on a login node or without the exact handoff commit.

- [ ] **Step 7: Run focused tests**

Run: `python -m pytest -q tests/evidence_localization/test_relations.py tests/evidence_localization/test_stage00.py --tb=short`  
Expected: PASS.

- [ ] **Step 8: Run the authorized 32-question smoke in a compute allocation**

Run the CLI with `--max-questions 32`. Expected: create-once artifacts, no V1 writes, deterministic rerun hashes, and a human-auditable overlay sample. Do not run the production build until the smoke report is reviewed.

- [ ] **Step 9: Commit Stage 00 implementation**

```bash
git add src/candidates/overlap_v1_relations.py src/evaluation/candidate_oracle.py \
  scripts/evidence_localization/stage00_candidates.py \
  sol/jobs/overlap_v1_evidence_localization/stage00_candidates.sbatch \
  tests/evidence_localization/test_relations.py \
  tests/evidence_localization/test_stage00.py
git commit -m "feat: complete stage 00 candidate oracle"
```

**Stage boundary:** Stop after Task 3. Produce/review `stage_00_completion.json`. Do not implement, download, or execute Stage 01 until a new binding SOL handoff names the passed Stage 00 hashes.

---

## Stage 01 — Pairwise Qwen R0/R1

### Task 4: Implement Shared Ranking Metrics and Paired Statistics

**Files:**
- Create: `src/evaluation/evidence_localization.py`
- Create: `scripts/evidence_localization/evaluate_rankings.py`
- Create: `tests/evidence_localization/test_metrics.py`
- Modify: `src/evaluation/README.md`

**Interfaces:**
- Produces: `evaluate_rankings(rows: Iterable[PredictionRow]) -> RankingReport`.
- Produces: `clustered_bootstrap_delta(left, right, *, cluster_key="canonical_document_id", seed=1729, replicates=10000) -> ConfidenceInterval`.
- Emits deterministic per-query rows, aggregate metrics, slices, costs, paired deltas, and integrity failures.

- [ ] **Step 1: Write hand-computable failing metric tests**

```python
def test_alternative_positive_rank_metrics() -> None:
    report = evaluate_rankings(predictions(scores=[0.9, 0.8, 0.1], positives={1}))
    assert report.recall_at_1 == 0.0
    assert report.recall_at_3 == 1.0
    assert report.mrr == 0.5


def test_bootstrap_clusters_documents() -> None:
    ci = clustered_bootstrap_delta(left_fixture(), right_fixture(), replicates=200)
    assert ci.cluster_key == "canonical_document_id"
```

- [ ] **Step 2: Confirm failure**

Run: `python -m pytest -q tests/evidence_localization/test_metrics.py --tb=short`  
Expected: FAIL because the evaluator does not exist.

- [ ] **Step 3: Implement ranking and integrity metrics**

Implement query-macro Recall@1/3/5, MRR, nDCG@5, positive-over-verified-negative pair accuracy, oracle-normalized reporting, deterministic tie-breaking by `candidate_id`, and slices required by `metrics_and_statistics.md`. Reject duplicates, missing eligible questions/candidates, non-finite scores, mixed revisions/views, relation leakage into model payloads, and holdout rows in primary evaluation.

- [ ] **Step 4: Implement document-clustered paired bootstrap**

Use a fixed NumPy generator, resample document IDs with replacement while retaining all their questions, compute paired arm deltas, and return percentile 2.5/97.5 bounds plus seed/replicate/cluster counts.

- [ ] **Step 5: Run focused tests and commit**

Run: `python -m pytest -q tests/evidence_localization/test_metrics.py --tb=short`  
Expected: PASS.

```bash
git add src/evaluation/evidence_localization.py \
  scripts/evidence_localization/evaluate_rankings.py \
  tests/evidence_localization/test_metrics.py src/evaluation/README.md
git commit -m "feat: add shared evidence ranking metrics"
```

### Task 5: Implement the Immutable Qwen Pairwise Adapter and R0 Scoring

**Files:**
- Create: `src/models/qwen_pairwise.py`
- Create: `configs/evidence_localization/qwen_r0.json`
- Create: `scripts/evidence_localization/stage01_score_r0.py`
- Create: `tests/evidence_localization/test_qwen_pairwise.py`
- Modify: `src/models/README.md`

**Interfaces:**
- Produces: `QwenPairwiseConfig.from_json(path)`, canonical hash, and immutable model/prompt/processor/environment lock.
- Produces: `format_candidate_instruction(query: str, candidate_image: Image.Image) -> list[dict[str, object]]`.
- Produces: `score_batch(batch: PairBatch) -> Tensor[B]` using the official yes/no logit difference.

- [ ] **Step 1: Write failing prompt, revision, and score-head tests**

```python
def test_original_query_precedes_candidate_image() -> None:
    messages = format_candidate_instruction("What is the total?", fixture_image())
    content = messages[1]["content"]
    assert next(i for i, x in enumerate(content) if x.get("text") == "What is the total?") < next(i for i, x in enumerate(content) if x.get("type") == "image")


def test_mutable_revision_is_rejected() -> None:
    with pytest.raises(ValueError, match="40-character"):
        QwenPairwiseConfig(model_revision="main")
```

- [ ] **Step 2: Confirm failure**

Run: `python -m pytest -q tests/evidence_localization/test_qwen_pairwise.py --tb=short`  
Expected: FAIL because the adapter does not exist.

- [ ] **Step 3: Implement the frozen official path**

Lock `Qwen/Qwen3-VL-Reranker-2B@4bd860ac4f15ad1897a214615cccc700f8f71818`, official source `QwenLM/Qwen3-VL-Embedding@393e2978d27852b0d0230d6994f37f9c15bed73c`, left padding, BF16, SDPA, disabled KV cache, the official task instruction, original query, and exactly one candidate crop/masked-page representation frozen by Stage 00. Build the scalar logit from `lm_head[yes] - lm_head[no]` and reproduce the official wrapper within `atol=5e-3, rtol=5e-3` on identical pixels/device.

- [ ] **Step 4: Add payload-leakage and resume tests**

Assert the serialized model input contains no answers, gold boxes, relation labels, source identity, conflict state, or audit decision. Resume keys are `(model_config_hash, view_hash, question_id, candidate_id)` and reject any lock mismatch.

- [ ] **Step 5: Run unit tests, then an eight-pair GPU smoke**

Unit run: `python -m pytest -q tests/evidence_localization/test_qwen_pairwise.py --tb=short`  
Expected: PASS without a real model.

GPU smoke expected: official parity passes, eight finite scores, exact input/pixel hashes, and measured latency/memory. Do not launch full R0 before smoke review.

- [ ] **Step 6: Commit**

```bash
git add src/models/qwen_pairwise.py configs/evidence_localization/qwen_r0.json \
  scripts/evidence_localization/stage01_score_r0.py \
  tests/evidence_localization/test_qwen_pairwise.py src/models/README.md
git commit -m "feat: add qwen pairwise r0 scorer"
```

### Task 6: Mine and Audit Training-Only Hard Non-Anchors

**Files:**
- Create: `src/training/hard_negative_mining.py`
- Create: `configs/evidence_localization/hard_negative_audit.json`
- Create: `scripts/evidence_localization/stage01_mine_negatives.py`
- Create: `tests/evidence_localization/test_hard_negative_mining.py`

**Interfaces:**
- Produces: `mine_negative_groups(scores, relations, cfg) -> tuple[NegativeGroup, ...]`.
- Produces a deterministic stratified audit sample of at least 200 hard candidates.
- Consumes only Stage 00 train rows and complete R0 train scores.

- [ ] **Step 1: Write failing guard tests**

```python
def test_partial_anchor_never_becomes_negative() -> None:
    groups = mine_negative_groups(score_fixture(), relation_fixture(partial=True), CFG)
    assert all("partial" not in group.candidate_ids for group in groups)


def test_validation_rows_are_rejected() -> None:
    with pytest.raises(ValueError, match="train only"):
        mine_negative_groups(validation_scores(), validation_relations(), CFG)
```

- [ ] **Step 2: Confirm failure**

Run: `python -m pytest -q tests/evidence_localization/test_hard_negative_mining.py --tb=short`  
Expected: FAIL because the miner does not exist.

- [ ] **Step 3: Implement guarded groups**

Exclude every positive/partial alternative, unresolved context, answer-string overlap flagged by the guard, same-table/section ambiguity, and candidates failing source/page/hash joins. Keep `hard_non_anchor`, `ordinary_non_anchor`, and `unknown` distinct. Select up to four hard negatives plus ordinary negatives per query under a deterministic config; a retriever score never certifies its own negative label.

- [ ] **Step 4: Implement deterministic audit sampling and ingestion**

Stratify at least 200 candidates across source, candidate type, score/rank bucket, answer-string presence, and geometry. Audit decisions are `verified_non_anchor`, `false_negative`, or `uncertain`; only the first enters negative loss. Freeze reviewer identity, instructions, sample hash, decisions, agreement, and rates.

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest -q tests/evidence_localization/test_hard_negative_mining.py --tb=short`  
Expected: PASS.

```bash
git add src/training/hard_negative_mining.py \
  configs/evidence_localization/hard_negative_audit.json \
  scripts/evidence_localization/stage01_mine_negatives.py \
  tests/evidence_localization/test_hard_negative_mining.py
git commit -m "feat: mine audited qwen hard negatives"
```

### Task 7: Train R1 and Complete Stage 01

**Files:**
- Create: `src/training/qwen_pairwise.py`
- Create: `configs/evidence_localization/qwen_r1.json`
- Create: `scripts/evidence_localization/stage01_train_r1.py`
- Create: `scripts/evidence_localization/stage01_report.py`
- Create: `sol/jobs/overlap_v1_evidence_localization/stage01_qwen.sbatch`
- Create: `tests/evidence_localization/test_qwen_training.py`
- Create: `tests/evidence_localization/test_stage01.py`

**Interfaces:**
- Produces deterministic R1 training groups from positive OR alternatives and audited verified negatives.
- Produces checkpoint/resume locks, validation predictions, matched R0/R1 comparison, and `stage_01_completion.json`.

- [ ] **Step 1: Write failing train-only, initialization, and resume tests**

```python
def test_r1_starts_from_exact_r0_lock() -> None:
    assert build_r1_state(R0_LOCK, TRAIN_VIEW).parent_model_hash == R0_LOCK.model_hash


def test_resume_rejects_changed_candidate_revision(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="candidate_revision"):
        resume_trainer(checkpoint_fixture(tmp_path), state_with_new_candidates())
```

- [ ] **Step 2: Confirm failure**

Run: `python -m pytest -q tests/evidence_localization/test_qwen_training.py tests/evidence_localization/test_stage01.py --tb=short`  
Expected: FAIL because the trainer/report do not exist.

- [ ] **Step 3: Implement R1 dataset/trainer/checkpoint plumbing**

Train from the exact R0 model lock on original queries and the frozen candidate representation. Use only eligible train questions with at least one active positive alternative and one audited verified negative. Validation alone chooses checkpoints; ties select the earlier step. Persist optimizer, scheduler, scaler, RNG, row-order, config, view, environment, and parent-model locks outside Git.

- [ ] **Step 4: Implement the Stage 01 report**

Require complete finite R0 train/validation and R1 validation scores; the audit sample/decisions; zero non-train optimizer rows; byte-identical validation candidates/inputs; ranking/slice/cost metrics; and document-clustered R1-minus-R0 intervals. Do not create internal-test or holdout predictions.

- [ ] **Step 5: Run focused tests and authorized training/evaluation**

Run: `python -m pytest -q tests/evidence_localization/test_qwen_training.py tests/evidence_localization/test_stage01.py --tb=short`  
Expected: PASS.

Execute smoke, full R0 scoring, audited mining, R1 training, and validation scoring only under the Stage 01 handoff. A non-improving R1 remains a valid completed result; an uncertifiable negative set fails the stage.

- [ ] **Step 6: Commit**

```bash
git add src/training/qwen_pairwise.py configs/evidence_localization/qwen_r1.json \
  scripts/evidence_localization/stage01_train_r1.py \
  scripts/evidence_localization/stage01_report.py \
  sol/jobs/overlap_v1_evidence_localization/stage01_qwen.sbatch \
  tests/evidence_localization/test_qwen_training.py \
  tests/evidence_localization/test_stage01.py
git commit -m "feat: complete qwen r0 r1 stage"
```

**Stage boundary:** Stop after Task 7. Review `stage_01_completion.json`, R0/R1 locks, negative-audit evidence, and validation comparison. Do not implement or execute MiniVGent until Stage 02 is separately activated.

---

## Stage 02 — MiniVGent Implementation and Systems Preflight

### Task 8: Add Frozen MiniVGent Config and Tensor Contracts

**Files:**
- Create: `src/models/minivgent/__init__.py`
- Create: `src/models/minivgent/config.py`
- Create: `src/data/minivgent_records.py`
- Create: `configs/evidence_localization/minivgent_two_block.json`
- Create: `configs/evidence_localization/minivgent_four_block.json`
- Create: `environments/overlap-v1-minivgent-poc.yml`
- Create: `tests/minivgent/test_config_and_records.py`

**Interfaces:**
- Produces: `MiniVGentConfig.from_json(path)`, canonical hash, and immutable `CandidateTensorBatch`/`AnchorTargetBatch`.
- Rejects unknown config keys, mutable revisions, illegal taps/dimensions, and inconsistent masks.

- [ ] **Step 1: Write failing exact-config and shape tests**

```python
def test_two_block_config_is_exact() -> None:
    cfg = MiniVGentConfig.from_json(TWO_BLOCK)
    assert cfg.memory_layers == (14, 28)
    assert cfg.decoder_width == 1024
    assert cfg.decoder_blocks == 2
    assert cfg.attention_heads == 16
    assert cfg.swiglu_hidden_width == 2736
    assert cfg.ocr_max_tokens == 128
    assert cfg.image_max_tokens == 1800


def test_target_requires_positive_and_verified_negative() -> None:
    with pytest.raises(ValueError, match="verified negative"):
        AnchorTargetBatch.from_fixture(positive=True, negative=False)
```

- [ ] **Step 2: Confirm failure**

Run: `python -m pytest -q tests/minivgent/test_config_and_records.py --tb=short`  
Expected: FAIL because the modules do not exist.

- [ ] **Step 3: Implement exact immutable batches**

Use `[B,C,M,4]` member boxes, `[B,C,M]` member masks/orders, `[B,C,128]` OCR IDs/masks, `[B,C]` type/candidate masks, `[B,A,C]` alternative-positive masks, `[B,A]` alternative masks, and `[B,C]` verified-negative masks. Enforce dtypes, common axes, `[0,1000]` boxes, PAD=0, six type IDs with UNKNOWN=5, inactive padding, and training-row target validity.

- [ ] **Step 4: Freeze exact configs and environment**

Two-block taps `(14,28)`; four-block taps `(7,14,21,28)`; width 1024; 16 heads; SwiGLU 2736; dropout 0.1; BF16; SDPA; cache disabled; image cap 1800; Qwen/source revisions from Task 5. Pin Python 3.11, torch 2.8.*, torchvision 0.23.*, transformers 4.57.3, accelerate 1.12.*, qwen-vl-utils 0.0.14, scipy, Pillow, and pytest. FlashAttention is excluded from the starting environment.

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest -q tests/minivgent/test_config_and_records.py --tb=short`  
Expected: PASS.

```bash
git add src/models/minivgent src/data/minivgent_records.py \
  configs/evidence_localization/minivgent_two_block.json \
  configs/evidence_localization/minivgent_four_block.json \
  environments/overlap-v1-minivgent-poc.yml \
  tests/minivgent/test_config_and_records.py
git commit -m "feat: define minivgent tensor contracts"
```

### Task 9: Implement Qwen Score Parity and Selective Page Memory

**Files:**
- Create: `src/models/minivgent/qwen_reranker.py`
- Create: `src/models/minivgent/qwen_memory.py`
- Create: `tests/minivgent/test_qwen_reranker.py`
- Create: `tests/minivgent/test_qwen_memory.py`

**Interfaces:**
- Produces the same prompt/yes-no diagnostic as R0/R1, now using one full supplied page.
- Produces immutable `PageMemory(memories, memory_mask, final_hidden, image_token_mask, image_grid_thw)`.
- `capture_page_memory(backbone, inputs, layers) -> PageMemory` removes hooks in `finally` and matches all-hidden-state extraction.

- [ ] **Step 1: Write failing prompt/head/hook tests**

```python
def test_layer_mapping_uses_human_one_based_numbers() -> None:
    assert module_index(14) == 13
    assert module_index(28) == 27


def test_hooks_are_removed_after_forward_error() -> None:
    model = exploding_fake_qwen()
    with pytest.raises(RuntimeError):
        capture_page_memory(model, fake_inputs(), (14, 28))
    assert count_forward_hooks(model) == 0
```

- [ ] **Step 2: Confirm failure**

Run: `python -m pytest -q tests/minivgent/test_qwen_reranker.py tests/minivgent/test_qwen_memory.py --tb=short`  
Expected: FAIL because the adapters do not exist.

- [ ] **Step 3: Implement official score diagnostic**

Use the official instruction, original query before one full page image, left padding, image budget, assistant-generation prefix, and `sigmoid((lm_head[yes]-lm_head[no]) dot h_final)`. Require BF16 same-device parity within `5e-3` absolute/relative tolerance.

- [ ] **Step 4: Implement selective taps**

Layer 14 is module 13 output before final RMSNorm; layer 28 is normalized final `last_hidden_state`. Hooks capture full sequence tensors, not only image tokens. `output_hidden_states=True` exists only as a parity oracle because it retains embedding plus 28 layers.

- [ ] **Step 5: Run fake-model tests and commit**

Run: `python -m pytest -q tests/minivgent/test_qwen_reranker.py tests/minivgent/test_qwen_memory.py --tb=short`  
Expected: PASS.

```bash
git add src/models/minivgent/qwen_reranker.py \
  src/models/minivgent/qwen_memory.py \
  tests/minivgent/test_qwen_reranker.py tests/minivgent/test_qwen_memory.py
git commit -m "feat: capture frozen qwen page memory"
```

### Task 10: Implement Visual Grid, ROI Pooling, and Candidate Encoder

**Files:**
- Create: `src/models/minivgent/visual_grid.py`
- Create: `src/models/minivgent/candidate_encoder.py`
- Create: `tests/minivgent/test_visual_grid.py`
- Create: `tests/minivgent/test_candidate_encoder.py`

**Interfaces:**
- Produces row-major `VisualGrid` from `image_grid_thw` and layer-28 image tokens.
- Produces `pool_member_union(grid, boxes_1000, member_mask) -> Tensor[B,C,4,2048]` using `torchvision.ops.roi_align(output_size=(2,2))`.
- Produces `CandidateEncoder.forward(batch, visual_grid) -> Tensor[B,C,1024]`.

- [ ] **Step 1: Write failing orientation/member-union tests**

```python
def test_row_major_grid_reconstruction() -> None:
    grid = reconstruct_grid(torch.arange(12).view(1, 12, 1), thw=(1, 3, 4))
    assert grid[0, 0, 2, 3].item() == 11


def test_disconnected_members_pool_separately() -> None:
    pooled = pool_member_union(numbered_grid(), boxes=((0,0,250,250),(750,750,1000,1000)))
    assert pooled.shape == (1, 1, 4, 2048)
```

- [ ] **Step 2: Confirm failure**

Run: `python -m pytest -q tests/minivgent/test_visual_grid.py tests/minivgent/test_candidate_encoder.py --tb=short`  
Expected: FAIL because the modules do not exist.

- [ ] **Step 3: Implement exact row-major visual mapping and ROI**

Validate image-token count equals `t*h*w`, reconstruct row-major spatial grids, scale each actual member box to feature coordinates, ROI-align every active member to 2x2, pool members with a masked mean, and never pool the enclosing envelope. Cover full-page, border-clipped, thin, table, and disconnected candidates.

- [ ] **Step 4: Implement the 1024-wide candidate encoder**

Concatenate/project visual member features, masked 128-token OCR embedding/encoder output, six-entry type embedding, 14 geometry features, and reading-order/member statistics into `[B,C,1024]`. Use configured normalization/dropout and zero inactive candidates. Do not add a second vision encoder or inferred hierarchy.

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest -q tests/minivgent/test_visual_grid.py tests/minivgent/test_candidate_encoder.py --tb=short`  
Expected: PASS.

```bash
git add src/models/minivgent/visual_grid.py \
  src/models/minivgent/candidate_encoder.py \
  tests/minivgent/test_visual_grid.py tests/minivgent/test_candidate_encoder.py
git commit -m "feat: encode minivgent document candidates"
```

### Task 11: Implement M0/M1 Decoder, Loss, Full Model, and Checkpoints

**Files:**
- Create: `src/models/minivgent/decoder.py`
- Create: `src/models/minivgent/model.py`
- Create: `src/models/minivgent/checkpoint.py`
- Create: `src/training/minivgent_losses.py`
- Create: `tests/minivgent/test_decoder.py`
- Create: `tests/minivgent/test_losses.py`
- Create: `tests/minivgent/test_model.py`
- Create: `tests/minivgent/test_checkpoint.py`

**Interfaces:**
- M0 masks all off-diagonal candidate self-attention; M1 permits bidirectional candidate interaction.
- Both cross-attend to the same frozen layer memories and produce independent logits `[B,C]`.
- Produces stable `L_or`, `L_rank`, `L_neg`, and total loss.
- Checkpoints store only added weights plus immutable locks.

- [ ] **Step 1: Write failing isolation, equivariance, and loss tests**

```python
def test_m0_candidate_isolation() -> None:
    logits_a = run_m0(candidate_batch(second_value=0.0))
    logits_b = run_m0(candidate_batch(second_value=999.0))
    torch.testing.assert_close(logits_a[:, 0], logits_b[:, 0])


def test_m1_permutation_equivariance() -> None:
    logits = run_m1(BATCH)
    permuted = run_m1(permute_candidates(BATCH, PERM))
    torch.testing.assert_close(permuted, logits[:, PERM])


@pytest.mark.parametrize("value", [-80.0, 0.0, 80.0])
def test_loss_is_finite_at_extreme_logits(value: float) -> None:
    loss = minivgent_loss(torch.full((2, 5), value), TARGETS)
    assert torch.isfinite(loss.total)
```

- [ ] **Step 2: Confirm failure**

Run: `python -m pytest -q tests/minivgent/test_decoder.py tests/minivgent/test_losses.py tests/minivgent/test_model.py tests/minivgent/test_checkpoint.py --tb=short`  
Expected: FAIL because decoder/loss/model/checkpoint modules do not exist.

- [ ] **Step 3: Implement two/four-block decoders**

Each 1024-wide block cross-attends to its configured 2048-wide Qwen memory, applies candidate self-attention with the M0/M1 mask, then SwiGLU width 2736, residuals, norms, 16 heads, and dropout 0.1. Inactive candidates cannot send/receive attention or affect logits.

- [ ] **Step 4: Implement exact stable loss**

```text
L_or   = -log(1 - product_{i in P}(1 - sigmoid(z_i)))
s_pos  = 0.1 * logsumexp(z_i / 0.1 for i in P)
N_8    = eight highest-logit verified negatives, or all if fewer
L_rank = mean_{j in N_8} softplus(0.2 - s_pos + z_j)
L_neg  = mean_{j in verified negatives} BCEWithLogits(z_j, 0)
L      = L_or + 0.5 * L_rank + 1.0 * L_neg
```

Use `logsigmoid`/`log1p` identities, OR-collapse active alternatives, ignore unknowns/padding, and reject rows without positives/verified negatives.

- [ ] **Step 5: Compose/freeze/count/checkpoint**

Set all Qwen parameters `requires_grad=False`, Qwen `eval()`, and exclude Qwen from the optimizer. Assert exact complete trainable counts: two-block M0/M1 `42,251,713`; four-block `80,074,881`; and preserved decoder-core counts `4,425,472`/`18,911,584` where their scoped assertions apply. Save added weights only with model/source/environment/prompt/candidate/view/config hashes and strict restore.

- [ ] **Step 6: Run tests and commit**

Run: `python -m pytest -q tests/minivgent/test_decoder.py tests/minivgent/test_losses.py tests/minivgent/test_model.py tests/minivgent/test_checkpoint.py --tb=short`  
Expected: PASS.

```bash
git add src/models/minivgent/decoder.py src/models/minivgent/model.py \
  src/models/minivgent/checkpoint.py src/training/minivgent_losses.py \
  tests/minivgent/test_decoder.py tests/minivgent/test_losses.py \
  tests/minivgent/test_model.py tests/minivgent/test_checkpoint.py
git commit -m "feat: implement exact minivgent selector"
```

### Task 12: Add Real-Qwen Preflight and Complete Stage 02

**Files:**
- Create: `src/training/minivgent_trainer.py`
- Create: `scripts/evidence_localization/stage02_preflight.py`
- Create: `scripts/evidence_localization/stage02_report.py`
- Create: `sol/jobs/overlap_v1_evidence_localization/stage02_preflight.sbatch`
- Create: `tests/minivgent/test_trainer.py`
- Create: `tests/minivgent/test_preflight.py`
- Create: `tests/evidence_localization/test_stage02.py`

**Interfaces:**
- Online trainer performs one frozen question-conditioned page forward per question and optimizes added modules only.
- Preflight executes all thirteen mandatory systems checks and a deterministic 16-question overfit.
- Produces Stage 02 locks/profiles/audits and `stage_02_completion.json`.

- [ ] **Step 1: Write failing freeze/overfit/report tests**

```python
def test_optimizer_contains_no_qwen_parameter() -> None:
    model = minivgent_fixture()
    optimizer = build_optimizer(model)
    assert parameter_ids(optimizer).isdisjoint(parameter_ids(model.qwen))


def test_stage02_fails_if_any_named_check_is_missing() -> None:
    with pytest.raises(ValueError, match="13 checks"):
        build_stage02_completion(preflight_fixture(checks=12))
```

- [ ] **Step 2: Confirm failure**

Run: `python -m pytest -q tests/minivgent/test_trainer.py tests/minivgent/test_preflight.py tests/evidence_localization/test_stage02.py --tb=short`  
Expected: FAIL because trainer/preflight/report modules do not exist.

- [ ] **Step 3: Implement online training plumbing**

Starting defaults: AdamW, LR `1e-4`, betas `(0.9,0.95)`, weight decay `0.05`, 5% linear warmup then cosine, clip `1.0`, BF16 autocast, microbatch 1, accumulation 16, screen seed 1729, validation Recall@1 selection with earlier-step tie break. Log loss components, LR, gradient norm, Qwen/decoder time, wall time, token/candidate counts, peak memory, and failures. Assert finite loss/gradients and zero Qwen gradients after backward.

- [ ] **Step 4: Implement all mandatory preflight checks**

Run official score parity; all-hidden versus selective tap parity for 14/28; exact image token count; row-major grid; candidate-to-grid overlays; ROI fixtures; M0 isolation; M1 equivariance; zero Qwen gradients/optimizer entries; finite nonzero added gradients; exact counts; added-weight round trip; and deterministic 16-question real overfit. The overfit requires at least 90% loss reduction and Recall@1 at least 0.95, with successful restore.

- [ ] **Step 5: Profile before choosing a candidate cap**

On a deterministic 128-512-question view, report candidate/member p50/p90/p95/max, OCR/image tokens, unknown types, overlaps, BF16 peak memory, Qwen/decoder latency, and OOMs. If a cap is necessary, create a new view hash and post-cap oracle report; never silently truncate.

- [ ] **Step 6: Run tests and authorized GPU preflight**

Run: `python -m pytest -q tests/minivgent tests/evidence_localization/test_stage02.py --tb=short`  
Expected: PASS before real Qwen execution.

Then run the Stage 02 wrapper only under its handoff. Any parity/orientation/ROI/gradient/count/checkpoint/OOM failure preserves evidence and stops.

- [ ] **Step 7: Commit**

```bash
git add src/training/minivgent_trainer.py \
  scripts/evidence_localization/stage02_preflight.py \
  scripts/evidence_localization/stage02_report.py \
  sol/jobs/overlap_v1_evidence_localization/stage02_preflight.sbatch \
  tests/minivgent/test_trainer.py tests/minivgent/test_preflight.py \
  tests/evidence_localization/test_stage02.py
git commit -m "feat: complete minivgent systems preflight"
```

**Stage boundary:** Stop after Task 12. Review every named preflight check, overlay, profile, and `stage_02_completion.json`. Do not run the scientific screen until Stage 03 is activated.

---

## Stage 03 — One-Seed M0/M1 Screen

### Task 13: Run the Matched Screen and Freeze the Confirmatory Choice

**Files:**
- Create: `scripts/evidence_localization/stage03_screen.py`
- Create: `scripts/evidence_localization/stage03_report.py`
- Create: `configs/evidence_localization/stage03_screen.json`
- Create: `sol/jobs/overlap_v1_evidence_localization/stage03_screen.sbatch`
- Create: `tests/evidence_localization/test_stage03.py`

**Interfaces:**
- Produces a deterministic source/document-balanced training sample of at most 10,000 eligible questions.
- Trains two-block M0/M1 from byte-identical added-module initialization and emits matched validation predictions/costs.
- Produces an immutable architecture-decision report and `stage_03_completion.json`.

- [ ] **Step 1: Write failing match and promotion tests**

```python
def test_m0_m1_screen_inputs_are_identical() -> None:
    left, right = build_screen_runs(SCREEN_CFG)
    assert left.initial_added_weights_sha256 == right.initial_added_weights_sha256
    assert left.row_order_sha256 == right.row_order_sha256
    assert left.loss_config_sha256 == right.loss_config_sha256


def test_four_blocks_require_gain_and_resource_fit() -> None:
    assert choose_depth(screen_fixture(stable=True, gain=True, fits=False)) == 2
```

- [ ] **Step 2: Confirm failure**

Run: `python -m pytest -q tests/evidence_localization/test_stage03.py --tb=short`  
Expected: FAIL because the screen/report do not exist.

- [ ] **Step 3: Implement deterministic sample and matched runs**

Freeze sample/view/order/config hashes. Train two-block M0/M1 for seed 1729 using identical rows, order, optimizer, schedule, objective, and checkpoint rule. Compare R1/M0/M1 validation rankings, answer-string slices, candidate-count slices, costs, and integrity checks. Internal test and holdout remain unopened.

- [ ] **Step 4: Implement depth decision**

Promote four-block M1 only when two-block M1 is stable, improves validation Recall@1 or MRR over M0 without material regression in the other, and fits the measured allocation. Otherwise freeze two blocks. No six-block or 2048-wide expansion is permitted.

- [ ] **Step 5: Freeze Stage 04 registration inputs**

Record exact candidate/view/model/source/environment/prompt/processor hashes, chosen depth, loss, optimizer, schedule, threshold rule, checkpoint rule, and three seeds. The report recommends but cannot start Stage 04.

- [ ] **Step 6: Run tests, execute under handoff, and commit**

Run: `python -m pytest -q tests/evidence_localization/test_stage03.py --tb=short`  
Expected: PASS.

```bash
git add scripts/evidence_localization/stage03_screen.py \
  scripts/evidence_localization/stage03_report.py \
  configs/evidence_localization/stage03_screen.json \
  sol/jobs/overlap_v1_evidence_localization/stage03_screen.sbatch \
  tests/evidence_localization/test_stage03.py
git commit -m "feat: add matched minivgent screen"
```

**Stage boundary:** Stop after Task 13. The control plane reviews the depth/config decision and must approve the immutable Stage 04 registration before any internal-test prediction.

---

## Stage 04 — Three-Seed Confirmatory POC

### Task 14: Register, Run, and Report the Confirmatory Comparison

**Files:**
- Create: `src/evaluation/confirmatory.py`
- Create: `scripts/evidence_localization/stage04_register.py`
- Create: `scripts/evidence_localization/stage04_run.py`
- Create: `scripts/evidence_localization/stage04_report.py`
- Create: `configs/evidence_localization/stage04_confirmatory.json`
- Create: `sol/jobs/overlap_v1_evidence_localization/stage04_confirmatory.sbatch`
- Create: `tests/evidence_localization/test_confirmatory.py`
- Create: `tests/evidence_localization/test_stage04.py`

**Interfaces:**
- Produces a create-once registration hash before internal test opens.
- Trains M0/M1 with three exact seeds and evaluates first R0/R1/M0/M1 internal-test predictions on the byte-identical view.
- Produces per-seed/aggregate deltas, clustered intervals, promotion/deployment decision, and `stage_04_completion.json`.

- [ ] **Step 1: Write failing registration and one-open tests**

```python
def test_internal_test_requires_frozen_registration() -> None:
    with pytest.raises(PermissionError, match="registration"):
        open_internal_test(unregistered_run())


def test_post_open_config_change_is_rejected() -> None:
    opened = internal_test_fixture(opened=True)
    with pytest.raises(ValueError, match="new experiment"):
        opened.with_config(changed_threshold_config())
```

- [ ] **Step 2: Confirm failure**

Run: `python -m pytest -q tests/evidence_localization/test_confirmatory.py tests/evidence_localization/test_stage04.py --tb=short`  
Expected: FAIL because confirmatory modules do not exist.

- [ ] **Step 3: Implement immutable registration and three-seed execution**

Validate all predecessor hashes, write registration/config/seed/checkpoint rules create-once, train M0/M1 from matched per-seed added initialization, select checkpoints on validation, freeze checkpoints/thresholds, then record the first internal-test opening and score R0/R1/M0/M1.

- [ ] **Step 4: Implement promotion and interpretation**

Promote M1 only if M1-minus-M0 Recall@1 is at least +2.0 absolute points, positive in all three seeds, has a document-clustered 95% interval excluding zero, persists on answer-string-absent rows, is no more than 2.0 points behind R1, and every integrity check passes. Always report `delta_pairwise`, `delta_shared_memory`, and `delta_interaction`. A failed scientific hypothesis remains a valid completion; recommend the strongest simpler system.

- [ ] **Step 5: Run tests, execute under handoff, and commit**

Run: `python -m pytest -q tests/evidence_localization/test_confirmatory.py tests/evidence_localization/test_stage04.py --tb=short`  
Expected: PASS.

```bash
git add src/evaluation/confirmatory.py \
  scripts/evidence_localization/stage04_register.py \
  scripts/evidence_localization/stage04_run.py \
  scripts/evidence_localization/stage04_report.py \
  configs/evidence_localization/stage04_confirmatory.json \
  sol/jobs/overlap_v1_evidence_localization/stage04_confirmatory.sbatch \
  tests/evidence_localization/test_confirmatory.py \
  tests/evidence_localization/test_stage04.py
git commit -m "feat: add confirmatory minivgent comparison"
```

**Stage boundary:** Stop after Task 14. No post-opening change may revise the confirmatory result. Stage 05 is optional and requires an independently reviewed frozen-system handoff.

---

## Stage 05 — Locked InfographicsVQA Robustness

### Task 15: Evaluate the Frozen Selected System Once

**Files:**
- Create: `scripts/evidence_localization/stage05_holdout.py`
- Create: `scripts/evidence_localization/stage05_report.py`
- Create: `configs/evidence_localization/stage05_holdout.json`
- Create: `sol/jobs/overlap_v1_evidence_localization/stage05_holdout.sbatch`
- Create: `tests/evidence_localization/test_stage05.py`

**Interfaces:**
- Verifies exact Stage 04 model/checkpoint/config/prompt/processor/threshold hashes.
- Builds a separately hashed holdout candidate/view/oracle under frozen rules, evaluates R0/R1/M0/M1 once, and produces `stage_05_completion.json`.

- [ ] **Step 1: Write failing frozen-hash and no-selection tests**

```python
def test_holdout_rejects_checkpoint_mismatch() -> None:
    with pytest.raises(ValueError, match="Stage 04"):
        prepare_holdout(stage04_lock(), changed_checkpoint_lock())


def test_holdout_report_cannot_change_model_choice() -> None:
    report = build_holdout_report(frozen_choice="R1", metrics=metrics_where_m1_wins())
    assert report.primary_choice == "R1"
```

- [ ] **Step 2: Confirm failure**

Run: `python -m pytest -q tests/evidence_localization/test_stage05.py --tb=short`  
Expected: FAIL because holdout modules do not exist.

- [ ] **Step 3: Implement one-shot robustness evaluation**

Verify every Stage 04 decision hash, construct the holdout view with the same candidate definition while reporting its distinct revision/oracle, score frozen arms once, report rankings/slices/costs/failures, and prevent threshold fitting, prompt repair, training, depth change, or model-choice changes.

- [ ] **Step 4: Run tests, execute only if separately activated, and commit**

Run: `python -m pytest -q tests/evidence_localization/test_stage05.py --tb=short`  
Expected: PASS.

```bash
git add scripts/evidence_localization/stage05_holdout.py \
  scripts/evidence_localization/stage05_report.py \
  configs/evidence_localization/stage05_holdout.json \
  sol/jobs/overlap_v1_evidence_localization/stage05_holdout.sbatch \
  tests/evidence_localization/test_stage05.py
git commit -m "feat: add locked infographics robustness run"
```

**Program boundary:** Stage 05 ends the active answer-anchor POC. It does not activate HierDoc/A1, answer-sufficiency data, page routing, multi-hop generation, or answer-feedback training.

---

## Control-Plane and Verification Tasks

### Task 16: Close Each Stage with Reproducibility and Handoff Evidence

**Files:**
- Create as stages complete: `reports/overlap_v1_evidence_localization/stage_NN.md`
- Modify after each reviewed completion: `agent-context/CURRENT_TASK.md`
- Modify after each reviewed completion: `sol/CURRENT_SOL_TASK.md`
- Modify only when real paths exist: `docs/EXPERIMENT_WORKSPACES.md`
- Test: `tests/test_overlap_v1_evidence_localization_spec.py`
- Test: `tests/evidence_dino/test_handoff.py`

**Interfaces:**
- Consumes a passed/failed stage completion JSON and exact artifact hashes.
- Produces a Git-safe factual report and either a stopped state or a separately reviewed next-stage handoff.

- [ ] **Step 1: Before Stage 00 implementation, create the isolated worktree and registry entry**

Use `superpowers:using-git-worktrees`. Create the exact proposed local worktree/branch, push the branch, create/verify the SOL checkout at the matching commit, and record actual local/SOL paths, scratch root, environment/cache locations, recovery authority, and commit parity. If any proposed destination cannot be created, stop and return for owner approval instead of substituting another path.

- [ ] **Step 2: Write the Stage 00-only SOL handoff**

The handoff pins source/plan/spec/commit hashes, Stage 00 config, CPU request, smoke and production commands, output roots, expected artifacts, no-model boundary, pass/stop criteria, return paths, and recovery. It explicitly prohibits Stage 01 work.

- [ ] **Step 3: After each stage, verify artifacts before updating authority**

Run focused stage tests, the complete new POC test suite, documentation/link tests, `git diff --check`, manifest/hash verification, duplicate/missing/non-finite scans, and create-once/resume checks. Record exact commands/output in the stage report.

- [ ] **Step 4: Keep next-stage activation separate**

Commit/push the completed stage report and code, verify local/remote/SOL parity and clean worktrees, then ask the owner/control plane to review. Only after approval may `sol/CURRENT_SOL_TASK.md` replace the completed handoff with the next stage's exact handoff.

- [ ] **Step 5: Preserve negative and failed results**

If a gate fails, set completion status `failed` or `stopped`, retain outputs/logs/hashes, document the cause, set `next_stage_eligible=false`, and do not weaken thresholds or silently change candidate/model/config definitions.

- [ ] **Step 6: Commit each control-plane transition separately**

```bash
git add reports/overlap_v1_evidence_localization/stage_NN.md \
  agent-context/CURRENT_TASK.md sol/CURRENT_SOL_TASK.md \
  docs/EXPERIMENT_WORKSPACES.md tests
git commit -m "docs: record overlap v1 stage NN review"
```

## Plan Self-Review

- **Spec coverage:** Tasks 1-3 cover candidates, supervision, artifact contracts, Stage 00 gates, and oracle reporting. Tasks 4-7 cover mandatory R0/R1, shared metrics, audited negatives, and Stage 01. Tasks 8-12 cover every exact MiniVGent tensor, Qwen hidden-state, ROI, decoder, loss, parameter, checkpoint, trainer, and Stage 02 system gate. Tasks 13-15 cover the screen, confirmatory comparison, one internal-test opening, and optional locked holdout. Task 16 covers stage-by-stage workspace/SOL authority and recovery.
- **Active/later boundary:** No task implements HierDoc, A1, answer sufficiency, page routing, synthetic multi-hop, or answer-feedback optimization.
- **Type consistency:** Candidate/relation/view identities flow from Tasks 1-3 into all prediction, training, metric, and resume interfaces. MiniVGent batches and locks defined in Task 8 are consumed unchanged by Tasks 9-14.
- **Test boundary:** Unit/fake-model tests precede real model loading; real inference/training is restricted to the separately activated stage and approved allocation.
- **Supersession:** This plan supersedes the execution ordering in `2026-08-11-overlap-v1-segment-reranker-baseline.md` and `2026-08-14-minivgent-qwen-implementation.md`. Those files remain provenance; their substantive candidate, Qwen, MiniVGent, tensor, loss, verification, and test requirements are represented here or in the binding architecture files.
