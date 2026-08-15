# Overlap-First V1 Segment-Reranker Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a reproducible stock-versus-hard-negative-fine-tuned multimodal segment-reranker experiment over the immutable overlap-first V1 corpus.

**Architecture:** A deterministic candidate builder converts packaged DeepSeek-OCR-2 output into a frozen segment universe, then maps V1 answer anchors to candidates without modifying V1. A model adapter scores identical `(segment, original_query)` pairs before and after training-only hard-negative fine-tuning; one evaluator produces query-macro rankings and a document-clustered paired comparison.

**Tech Stack:** Python 3.11, PyTorch, Hugging Face Transformers and Hub client, Pillow, JSONL artifacts, pytest, Slurm on SOL.

## Global Constraints

- Do not begin execution until `docs/EXPERIMENT_WORKSPACES.md` names the approved checkout, branch, scratch root, and recovery authority.
- Treat `/home/lmalveau/overlap_first_document_corpus/v1` as immutable input.
- Preserve the V1 document-grouped splits exactly and require `usable_in_v1=true`.
- Do not use `infographicsvqa_holdout` for model choice, threshold fitting, early stopping, hard-negative mining, or training.
- Freeze one `candidate_revision` before stock scoring; a changed revision starts a new experiment.
- Labels are answer-anchor coverage labels, not complete-evidence judgments.
- Use only the original question and rendered segment in the primary model input.
- Mine hard negatives from `train` only; validation and test never enter optimizer updates.
- Compare stock and tuned checkpoints on byte-identical held-out manifests.
- Store datasets, crops, predictions, model weights, caches, and Slurm logs outside Git.
- Use a compute allocation for image processing, model inference, environment creation, and training; login nodes are for light inspection and submission only.

---

### Task 1: Freeze the Candidate and Eligibility Manifests

**Files:**
- Create: `src/candidates/overlap_v1_segments.py`
- Create: `scripts/reranker/build_overlap_v1_segment_pairs.py`
- Create: `tests/reranker/test_overlap_v1_segment_pairs.py`
- Modify: `src/candidates/README.md`

**Interfaces:**
- Consumes: V1 `questions.jsonl`, `pages.jsonl`, `ocr.jsonl`, image paths, and the rules in `docs/specifications/deepseek_semantic_segmentation.md`.
- Produces: `candidates.jsonl`, `relations.jsonl`, `eligible_questions.jsonl`, `excluded_questions.jsonl`, `candidate_manifest.json`, and `candidate_oracle_report.json`.
- Produces: `build_candidate_revision(v1_manifest_sha256: str, parser_config: dict[str, object]) -> str`.
- Produces: `gold_coverage(gold_boxes: list[Box], member_boxes: list[Box]) -> float`.

- [ ] **Step 1: Write failing geometry and eligibility tests**

```python
def test_gold_coverage_uses_member_union_not_envelope() -> None:
    gold = [[0, 0, 100, 100]]
    members = [[0, 0, 50, 100], [75, 0, 100, 100]]
    assert gold_coverage(gold, members) == 0.75


def test_missing_anchor_candidate_is_audited_not_deleted() -> None:
    result = map_question_to_candidates(question_fixture(), candidate_fixture([]))
    assert result.eligible is False
    assert result.exclusion_reason == "no_anchor_covering_segment"
```

- [ ] **Step 2: Run the focused tests and confirm the missing implementation failure**

Run: `python -m pytest -q tests/reranker/test_overlap_v1_segment_pairs.py --tb=short`  
Expected: FAIL because `src.candidates.overlap_v1_segments` does not exist.

- [ ] **Step 3: Implement deterministic candidate records and coverage mapping**

```python
@dataclass(frozen=True)
class SegmentCandidate:
    candidate_id: str
    canonical_page_id: str
    candidate_revision: str
    member_boxes_1000: tuple[tuple[int, int, int, int], ...]
    crop_sha256: str
    crop_path: str
    text: str | None
    candidate_type: str


@dataclass(frozen=True)
class QuerySegmentRelation:
    canonical_question_id: str
    candidate_id: str
    split: str
    label: str
    gold_coverage: float
```

Implement union-area coverage without converting disconnected members into an
enclosing rectangle. Map valid alternative answer occurrences as OR groups.
Assign `positive_anchor` at coverage `>= 0.70`, `partial_anchor` in `(0, 0.70)`,
and `non_anchor` at `0`. Exclude unresolved target conflicts and emit exactly
one exclusion row per ineligible V1 question.

- [ ] **Step 4: Add deterministic manifest and split-integrity tests**

Test that repeated builds are byte-identical, every derived question exists in
V1, no document crosses splits, holdout rows are absent from primary manifests,
and every eligible question has at least one positive candidate.

- [ ] **Step 5: Run the focused tests**

Run: `python -m pytest -q tests/reranker/test_overlap_v1_segment_pairs.py --tb=short`  
Expected: PASS.

- [ ] **Step 6: Run a small compute-allocation smoke build**

Run inside an approved compute allocation:

```bash
python scripts/reranker/build_overlap_v1_segment_pairs.py \
  --v1-root /home/lmalveau/overlap_first_document_corpus/v1 \
  --output-root "$RERANKER_RUN_ROOT/candidate_smoke" \
  --max-questions 32 \
  --verify-source-hashes
```

Expected: six derived artifacts, 32 or fewer source questions, no mutation
under the V1 root, and a candidate-oracle report with explicit exclusions.

- [ ] **Step 7: Commit the candidate stage**

```bash
git add src/candidates/overlap_v1_segments.py \
  scripts/reranker/build_overlap_v1_segment_pairs.py \
  tests/reranker/test_overlap_v1_segment_pairs.py src/candidates/README.md
git commit -m "feat: build overlap v1 reranker candidates"
```

### Task 2: Implement Shared Ranking Metrics and Paired Comparison

**Files:**
- Create: `src/training/reranker_metrics.py`
- Create: `scripts/reranker/evaluate_reranker.py`
- Create: `tests/reranker/test_reranker_metrics.py`

**Interfaces:**
- Consumes: prediction JSONL with `canonical_question_id`, `canonical_document_id`, `candidate_id`, `label`, `score`, and `rank`.
- Produces: `metrics.json`, `per_query_metrics.jsonl`, and optional `paired_delta.json`.
- Produces: `evaluate_rankings(rows: Iterable[PredictionRow]) -> RankingReport`.
- Produces: `clustered_bootstrap_delta(stock: RankingReport, tuned: RankingReport, seed: int = 1729, replicates: int = 10000) -> ConfidenceInterval`.

- [ ] **Step 1: Write failing metric tests with hand-computable rankings**

```python
def test_recall_and_mrr_with_alternative_positives() -> None:
    rows = prediction_fixture(scores=[0.9, 0.8, 0.1], positives={1})
    report = evaluate_rankings(rows)
    assert report.recall_at_1 == 0.0
    assert report.recall_at_3 == 1.0
    assert report.mrr == 0.5


def test_cluster_bootstrap_samples_documents_not_questions() -> None:
    interval = clustered_bootstrap_delta(stock_fixture(), tuned_fixture(), seed=1729, replicates=200)
    assert interval.unit == "canonical_document_id"
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `python -m pytest -q tests/reranker/test_reranker_metrics.py --tb=short`  
Expected: FAIL because the metric module does not exist.

- [ ] **Step 3: Implement deterministic query-macro metrics**

Implement Recall@1/3/5, MRR, nDCG@5, pairwise positive-over-negative
accuracy, per-source slices, and selected candidate cost. Reject prediction
files with duplicate candidate rows, missing eligible queries, non-finite
scores, mixed candidate revisions, or holdout rows in primary evaluation.

- [ ] **Step 4: Implement document-clustered paired bootstrap**

Use a fixed NumPy generator seed, resample document IDs with replacement, keep
all questions belonging to each sampled document, and compute the 2.5th and
97.5th percentiles of tuned-minus-stock Recall@1.

- [ ] **Step 5: Run focused tests**

Run: `python -m pytest -q tests/reranker/test_reranker_metrics.py --tb=short`  
Expected: PASS.

- [ ] **Step 6: Commit the evaluator**

```bash
git add src/training/reranker_metrics.py \
  scripts/reranker/evaluate_reranker.py \
  tests/reranker/test_reranker_metrics.py
git commit -m "feat: add reranker ranking evaluation"
```

### Task 3: Add Stock Qwen and Jina Scoring Adapters

**Files:**
- Create: `src/training/reranker_adapters.py`
- Create: `scripts/reranker/run_stock_reranker.py`
- Create: `configs/reranker/qwen3_vl_reranker_2b_stock.json`
- Create: `configs/reranker/jina_reranker_m0_stock.json`
- Create: `tests/reranker/test_reranker_adapters.py`
- Modify: `src/training/README.md`

**Interfaces:**
- Consumes: frozen candidate and eligible-question manifests.
- Produces: a model lock containing resolved Hub commit SHA, license metadata, preprocessing configuration, and package versions.
- Produces: prediction JSONL matching Task 2.
- Produces: `RerankerAdapter.score(query: str, segment_path: Path) -> float`.

- [ ] **Step 1: Write failing adapter-contract tests with fake processors/models**

```python
@pytest.mark.parametrize("family", ["qwen3_vl_reranker", "jina_reranker_m0"])
def test_adapter_returns_one_finite_scalar_per_pair(family: str) -> None:
    adapter = adapter_fixture(family)
    score = adapter.score("What is the total?", fixture_segment_path())
    assert math.isfinite(score)
```

Also test that model locks reject mutable branch names after resolution,
predictions never contain answers or gold boxes, and the runner refuses mixed
candidate revisions.

- [ ] **Step 2: Run tests and confirm failure**

Run: `python -m pytest -q tests/reranker/test_reranker_adapters.py --tb=short`  
Expected: FAIL because the adapter module does not exist.

- [ ] **Step 3: Implement model locking and common scoring protocol**

```python
class RerankerAdapter(Protocol):
    model_id: str
    resolved_revision: str

    def score(self, query: str, segment_path: Path) -> float:
        ...
```

Resolve each Hub model ID to its immutable commit SHA before loading. Keep the
official model-specific processor/template and score extraction inside its
adapter. The runner supplies the same raw query and same segment pixels to
both families and writes model-independent scalar scores.

- [ ] **Step 4: Implement resumable stock prediction writing**

Write one append-safe shard per process, validate each row before flush, and
merge shards only when their model lock, candidate revision, and input
manifest hashes match. Resume by exact `(question_id, candidate_id)` key rather
than line count.

- [ ] **Step 5: Run unit tests and one eight-pair GPU smoke test**

Run: `python -m pytest -q tests/reranker/test_reranker_adapters.py --tb=short`  
Expected: PASS.

Run inside an approved GPU allocation with one selected config:

```bash
python scripts/reranker/run_stock_reranker.py \
  --config configs/reranker/qwen3_vl_reranker_2b_stock.json \
  --candidate-root "$RERANKER_RUN_ROOT/candidate_smoke" \
  --output-root "$RERANKER_RUN_ROOT/qwen_stock_smoke" \
  --max-pairs 8
```

Expected: eight finite scores, one immutable model lock, and no gold fields in
the prediction input payload.

- [ ] **Step 6: Commit the stock adapters and runner**

```bash
git add src/training/reranker_adapters.py \
  scripts/reranker/run_stock_reranker.py configs/reranker \
  tests/reranker/test_reranker_adapters.py src/training/README.md
git commit -m "feat: benchmark stock multimodal rerankers"
```

### Task 4: Mine and Audit Training-Only Hard Non-Anchor Candidates

**Files:**
- Create: `src/training/hard_negative_mining.py`
- Create: `scripts/reranker/mine_hard_negatives.py`
- Create: `tests/reranker/test_hard_negative_mining.py`

**Interfaces:**
- Consumes: training relations, candidate records, accepted answer strings, and stock training predictions.
- Produces: `hard_negative_groups.jsonl`, `hard_negative_audit_sample.jsonl`, and `hard_negative_mining_report.json`.
- Produces: `mine_group(query: QueryRecord, candidates: Sequence[ScoredCandidate], hard_limit: int = 4, ordinary_limit: int = 4) -> TrainingGroup`.

- [ ] **Step 1: Write failing leakage and false-negative-guard tests**

```python
def test_miner_rejects_non_train_rows() -> None:
    with pytest.raises(ValueError, match="training split only"):
        mine_group(validation_query(), scored_candidates())


def test_answer_string_candidate_is_not_a_strong_negative() -> None:
    group = mine_group(query(answer="42"), candidates_with_text("The total is 42"))
    assert group.hard_non_anchor == ()
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `python -m pytest -q tests/reranker/test_hard_negative_mining.py --tb=short`  
Expected: FAIL because the mining module does not exist.

- [ ] **Step 3: Implement deterministic mining**

Remove positives, partial overlaps, accepted alternative overlaps, and
normalized answer-string matches before selecting up to four top-scoring
`hard_non_anchor` candidates. Select up to four ordinary candidates by a
stable SHA-256 ordering over question and candidate IDs. Record every removal
reason and never label a row simply `irrelevant`.

- [ ] **Step 4: Implement the 200-row stratified audit sampler**

Allocate the sample proportionally across source families with at least 20
rows per represented family, then stratify within family by hard-negative rank
and OCR quality flag. The audit schema contains `valid_hard_non_anchor`,
`likely_false_negative`, and `uncertain`; training consumes only the first.

- [ ] **Step 5: Run tests and a 32-query smoke mine**

Run: `python -m pytest -q tests/reranker/test_hard_negative_mining.py --tb=short`  
Expected: PASS.

Run:

```bash
python scripts/reranker/mine_hard_negatives.py \
  --candidate-root "$RERANKER_RUN_ROOT/candidate_smoke" \
  --stock-predictions "$RERANKER_RUN_ROOT/qwen_stock_smoke/predictions.jsonl" \
  --output-root "$RERANKER_RUN_ROOT/hard_negative_smoke"
```

Expected: training rows only, no positive or answer-string-matching strong
negative, deterministic group hashes, and an audit sample manifest.

- [ ] **Step 6: Commit the miner**

```bash
git add src/training/hard_negative_mining.py \
  scripts/reranker/mine_hard_negatives.py \
  tests/reranker/test_hard_negative_mining.py
git commit -m "feat: mine reranker hard non-anchor pairs"
```

### Task 5: Fine-Tune the Selected Reranker

**Files:**
- Create: `src/training/pairwise_reranker_trainer.py`
- Create: `scripts/reranker/train_segment_reranker.py`
- Create: `configs/reranker/pairwise_hard_negative_finetune.json`
- Create: `tests/reranker/test_pairwise_reranker_trainer.py`

**Interfaces:**
- Consumes: audited training groups, selected stock model lock, and validation ranking manifest.
- Produces: checkpoint directory, `training_manifest.json`, step metrics JSONL, and selected-checkpoint lock.
- Produces: `pairwise_logistic_loss(positive_scores: Tensor, negative_scores: Tensor, query_index: Tensor) -> Tensor`.

- [ ] **Step 1: Write failing loss, masking, and split tests**

```python
def test_pairwise_loss_decreases_when_positive_score_increases() -> None:
    low = pairwise_logistic_loss(tensor([0.0]), tensor([1.0]), tensor([0]))
    high = pairwise_logistic_loss(tensor([2.0]), tensor([1.0]), tensor([0]))
    assert high < low


def test_training_loader_rejects_validation_and_test_groups() -> None:
    with pytest.raises(ValueError, match="train groups only"):
        build_training_loader([validation_group()])
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `python -m pytest -q tests/reranker/test_pairwise_reranker_trainer.py --tb=short`  
Expected: FAIL because the trainer module does not exist.

- [ ] **Step 3: Implement query-balanced pairwise training**

Compute `softplus(-(positive-negative))` for each valid pair, average within
question, then average across questions. Initialize only from the selected
stock model lock. Refuse changed preprocessing, candidate revision, or score
head. Save optimizer state and deterministic seeds for resume.

- [ ] **Step 4: Implement validation-only checkpoint selection**

At each configured evaluation interval, score the complete validation
manifest and select the checkpoint with highest validation MRR; break ties by
Recall@1, then earliest optimizer step. Never inspect internal-test metrics
during training.

- [ ] **Step 5: Run unit tests and a two-step GPU overfit smoke test**

Run: `python -m pytest -q tests/reranker/test_pairwise_reranker_trainer.py --tb=short`  
Expected: PASS.

Run inside an approved GPU allocation:

```bash
python scripts/reranker/train_segment_reranker.py \
  --config configs/reranker/pairwise_hard_negative_finetune.json \
  --model-lock "$RERANKER_RUN_ROOT/qwen_stock_smoke/model_lock.json" \
  --training-groups "$RERANKER_RUN_ROOT/hard_negative_smoke/hard_negative_groups.jsonl" \
  --candidate-root "$RERANKER_RUN_ROOT/candidate_smoke" \
  --output-root "$RERANKER_RUN_ROOT/train_smoke" \
  --max-steps 2
```

Expected: finite loss, one resumable checkpoint, validation code path invoked,
and no non-train row in the optimizer loader.

- [ ] **Step 6: Commit the trainer**

```bash
git add src/training/pairwise_reranker_trainer.py \
  scripts/reranker/train_segment_reranker.py \
  configs/reranker/pairwise_hard_negative_finetune.json \
  tests/reranker/test_pairwise_reranker_trainer.py
git commit -m "feat: fine tune segment reranker on hard negatives"
```

### Task 6: Run the Matched Benchmark and Publish the Handoff

**Files:**
- Create: `scripts/reranker/compare_stock_and_tuned.py`
- Create: `tests/reranker/test_stock_tuned_comparison.py`
- Create: `reports/overlap_v1_segment_reranker/README.md`
- Modify: `experiments/README.md`
- Modify: `agent-context/CURRENT_TASK.md`
- Modify: `agent-context/modules/overlap_v1_segment_reranker.md`
- Modify: `docs/ACTIVE_PROJECTS.md`
- Modify: `docs/EXPERIMENT_WORKSPACES.md`

**Interfaces:**
- Consumes: stock and tuned predictions over the same validation and internal-test manifest hashes.
- Produces: `paired_comparison.json`, `per_query_delta.jsonl`, and a Git-safe Markdown report.

- [ ] **Step 1: Write failing matched-input tests**

```python
def test_comparison_rejects_different_candidate_revisions() -> None:
    with pytest.raises(ValueError, match="candidate revision"):
        compare_runs(stock_run(revision="a"), tuned_run(revision="b"))


def test_comparison_reports_tuned_minus_stock() -> None:
    report = compare_runs(stock_run(recall1=0.40), tuned_run(recall1=0.55))
    assert report.delta_recall_at_1 == pytest.approx(0.15)
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `python -m pytest -q tests/reranker/test_stock_tuned_comparison.py --tb=short`  
Expected: FAIL because the comparison module does not exist.

- [ ] **Step 3: Implement strict matched-run validation and comparison**

Require identical V1 hashes, eligible-question hashes, candidate revision,
prompt, preprocessing, score extraction, and evaluation hardware class. Report
Recall@1 as primary, every secondary metric, clustered 95% confidence interval,
source slices, wins/losses/unchanged counts, latency, and memory.

- [ ] **Step 4: Run the full focused test suite**

Run:

```bash
python -m pytest -q \
  tests/reranker/test_overlap_v1_segment_pairs.py \
  tests/reranker/test_reranker_metrics.py \
  tests/reranker/test_reranker_adapters.py \
  tests/reranker/test_hard_negative_mining.py \
  tests/reranker/test_pairwise_reranker_trainer.py \
  tests/reranker/test_stock_tuned_comparison.py \
  --tb=short
```

Expected: PASS.

- [ ] **Step 5: Run production baseline, mine, train, and matched evaluation on SOL**

Submit only through the approved SOL handoff. Verify each artifact hash and
candidate-oracle gate before advancing. Stop after the stock run if candidate
coverage is poor; stop before training if the 200-row negative audit is not
complete.

- [ ] **Step 6: Write the final report and update control-plane status**

The report must distinguish measured counts from planned counts, stock from
tuned results, candidate misses from reranker misses, and answer-anchor ranking
from complete-evidence selection. Record exact job IDs, commit, model lock,
candidate revision, data hashes, exclusions, hard-negative audit rate, metrics,
confidence interval, failures, and follow-up recommendation.

- [ ] **Step 7: Run documentation and diff verification**

Run:

```bash
git diff --check
python -m pytest -q tests/test_documentation_contract.py --tb=short
```

Expected: no whitespace errors and PASS.

- [ ] **Step 8: Commit the comparison and final handoff**

```bash
git add scripts/reranker/compare_stock_and_tuned.py \
  tests/reranker/test_stock_tuned_comparison.py \
  reports/overlap_v1_segment_reranker/README.md \
  experiments/README.md agent-context/CURRENT_TASK.md \
  agent-context/modules/overlap_v1_segment_reranker.md \
  docs/ACTIVE_PROJECTS.md docs/EXPERIMENT_WORKSPACES.md
git commit -m "docs: report overlap v1 reranker experiment"
```
