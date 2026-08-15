# ColQwen Verifier Experiment Log

This log tracks verifier experiments for the frozen dataset at
`/scratch/$USER/mmdocir_colqwen2_verifier_training_dataset_v1`.

Hard constraints:

- Do not modify or train on the dataset artifact in place.
- Do not run VLM/ColQwen forward passes for verifier training.
- Use frozen query/page embeddings from the validated ColQwen2 retrieval run.
- Do not present union eval as the reranking headline.
- Do not push, pull, commit, or open PRs without user direction.

## Frozen Inputs

- Dataset: `/scratch/lmalveau/mmdocir_colqwen2_verifier_training_dataset_v1`
- Retrieval cache: `/scratch/lmalveau/mmdocir_colqwen2_docscoped_runs/20260625T035758Z`
- Retrieval provenance: `vidore/colqwen2-v1.0`, `ColQwen2`, `ColQwen2Processor`
- Cache files:
  - `page_embeddings.pt`: 6817 page tensors, bf16, shape examples `(747-755, 128)`
  - `query_embeddings.pt`: 4000 query tensors, bf16, shape examples `(33-56, 128)`
- Verified mapping:
  - All verifier rows map to one query embedding and one page embedding.
  - Reconstructed `sum_i max_j(q_i @ p_j)` matches saved retrieval scores up to bf16/rounding noise.

## Implementation Notes

- Model: [colqwen_verifier_model.py](../../../archive/code/colqwen/colqwen_verifier_model.py)
- Training/eval: [train_colqwen_verifier.py](../../../archive/code/colqwen/train_colqwen_verifier.py)
- Aggregation/table: [evaluate_colqwen_verifier.py](../../../archive/code/colqwen/evaluate_colqwen_verifier.py)
- SOL wrapper: [run_colqwen_verifier_v1.sbatch](../jobs/colqwen/run_colqwen_verifier_v1.sbatch)

Strict top-5 filtering uses `candidate_source in {"colqwen_top5", "combined"}`.
This is intentional: rows tagged `combined` are still ColQwen top-5 rows when
they overlap with gold or annotation-negative sources. Literal
`candidate_source == "colqwen_top5"` drops the gold positive in every group.

## Verification

- Added verifier unit tests in [test_colqwen_verifier.py](../../../archive/tests/colqwen/test_colqwen_verifier.py).
- Latest local verification after identity-calibrator fix:
  - Focused verifier tests: `7/7` pass.
  - Syntax check: `py_compile` passes for model/train/eval scripts.
  - Full repo tests: `33/33` pass.

## Runs

### 2026-06-26 06:21 UTC - v1 Initial Run, Random MaxSim Calibrator

- Job: `57369943`
- Output: `/scratch/lmalveau/mmdocir_colqwen2_verifier_runs/v1/20260626T062103Z`
- Status: stale pre-fix run; do not use for final verdict.
- Finding:
  - Group assembly audit passed: train/val/test groups had exactly one positive and matched `query_groups.jsonl`.
  - Seed 13 learned a negative late-fusion MaxSim slope.
  - Seed 13 strict rank@1 collapsed below B0.

Seed snapshots:

| Seed | B0 Strict R@1 | B1 Strict R@1 | Full Strict R@1 | Ablated Strict R@1 | Note |
| ---: | ---: | ---: | ---: | ---: | --- |
| 13 | 0.6988 | 0.6738 | 0.5119 | 0.2524 | `g` weight `-0.149` |
| 17 | 0.6988 | 0.6762 | 0.6464 | 0.2357 | `g` weight `0.485` |

Root-cause hypothesis:

`g(maxsim_scalar)` used default random `nn.Linear(1, 1)` initialization. A
negative initial slope can invert the validated MaxSim signal, which makes the
full model spend scarce data relearning that MaxSim should be positive.

Fix shipped:

```python
nn.init.ones_(self.maxsim_calibrator.weight)
nn.init.zeros_(self.maxsim_calibrator.bias)
```

### 2026-06-26 07:48 UTC - v1 Fixed Run, Identity MaxSim Calibrator

- Job: `57379992`
- Node: `sg020`
- Output: `/scratch/lmalveau/mmdocir_colqwen2_verifier_runs/v1/20260626T074814Z`
- Status: complete.
- Seeds configured: `13 17 23`
- Label-shuffle control configured after main seeds.

Strict test mean +/- std:

| Method | Rank@1 | R@2 | R@3 | R@5 | PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0 MaxSim | 0.6988 +/- 0.0000 | 0.8726 +/- 0.0000 | 0.9452 +/- 0.0000 | 1.0000 +/- 0.0000 | 0.3166 +/- 0.0000 |
| B1 alignment | 0.6734 +/- 0.0030 | 0.8548 +/- 0.0012 | 0.9302 +/- 0.0025 | 1.0000 +/- 0.0000 | 0.2700 +/- 0.0003 |
| Full verifier | 0.7008 +/- 0.0086 | 0.8587 +/- 0.0084 | 0.9401 +/- 0.0045 | 1.0000 +/- 0.0000 | 0.3161 +/- 0.0015 |
| MaxSim ablated | 0.2460 +/- 0.0168 | 0.4663 +/- 0.0225 | 0.6786 +/- 0.0183 | 1.0000 +/- 0.0000 | 0.2156 +/- 0.0079 |

Per-seed snapshot:

| Seed | B0 Strict R@1 | B1 Strict R@1 | Full Strict R@1 | Ablated Strict R@1 | Union Full R@1 | Note |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 13 | 0.6988 | 0.6738 | 0.6964 | 0.2655 | 0.6952 | Fixed run, near B0 |
| 17 | 0.6988 | 0.6762 | 0.7107 | 0.2357 | 0.7107 | Only seed clearly above B0 |
| 23 | 0.6988 | 0.6702 | 0.6952 | 0.2369 | 0.6940 | Near B0 |

Current verdict before label-shuffle result:

- Identity initialization removed the random-sign failure mode.
- Full verifier mean is only `+0.0020` absolute over B0 rank@1, below its
  `0.0086` seed std.
- Full verifier R@2/R@3 are worse than B0, and PR-AUC is essentially tied.
- MaxSim-ablated content branch is weak, with rank@1 `0.2460 +/- 0.0168`.
- Current decision-rule verdict from the aggregator: cross-encoder not
  justified because it ties B0 or collapses under MaxSim ablation.
- Label-shuffle control completed with seed `101`.

Label-shuffle control:

| Seed | Full Strict R@1 | Ablated Strict R@1 | Note |
| ---: | ---: | ---: | --- |
| 101 | 0.6917 | 0.1607 | Full remains near B0 because identity late fusion still injects frozen MaxSim; ablated learned-content control is below chance-ish. |

Interpretation of label-shuffle:

- The full shuffled-label result is not a valid leakage alarm by itself because
  `final_logit = content + maxsim` still carries the frozen retrieval baseline.
- The MaxSim-ablated shuffled result is the cleaner check for learned content
  leakage; it does not show a suspicious high score.

### 2026-06-26 08:16 UTC - v1 Content-Only Variant

- Job: `57383130`
- Node: `sg027`
- Output: `/scratch/lmalveau/mmdocir_colqwen2_verifier_runs/v1_content_only/20260626T081632Z`
- Status: complete.
- Seeds configured: `13 17 23`
- Label-shuffle control disabled for this run.

Purpose:

Train the verifier with the same group-softmax objective but remove late-fused
`g(maxsim_scalar)` from train/eval logits. The model output depends on the
cross-attention content branch plus the alignment features, not the MaxSim
scalar.

Implementation:

- CLI flag: `scripts/train_colqwen_verifier.py --content-only`
- SOL toggle: `CONTENT_ONLY=1 sol/run_colqwen_verifier_v1.sbatch`
- Forward path: `disable_late_fusion=True`, while `ablate_maxsim=False`

Strict test mean +/- std:

| Method | Rank@1 | R@2 | R@3 | R@5 | PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0 MaxSim | 0.6988 +/- 0.0000 | 0.8726 +/- 0.0000 | 0.9452 +/- 0.0000 | 1.0000 +/- 0.0000 | 0.3166 +/- 0.0000 |
| B1 alignment | 0.6734 +/- 0.0030 | 0.8548 +/- 0.0012 | 0.9302 +/- 0.0025 | 1.0000 +/- 0.0000 | 0.2700 +/- 0.0003 |
| Content-only verifier | 0.5460 +/- 0.0068 | 0.7635 +/- 0.0147 | 0.8925 +/- 0.0054 | 1.0000 +/- 0.0000 | 0.2464 +/- 0.0014 |
| MaxSim ablated | 0.2579 +/- 0.0097 | 0.4813 +/- 0.0153 | 0.6897 +/- 0.0131 | 1.0000 +/- 0.0000 | 0.2088 +/- 0.0016 |

Provisional seed snapshot:

| Seed | B0 Strict R@1 | B1 Strict R@1 | Content-Only Strict R@1 | Ablated Strict R@1 | Note |
| ---: | ---: | ---: | ---: | ---: | --- |
| 13 | 0.6988 | 0.6738 | 0.5536 | 0.2512 |  |
| 17 | 0.6988 | 0.6762 | 0.5405 | 0.2536 |  |
| 23 | 0.6988 | 0.6702 | 0.5440 | 0.2690 |  |

Interpretation:

- This is the cleanest test so far of whether the cross-attention content branch
  plus alignment features can rank without the late MaxSim scalar.
- The answer for v1 is no: content-only is far below both B0 and B1.
- The frozen embeddings likely carry latent relevance signal, but this v1
  architecture is not extracting it competitively without the scalar MaxSim
  baseline.

### 2026-06-26 17:59 UTC - v1 Content-Only, No Input LayerNorm

- Job: `57438641`
- Node: `sg043`
- Output: `/scratch/lmalveau/mmdocir_colqwen2_verifier_runs/v1_content_only_no_input_ln/20260626T175922Z`
- Status at 2026-06-26 17:59 UTC: running.
- Seeds configured: `13 17 23`
- Label-shuffle control disabled for this run.

Purpose:

Test whether `LayerNorm(Q)` and `LayerNorm(P)` before cross-attention flattened
ColQwen geometry that carries relevance signal. This keeps the content-only
setup from the prior run but bypasses input LayerNorm on Q/P.

Implementation:

- CLI flags: `--content-only --no-input-layernorm`
- SOL toggles: `CONTENT_ONLY=1 NO_INPUT_LAYERNORM=1`
- Forward path:
  - attention queries/keys/values use raw frozen Q/P embeddings
  - query baseline pooling uses raw Q
  - residual/pool LayerNorms remain in place
  - late-fused `g(maxsim_scalar)` remains disabled

Verification before submit:

- Focused verifier tests: `9/9` pass.
- Syntax check: model/train/eval scripts compile.
- SOL wrapper `bash -n` passes.
- Full repo tests: `35/35` pass.

Pending:

- Report strict rank@1 for all three seeds and compare to content-only with
  input LayerNorm (`0.5460 +/- 0.0068`) and B1 (`0.6734 +/- 0.0030`).

## Open Questions

- Does the fixed full verifier beat B0 by more than seed std over all seeds?
- Does the fixed full verifier beat B1 by more than seed std?
- Does MaxSim ablation retain meaningful edge, or does the model collapse to
  chance-like behavior without MaxSim-derived inputs?
- If v1 ties B0/B1, should v2 model the full query-token x page-patch
  similarity maps more directly: top-k patch scores, score entropy, patch
  overlap across query tokens, and patch coordinates/layout buckets?

## Visualization Tools

### Static Strict-Failure Casebook

- Script: `scripts/build_colqwen_failure_casebook.py`
- Test: `tests/test_colqwen_failure_casebook.py`
- Sample report: `outputs/casebooks/test_false_rank1_top20.html`
- Generated with:

```bash
/home/lmalveau/mamba-envs/colqwen25/bin/python scripts/build_colqwen_failure_casebook.py \
  --split test \
  --limit 20 \
  --output outputs/casebooks/test_false_rank1_top20.html
```

Purpose:

Show strict-top-5 false-rank-1 cases where ColQwen chooses a negative page at
rank 1 while the gold page remains in the top 5. Each case displays the query,
domain, document, score margin, rank-1 false-positive page image, gold page
image, and expandable top-5 candidate strip.

Current scope:

- Uses frozen split JSONL rows and page PNG paths.
- Does not modify the dataset artifact.
- Does not run ColQwen/VLM inference.
- Does not yet draw patch-level heatmaps because the cached page embeddings do
  not include explicit patch-to-image coordinates.

## Entry Template

### YYYY-MM-DD HH:MM UTC - Short Run Name

- Job:
- Output:
- Code state:
- Seeds:
- Status:
- Headline strict test metrics:
- Union metrics:
- Ablation:
- Label-shuffle:
- Verdict:
- Notes:
