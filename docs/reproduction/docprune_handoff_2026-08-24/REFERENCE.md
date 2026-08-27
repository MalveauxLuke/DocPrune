# Exact reference

Companion to `README.md`. This file favors exact recovery data over narrative.

## Source and environment pins

```text
working repo: /home/lmalveau/DocPrune-fix-evaluate-measurement
working branch: fix/evaluate-measurement
handoff-time HEAD: a2bd8f27d6e38009daf0898926abb6787ad7d216

production runtime:
  /home/lmalveau/DocPrune-runtime-4e2473b
  4e2473bdbbc2e4eca0e92c30d4a0633044501ccf
M3DocRAG:
  /home/lmalveau/src/m3docrag-benchmark-29e6ac2
  29e6ac2294d6b87075a1d45b8a8df175b214248a
environment: /home/lmalveau/mamba-envs/docprune-sol
PDF tools: /home/lmalveau/mamba-envs/m3docvqa-acquisition
corpus: /scratch/lmalveau/docprune/datasets/m3docvqa
HF cache: /scratch/lmalveau/hf_cache

Qwen/Qwen2-VL-7B-Instruct:
  eed13092ef92e448dd6875b2a00151bd3f7db0ac
vidore/colpali-v1.2:
  961b51745de3e9adb3468ac5c9ccca0ac626c217
vidore/colpaligemma-3b-pt-448-base:
  30ab955d073de4a91dc5a288e8c97226647e3e5a
```

Recent relevant commits:

- `a2bd8f2`: incremental stage-245 diagnostic launcher/test
- `607fc38`: replace invalid infinite diagnostic threshold with finite `1e9`
- `fdb5e91`: mixed-hardware quality shards

Important current uncommitted/user-owned files: `agent-context/CURRENT_TASK.md`,
`docs/NAVIGATION.md`, `sol/AGENTS.md`, `sol/CURRENT_SOL_TASK.md`, and
`docs/reproduction/DISCREPANCY_AUDIT.md`. Preserve them.

## Paper contract

Paper: arXiv `2604.22281`, including supplementary material.

Top-4 Table 2:

```text
all-kept Qwen: single 43.9; multi 24.9; EM 31.5; F1 36.3;
               encoder drop 0; decoder drop 0
DocPrune:      single 45.6; multi 24.8; EM 33.0; F1 37.3;
               encoder drop 0.60; decoder drop 0.74
```

Supplement Table B top-4 parameters:

```text
retrieval tau_bg = 1.0
QA tau_bg        = 0.8
tau_e            = 1
tau_q            = 0.4
tau_info/comp    = 45
tau_att          = 0.075
```

Local `configs/docprune-m3docvqa.toml` matches these values. Paper-matched
architecture: BTP/QTP prune 2x2 blocks before Qwen2-VL's spatial merger; CTP
recomputes attention only for the last token at the selected layer to remain
FlashAttention-compatible. Relevant code:

```text
src/docprune/ctp.py
src/docprune/qwen2vl/decoder.py
```

Author-undocumented reconstruction detail: exact head aggregation and
normalization of last-token attention before `tau_att`. Current code takes the
first L2 threshold crossing and scales mean-head visual attention by the
current visual-token count.

## Full top-4 benchmark artifacts

```text
root:
  /scratch/lmalveau/docprune/benchmark-4e2473b/attempt-2
control:
  /scratch/lmalveau/docprune/benchmark-4e2473b/attempt-2/control.json
analysis:
  /scratch/lmalveau/docprune/benchmark-4e2473b/attempt-2/analysis/top4-paired-quality.json
analysis SHA-256:
  2770fc16f66cc5f33d856179032335a03f8da8a0abc5e967780eff5dedd6ea14
all-kept completion array: 62068296
DocPrune completion array: 62068302
completion: 39/39 valid shards and 2441/2441 rows in each mode
```

Exact full results:

| Slice | N | All-kept F1 | DocPrune F1 | Paired delta | 95% CI |
|---|---:|---:|---:|---:|---:|
| Overall | 2441 | 37.7911 | 36.7603 | -1.0307 | [-2.1188, +0.0606] |
| Single-hop | 1461 | 46.4497 | 44.8303 | -1.6194 | [-3.0910, -0.1609] |
| Multi-hop | 980 | 24.8827 | 24.7296 | -0.1531 | [-1.7571, +1.4592] |
| Retrieval-identical | 2126 | -- | -- | -1.1980 | [-2.3471, -0.0640] |
| Retrieval-drifted | 315 | -- | -- | +0.0984 | [-3.2794, +3.4984] |
| Single-hop, retrieval-identical | 1334 | -- | -- | -1.6822 | [-3.1874, -0.1784] |

Overall EM: all-kept `32.4867`, DocPrune `31.8722`, delta `-0.6145`.
Ordered retrieval identities match on 2,126/2,441 questions; exact retrieval
scores match on 1,600/2,441.

Full DocPrune visual-token counts:

```text
original: 24,488,112 (100%)
post-BTP: 13,031,372 (53.2151% retained; 46.7849% drop)
post-QTP: 11,289,528 (46.1021% retained; 53.8979% drop)
post-CTP:  4,477,474 (18.2843% retained; 81.7157% drop)
CTP layer counts, zero-based: layer14=1491, layer15=147, layer16=797, layer17=6
```

## Stage diagnostic artifacts

```text
root:
  /scratch/lmalveau/docprune/benchmark-4e2473b/attempt-2/diagnostics/stage245-v1-incremental
analysis:
  /scratch/lmalveau/docprune/benchmark-4e2473b/attempt-2/diagnostics/stage245-v1-incremental/stage245-analysis.json
analysis SHA-256:
  d1a2479528aa775acbe8af7bb6e94718af034f2687867ae310ff2238dfb1b3af
BTP array: 62075807 (12/12 exit 0)
BTP+QTP array: 62075816 (12/12 exit 0)
runtime:
  /home/lmalveau/DocPrune-stage245-runtime-a2bd8f2
  a2bd8f27d6e38009daf0898926abb6787ad7d216
```

Population: all 245 then-available paired single-hop questions. The first 64
were reused; remaining 181 ran as <=16-question A100-40GB shards. Retrieval is
identical across all stages.

| Contrast | EM delta | F1 delta | F1 95% CI |
|---|---:|---:|---:|
| all-kept -> BTP | -2.0408 | -1.7469 | [-4.6816, +1.0735] |
| BTP -> BTP+QTP | +0.8163 | +0.5184 | [-2.1388, +3.2327] |
| BTP+QTP -> full | -2.0408 | -2.2980 | [-4.2980, -0.6286] |
| all-kept -> full | -3.2653 | -3.5265 | [-7.0286, -0.0939] |

The direct all-kept -> BTP+QTP mean is `-1.2286` F1; no direct bootstrap CI is
stored in the sealed analysis. Do not invent one.

Earlier 64-question diagnostic: all-kept `41.0625`, BTP `39.265625`, BTP+QTP
`39.5625`, full `39.625`; adjacent-stage intervals included zero. H100 BTP
shard `62072829_1` had retrieval drift on 9/32 questions and is excluded.
A100-40GB recovery `62074548_1` matched exactly.

## Packaging blocker

Canonical quality directories are not yet published. No benchmark answers are
missing or corrupted.

Failed merge jobs, all with no published output:

```text
32 GB: all-kept 62086624; DocPrune 62086625; OOM near 33.55 GB
64 GB: all-kept 62087864; DocPrune 62087740; OOM near 67.1 GB
local/lightwork parallel attempts: exit 137; no output
```

Root cause: `merge_evaluation_quality_shards` in
`src/docprune/evaluation_shards.py` calls `validate_benchmark_run` for every
already-valid shard. `_load_index_manifest` -> `IndexManifest.validate_files`
materializes all of:

```text
embeddings safetensors: 23,908,112,984 bytes
token2pageuid JSON:      2,824,013,832 bytes; expands heavily as Python objects
FAISS index:            23,540,295,725 bytes
plus finiteness-check temporaries
```

Required fix: lightweight quality-merge validation over saved successful shard
validation/result hashes plus immutable manifest identity. Keep atomic publish
semantics. Exclude efficiency aggregation. Do not solve this by requesting a
larger memory tier without changing the algorithm.

## Retrieval/corpus context

M3DocVQA open-domain retrieval correctly searches the full dev corpus, not the
gold support-document group. Local corpus: 3,366 PDFs, 44,638 pages. Searching
only support documents would leak labels and would not reproduce the paper.

Local retrieval uses one offline global exact FlatIP index over 45,977,140
token rows. The paper-era M3DocRAG setup used IVFFlat over approximately 40k
pages. Corpus regeneration, exact-versus-approximate index, and retriever
version can affect absolute scores and shared page rankings.

The M3DocRAG authors confirmed in GitHub issue 8 that paper experiments used
original ColPali v1.0, while the released recipe/local run uses v1.2; live
Wikipedia/PDF regeneration also differs. These facts may explain absolute
paper/local differences, but not a controlled loss after local retrieval is
held fixed.

## Job/history rules

- No production benchmark jobs remain active; full top-4 computation is done.
- Top-1/top-2 were never submitted.
- Failed preflights and OOM jobs evaluated/published nothing; retain records.
- Original retry/recovery history is documented in `sol/CURRENT_SOL_TASK.md`.
- Do not treat old pending/canceled job IDs as work to resume.
- Do not recompute valid shards or write concurrently to an existing root.
- Use short HTC arrays. Broaden compatible GPU types/concurrency when fidelity
  permits. A100-40GB only for the existing controlled diagnostic protocol.
- Never combine timing/efficiency metrics across mixed GPU models.

