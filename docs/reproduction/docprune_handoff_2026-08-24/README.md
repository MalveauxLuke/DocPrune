# DocPrune reproduction handoff (2026-08-24)

Read this first. `REFERENCE.md` contains exact pins, paths, hashes, jobs, and
failure history. Current execution authority remains `sol/CURRENT_SOL_TASK.md`.

## Goal and current truth

Reproduce the training-free DocPrune M3DocVQA results from arXiv `2604.22281`,
starting with top-4, and determine whether the local implementation follows the
paper. The full paired top-4 quality benchmark is complete: both all-kept and
DocPrune have 39/39 valid shards and 2,441/2,441 answers. Top-1 and top-2 were
not submitted.

The benchmark is usable. Its main finding is not an absolute-score failure:
local DocPrune reaches 36.76 F1 versus the paper's 37.3. The discrepancy is the
effect of pruning. The paper gains 1.0 F1 over its baseline; locally, pruning
loses 1.03 F1. A controlled stage diagnostic localizes most of the loss to CTP.

## Headline results

### Full top-4 benchmark

| Result | EM | F1 | Single-hop F1 | Multi-hop F1 |
|---|---:|---:|---:|---:|
| Paper all-kept | 31.5 | 36.3 | 43.9 | 24.9 |
| Paper DocPrune | 33.0 | 37.3 | 45.6 | 24.8 |
| Local all-kept | 32.49 | 37.79 | 46.45 | 24.88 |
| Local DocPrune | 31.87 | 36.76 | 44.83 | 24.73 |
| Paper pruning delta | +1.5 | +1.0 | +1.7 | -0.1 |
| Local paired delta | -0.61 | -1.03 | -1.62 | -0.15 |

Local paired F1 delta: `-1.0307`, 100k-bootstrap 95% CI
`[-2.1188, +0.0606]`. Single-hop (`N=1461`) is the material mismatch:
`-1.6194`, CI `[-3.0910, -0.1609]`. Multi-hop (`N=980`) nearly exactly
matches the paper and has a near-zero local delta.

Retrieved page identities are ordered-identical for 2,126/2,441 questions.
Within those questions, DocPrune still loses `1.1980` F1, CI
`[-2.3471, -0.0640]`; within the 1,334 retrieval-identical single-hop
questions it loses `1.6822`, CI `[-3.1874, -0.1784]`. Therefore retrieval
drift does not explain the controlled pruning loss.

### Controlled 245-question single-hop stage diagnostic

Every stage used the same questions, retrieved pages, and answer model.

| Stage | EM | F1 | Visual tokens retained |
|---|---:|---:|---:|
| All-kept | 37.96 | 44.86 | 100.00% |
| BTP only | 35.92 | 43.11 | 52.83% |
| BTP + QTP | 36.73 | 43.63 | 45.43% |
| Full DocPrune | 34.69 | 41.33 | 18.80% |

Paired F1 transitions:

- all-kept -> BTP: `-1.7469`, CI `[-4.6816, +1.0735]`
- BTP -> BTP+QTP: `+0.5184`, CI `[-2.1388, +3.2327]`
- BTP+QTP -> full/CTP: `-2.2980`, CI `[-4.2980, -0.6286]`
- all-kept -> full: `-3.5265`, CI `[-7.0286, -0.0939]`

BTP+QTP alone is 1.23 F1 below all-kept on this subset, but that direct
contrast has not been bootstrapped in the sealed analysis. It is much better
than full DocPrune and retains 45.4% of tokens, close to the paper's 40%
encoder retention. It omits CTP and therefore also omits the intended decoder
efficiency gain.

## Interpretation, ranked by confidence

1. **High:** The local quality loss is principally introduced after BTP+QTP,
   at CTP. Its controlled `-2.30` F1 interval excludes zero.
2. **High:** Local CTP is more aggressive than the paper: total local visual
   token drop is 81.72%, versus the paper's 74% decoder drop. Pre-decoder local
   drop is only 53.90% versus the paper's 60% encoder drop.
3. **High:** Retrieval cannot explain the within-local controlled loss because
   retrieved pages and answer model are held fixed in the diagnostic.
4. **Medium-high:** CTP attention aggregation, normalization, or threshold
   semantics is the leading implementation discrepancy. The supplement says
   to recompute last-token attention but does not specify the exact head/token
   normalization used before comparing with `tau_att`. The local implementation
   multiplies mean-head visual attention by the current visual-token count.
5. **Medium:** Paper-era corpus and ColPali/runtime differences explain some
   absolute-score differences, especially retrieval. They do not explain why
   the local all-kept/DocPrune pair diverges after identical retrieval.
6. **Low:** Ordinary sample noise explains the controlled 245-question CTP
   result; its paired interval excludes zero. The full overall interval narrowly
   includes zero, but single-hop and retrieval-identical slices do not.

The local final DocPrune score resembling the paper is partly coincidental:
the local baseline starts 1.49 F1 higher, then loses 1.03; the paper starts
lower and gains 1.0.

## Next actions

1. Implement a lightweight quality-only shard merge validator. The current
   merger repeatedly loads the 23.9 GB embeddings, expanded token map, and
   23.5 GB FAISS index; both 32 GB and 64 GB merge jobs OOMed. Authenticate
   already-successful shard validations/result hashes and immutable manifest
   identity without rematerializing the full index. Publish two atomic
   quality-only runs. Never aggregate efficiency across mixed GPU types.
2. Audit CTP against the paper/supplement, especially last-token attention head
   aggregation, scale/normalization, threshold direction, zero-based layer,
   and token-count basis before/after prior pruning. Use the controlled 245
   questions before launching another full benchmark.
3. If needed, sweep CTP semantics/thresholds on the existing controlled subset
   and compare both F1 and retained-token rate to the paper. Do not tune only
   for score or silently change the reproduction configuration.
4. Run top-1/top-2 only after the top-4 implementation question is resolved.

## Operating constraints

- Use SBATCH/HTC for GPU compute; never use a 24-hour request for short
  diagnostics. Use short recoverable shards.
- When speed matters, request the broadest compatible GPU set and maximum safe
  concurrency. Controlled diagnostics currently require A100-40GB because an
  H100 shard changed 9/32 retrieved page lists.
- Preserve completed artifacts and failed-job evidence. Avoid concurrent writes
  to one result root. Do not rerun valid shards.
- Mixed hardware is acceptable for quality only, not efficiency aggregation.
- Keep large data, indexes, caches, model weights, and generated outputs out of
  Git.

