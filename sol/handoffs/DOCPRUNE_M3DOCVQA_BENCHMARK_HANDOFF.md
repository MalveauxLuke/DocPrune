# DocPrune M3DocVQA benchmark handoff

## Authority and scope

This is the active handoff for the complete M3DocVQA benchmark. It authorizes
the processor/mapping gate, six page-specific index builds, and six immutable
evaluation cells. It authorizes no training or fine-tuning, no dataset
mutation, and no browser download. Large artifacts, model weights, HF caches,
page images, indexes, predictions, and raw profiles remain under
`/scratch/lmalveau/docprune/`.

No job has been submitted by this handoff yet. Submit only from the clean
control checkout after the local checks at the end of this document pass.

## Immutable sources

```text
control checkout: /home/lmalveau/DocPrune-benchmark
runtime checkout: /home/lmalveau/DocPrune-runtime-d5cefb3 (detached, clean)
runtime commit: d5cefb33f7ca97ce0ef2104fa5e63bd3ad8a5761
control commit: the full Task 7 handoff commit recorded below
M3DocRAG checkout: /home/lmalveau/src/m3docrag-runtime-29e6ac2
M3DocRAG commit: 29e6ac2294d6b87075a1d45b8a8df175b214248a
environment: /home/lmalveau/mamba-envs/docprune-sol
corpus: /scratch/lmalveau/docprune/datasets/m3docvqa
factory: docprune.m3docvqa_factory:build_workload
```

Set `CONTROL_COMMIT` to the exact full control-checkout commit named below;
every launcher verifies it before doing work. Every Python command runs from the detached runtime checkout with
`PYTHONPATH=$RUNTIME_DIR/src`. The M3DocRAG checkout must be clean at its exact
commit. The control checkout is used only to submit these wrappers.

## Corpus identity

The corpus is the acquired official MMQA/M3DocVQA dev material, not a Hub
dataset revision. These values are fixed by `docprune.benchmark_config` and
must appear in every final run config:

```text
integrity_sha256:                 e2581c9766157e800ba195d37905c0c25611cf4057d03e2fa32fc23e79280e39
archive_checksum_manifest_sha256: 8ff8f1dca284a16d9a0a726ea5f0dac58f7e05a2aafaa7c2d88a56dfecd46f6d
questions_sha256:                 31192a64bfc4ffc23123c1e6657a5b57dbb9515e9a37ecbbc8578cf894dd0e3b
document_ids_sha256:              2d9e09689b2d1e867c566e0e893e9b53955487202921bdd8664aaa6e4037e429
expected_question_count:          2441
expected_pdf_count:               3366
expected_page_count:              44638
is_fixture:                       false
```

The five preserved archive digests are the values in
`src/docprune/benchmark_config.py`. `attempt-3-integrity.json` is the
production integrity report. Failed acquisition/integrity attempts are never
deleted or overwritten.

## Model and generation pins

```text
Qwen/Qwen2-VL-7B-Instruct@eed13092ef92e448dd6875b2a00151bd3f7db0ac
vidore/colpali-v1.2@961b51745de3e9adb3468ac5c9ccca0ac626c217
vidore/colpaligemma-3b-pt-448-base@30ab955d073de4a91dc5a288e8c97226647e3e5a
max_new_tokens=128; do_sample=false; num_beams=1
prompt=question: $question\noutput only answer.
```

The processor contract is schema 2 with all three checks true:
`colpali_visual_grid_inferred`, `qwen_merge_groups_valid`, and
`raster_order_verified`. It is written at
`$ATTEMPT_ROOT/gate/processor-contract.json`; final run configs include its
SHA-256.

## Resources and artifacts

Every job requests one A100 80 GB, 8 CPUs, and 128 GB RAM on `public`/`public`.
Use `/scratch/lmalveau/docprune/benchmark-d5cefb3/attempt-N/` for each new
attempt; never reuse a failed attempt root.

```text
$ATTEMPT_ROOT/gate/gate.json
$ATTEMPT_ROOT/gate/processor-contract.json
$ATTEMPT_ROOT/indexes/all-kept/top{1,2,4}/all-kept/manifest.json
$ATTEMPT_ROOT/indexes/docprune/top{1,2,4}/docprune/manifest.json
$ATTEMPT_ROOT/eval/{all-kept,docprune}/top{1,2,4}/run/
```

The six index roots are distinct because BTP is page-count-specific. A
manifest from one mode/page count is never reused for another.

## Run-config construction

The gate receives a provisional `gate-top1.json` whose
`processor_contract_path` points at the not-yet-created gate contract and
which omits `processor_contract_sha256`. After the gate writes the contract,
create six final configs under `$RUN_CONFIG_ROOT` with the contract digest.
Each final JSON must contain the following fields, plus the complete nested
corpus identity shown in the Corpus section:

```json
{
  "mode": "all-kept",
  "page_count": 1,
  "runtime_commit": "d5cefb33f7ca97ce0ef2104fa5e63bd3ad8a5761",
  "m3docrag_commit": "29e6ac2294d6b87075a1d45b8a8df175b214248a",
  "qwen_model": "Qwen/Qwen2-VL-7B-Instruct",
  "qwen_revision": "eed13092ef92e448dd6875b2a00151bd3f7db0ac",
  "colpali_model": "vidore/colpali-v1.2",
  "colpali_revision": "961b51745de3e9adb3468ac5c9ccca0ac626c217",
  "colpali_backbone_model": "vidore/colpaligemma-3b-pt-448-base",
  "colpali_backbone_revision": "30ab955d073de4a91dc5a288e8c97226647e3e5a",
  "processor_contract_path": "/scratch/.../gate/processor-contract.json",
  "processor_contract_sha256": "<sha256 of that regular file>",
  "max_new_tokens": 128,
  "do_sample": false,
  "num_beams": 1,
  "prompt": "question: $question\noutput only answer.",
  "m3docrag_root": "/home/lmalveau/src/m3docrag-runtime-29e6ac2",
  "corpus": {"...": "the exact pinned identity and archive_hashes"}
}
```

Construct these JSON files on a compute allocation with
`CorpusIdentity.from_root`, the constants in `benchmark_config.py`, and the
actual gate-contract SHA. Do not substitute fixture values or
`DATASET_REVISION`.

## Ordered submission

Set the common environment from the exact values above, including all six
model resource/revision variables. The launchers are:

1. `12_docprune_m3docvqa_gate.sbatch`, with a new `GATE_ROOT`, provisional
   `RUN_CONFIG`, deterministic 144-DPI `PROBE_IMAGE`, and fixed comma-separated
   `GATE_SAMPLE_IDS`. It runs the pinned processor probe, exact span/grid/raster
   checks, all six deterministic dry-run selectors, repeated all-kept fixed
   answers, and one real DocPrune answer at pages 1/2/4. It writes
   `gate.json` with `status=passed`; every failed invariant exits nonzero.

2. `13_docprune_m3docvqa_index.sbatch`, submitted six times with
   `--dependency=afterok:$GATE_JOB`, one mode/page-specific config, and a new
   root for each index path. It validates schema 4, source/corpus/model/runtime
   identity, artifact checksums, FAISS width, token-to-page mapping, and the
   immutable completion ledger.

3. `14_docprune_m3docvqa_eval_array.sbatch`, submitted with
   `--dependency=afterok:$GATE_JOB:$ALL_INDEX_JOB_IDS`, plus
   `RUN_CONFIG_ROOT`, both index-family roots, and `ATTEMPT_ROOT`. Its fixed
   array map is:

   ```text
   0 all-kept top1   1 all-kept top2   2 all-kept top4
   3 docprune top1   4 docprune top2   5 docprune top4
   ```

   Each task selects its matching page-specific manifest, runs all 2,441
   source-ordered questions, resumes only a manifest-identical prefix, then
   runs `validate-run --expected-questions 2441` and preserves an independent
   summary.

## Pass conditions and recovery

The gate must report schema 1 `status=passed`, exact runtime/upstream commits,
the processor-contract digest, and all three page-count traces. Every index
must have schema 4 and a valid manifest digest. Every evaluation must have
exactly 2,441 unique source-ordered qids, valid monotonic traces, finite
measurements, positive production GPU peak allocation, warmup excluded from
recorded rows, and a summary reproducible from immutable JSONL.

On preemption or time limit, use `--resume` only when the run/index manifest,
source files, runtime, model revisions, mode, page count, and output root are
identical. On corruption or any identity mismatch, preserve the failed
attempt and submit a new `attempt-N+1`; never repair an artifact in place. A
failed semantic gate stops all downstream jobs.

## Local preflight before submission

From the control checkout, run `bash -n` on all four wrappers, the complete
CPU pytest suite in `/home/lmalveau/mamba-envs/docprune-sol`, Ruff on `src/`
and tests, `inspect` and CLI dry runs, Markdown link checks, `git diff --check`,
and scans proving no weights, caches, datasets, indexes, predictions,
profiles, secrets, or `DATASET_REVISION` are tracked. Confirm a fresh detached
runtime checkout at the exact runtime commit is clean, M3DocRAG is clean, and
the control checkout contains only reviewed source/docs/launchers.

## Control commit

After the handoff and state files are committed, replace this sentence with
the resulting full commit hash and use that exact control checkout for every
`sbatch` submission. The runtime remains pinned separately to
`d5cefb33f7ca97ce0ef2104fa5e63bd3ad8a5761`.
