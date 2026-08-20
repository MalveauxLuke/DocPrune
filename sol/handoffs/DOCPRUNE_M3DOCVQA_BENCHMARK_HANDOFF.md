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
control commit: sealed per-attempt in `$ATTEMPT_ROOT/control.json`
M3DocRAG checkout: /home/lmalveau/src/m3docrag-benchmark-29e6ac2
M3DocRAG commit: 29e6ac2294d6b87075a1d45b8a8df175b214248a
environment: /home/lmalveau/mamba-envs/docprune-sol
PDF tools: /home/lmalveau/mamba-envs/m3docvqa-acquisition
Poppler: 26.05.0, build hfdef1ce_3
Poppler package SHA-256: a5737f253f6301dac019dc9b9cfaefae40d1ee6f846410d43eed56ff7588cbd3
pdfinfo SHA-256: 5d0e1caa04f15391324c9e5f1d65753d8b925761eedc710c70e02f49b5080aae
pdftoppm SHA-256: 1102bc3f4a12f3d3d207ac8e39f463d0fb3c403511fb4c817e776e5251b53f33
corpus: /scratch/lmalveau/docprune/datasets/m3docvqa
factory: docprune.m3docvqa_factory:build_workload
```

Every launcher validates `CONTROL_RECORD` before doing work. The record binds
the full reviewed control commit and its Git tree SHA, so a later commit cannot
silently change the meaning of an already-created attempt. Every Python command
runs from the detached runtime checkout with `PYTHONPATH=$RUNTIME_DIR/src`.
The M3DocRAG checkout must be a fresh, dedicated detached worktree, clean at
its exact commit. The control checkout is used only to submit these wrappers.
Every launcher also requires `PDFTOOLS_DIR`, validates the two regular
non-symlink Poppler executables against the hashes above, checks version
26.05.0, and prepends `$PDFTOOLS_DIR/bin` before `$ENV_DIR/bin` in `PATH`.
The local gate-config and run-config generators perform the same Python
preflight. The package/build values above are provenance; the executable
hashes are the load-bearing job checks.

## Seal the reviewed control checkout

Run this after review, with no jobs submitted and before creating the gate
configuration. It fails closed on dirty control state, writes outside the
checkout, and makes the record read-only. Do not edit or overwrite this file;
a failed attempt gets a new attempt root and a new record.

```bash
export PROJECT_DIR=/home/lmalveau/DocPrune-benchmark
export ATTEMPT_ROOT=/scratch/lmalveau/docprune/benchmark-d5cefb3/attempt-N
export CONTROL_RECORD="$ATTEMPT_ROOT/control.json"
test ! -e "$ATTEMPT_ROOT"
mkdir "$ATTEMPT_ROOT"
test -z "$(git -C "$PROJECT_DIR" status --porcelain --untracked-files=all)"
CONTROL_COMMIT="$(git -C "$PROJECT_DIR" rev-parse HEAD)"
CONTROL_TREE="$(git -C "$PROJECT_DIR" rev-parse "${CONTROL_COMMIT}^{tree}")"
python - "$CONTROL_RECORD" "$PROJECT_DIR" "$CONTROL_COMMIT" "$CONTROL_TREE" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "project_dir": str(Path(sys.argv[2]).resolve()),
    "control_commit": sys.argv[3],
    "tree_sha": sys.argv[4],
}
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
chmod 0444 "$CONTROL_RECORD"
export CONTROL_COMMIT
test "$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["control_commit"])' "$CONTROL_RECORD")" = "$CONTROL_COMMIT"
test "$(git -C "$PROJECT_DIR" rev-parse HEAD)" = "$CONTROL_COMMIT"
test "$(git -C "$PROJECT_DIR" rev-parse "${CONTROL_COMMIT}^{tree}")" = "$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["tree_sha"])' "$CONTROL_RECORD")"
```

## Prepare pinned runtime and upstream checkouts

This is executable and preserves any pre-existing directory; it never removes
user data. An absent runtime is created as a detached worktree at the exact
runtime commit. The upstream source checkout is used only to create a new,
dedicated detached worktree; the new destination must not already exist. Git's
`--untracked-files=all` check reports every file, and every launcher and the
config generator require a strictly empty upstream status. All jobs set
`PYTHONDONTWRITEBYTECODE=1` so the dedicated checkout remains clean.

```bash
export RUNTIME_DIR=/home/lmalveau/DocPrune-runtime-d5cefb3
export EXPECTED_COMMIT=d5cefb33f7ca97ce0ef2104fa5e63bd3ad8a5761
export M3DOCRAG_SOURCE=/home/lmalveau/src/m3docrag-runtime-29e6ac2
export M3DOCRAG_DIR=/home/lmalveau/src/m3docrag-benchmark-29e6ac2
export M3DOCRAG_COMMIT=29e6ac2294d6b87075a1d45b8a8df175b214248a
if [[ ! -e "$RUNTIME_DIR" ]]; then
  git -C "$PROJECT_DIR" worktree add --detach "$RUNTIME_DIR" "$EXPECTED_COMMIT"
fi
test "$(git -C "$RUNTIME_DIR" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$RUNTIME_DIR" status --porcelain --untracked-files=all)"
test -d "$M3DOCRAG_SOURCE"
test "$(git -C "$M3DOCRAG_SOURCE" rev-parse "$M3DOCRAG_COMMIT^{commit}")" = "$M3DOCRAG_COMMIT"
test ! -e "$M3DOCRAG_DIR"
git -C "$M3DOCRAG_SOURCE" worktree add --detach "$M3DOCRAG_DIR" "$M3DOCRAG_COMMIT"
test "$(git -C "$M3DOCRAG_DIR" rev-parse HEAD)" = "$M3DOCRAG_COMMIT"
test -z "$(git -C "$M3DOCRAG_DIR" status --porcelain --untracked-files=all)"
```

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
$ATTEMPT_ROOT/inputs/gate-top1.json
$ATTEMPT_ROOT/gate/gate.json
$ATTEMPT_ROOT/gate/probe-page.png
$ATTEMPT_ROOT/gate/processor-contract.json
$ATTEMPT_ROOT/run-configs/{all-kept,docprune}-top{1,2,4}.json
$ATTEMPT_ROOT/indexes/all-kept/top{1,2,4}/all-kept/manifest.json
$ATTEMPT_ROOT/indexes/docprune/top{1,2,4}/docprune/manifest.json
$ATTEMPT_ROOT/eval/{all-kept,docprune}/top{1,2,4}/run/
```

The six index roots are distinct because BTP is page-count-specific. A
manifest from one mode/page count is never reused for another.

## Run-config construction

The gate receives a provisional `inputs/gate-top1.json` whose
`processor_contract_path` points at the not-yet-created
`$ATTEMPT_ROOT/gate/processor-contract.json` and which omits
`processor_contract_sha256`. After the gate writes and authenticates
`gate.json`, it invokes the repository-native generator before reporting
`GATE_STATUS=passed`. The generator creates exactly six final configs under
`$RUN_CONFIG_ROOT`, validates each through the production run-config identity
loader, authenticates all gate evidence, and pins the actual contract digest.
The executable is `examples/m3docvqa/make_run_configs.py`.
Operators do not run this generator as a separate pre-gate step.

Every final JSON contains the following fields, plus the complete nested
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
  "processor_contract_sha256": "actual SHA-256 of processor_contract_path",
  "max_new_tokens": 128,
  "do_sample": false,
  "num_beams": 1,
  "prompt": "question: $question\noutput only answer.",
  "m3docrag_root": "/home/lmalveau/src/m3docrag-benchmark-29e6ac2",
  "gate_path": "/scratch/.../gate/gate.json",
  "gate_sha256": "canonical digest of the authenticated gate JSON",
  "semantic_samples_path": "/scratch/.../gate/semantic-samples.json",
  "semantic_samples_sha256": "digest of the authenticated semantic evidence",
  "gate_sample_ids": ["the exact fixed five-qid tuple"],
  "corpus": {"...": "the exact pinned identity and archive_hashes"}
}
```

The generator uses `CorpusIdentity.from_root`, the constants in
`benchmark_config.py`, and the actual gate-contract SHA. Do not substitute
fixture values or `DATASET_REVISION`.

## Ordered submission

Set the common environment from the exact values above, including all six
model resource/revision variables. The launchers are:

1. `12_docprune_m3docvqa_gate.sbatch`, with an absent `GATE_ROOT`, a
   provisional input config outside that root, and the exact fixed
   comma-separated `GATE_SAMPLE_IDS`. It renders the deterministic 144-DPI
   `PROBE_IMAGE` inside the gate from the pinned corpus/supporting document,
   runs the pinned processor probe, exact span/grid/raster checks, all six
   deterministic dry-run selectors, repeated all-kept fixed answers, and one
   real DocPrune answer at pages 1/2/4. It writes/authenticates `gate.json`,
   produces the six final configs, and only then reports `status=passed`.

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

## Exact Slurm submission commands

This block supplies every launcher variable and directs logs to an absolute
artifact directory outside the control checkout. `--export=ALL` carries the
explicitly exported values into each job; comma-separated sample IDs remain
safe because they are exported through the environment rather than embedded in
the Slurm export list. Use a fresh `attempt-N` and never reuse a failed root.

```bash
export PROJECT_DIR=/home/lmalveau/DocPrune-benchmark
export RUNTIME_DIR=/home/lmalveau/DocPrune-runtime-d5cefb3
export ENV_DIR=/home/lmalveau/mamba-envs/docprune-sol
export PDFTOOLS_DIR=/home/lmalveau/mamba-envs/m3docvqa-acquisition
export EXPECTED_COMMIT=d5cefb33f7ca97ce0ef2104fa5e63bd3ad8a5761
export M3DOCRAG_DIR=/home/lmalveau/src/m3docrag-benchmark-29e6ac2
export M3DOCRAG_COMMIT=29e6ac2294d6b87075a1d45b8a8df175b214248a
export CORPUS_ROOT=/scratch/lmalveau/docprune/datasets/m3docvqa
export DOCPRUNE_FACTORY=docprune.m3docvqa_factory:build_workload
export QWEN_MODEL=Qwen/Qwen2-VL-7B-Instruct
export QWEN_REVISION=eed13092ef92e448dd6875b2a00151bd3f7db0ac
export COLPALI_MODEL=vidore/colpali-v1.2
export COLPALI_REVISION=961b51745de3e9adb3468ac5c9ccca0ac626c217
export COLPALI_BACKBONE_MODEL=vidore/colpaligemma-3b-pt-448-base
export COLPALI_BACKBONE_REVISION=30ab955d073de4a91dc5a288e8c97226647e3e5a
export ATTEMPT_ROOT=/scratch/lmalveau/docprune/benchmark-d5cefb3/attempt-N
export CONTROL_RECORD="$ATTEMPT_ROOT/control.json"
export CONTROL_COMMIT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["control_commit"])' "$CONTROL_RECORD")"
export GATE_ROOT="$ATTEMPT_ROOT/gate"
export INPUT_ROOT="$ATTEMPT_ROOT/inputs"
export RUN_CONFIG="$INPUT_ROOT/gate-top1.json"
export GATE_SAMPLE_IDS=a33985b1e8b2502fc18cc8147dc27db8,710a6d2254076ea58756c6c7cc211f1e,0d8f2779137fb47db953c4af5247ffe5,e240f5fe65b39eee70d3576cff88fe5a,18ecd2ac6c0ac69993b92dc4b30137e8
export SLURM_LOG_DIR="$ATTEMPT_ROOT/slurm-logs"
export RUN_CONFIG_ROOT="$ATTEMPT_ROOT/run-configs"
source "$PROJECT_DIR/examples/m3docvqa/pdf_tools_preflight.sh"
mkdir -p "$SLURM_LOG_DIR" "$INPUT_ROOT"

PYTHONPATH="$PROJECT_DIR/examples/m3docvqa:$RUNTIME_DIR/src" "$ENV_DIR/bin/python" "$PROJECT_DIR/examples/m3docvqa/make_gate_config.py" \
  --corpus-root "$CORPUS_ROOT" \
  --processor-contract "$GATE_ROOT/processor-contract.json" \
  --m3docrag-root "$M3DOCRAG_DIR" \
  --output "$RUN_CONFIG"

GATE_JOB="$(sbatch --parsable \
  --chdir="$SLURM_LOG_DIR" \
  --output="$SLURM_LOG_DIR/gate-%j.out" \
  --error="$SLURM_LOG_DIR/gate-%j.err" \
  --export=ALL \
  "$PROJECT_DIR/examples/sbatch/12_docprune_m3docvqa_gate.sbatch")"

ALL_KEPT_INDEX_ROOT="$ATTEMPT_ROOT/indexes/all-kept"
DOCPRUNE_INDEX_ROOT="$ATTEMPT_ROOT/indexes/docprune"
INDEX_JOB_IDS=()
for MODE in all-kept docprune; do
  for PAGES in 1 2 4; do
    export MODE PAGES RESUME=0
    export RUN_CONFIG="$RUN_CONFIG_ROOT/${MODE}-top${PAGES}.json"
    export INDEX_ROOT="$ATTEMPT_ROOT/indexes/${MODE}/top${PAGES}"
    INDEX_JOB_IDS+=("$(sbatch --parsable \
      --dependency="afterok:$GATE_JOB" \
      --chdir="$SLURM_LOG_DIR" \
      --output="$SLURM_LOG_DIR/index-${MODE}-top${PAGES}-%j.out" \
      --error="$SLURM_LOG_DIR/index-${MODE}-top${PAGES}-%j.err" \
      --export=ALL \
      "$PROJECT_DIR/examples/sbatch/13_docprune_m3docvqa_index.sbatch")")
  done
done

export ALL_KEPT_INDEX_ROOT DOCPRUNE_INDEX_ROOT ATTEMPT_ROOT RUN_CONFIG_ROOT
INDEX_DEPENDENCY="$(IFS=:; echo "${INDEX_JOB_IDS[*]}")"
EVAL_JOB="$(sbatch --parsable \
  --dependency="afterok:$GATE_JOB:$INDEX_DEPENDENCY" \
  --chdir="$SLURM_LOG_DIR" \
  --output="$SLURM_LOG_DIR/eval-%A_%a.out" \
  --error="$SLURM_LOG_DIR/eval-%A_%a.err" \
  --export=ALL \
  "$PROJECT_DIR/examples/sbatch/14_docprune_m3docvqa_eval_array.sbatch")"
printf 'gate=%s indexes=%s eval=%s\n' "$GATE_JOB" "${INDEX_JOB_IDS[*]}" "$EVAL_JOB"
```

The gate is the only job without a dependency. Each of the six index jobs is
blocked by `afterok:$GATE_JOB`; the six-cell array is blocked by the gate and
all six index jobs. No command in this handoff submits work before the seal,
runtime/upstream checks, provisional config, and local preflight succeed.

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
CPU pytest suite, Ruff on `src/`, tests, and the config generators, `inspect`
and CLI dry runs, Markdown link checks, `git diff --check`, and scans proving
no weights, caches, datasets, indexes, predictions, profiles, secrets, or
`DATASET_REVISION` are tracked:

```bash
export PROJECT_DIR=/home/lmalveau/DocPrune-benchmark
export ENV_DIR=/home/lmalveau/mamba-envs/docprune-sol
export PDFTOOLS_DIR=/home/lmalveau/mamba-envs/m3docvqa-acquisition
source "$PROJECT_DIR/examples/m3docvqa/pdf_tools_preflight.sh"
for wrapper in examples/sbatch/{11,12,13,14}_docprune_m3docvqa*.sbatch; do bash -n "$wrapper"; done
PYTHONPATH=src /home/lmalveau/mamba-envs/docprune-sol/bin/pytest -q
/home/lmalveau/mamba-envs/docprune-sol/bin/ruff check src tests examples/m3docvqa
PYTHONPATH=src /home/lmalveau/mamba-envs/docprune-sol/bin/docprune-m3docvqa inspect --config configs/docprune-m3docvqa.toml --pages 1
PYTHONPATH=src /home/lmalveau/mamba-envs/docprune-sol/bin/pytest -q tests/test_cli.py -k dry_run
git diff --check
test -z "$(git ls-files | rg 'DATASET_REVISION|(^|/)(weights?|checkpoints?|.*\\.(safetensors|bin|pt|ckpt|gguf|onnx|npz|npy|faiss|parquet|arrow|sqlite|db))$' || true)"
```

Then run the pinned runtime/upstream preparation block above. Confirm the
detached runtime and the newly created dedicated M3DocRAG worktree are at their
exact commits and strictly clean; and the control checkout contains only
reviewed source/docs/launchers.

The runtime remains pinned separately to
`d5cefb33f7ca97ce0ef2104fa5e63bd3ad8a5761`; the reviewed control commit is
always the full SHA sealed in each attempt's `control.json`.
