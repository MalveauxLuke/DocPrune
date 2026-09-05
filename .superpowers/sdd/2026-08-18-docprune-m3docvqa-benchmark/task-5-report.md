# Task 5 report: concrete M3DocVQA factory, safe resume, and benchmark CLI

## RED evidence

The first targeted collection failed because
`docprune.m3docvqa_factory` did not exist. The new retrieval regression then
failed until the official boundary over-fetched token neighbors and deduped
structured `(doc_id, page_index)` identities.

## GREEN evidence

- Targeted Task 5 and adjacent CLI/retrieval tests pass: **39 passed**.
- Complete CPU suite passes: **215 passed, 1 skipped**. The skip is the
  existing opt-in CUDA/local-Qwen parity probe.
- Ruff format/check pass for all changed Python files.
- Six CLI dry runs pass for `all-kept` and `docprune` at page counts 1, 2, and
  4. Each emits the concrete factory name and resolved mode/page count without
  importing model weights.
- `git diff --check` passes.

## Fix-round evidence

- Fresh custom evaluation workloads are checked against the independently
  resolved invocation authority for runtime commit, model resources, corpus
  identity, pruning settings, selection, and the complete index manifest.
- Existing complete manifests must retain every CLI invocation selector,
  including factory/config paths, mode/page count, limits, sample IDs, and
  raw run-config/index source path and SHA-256 fields; those selectors are
  checked before a factory payload is merged.
- Workload sample order must equal the manifest's resolved question-ID order,
  and the authoritative corpus rows are compared by qid, question, and
  accepted answers before any runner call.
- Final embed manifests are validated against the complete schema/status
  contract before atomic publication.

## Fix-round 4 evidence

- Evaluation workloads reject a supplied `command` unless it is exactly
  `evaluate`; the final merged operation, command, schema, status, invocation
  selectors, source fields, and digest are checked again after the workload
  returns.
- Embed publication is rebuilt from the original CLI invocation plus the
  independently resolved run identity and returned `IndexManifest`; a
  factory-rewritten `run_manifest.json` cannot alter command, config, factory,
  output, mode/page count, selectors, or raw source path/SHA fields.
- Embed and evaluation preflight require a nonempty canonical dataset
  `source_order_sha256`, and returned/index manifests must match it exactly.
- Returned embed manifests must be regular non-symlink JSON files with schema
  4, a valid canonical digest, and byte-level JSON equality to
  `IndexManifest.to_dict()`. JSON index manifests likewise must exactly equal
  the reconstructed `IndexManifest`; arbitrary objects exposing
  `validate_files()` are rejected.
- Focused CLI/factory tests pass: **44 passed**; full CPU suite passes:
  **215 passed, 1 skipped**. Ruff format/check and `git diff --check` pass.
- Six model-free CLI dry runs pass for `all-kept` and `docprune` crossed with
  page counts 1, 2, and 4.

## Fix-round 5 evidence

- Every custom factory receives a fresh `copy.deepcopy` of the invocation
  manifest. Evaluation and embed publication retain separate trusted deep
  copies for all identity comparisons and final manifest composition, so
  nested or scalar in-place factory mutations cannot become the authority.
- Regression tests reproduce nested `paper_values` plus scalar `factory`
  mutation for both evaluation and embedding and require fail-closed
  publication.
- `_validate_embed_result` requires the exact canonical `IndexBuildResult` and
  `IndexManifest` types. Persisted embed manifests are reconstructed and
  validated through the canonical index-manifest loader before exact payload
  comparison; `_load_index_manifest` likewise rejects malicious subclasses.
- Focused Task 5/CLI tests pass: **47 passed**; complete CPU suite passes:
  **218 passed, 1 skipped**. Ruff format/check and `git diff --check` pass.
- Six model-free CLI dry runs pass for `all-kept` and `docprune` crossed with
  page counts 1, 2, and 4.

## Implementation decisions

- `build_workload` is the concrete factory for both `embed` and `evaluate`.
  It validates the pinned M3DocRAG checkout, immutable runtime/model
  identities, processor contract, corpus, index manifest, source order, and
  mode/page count before model loading. Model snapshots are resolved
  local-only and loaded lazily after those checks.
- Evaluation uses the pinned corpus adapter, manifest-bound FAISS/safetensors
  artifacts, the same ColPali instance for retrieval and DocPrune QA, and
  mode-specific all-kept or DocPrune Qwen answerers.
- Indexed retrieval reproduces upstream per-query-token MaxSim (maximum per
  page followed by query-token summation), doubles token-neighbor search
  deterministically until enough unique pages exist, and consumes Task 3's
  structured `token2pageuid` rows without string parsing. The non-indexed
  compatibility path applies the same deterministic over-fetch/dedup rule to
  the official upstream API.
- The CLI defaults to the concrete factory and exposes `--mode`,
  `--run-config`, `--index-manifest`, `--limit`, and `--sample-ids`. Dry runs
  remain model-free and include these identity selectors in the run manifest.
- Resume validates the exact generic and complete run identities, requires an
  exact source-order prefix, checks qid uniqueness plus source question and
  answer identity, skips only already-completed qids, and appends each result
  with a single durable `O_APPEND`/`fsync` write. Existing output collisions
  fail closed. Complete manifests are atomically published with file and
  parent-directory fsyncs; missing/incomplete/tampered resume state, symlinked
  results, malformed records, and source-record drift are rejected before
  model loading or further writes. JSON/mapping run configurations are
  authoritative and include all pinned resources, generation settings, and
  the complete corpus identity; selection and pruning identities are bound to
  the run manifest. CLI resume independently verifies complete canonical
  manifests and hashes supplied run-config/index files, while embed preflight
  resolves the authoritative environment or fails closed before custom
  factories run. Legacy `qid` records are canonicalized to `question_id`, and
  completed records must contain exactly the requested unique page count.

## Limitations

The real 7B model path and the full 2,441-question benchmark are intentionally
not executed in this CPU task. They require the approved SOL allocation,
cached immutable snapshots, processor probe, and validated Task 3 indexes.
