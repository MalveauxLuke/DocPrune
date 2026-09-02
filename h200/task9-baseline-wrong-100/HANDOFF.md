# H200 handoff: Task 9 baseline-wrong 100 confirmation

## Outcome to produce

Run the already implemented confirmation experiment on 100 new ordinary
baseline-wrong questions and publish one unified analysis JSON. Do not redesign
the experiment.

The development pilot found four exact rescues among 24 baseline-wrong
questions, no harms among 24 baseline-correct questions, and a baseline-wrong
mean F1 advantage of `+0.24375` over native dynamic DocPrune. The later frozen
192-mask comparison was too weak on difficult/rescue cases, so this run retains
256 fit masks. The expensive 64 surrogate holdouts are removed because the new
questions and actual generated answers provide the confirmation test.

This remains an adapted region-level, answer-conditioned ContextCite diagnostic,
not vanilla ContextCite or a token oracle.

## Frozen experiment

- 100 new baseline-wrong QIDs, randomly selected from the authenticated Task 6
  holdout failures.
- 100 distinct support-document components; no document fallback was needed.
- Exact cached ordered top-4 pages and persisted features; no retrieval.
- Qwen/Qwen2-VL-7B-Instruct at the existing pinned revision, prompt, and greedy
  decoding.
- Native aggregate-threshold DocPrune independently chooses each question's
  dynamic layer and retained-token budget.
- 256 Bernoulli whole-region fit masks, one scoring pass per mask providing
  accepted-answer and generated-answer supervision.
- Zero global holdouts and zero budget-local holdouts.
- Generated arms: unpruned, native DocPrune, gold-support ContextCite,
  gold-margin ContextCite, and region-size-aware random at the same whole-region
  budget.
- Primary outputs: paired normalized token-F1, exact match, win/tie/loss,
  rescues, gold likelihood, and gold-versus-alternative margin. Uncertainty is
  clustered by support component.

## Sealed cohort

Local source before transfer:

```text
/home/lmalveau/task9-h200-artifacts/task9-baseline-wrong100-confirmation-v1/cohort.json
file SHA-256: afead001a93666628126260deccfd1493ba5b1f2071e646da6d220a8969ab9fb
internal SHA-256: 0bdfdde29b568f54ea541453abeca196cfb49f4b6630f084f80e6c21f158514e
selected: 100
eligible baseline-wrong: 731
eligible support components: 724
fallback: false
```

Preserve its bytes during transfer. Large source data, cached features, PDFs,
model files, mappings, and outputs stay outside Git.

## Phase 0 — mandatory read-only survey

1. Sparse-clone the prepared branch into `/mnt/data1/eunwooim/DocPrune` using
   `SPARSE_CHECKOUT_PATHS.txt`. This is the only permitted pre-survey setup.
2. Read the scoped `AGENTS.md` and `CORAL_POLICY.md`.
3. Run `bash SURVEY_COMMANDS.sh`. It is read-only.
4. Fill `ENVIRONMENT_SURVEY.md` with the observed output and conclusions.
5. Stop if `/mnt/data1` or `/mnt/data2` naming/ownership differs, if CoRAL GPU
   identity is unclear, or if a safe cache/environment root is unavailable.

Apart from the sparse checkout, do not install, transfer, load a model, or
touch a GPU before this phase is recorded.

## Phase 1 — environments while transfer continues

After the survey:

1. Copy `paths.env.example` to `paths.env` and change only survey-disproved
   paths.
2. Manually accept any Conda channel terms reported by the previous failed
   setup attempt. Do not bypass or automate account acceptance.
3. Export every cache/temp variable from `paths.env`, then run
   `setup_environments.sh`. It creates separate environments under
   `/mnt/data2/eunwooim/.conda/envs/`: `docprune-h200` uses the frozen
   Python 3.10/PyTorch 2.4.1/Transformers 4.46.3 stack; `mineru-h200` uses the
   pinned MinerU source at `d9cd58a` and its compatible Python 3.11 stack.
   A single environment is prohibited because the validated dependency sets
   conflict.
4. The setup also checks out M3DocRAG at `29e6ac2` and downloads the exact
   MinerU model revision under `/mnt/data2/eunwooim` caches. Run only its
   targeted import/version checks; do not run the full test suite or load a GPU
   model.
5. Record the clean 40-character execution commit. Keep survey notes outside
   the execution checkout if necessary.

Environment setup may overlap the data transfer. Everything below waits for a
complete verified transfer.

## Phase 2 — authenticate and relocate sealed inputs

The transferred bundle is the source-built CPU cohort, not completed MinerU,
geometry, or mapping output. It contains the unchanged cohort, selected source
rows, 400 sealed PNG inputs, the exact required PDFs and feature shards, fixed
fixture/input manifests, and their provenance metadata. It excludes the global
retrieval index and model snapshots.

1. Require `rsync` to finish and reject any remaining partial file.
2. Verify `MANIFEST.sha256.sha256`, then all 1,014 entries in
   `MANIFEST.sha256`; reject missing, mismatched, or extra raw files.
3. Place the verified directory at
   `task9-h200-local-data/inputs/transferred/` and run
   `relocate_task9_h200_inputs.py` once. This writes new authenticated H200-local
   fixture/input manifests while preserving transferred source bytes.

   ```bash
   "$TASK9_ENV_PREFIX/bin/python" \
     "$TASK9_REPO/examples/relocate_task9_h200_inputs.py" \
     --bundle-root "$TASK9_TRANSFER_ROOT" \
     --source-fixed-root "$TASK9_TRANSFER_ROOT/raw/scratch/lmalveau/docprune/task9-baseline-wrong100-inputs-c0c9bee-v1" \
     --output-root "$TASK9_FIXED_ROOT"
   ```

4. Inventory the cohort, selected rows, cached page identities, 400 page bytes,
   required PDFs/features, and model snapshots. The small source run config,
   index manifest, and processor contract are tracked in `runtime-metadata/`;
   preserve their recorded hashes when preparing H200-local runtime metadata.
   Stop if any fixed identity is missing or retrieval would be required.

## Phase 3 — MinerU, geometry, and mappings

1. Before GPU use, check all GPUs and propose one physical CoRAL GPU ID. Obtain
   explicit approval for that exact GPU. Normal CoRAL use needs no per-job
   notice; check first and post for use beyond 15 minutes only when borrowing
   outside the CoRAL allocation.
2. Run a one-page pinned MinerU smoke, recording tool/model revisions, input and
   output hashes, runtime, completion manifest, and GPU identity.
3. After admission, process the 400 sealed pages resumably on the same approved
   GPU. Never perform retrieval or inspect answer outcomes.
4. Run a one-question geometry smoke in the DocPrune environment, then capture
   all 100 QIDs resumably after admission. Authenticate page order, features,
   model resources, and token geometry.
5. Build exactly 100 mappings on CPU and run
   `run_task9_confirmation_batch.py --validate-only` for ordinals 0–99. Require
   256 fit masks, zero holdouts, complete region/residual coverage, and no
   retrieval/global index.

## Phase 4 — experiment smoke and admission

1. Check current processes and memory on GPUs 4–7 with `nvidia-smi`.
2. Propose exactly one CoRAL physical GPU ID and obtain explicit approval for
   that allocation.
3. Confirm the operator is present in the shared channel or has a relay, then
   run `launch_smoke.sh GPU_ID` on only the approved GPU.
4. In the same smoke stage, confirm completion-manifest admission, exactly 256
   fit masks, zero holdouts, exact fixture/mapping hashes, dynamic native
   layer/budget, all five arms, finite likelihoods, and recorded H200
   identity/peak VRAM.
5. Do not launch production if the smoke changes code or experiment semantics.

The admitted smoke is question 0 and is retained as canonical production data.

## Phase 5 — production and aggregation

Normal use of an assigned CoRAL GPU needs no per-job announcement. After
separate approval, recheck the selected GPU and run
`launch_production.sh GPU_ID`. It processes questions 1–99 sequentially as
resumable one-question units on one approved GPU; question 0 comes from the
admitted smoke. Failed units can be rerun individually without repeating
completed questions. Multiple GPUs are never implicitly authorized and require separate approval for
the exact count and IDs.

After all 100 unique QIDs are admitted, run `aggregate_task9_confirmation.py`
with both the smoke root and production root. Preserve its unified JSON, file
SHA-256, internal SHA-256, per-question rows, logs, GPU identities, and any
retries in the experiment log.

## Runtime code entry points

- Intervention run: [`../../examples/run_task9_regional_development.py`](../../examples/run_task9_regional_development.py)
- H200 input relocation: [`../../examples/relocate_task9_h200_inputs.py`](../../examples/relocate_task9_h200_inputs.py)
- MinerU preparation/finalization: [`../../examples/run_task8_mineru_smoke.py`](../../examples/run_task8_mineru_smoke.py)
- Geometry capture: [`../../examples/run_task9_preliminary_geometry_capture.py`](../../examples/run_task9_preliminary_geometry_capture.py)
- Region mapping: [`../../examples/build_task9_preliminary_region_mapping.py`](../../examples/build_task9_preliminary_region_mapping.py)
- Fit-only analysis: [`../../examples/analyze_task9_confirmation_attribution.py`](../../examples/analyze_task9_confirmation_attribution.py)
- Batch driver: [`../../examples/run_task9_confirmation_batch.py`](../../examples/run_task9_confirmation_batch.py)
- Selected arms: [`../../examples/run_task9_preliminary_selected_arms.py`](../../examples/run_task9_preliminary_selected_arms.py)
- Unified result: [`../../examples/aggregate_task9_confirmation.py`](../../examples/aggregate_task9_confirmation.py)

## Stop conditions

Stop before GPU work if transfer verification is incomplete, the cohort hash
differs, a selected QID/page/feature differs, retrieval would be required, the
checkout is dirty, the model/prompt/decoding revision cannot be preserved, GPU
ownership is unclear, or caches would write to `/` or cross-lab storage. Stop
before experiment smoke if mappings are incomplete. Record the exact blocker;
do not improvise a scientifically different run.
