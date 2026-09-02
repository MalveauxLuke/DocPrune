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

1. Read the scoped `AGENTS.md` and `CORAL_POLICY.md`.
2. Run `bash SURVEY_COMMANDS.sh`. It is read-only.
3. Fill `ENVIRONMENT_SURVEY.md` with the observed output and conclusions.
4. Stop if `/mnt/data1` or `/mnt/data2` naming/ownership differs, if CoRAL GPU
   identity is unclear, or if a safe cache/environment root is unavailable.

Do not clone, install, transfer, load a model, or touch a GPU before this phase
is recorded.

## Phase 1 — checkout and environment

After the survey:

1. Clone the prepared branch into `/mnt/data1/eunwooim/DocPrune` (or the
   survey-confirmed CoRAL equivalent). Use the paths in
   `SPARSE_CHECKOUT_PATHS.txt` if a trimmed checkout is desired.
2. Copy `paths.env.example` to `paths.env` and change only survey-disproved
   paths.
3. Create the environment under `/mnt/data2/eunwooim/.conda/`, never under `/`
   or the default home cache. `environment-h200.yml` preserves the successful
   SOL versions and adds scikit-learn 1.7.2. Treat those versions as a strong
   freeze; if an H200/driver incompatibility makes one impossible, record the
   minimal substitution before proceeding.
4. Export every cache/temp variable from `paths.env` before installation or
   model access.
5. Run only targeted import/version checks. Do not run the full test suite.
6. The execution checkout must be clean and its exact 40-character commit
   recorded before the smoke.

## Phase 2 — fixed inputs and mappings

Follow `TRANSFER_MANIFEST.md`. The transfer strategy is intentionally finalized
after the H200 survey because mount paths and available local caches are not yet
known. No code needs to be written.

Once artifacts are present, run `seal_task9_confirmation_inputs.py` to create
the 100-question fixed-page fixture. It authenticates the cohort and creates
retrieval-free QID inputs. Use the existing prepared mapping pipeline:

- `run_task8_mineru_smoke.py` for each selected fixed-page input;
- `run_task9_preliminary_geometry_capture.py` for the frozen post-QTP geometry;
- `build_task9_preliminary_region_mapping.py` to combine authenticated MinerU
  and geometry artifacts.

The filenames placed in `TASK9_MAPPINGS` must begin with the zero-padded
ordinal and include the QID, for example `000-<qid>.json`. Validate all 100
mappings before any attribution run. No retrieval command is permitted.

## Phase 3 — smoke

1. Check GPUs 4–7 with `nvidia-smi`.
2. If the smoke may exceed 15 minutes, post in the channel first.
3. Choose one idle CoRAL physical GPU and run `launch_smoke.sh GPU_ID`.
4. Confirm completion-manifest admission, exactly 256 fit masks, zero holdouts,
   exact fixture/mapping hashes, dynamic native layer/budget, all five arms,
   finite likelihoods, and recorded H200 identity/peak VRAM.
5. Do not launch production if the smoke changes code or experiment semantics.

The admitted smoke is question 0 and is retained as canonical production data.

## Phase 4 — production and aggregation

Production is expected to exceed 15 minutes, so post in the channel first.
After confirming GPUs 4–7 remain available, run `launch_production.sh`. It
processes questions 1–99 as one-question units across the four CoRAL GPUs;
question 0 comes from the admitted smoke. Failed units can be rerun individually
without repeating completed questions.

After all 100 unique QIDs are admitted, run `aggregate_task9_confirmation.py`
with both the smoke root and production root. Preserve its unified JSON, file
SHA-256, internal SHA-256, per-question rows, logs, GPU identities, and any
retries in the experiment log.

## Prepared code entry points

- Cohort: [`../../examples/seal_task9_baseline_wrong100.py`](../../examples/seal_task9_baseline_wrong100.py)
- Fixed pages: [`../../examples/seal_task9_confirmation_inputs.py`](../../examples/seal_task9_confirmation_inputs.py)
- Intervention run: [`../../examples/run_task9_regional_development.py`](../../examples/run_task9_regional_development.py)
- Fit-only analysis: [`../../examples/analyze_task9_confirmation_attribution.py`](../../examples/analyze_task9_confirmation_attribution.py)
- Batch driver: [`../../examples/run_task9_confirmation_batch.py`](../../examples/run_task9_confirmation_batch.py)
- Selected arms: [`../../examples/run_task9_preliminary_selected_arms.py`](../../examples/run_task9_preliminary_selected_arms.py)
- Unified result: [`../../examples/aggregate_task9_confirmation.py`](../../examples/aggregate_task9_confirmation.py)

## Stop conditions

Stop before GPU work if the cohort hash differs, a selected QID/page differs,
retrieval would be required, the checkout is dirty, the model/prompt/decoding
revision cannot be preserved, mappings are incomplete, GPU ownership is
unclear, or caches would write to `/` or cross-lab storage. Record the exact
blocker; do not improvise a scientifically different run.
