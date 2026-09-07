# H200: evidence-reviewed correction corpus and deletion-depth comparison

## Objective and scope

Establish a matching baseline on the 40 curated development candidates, then
compare separately fitted regional ContextCite oracles at **input**, before the
first answer-decoder block, and **dynamic**, after the zero-based block selected
by native DocPrune for that question. `0` means after block zero; it is not input.
These two boundaries are the initial comparison; an expanded numerical grid is
not yet fixed. Preserve and report each actual boundary, rather than calling all
intermediate interventions B13.

The corpus contains 10 minimally revised questions and nine repaired four-page
fixtures. These are explicitly designed experimental inputs. Report them
separately from unchanged questions with original retrieved pages. Forty is the
candidate count, not a promise of forty genuinely incorrect matching baselines.
The initial audit found three matching-baseline errors; the remaining candidates
need matching baselines. Baseline all forty under the packaged controlled setup.

The 600 questions are already on H200; their preparation continues separately.
The new assembler can reuse authenticated files from that transfer or other
existing inputs. Do not transfer the 600 again, alter their split, prepare them,
or run their shared-probe experiment through this handoff. The 100 confirmation
questions are excluded. The new corpus records a protected-document exclusion
check; do not infer independence of historical cohorts from their names.

## Package and machine setup

Use the **existing `/mnt/data1/eunwooim/DocPrune` checkout**. Do not create a
second clone or extract another source tree. Keep correction-specific data,
preparation and results under its ignored
`task9-h200-local-data/correction-depth40/` directory.

Update the existing H200 branch by fast-forward only after checking that no
active job is reading code being updated. Preserve local changes and the
600/100 artifacts. Extend an existing sparse checkout with the paths listed in
`SPARSE_CHECKOUT_PATHS.txt`; do not replace its patterns and hide another
track's files. This handoff is the correction track's execution entrypoint;
older 100/600 instructions remain scoped to those separate jobs.

Record `git rev-parse HEAD` and the runtime's source-tree hash. Verify the input
package's `MANIFEST.sha256` before assembly. The older SOL source archive is
an optional backup; do not transfer or extract it for this checkout workflow.

Inspect mounts, free storage, GPU processes, and the existing environments first.
Use CoRAL physical GPUs 4–7; GPUs 0–3 belong to ARC. Do not borrow those GPUs
through this handoff. Be in the shared channel or have an active relay. Keep
data/code on `/mnt/data1/eunwooim` and caches/environments/temp on
`/mnt/data2/eunwooim`, after verifying the mounts. Preserve existing workloads.
Copy `paths.env.example` into the ignored correction data directory, fill the existing manifest
paths, and source that copy. Reuse the validated Qwen and separate MinerU
environments; no reinstall or model download is part of this package.
The existing H200 oracle environment specifies scikit-learn 1.7.2, frozen
Transformers 4.46.3 and qwen-vl-utils 0.0.8. Confirm those imports before model
execution; the Git checkout is not a replacement model environment.

Set `DOCPRUNE_TASK6_POPPLER_BIN` to the existing transferred Poppler `bin`
directory. The path is relocatable; the pinned `pdftoppm` SHA-256 remains
`1102bc3f4a12f3d3d207ac8e39f463d0fb3c403511fb4c817e776e5251b53f33`.
An arbitrary system renderer is rejected. Ensure its companion `pdfinfo` and
shared libraries are available, and the existing Python environment has
`pdf2image`. No SOL filesystem symlink needs to be created on H200.

## Assemble and validate on CPU

The small assembly recipe is committed under `h200/correction-depth/recipe/`
and arrives through the existing sparse Git checkout. It contains all 40 case
specifications, answer contracts, source-ledger metadata and expected hashes;
no PDF, feature or segmentation bytes. No separate metadata/archive transfer
is required. `CORRECTION_PACKAGE` points to this recipe directory.

First run `bash h200/correction-depth/check_inputs.sh`. Pass existing transfer
roots or exact PDF/feature directories as additional arguments when their paths
are known. This only checks files and writes `asset-check.json` plus
`available-case-ids.json` under the ignored correction data folder. If files
appear missing, inspect the existing H200 fixture/manifests to locate their
actual directories and repeat with those roots before requesting any transfer.
The check does not recursively search all storage or run retrieval. Never rebuild
features or change pages to bypass missing assets. An unsealed PDF identity must
be recovered before that case can be admitted.

```bash
"$CORRECTION_PYTHON" "$CORRECTION_CODE/examples/package_correction_depth.py" assemble \
  --package "$CORRECTION_PACKAGE" --output "$CORRECTION_FIXED" \
  --reuse-root "$CORRECTION_REUSE_ROOT" \
  --config "$CORRECTION_CODE/configs/docprune-m3docvqa.toml" \
  --run-config "$CORRECTION_RUN_CONFIG" \
  --index-manifest "$CORRECTION_INDEX_MANIFEST"
```

Additional `--reuse-root` values may identify existing transfer roots or exact
PDF/document-feature directories. Reads are limited to known candidate paths.
The index **manifest** is provenance metadata; the global index itself must
never be loaded. Assembly seals the exact four rendered page identities and
preserves question edits, complete accepted answers, and original gold labels.
Use the assembled `corpus.json` for every subsequent command.

Set `CORRECTION_REVISION` to the current `git rev-parse HEAD`.
Run `run_correction_depth.py validate` with the common arguments below. This
validates fixed resources without loading a model. If assets are unavailable,
finish assembly for explicitly selected available cases and log omissions;
never count missing cases as unsuccessful oracle experiments.
The H200 check writes `available-case-ids.json` from its actual local inventory.
Pass it with `--case-ids-file FILE` to assemble an explicit nonempty available
subset if some cases remain blocked. The SOL package had 36 complete cases;
that is not a measurement of H200 availability. Use a fresh output directory
for each distinct assembled membership.

## Execution order

Each command accepts explicit case IDs for a one-case smoke and selective
resumption. Start with one case through the new path. Check baseline outputs,
geometry, mapping, all-keep parity and selected answers before admitting the
remaining cases. Do not repeat unchanged successful units.

Common runner arguments:

```bash
"$CORRECTION_PYTHON" "$CORRECTION_CODE/examples/run_correction_depth.py" validate \
  --corpus "$CORRECTION_FIXED/corpus.json" \
  --resources "$CORRECTION_FIXED/resources.json" \
  --output "$CORRECTION_RESULTS" --runtime-commit "$CORRECTION_REVISION"
```

1. **Segment first:** seal the exact fixture images and run pinned MinerU on
   all four supplied pages for each admitted case. This does not require a
   baseline; the `--baseline-root` argument is only consumed by geometry. Reuse
   authenticated completed outputs when their exact image identities agree.
   Run one case first, then the remaining cases after checking its output.
   No completed segmentation is assumed for the new 40-candidate corpus.
2. **Baseline:** replace `validate` with `baseline`; on an idle assigned GPU,
   set `CUDA_VISIBLE_DEVICES` explicitly. Frozen Qwen uses initial BTP/QTP and
   no CTP for the matching reference. Also run native DocPrune on identical
   pages and capture its boundary and achieved token budget. Save the exact
   fixed self-generation token IDs, complete gold target IDs, prompt identities,
   raw likelihoods, generation text, timing, memory, and preprocessing counts.
3. **Review:** inspect baseline contract scores. `review` is neither incorrect
   nor correct. A baseline adjudication file maps case IDs to
   `{baseline_sha256, status, reviewer, rationale}`; the hash is the baseline's
   `artifact_sha256`. Resolve meaning against the supplied evidence before
   passing `--adjudications FILE` to comparison. Correct baselines become
   preservation controls; only verified incorrect baselines enter the
   correction denominator. Freeze reviewed answer contracts before oracle work.
4. **Geometry and mapping:** after baseline, run the preparation `geometry`
   phase against the saved `reference.jsonl`, then the CPU `map` phase to
   publish mappings into `resources.json`. Reuse the segmented exact pages;
   do not rerun MinerU simply because baselines finished. Geometry remains
   question-specific and must match the actual post-BTP/QTP population.
5. **Compare:** use runner phase `compare --boundaries input dynamic`, plus any
   baseline adjudications. Fit **256 masks separately at each boundary** using
   identical region-mask vectors, question/pages/regions, and fixed targets.
   Gold support is primary; G−S and the coefficient residual below are secondary.
   The same 256 masks score both gold and fixed self: no second mask sweep is
   needed. There is no additional holdout by
   default; diagnostics explicitly say unmeasured. An optional holdout is extra
   masks, never a reduction of the 256 fitting masks.
6. **Summarize:** use phase `summarize` with the same paths. Preserve raw outputs
   and write a short experiment log recording source identity, inputs, GPU,
   admissions, failures, completed cases and evidence-linked results.

Preparation uses the following common arguments (shown for CPU image sealing):

```bash
"$CORRECTION_PYTHON" "$CORRECTION_CODE/examples/prepare_correction_depth.py" seal \
  --corpus "$CORRECTION_FIXED/corpus.json" \
  --resources "$CORRECTION_FIXED/resources.json" \
  --baseline-root "$CORRECTION_RESULTS" --output "$CORRECTION_PREP" \
  --runtime-commit "$CORRECTION_REVISION"
```

Run phase `segment` with `--execute-gpu`,
`--mineru-template-run-manifest "$CORRECTION_MINERU_MANIFEST"` and
`--mineru-executable "$CORRECTION_MINERU"`. Then run phase `geometry` with
`--execute-gpu`, and phase `map` without GPU authorization. Set an idle assigned
physical device with `CUDA_VISIBLE_DEVICES` for each GPU process. These separate
processes release model memory between segmentation and geometry. Append
`--case-id ID` to each phase for the initial smoke.

## Interpretation and admission checks

The runner requires all-keep continuation likelihood parity and generation
parity before accepting a depth's selected generations. Keep original prompt
order, frozen raster/processor settings, BTP/QTP, surviving text states,
original positional identities, greedy generation and EOS behavior. A later
state can already contain deleted-region information; success there does not
establish sufficient evidence before the first decoder block.

Native DocPrune remains the main pruning comparator. The regional random arm
is additional. Regional gold-support, margin and random sets use a common
achievable whole-region token budget at or below the native target. Among sets
at that cost, the existing knapsack maximizes the summed coefficients (random
scores for the random arm). Report both achieved counts: a region partition
may prevent exact equality to native token-level selection. Do not call a
region-count match a token-budget match.

Report genuine correction and preservation separately, with their respective
denominators; retain unknown pruned generations for semantic review. Legacy
F1/EM remain auxiliary and cannot turn a superficial overlap into a correction.
Save G and S and each change from its own boundary's all-keep reference. A
positive change in G−S can coexist with decreasing G. When gold and self are
the same sequence, margin is not a preservation measure.

### Gold, self and residual comparison

Fit gold support `G(m)` and fixed-original-answer support `S(m)` separately on
the same mask bank at each depth. Also fit `G(m)−S(m)` directly. Export a fourth
coefficient vector, `beta_G−beta_S`, as the **coefficient residual**. Lasso
regularization means subtracting separately fitted coefficients need not equal
fitting the difference target; retain distinct names and results.

Select self-support and coefficient-residual contexts at the same achieved
whole-region budget as gold support, direct margin and regional random. These
two additional selections require two final answer generations per depth;
their offline probabilities and fits reuse the original 256 mask evaluations.
The self answer stays fixed to the original no-CTP baseline at both depths;
do not replace it with a mask's generated answer or the native DocPrune answer.

Export the actual gold/self selected-set overlap and differences in region IDs
and visual-token IDs. Set differences describe selected contexts; coefficient
differences describe the fitted additive response. Neither establishes which
individual region caused the model's answer. Save all raw G/S values so these
comparisons can be recomputed without GPU work.

This is a depth experiment, not selector training. Do not choose an attached,
compact, pretrained 2B or hybrid selector from oracle scores alone. A future
learned experiment needs document-level separation and targets generated at its
own deployment boundary. The intended early contract remains one shared vision
pass after initial BTP/QTP, all surviving visual tokens and region identities
available to the selector, and selected original tokens entering the frozen
decoder. Gold, self answers and oracle masks are offline targets only.

## Current validation status

SOL prepares and CPU-tests this package. No GPU experiment is launched as part
of packaging. The one-case H200 smoke remains necessary to verify the new
orchestration against the installed model stack. Storage omissions are recorded
in the input package; they are not hidden by the source package's readiness.

A real CPU smoke rendered all four pages of the revised Tony Ricciardello case
with the pinned 144-DPI renderer, assembled its fixture, and passed the runner's
CPU validation. The revised question and complete four-year answer survived
through `SampleInput`. This validates input assembly, not GPU model execution.
