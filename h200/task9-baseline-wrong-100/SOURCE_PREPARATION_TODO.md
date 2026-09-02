# Remaining source/SOL preparation

These tasks run on the current source/SOL computer, not on H200. Keep all
generated data outside Git. H200 may survey and prepare its environment in
parallel, but it must wait for this completed bundle before smoke execution.

1. **Seal fixed inputs.** Run `seal_task9_confirmation_inputs.py` against the
   sealed cohort and exact cached top-4 pages/features. Require exactly 100 QIDs,
   400 pages, `retrieval_run: false`, and `global_index_loaded: false`.
2. **Run pinned MinerU.** Prepare, execute, and finalize MinerU for each QID's
   four sealed pages using commit `d9cd58add047c2364c1198eefcb1ee9cd63a971a`
   and model revision `d3f5e08d073c21466bbabe21c71bb1e9c2e595da`.
   Preserve completion manifests, raw hashes, and actual GPU identities.
3. **Capture frozen geometry.** Run
   `run_task9_preliminary_geometry_capture.py` once per QID and authenticate
   fixture, source-row, page, feature, model, and geometry identities.
4. **Build mappings.** Run `build_task9_preliminary_region_mapping.py` to create
   exactly `000-<qid>.json` through `099-<qid>.json`, with complete whole-region
   or bounded-residual assignment for every post-QTP token.
5. **Validate all inputs.** Run `run_task9_confirmation_batch.py` with
   `--validate-only` for ordinals 0–99. Require one unique mapping per QID,
   exact cohort/fixture identities, 256 fit masks, and zero holdout masks.
6. **Package the bundle.** Include the unchanged cohort, selected source rows,
   400 cached pages, persisted features, fixed fixture/QID inputs, completed
   MinerU and geometry artifacts, final mappings, and required runtime metadata.
   Exclude the global retrieval index, unrelated scratch data, and MinerU
   executables/model weights.
7. **Seal transfer provenance.** Record every relative path, byte count, and
   SHA-256 in a content-addressed manifest. Verify the bundle from that manifest
   before transfer. Add the pinned Qwen snapshot only if the H200 survey shows
   that exact revision is absent there.

After these seven steps pass, transfer the bundle to the survey-confirmed H200
artifact root. H200 then authenticates it, repeats only the cheap CPU validation,
and proceeds to smoke and production.
