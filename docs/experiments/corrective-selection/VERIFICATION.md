# Repository preparation verification — 2026-09-10

## Integrity and scope

- Archived 150 retired experiment files, preserving contents and file modes.
- Verified SHA-256 for all 184 archived files/original snapshots.
- `docs/ExperimentPlan.md` is unchanged: `cfc5f54dcdc7a73ab8212e0de04f8eb9a88e68b4bdcd0e3c571b4eda48aeeb5d`.
- All `src/docprune/` library files and baseline dependency/configuration pins are unchanged.
- Changed Python test files parse successfully; their edits redirect historical fixture access.
- `git diff --check` passed. Current navigation was checked for local link targets.
- No training, inference, remote operation, commit or push was performed.

## Matched CPU regression comparison

Both runs used the original repository location, Python 3.12.13 from the available
`mineru34` environment, and identical test selection. Two small missing packages
(`word2number` and `pdf2image`) were installed only into a temporary directory and
exposed through PYTHONPATH; the user's environments were not modified.

| Checkout | Result |
| --- | --- |
| Before reorganization | 34 failed, 639 passed, 22 skipped in 40.81s |
| After reorganization | 34 failed, 639 passed, 22 skipped in 41.27s |

New failing test IDs after the reorganization: **0**.
This is a matched regression check, not a clean full-suite pass. The existing
failures include missing pinned cluster tools/artifacts, Linux-specific filesystem
behavior on macOS, missing scikit-learn, and subprocesses that replace PYTHONPATH
and consequently cannot see the temporary normalization dependency.

The selected run excluded `tests/colpali/`, `tests/qwen2vl/`,
`test_indexing.py`, `test_m3docvqa_factory.py`, `test_artifacts.py` and
`test_m3docrag.py`: model/retrieval dependencies such as colpali_engine and faiss
are unavailable. No claim of model integration or GPU validation is made.

## Existing failed tests retained

- `tests/test_experiment_design.py::test_atomic_publish_does_not_replace_destination_created_after_precheck`
- `tests/test_experiment_design.py::test_durable_development_registry_authenticates_sources_and_union`
- `tests/test_experiment_design.py::test_task9_preliminary_sealer_authenticates_sources_and_writes_24_per_stratum`
- `tests/test_m3docvqa_launchers.py::test_benchmark_runtime_pin_and_root_are_consistent_across_benchmark_docs`
- `tests/test_m3docvqa_launchers.py::test_embedded_python_heredocs_are_f821_clean`
- `tests/test_m3docvqa_launchers.py::test_handoff_submission_graph_executes_against_fake_sbatch`
- `tests/test_m3docvqa_launchers.py::test_probe_helper_uses_first_pinned_supporting_pdf_and_no_pdf_path_method`
- `tests/legacy/test_task6_analysis.py::test_analysis_cli_authenticates_gate_and_publishes_report`
- `tests/legacy/test_task6_analysis.py::test_fixed_sensitivity_cli_authenticates_trigger_file`
- `tests/legacy/test_task6_runtime.py::test_task6_isolated_run_config_drives_the_factory_checkout_validation`
- `tests/legacy/test_task7_native_boundary.py::test_native_boundary_close_failure_does_not_leak_other_descriptor_or_mask_state`
- `tests/legacy/test_task7_native_boundary.py::test_native_boundary_failed_rename_preserves_exact_fsynced_temporary`
- `tests/legacy/test_task7_native_boundary.py::test_native_boundary_fifo_source_substitution_is_nonblocking_and_rejected`
- `tests/legacy/test_task7_native_boundary.py::test_native_boundary_final_parent_replacement_reports_preserved_destination`
- `tests/legacy/test_task7_native_boundary.py::test_native_boundary_post_publish_failure_never_rolls_back_by_name`
- `tests/legacy/test_task7_native_boundary.py::test_native_boundary_post_rename_fsync_failure_reports_preserved_destination`
- `tests/legacy/test_task7_native_boundary.py::test_native_boundary_rename_success_then_raise_reports_verified_destination`
- `tests/legacy/test_task7_native_boundary.py::test_native_boundary_verifier_close_failure_cannot_return_success`
- `tests/legacy/test_task7_native_boundary.py::test_task7_l40s_probe_uses_exactly_one_cuda_visible_device_without_sigpipe`
- `tests/legacy/test_task7_native_report.py::test_native_admission_cli_exposes_all_immutable_pins`
- `tests/legacy/test_task7_report_driver.py::test_atomic_publication_never_replaces_a_competing_bundle`
- `tests/legacy/test_task7_report_driver.py::test_destination_swap_during_parent_fsync_cannot_return_success`
- `tests/legacy/test_task7_report_driver.py::test_driver_admits_exact_sealed_order_and_publishes_once`
- `tests/legacy/test_task7_report_driver.py::test_report_driver_import_is_artifact_only`
- `tests/legacy/test_task7_report_driver.py::test_snapshot_closes_early_members_before_late_shard_admission`
- `tests/legacy/test_task7_report_driver.py::test_snapshot_remains_closed_after_analysis_compilation`
- `tests/legacy/test_task7_report_driver.py::test_snapshot_uses_relative_authority_and_survives_source_mutation_at_publish`
- `tests/legacy/test_task7_report_driver.py::test_source_name_swap_is_detected_and_no_unverified_tree_is_deleted`
- `tests/legacy/test_task7_runtime.py::test_actual_pure_task7_artifact_compilation_imports_no_model_or_image_stack`
- `tests/legacy/test_task8_runtime.py::test_task8_gpu_probe_uses_the_single_cuda_visible_device_without_sigpipe`
- `tests/legacy/test_task9_attribution.py::test_mask_count_ablation_reuses_deterministic_subsets_and_canonical_reference`
- `tests/legacy/test_task9_attribution.py::test_region_mask_design_allows_fit_only_confirmation`
- `tests/legacy/test_task9_h200_mineru.py::test_gpu_monitor_accepts_owned_descendant_in_a_new_session`
- `tests/legacy/test_task9_shared_probe.py::test_shared_probe_cli_publishes_atomically_and_never_replaces`
